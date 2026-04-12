"""
graph/pipeline.py —— Agent 状态机调度器

这是整个系统的"大脑"，负责：
  1. 按照 Stage 枚举依次调度各专科 Agent
  2. 处理异常和超时
  3. 触发 RAG 检索（在 ECG/Echo 分析后，诊断前）
  4. 管理多轮对话（检测到追问时暂停等待用户输入）

状态机流转：
  INIT → ECG_ANALYSIS → ECHO_ANALYSIS → RAG_RETRIEVAL → DIAGNOSIS → CRITIQUE → DONE
                                                                        ↑__↓（最多2轮）
"""

import logging
import time
import uuid

from graph.state import AgentState, Stage
from agents.agents import DispatcherAgent, ECGAgent, EchoAgent, DiagnosisAgent
from memory.memory import MemoryManager
from rag.retriever import FAISSRetriever
from models.llm import LocalLLM
from config.config import cfg

logger = logging.getLogger(__name__)


class CardiacDiagnosisPipeline:
    """
    心脏病诊断 Agent 流水线。

    用法：
        pipeline = CardiacDiagnosisPipeline()
        result = pipeline.run(
            patient_info={"age": 62, "gender": "男", "chief_complaint": "胸闷"},
            ecg_image_paths=["/data/ecg/001.png"],
            echo_paths=["/data/echo/001.mp4"],
            user_query="请帮我分析这位患者的心脏情况",
        )
        print(result.diagnosis_report.primary_diagnosis)
    """

    def __init__(self):
        logger.info("初始化 CardiacDiagnosisPipeline...")

        # 加载本地 LLM（单例，只加载一次）
        self.llm = LocalLLM(
            model_path=cfg.model.llm_path,
            device=cfg.model.llm_device,
            dtype=cfg.model.llm_dtype,
            max_new_tokens=cfg.model.llm_max_new_tokens,
            temperature=cfg.model.llm_temperature,
            do_sample=cfg.model.llm_do_sample,
        )

        # 初始化各 Agent
        self.dispatcher = DispatcherAgent(self.llm)
        self.ecg_agent = ECGAgent(self.llm)
        self.echo_agent = EchoAgent(self.llm)
        self.diagnosis_agent = DiagnosisAgent(self.llm)

        # 初始化 RAG 检索器
        self.retriever = FAISSRetriever(
            embed_model_path=cfg.model.embed_model_path,
            index_path=cfg.rag.index_path,
            guidelines_dir=cfg.rag.guidelines_dir,
            top_k=cfg.rag.top_k,
            embed_device=cfg.model.embed_device,
            confidence_threshold=cfg.rag.confidence_threshold,
        )

        logger.info("Pipeline 初始化完成")

    def run(
        self,
        patient_info: dict,
        ecg_image_paths: list[str] = None,
        echo_paths: list[str] = None,
        user_query: str = "",
        patient_id: str = "",
        session_id: str = None,
    ) -> AgentState:
        """
        执行完整的诊断流水线。

        Args:
            patient_info   : 患者基本信息字典
            ecg_image_paths: ECG 图像路径列表
            echo_paths     : Echo 视频/图像路径列表
            user_query     : 用户原始问题
            patient_id     : 患者唯一ID（用于长期记忆）
            session_id     : 会话ID（不传则自动生成）

        Returns:
            处理完成的 AgentState，通过 .diagnosis_report 获取报告
        """
        start_time = time.time()
        session_id = session_id or str(uuid.uuid4())[:8]

        logger.info(f"开始诊断流水线 | session={session_id} | patient={patient_id}")

        # ── 初始化状态 ──────────────────────────────────────────────────────
        state = AgentState(
            patient_info=patient_info,
            ecg_image_paths=ecg_image_paths or [],
            echo_paths=echo_paths or [],
            user_query=user_query,
            session_id=session_id,
            stage=Stage.INIT,
        )

        # ── 初始化记忆管理器 ────────────────────────────────────────────────
        memory = MemoryManager(
            session_id=session_id,
            patient_id=patient_id,
            window_size=cfg.agent.short_term_window,
        )
        memory.add_user_message(user_query)

        # 注入患者历史记录到 patient_info（诊断 Agent 会读取）
        history_summary = memory.get_patient_context()
        if history_summary:
            state.patient_info["_history_summary"] = history_summary

        # ── 状态机主循环 ────────────────────────────────────────────────────
        try:
            state = self._run_pipeline(state, memory)
        except Exception as e:
            logger.error(f"Pipeline 执行异常: {e}", exc_info=True)
            state.stage = Stage.ERROR
            state.error_message = str(e)
            return state

        # ── 保存诊断结果到长期记忆 ──────────────────────────────────────────
        if state.stage == Stage.DONE and state.diagnosis_report:
            key_findings = []
            if state.ecg_result:
                key_findings.extend(state.ecg_result.arrhythmia_labels)
            if state.echo_result and state.echo_result.wall_motion_abnormalities:
                key_findings.extend(state.echo_result.wall_motion_abnormalities)

            memory.save_diagnosis(
                diagnosis=state.diagnosis_report.primary_diagnosis,
                confidence=state.diagnosis_report.overall_confidence,
                patient_info=patient_info,
                key_findings=key_findings,
            )

        elapsed = time.time() - start_time
        logger.info(f"诊断流水线完成 | 耗时: {elapsed:.1f}s | 阶段: {state.stage}")
        return state

    def _run_pipeline(self, state: AgentState, memory: MemoryManager) -> AgentState:
        """状态机核心：按 Stage 依次执行各 Agent。"""

        # ── Stage 1: 调度 ────────────────────────────────────────────────────
        state = self.dispatcher.run(state)

        # 如果需要追问，暂停流程返回追问内容
        if state.needs_clarification and state.clarification_questions:
            clarification_text = memory.needs_clarification_prompt(state.clarification_questions)
            memory.add_assistant_message(clarification_text)
            logger.info(f"需要追问，暂停流程: {state.clarification_questions}")
            # 实际多轮对话中，这里应该返回给前端等待用户输入
            # 这里为了演示，跳过追问继续执行
            logger.info("（演示模式：跳过追问，继续执行）")

        # ── Stage 2: ECG 分析 ────────────────────────────────────────────────
        if state.stage == Stage.ECG_ANALYSIS:
            state = self.ecg_agent.run(state)
            if state.ecg_result:
                memory.add_tool_result("ecg_analyze", state.ecg_result.summary)

        # ── Stage 3: Echo 分析 ───────────────────────────────────────────────
        if state.stage == Stage.ECHO_ANALYSIS:
            state = self.echo_agent.run(state)
            if state.echo_result:
                memory.add_tool_result("echo_analyze", state.echo_result.summary)

        # ── Stage 4: RAG 检索 ────────────────────────────────────────────────
        if state.stage == Stage.RAG_RETRIEVAL:
            state = self._run_rag(state)

        # ── Stage 5: 综合诊断 ────────────────────────────────────────────────
        if state.stage in (Stage.RAG_RETRIEVAL, Stage.DIAGNOSIS):
            state.stage = Stage.DIAGNOSIS
            state = self.diagnosis_agent.run(state)

        return state

    def _run_rag(self, state: AgentState) -> AgentState:
        """执行 RAG 检索，决定是否触发由置信度驱动。"""
        logger.info("=== RAG 检索开始 ===")

        # 构建查询：把 ECG + Echo 的关键发现组合成查询文本
        query_parts = []
        if state.ecg_result and state.ecg_result.arrhythmia_labels:
            query_parts.extend(state.ecg_result.arrhythmia_labels)
        if state.echo_result:
            if state.echo_result.wall_motion_abnormalities:
                query_parts.extend(state.echo_result.wall_motion_abnormalities)
            if state.echo_result.lvef is not None and state.echo_result.lvef < 0.5:
                query_parts.append(f"左室射血分数降低 LVEF {state.echo_result.lvef:.0%}")

        if not query_parts:
            query_parts = [state.user_query]

        query = " ".join(query_parts)

        # 计算综合置信度（ECG 和 Echo 置信度的加权平均）
        confidences = []
        if state.ecg_result:
            confidences.append(state.ecg_result.max_confidence)
        if state.echo_result:
            confidences.append(state.echo_result.confidence)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # 自适应检索：低置信度或标记触发时才检索
        docs = self.retriever.retrieve(
            query=query,
            confidence=avg_confidence if not state.trigger_rag else None,
            force=state.trigger_rag,
        )

        state.retrieved_docs = docs
        state.stage = Stage.DIAGNOSIS

        if docs:
            logger.info(f"RAG 检索完成，获取 {len(docs)} 条指南参考")
        else:
            logger.info("RAG 跳过或无结果（置信度充足）")

        return state

    def format_report(self, state: AgentState) -> str:
        """将 AgentState 格式化为可读的诊断报告文本。"""
        if state.stage == Stage.ERROR:
            return f"❌ 诊断流程发生错误：{state.error_message}"

        if not state.diagnosis_report:
            return "⚠️ 诊断未完成"

        r = state.diagnosis_report
        lines = [
            "=" * 60,
            "          心脏病辅助诊断报告",
            "=" * 60,
            "",
            f"【主诊断】{r.primary_diagnosis}",
            f"【诊断置信度】{r.overall_confidence:.0%}",
            "",
        ]

        if r.differential_diagnosis:
            lines.append("【鉴别诊断】")
            for d in r.differential_diagnosis:
                lines.append(f"  • {d}")
            lines.append("")

        if r.evidence:
            lines.append("【支撑证据】")
            for e in r.evidence:
                lines.append(f"  • {e}")
            lines.append("")

        if r.treatment_recommendations:
            lines.append("【治疗建议】")
            for t in r.treatment_recommendations:
                lines.append(f"  • {t}")
            lines.append("")

        if r.follow_up:
            lines.append(f"【随访建议】{r.follow_up}")
            lines.append("")

        # 质控信息
        critique_status = "✅ 通过" if r.critique_passed else "⚠️ 未通过（已标注）"
        lines.append(f"【Self-Critique 校验】{critique_status}")
        if r.critique_notes:
            lines.append(f"  备注: {r.critique_notes}")

        lines.append("")
        lines.append("⚠️  本报告仅供临床参考，最终诊断需由执业医师确认。")
        lines.append("=" * 60)

        return "\n".join(lines)

if __name__ == "__main__":
    # 演示模式：使用模拟数据跑通流程
    pipeline = CardiacDiagnosisPipeline()
    result = pipeline.run(
        patient_info={"age": 62, "gender": "男", "chief_complaint": "胸闷"},
        ecg_image_paths=["/data/ecg/001.png"],
        echo_paths=["/data/echo/001.mp4"],
        user_query="请帮我分析这位患者的心脏情况",
    )
    
    print(result.diagnosis_report.primary_diagnosis)
