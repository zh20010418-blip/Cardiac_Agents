"""
agents/agents.py —— 四类专科 Agent 实现

Agent 基类 + 四个子类：
  1. DispatcherAgent   —— 主控调度，意图解析，决定执行哪些专科 Agent
  2. ECGAgent          —— ECG 心律失常分析（调用 ecg_analyze 工具）
  3. EchoAgent         —— Echo 超声分析（调用 echo_analyze 工具）
  4. DiagnosisAgent    —— 综合诊断 + Self-Critique 校验

每个 Agent 的核心是 run(state) 方法：
  接收全局 AgentState → 处理 → 更新 state → 返回更新后的 state
"""

import json
import logging
import re
from abc import ABC, abstractmethod

from graph.state import (
    AgentState, Stage,
    ECGResult, EchoResult, DiagnosisReport, RetrievedDoc
)
from models.llm import LocalLLM
from tools.registry import ToolDispatcher, get_tool_prompt
from config.config import cfg

logger = logging.getLogger(__name__)


# ── Agent 基类 ────────────────────────────────────────────────────────────────
class BaseAgent(ABC):
    """
    所有 Agent 的基类。

    子类只需实现 run() 方法，基类提供：
      - LLM 访问（self.llm）
      - 工具调用（self.dispatcher）
      - ReAct 推理循环（self._react_loop）
    """

    def __init__(self, llm: LocalLLM):
        self.llm = llm
        self.dispatcher = ToolDispatcher()

    @abstractmethod
    def run(self, state: AgentState) -> AgentState:
        """处理 state 并返回更新后的 state。"""
        ...

    def _react_loop(
        self,
        system_prompt: str,
        user_message: str,
        history: list[dict],
        max_steps: int = 5,
    ) -> str:
        """
        ReAct（Reasoning + Acting）推理循环。

        流程：
          1. LLM 输出 Thought（推理）或 Action（工具调用 JSON）
          2. 如果是工具调用 → 执行工具 → 把 Observation 追加到历史
          3. 继续让 LLM 推理，直到 LLM 输出最终答案（不含工具调用）
          4. 最多循环 max_steps 次，防止死循环

        Returns:
            LLM 最终输出的文本（不含工具调用指令）
        """
        messages = history + [{"role": "user", "content": user_message}]

        for step in range(max_steps):
            response = self.llm.generate(messages, system_prompt=system_prompt)
            logger.debug(f"ReAct step {step+1}: {response[:100]}...")

            # 尝试解析工具调用
            has_call, observation = self.dispatcher.run(response)

            if not has_call:
                # 没有工具调用，说明 LLM 已给出最终答案
                return response

            # 有工具调用：把工具结果作为新消息追加，继续推理
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Observation:\n{observation}\n\n请继续分析。"
            })
            logger.info(f"工具调用完成，继续推理 (step {step+1}/{max_steps})")

        # 超过最大步数，返回最后一次输出
        logger.warning(f"ReAct 循环超过最大步数 {max_steps}，返回最后输出")
        return response

    def _parse_json_output(self, text: str) -> dict:
        """从 LLM 输出中提取 JSON（兼容带前后文字的情况）。"""
        # 优先提取 ```json 代码块
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # fallback: 提取第一个 { ... }
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {}


# ── Agent 1：主控调度 ─────────────────────────────────────────────────────────
class DispatcherAgent(BaseAgent):
    """
    主控调度 Agent。

    职责：
      1. 解析用户输入，提取患者信息
      2. 判断需要执行哪些专科 Agent（有无 ECG/Echo 数据）
      3. 发现信息不足时，生成追问问题
    """

    SYSTEM_PROMPT = """你是心脏科诊断系统的主控调度器。

你的任务：
1. 分析用户提供的患者信息，判断数据完整性
2. 如果信息不足，列出需要补充的内容
3. 提取结构化的患者基本信息

请以 JSON 格式输出：
```json
{
  "has_ecg": true/false,
  "has_echo": true/false,
  "needs_clarification": true/false,
  "clarification_questions": ["问题1", "问题2"],
  "patient_summary": "患者简要描述",
  "urgency": "urgent/routine"
}
```
"""

    def run(self, state: AgentState) -> AgentState:
        logger.info("=== DispatcherAgent 开始 ===")

        # 构造 prompt：把所有输入信息汇总
        user_content = f"""
患者信息: {json.dumps(state.patient_info, ensure_ascii=False)}
ECG 图像: {state.ecg_image_paths if state.ecg_image_paths else '未提供'}
Echo 数据: {state.echo_paths if state.echo_paths else '未提供'}
用户问题: {state.user_query}
""".strip()

        response = self.llm.generate(
            messages=[{"role": "user", "content": user_content}],
            system_prompt=self.SYSTEM_PROMPT,
        )

        parsed = self._parse_json_output(response)

        # 根据解析结果更新 state
        state.skip_echo = not bool(state.echo_paths) or not parsed.get("has_echo", True)
        state.needs_clarification = parsed.get("needs_clarification", False)
        state.clarification_questions = parsed.get("clarification_questions", [])

        # 紧急病例直接跳过追问
        if parsed.get("urgency") == "urgent":
            state.needs_clarification = False
            logger.warning("检测到紧急病例，跳过追问，直接进入分析流程")

        state.stage = Stage.ECG_ANALYSIS
        logger.info(f"调度完成: ECG={'是' if not state.skip_echo else '否'}, "
                    f"Echo={'否' if state.skip_echo else '是'}, "
                    f"追问={state.needs_clarification}")
        return state


# ── Agent 2：ECG 专科 ─────────────────────────────────────────────────────────
class ECGAgent(BaseAgent):
    """
    ECG 专科 Agent。

    用 ReAct 循环：先调用 ecg_analyze 工具获取模型预测，
    再用 LLM 对原始结果做临床解读，生成自然语言摘要。
    """

    SYSTEM_PROMPT = """你是心脏科 ECG 专家。

你有以下工具可用：
{tool_prompt}

分析步骤：
1. 调用 ecg_analyze 工具获取 ECG 机器分析结果
2. 结合患者年龄、性别、主诉，对结果进行临床解读
3. 判断是否存在高危心律（心室颤动、室速、三度房室传导阻滞等）
4. 输出临床意见，不要重复机器输出的数字，要给出临床意义

注意：你的输出会直接传给诊断综合 Agent，请清晰、专业。
"""

    def run(self, state: AgentState) -> AgentState:
        logger.info("=== ECGAgent 开始 ===")

        if not state.ecg_image_paths:
            logger.info("无 ECG 数据，跳过")
            state.ecg_result = ECGResult(summary="未提供 ECG 数据")
            state.stage = Stage.ECHO_ANALYSIS
            return state

        system = self.SYSTEM_PROMPT.format(tool_prompt=get_tool_prompt())
        patient_ctx = json.dumps(state.patient_info, ensure_ascii=False)
        user_msg = f"""
患者信息: {patient_ctx}
ECG 图像路径: {state.ecg_image_paths[0]}
患者年龄: {state.patient_info.get('age', 50)}

请分析该患者的 ECG，给出心律失常诊断和临床解读。
""".strip()

        # ReAct 循环：LLM 会自动调用 ecg_analyze 工具
        analysis = self._react_loop(
            system_prompt=system,
            user_message=user_msg,
            history=[],
            max_steps=3,
        )

        # 解析工具结果（从 dispatcher 缓存中取，或解析 LLM 输出）
        # 这里简化：直接调用工具获取原始结果
        from tools.medical_tools import ecg_analyze
        raw = ecg_analyze(
            image_path=state.ecg_image_paths[0],
            patient_age=state.patient_info.get("age", 50)
        )

        state.ecg_result = ECGResult(
            arrhythmia_labels=raw.get("arrhythmia_labels", []),
            confidence_scores=raw.get("confidence_scores", {}),
            max_confidence=raw.get("max_confidence", 0.0),
            is_high_risk=raw.get("is_high_risk", False),
            summary=analysis,  # LLM 的临床解读
        )

        # 决定是否触发 RAG（低置信度或高危时触发）
        if raw.get("max_confidence", 1.0) < cfg.rag.confidence_threshold or raw.get("is_high_risk"):
            state.trigger_rag = True

        state.stage = Stage.ECHO_ANALYSIS
        logger.info(f"ECG 分析完成: {raw.get('arrhythmia_labels')}")
        return state


# ── Agent 3：Echo 专科 ────────────────────────────────────────────────────────
class EchoAgent(BaseAgent):
    """Echo 超声专科 Agent，逻辑与 ECGAgent 类似。"""

    SYSTEM_PROMPT = """你是心脏超声（Echo）专家。

你有以下工具可用：
{tool_prompt}

分析步骤：
1. 调用 echo_analyze 工具获取超声分析结果
2. 重点关注 LVEF（左室射血分数）、室壁运动异常、瓣膜病变
3. 结合 ECG 发现（如已有），综合评估心功能状态
4. 给出心功能分级（建议参考 NYHA 分级）
"""

    def run(self, state: AgentState) -> AgentState:
        logger.info("=== EchoAgent 开始 ===")

        if state.skip_echo or not state.echo_paths:
            logger.info("无 Echo 数据或已跳过")
            state.echo_result = EchoResult(summary="未提供 Echo 数据")
            state.stage = Stage.RAG_RETRIEVAL
            return state

        system = self.SYSTEM_PROMPT.format(tool_prompt=get_tool_prompt())

        # 把 ECG 结果作为上下文传给 Echo Agent
        ecg_context = ""
        if state.ecg_result:
            ecg_context = f"\nECG 分析结果（供参考）：{state.ecg_result.summary}"

        user_msg = f"""
患者信息: {json.dumps(state.patient_info, ensure_ascii=False)}
Echo 路径: {state.echo_paths[0]}
{ecg_context}

请分析该患者的心脏超声，评估心功能状态。
""".strip()

        analysis = self._react_loop(
            system_prompt=system,
            user_message=user_msg,
            history=[],
            max_steps=3,
        )

        from tools.medical_tools import echo_analyze
        raw = echo_analyze(echo_path=state.echo_paths[0])

        state.echo_result = EchoResult(
            lvef=raw.get("lvef"),
            wall_motion_abnormalities=raw.get("wall_motion_abnormalities", []),
            structural_findings=raw.get("structural_findings", []),
            confidence=raw.get("lvef_confidence", 0.0),
            is_severe=raw.get("is_severe", False),
            summary=analysis,
        )

        # LVEF 严重降低时强制触发 RAG
        if raw.get("lvef", 1.0) < 0.35:
            state.trigger_rag = True
            logger.warning(f"LVEF={raw.get('lvef'):.0%}，严重降低，强制触发 RAG")

        state.stage = Stage.RAG_RETRIEVAL
        logger.info(f"Echo 分析完成: LVEF={raw.get('lvef')}")
        return state


# ── Agent 4：诊断综合 + Self-Critique ─────────────────────────────────────────
class DiagnosisAgent(BaseAgent):
    """
    诊断综合 Agent。

    职责：
      1. 汇总 ECG + Echo + RAG 结果，通过 CoT 推理生成结构化诊断报告
      2. Self-Critique：对自己的诊断进行一致性自检，发现逻辑矛盾时修正
    """

    DIAGNOSIS_SYSTEM = """你是高级心脏科主治医师，负责综合诊断。

请基于以下信息，进行系统性诊断推理（Chain-of-Thought）：

{patient_history}

诊断步骤：
1. 整合 ECG 和 Echo 发现，识别核心异常
2. 结合患者基本信息（年龄、既往史、危险因素）分析
3. 参考指南建议（如有），制定诊断意见
4. 评估整体诊断置信度

请以如下 JSON 格式输出：
```json
{{
  "primary_diagnosis": "主诊断（中文）",
  "differential_diagnosis": ["鉴别诊断1", "鉴别诊断2"],
  "evidence": ["支撑证据1", "支撑证据2", "支撑证据3"],
  "treatment_recommendations": ["建议1", "建议2"],
  "follow_up": "随访建议",
  "overall_confidence": 0.0~1.0,
  "reasoning": "推理过程简述（供自检使用）"
}}
```
"""

    CRITIQUE_SYSTEM = """你是医疗质控专家，负责审核诊断报告的逻辑一致性。

检查以下问题：
1. 诊断与 ECG/Echo 证据是否一致？
2. 置信度评估是否合理？
3. 高危情况是否有对应的紧急处理建议？
4. 是否存在明显遗漏的鉴别诊断？

如果诊断合理，输出：{{"passed": true, "notes": ""}}
如果发现问题，输出：{{"passed": false, "notes": "具体问题描述", "corrections": "建议修正方向"}}
"""

    def run(self, state: AgentState) -> AgentState:
        logger.info("=== DiagnosisAgent 开始 ===")

        # ── 1. 构建诊断输入 ──────────────────────────────────────────────────
        context_parts = [
            f"**患者基本信息**: {json.dumps(state.patient_info, ensure_ascii=False)}",
        ]

        if state.ecg_result:
            context_parts.append(f"**ECG 分析结果**:\n{state.ecg_result.summary}")
            if state.ecg_result.arrhythmia_labels:
                context_parts.append(f"  检出心律: {', '.join(state.ecg_result.arrhythmia_labels)}")

        if state.echo_result:
            context_parts.append(f"**Echo 分析结果**:\n{state.echo_result.summary}")
            if state.echo_result.lvef is not None:
                context_parts.append(f"  LVEF: {state.echo_result.lvef:.0%}")

        if state.retrieved_docs:
            from rag.retriever import FAISSRetriever
            doc_text = "\n\n".join(
                f"[{doc.source}]\n{doc.content}" for doc in state.retrieved_docs
            )
            context_parts.append(f"**相关指南参考**:\n{doc_text}")

        full_context = "\n\n".join(context_parts)

        # ── 2. 主诊断推理 ────────────────────────────────────────────────────
        system = self.DIAGNOSIS_SYSTEM.format(
            patient_history=state.patient_info.get("_history_summary", "无历史记录")
        )

        response = self.llm.generate(
            messages=[{"role": "user", "content": full_context}],
            system_prompt=system,
        )

        parsed = self._parse_json_output(response)
        if not parsed:
            # LLM 输出格式异常，用默认值
            logger.warning("诊断输出 JSON 解析失败，使用原始文本")
            parsed = {
                "primary_diagnosis": response[:200],
                "overall_confidence": 0.5,
            }

        report = DiagnosisReport(
            primary_diagnosis=parsed.get("primary_diagnosis", ""),
            differential_diagnosis=parsed.get("differential_diagnosis", []),
            evidence=parsed.get("evidence", []),
            treatment_recommendations=parsed.get("treatment_recommendations", []),
            follow_up=parsed.get("follow_up", ""),
            overall_confidence=parsed.get("overall_confidence", 0.5),
        )

        # ── 3. Self-Critique 自检 ────────────────────────────────────────────
        report = self._self_critique(report, full_context, state)

        state.diagnosis_report = report
        state.stage = Stage.DONE
        logger.info(f"诊断完成: {report.primary_diagnosis} (置信度: {report.overall_confidence:.0%})")
        return state

    def _self_critique(
        self,
        report: DiagnosisReport,
        context: str,
        state: AgentState,
    ) -> DiagnosisReport:
        """
        Self-Critique：让另一个 LLM 调用审核诊断逻辑。

        实现思路：
          - 把诊断报告 + 原始证据一起喂给 LLM
          - 让 LLM 扮演质控专家，检查逻辑一致性
          - 如果发现问题，把修正意见喂回给诊断 LLM 重新生成
          - 最多重试 max_critique_rounds 次
        """
        max_rounds = cfg.agent.max_critique_rounds

        for round_i in range(max_rounds):
            state.critique_round = round_i + 1
            logger.info(f"Self-Critique 第 {round_i+1} 轮")

            critique_input = f"""
## 原始证据
{context}

## 当前诊断报告
主诊断: {report.primary_diagnosis}
鉴别诊断: {report.differential_diagnosis}
支撑证据: {report.evidence}
治疗建议: {report.treatment_recommendations}
置信度: {report.overall_confidence:.0%}

请审核此诊断报告是否存在逻辑问题。
""".strip()

            critique_response = self.llm.generate(
                messages=[{"role": "user", "content": critique_input}],
                system_prompt=self.CRITIQUE_SYSTEM,
            )

            critique_result = self._parse_json_output(critique_response)

            if critique_result.get("passed", True):
                logger.info(f"Self-Critique 第 {round_i+1} 轮通过")
                report.critique_passed = True
                report.critique_notes = critique_result.get("notes", "")
                break

            # 未通过：把修正意见喂回重新生成
            corrections = critique_result.get("corrections", "")
            logger.info(f"Self-Critique 发现问题: {corrections[:100]}")
            report.critique_notes = critique_result.get("notes", "")

            # 重新生成（带修正意见）
            revised_response = self.llm.generate(
                messages=[{
                    "role": "user",
                    "content": f"{context}\n\n## 质控意见（请据此修正）\n{corrections}"
                }],
                system_prompt=self.DIAGNOSIS_SYSTEM.format(patient_history=""),
            )

            revised = self._parse_json_output(revised_response)
            if revised:
                report.primary_diagnosis = revised.get("primary_diagnosis", report.primary_diagnosis)
                report.differential_diagnosis = revised.get("differential_diagnosis", report.differential_diagnosis)
                report.evidence = revised.get("evidence", report.evidence)
                report.treatment_recommendations = revised.get("treatment_recommendations", report.treatment_recommendations)
                report.overall_confidence = revised.get("overall_confidence", report.overall_confidence)
        else:
            # 达到最大轮次仍未通过，标记但继续输出
            report.critique_passed = False
            logger.warning(f"Self-Critique 达到最大轮次 {max_rounds}，保留当前结果")

        return report