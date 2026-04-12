"""
agents/agents.py —— 四类专科 Agent 实现

工具归属：
  DispatcherAgent : parse_patient_info、assess_vitals、patient_risk_score
  ECGAgent        : ecg_analyze、ecg_interval_measure
  EchoAgent       : echo_analyze、echo_wall_motion_detail
  DiagnosisAgent  : guideline_query、drug_recommendation
"""

import json
import logging
import re
from abc import ABC, abstractmethod

from graph.state import (
    AgentState, Stage,
    ECGResult, EchoResult, DiagnosisReport,
)
from models.llm import LocalLLM
from tools.registry import ToolDispatcher, get_tool_prompt
from config.config import cfg

logger = logging.getLogger(__name__)


# ── Agent 基类 ────────────────────────────────────────────────────────────────
class BaseAgent(ABC):
    def __init__(self, llm: LocalLLM):
        self.llm = llm
        self.dispatcher = ToolDispatcher()

    @abstractmethod
    def run(self, state: AgentState) -> AgentState: ...

    def _react_loop(self, system_prompt: str, user_message: str,
                    history: list, max_steps: int = 5) -> str:
        """ReAct 推理循环：思考 → 工具调用 → 观察 → 继续推理。"""
        messages = history + [{"role": "user", "content": user_message}]
        for step in range(max_steps):
            response = self.llm.generate(messages, system_prompt=system_prompt)
            logger.debug(f"ReAct step {step+1}: {response[:100]}...")
            has_call, observation = self.dispatcher.run(response)
            if not has_call:
                return response
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation:\n{observation}\n\n请继续分析。"})
            logger.info(f"工具调用完成，继续推理 (step {step+1}/{max_steps})")
        logger.warning(f"ReAct 超过最大步数 {max_steps}")
        return response

    def _parse_json_output(self, text: str) -> dict:
        """从 LLM 输出中提取 JSON，三级兜底策略。"""
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
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
    专属工具：parse_patient_info、assess_vitals、patient_risk_score
    """

    SYSTEM_PROMPT = """你是心脏科诊断系统的主控调度器。

你有以下工具可用：
{tool_prompt}

工作步骤：
1. 调用 parse_patient_info 提取结构化患者信息
2. 如果有生命体征数据，调用 assess_vitals 评估异常指标
3. 调用 patient_risk_score 计算风险评分，决定分诊优先级
4. 根据以上信息判断数据完整性，决定是否需要追问

请以 JSON 格式输出：
```json
{{
  "has_ecg": true/false,
  "has_echo": true/false,
  "needs_clarification": true/false,
  "clarification_questions": ["问题1"],
  "patient_summary": "患者简要描述",
  "urgency": "urgent/routine"
}}
```
"""

    def run(self, state: AgentState) -> AgentState:
        logger.info("=== DispatcherAgent 开始 ===")

        user_content = f"""
患者信息: {json.dumps(state.patient_info, ensure_ascii=False)}
ECG 图像: {state.ecg_image_paths if state.ecg_image_paths else '未提供'}
Echo 数据: {state.echo_paths if state.echo_paths else '未提供'}
用户问题: {state.user_query}
""".strip()

        # ReAct 循环：自动调用 parse_patient_info、assess_vitals、patient_risk_score
        response = self._react_loop(
            system_prompt=self.SYSTEM_PROMPT.format(tool_prompt=get_tool_prompt()),
            user_message=user_content,
            history=[],
            max_steps=4,
        )

        # 直接调用风险评分工具，结果写入 state 供后续 Agent 参考
        from tools.medical_tools import patient_risk_score
        risk = patient_risk_score(state.patient_info, "GRACE")
        state.patient_info["_risk_score"] = risk
        logger.info(f"风险评分: {risk['score_type']} {risk['score']} ({risk['risk_level']})")

        parsed = self._parse_json_output(response)
        state.skip_echo = not bool(state.echo_paths) or not parsed.get("has_echo", True)
        state.needs_clarification = parsed.get("needs_clarification", False)
        state.clarification_questions = parsed.get("clarification_questions", [])

        # 高危病例跳过追问
        if parsed.get("urgency") == "urgent" or risk.get("urgent", False):
            state.needs_clarification = False
            logger.warning("检测到紧急病例，跳过追问")

        state.stage = Stage.ECG_ANALYSIS
        logger.info(f"调度完成: skip_echo={state.skip_echo}, 追问={state.needs_clarification}")
        return state


# ── Agent 2：ECG 专科 ─────────────────────────────────────────────────────────
class ECGAgent(BaseAgent):
    """
    ECG 专科 Agent。
    专属工具：ecg_analyze（心律失常分类）、ecg_interval_measure（间期测量）

    两个工具分工：
      ecg_analyze         → 识别心律类型和置信度
      ecg_interval_measure → 测量 PR/QRS/QTc，判断传导阻滞等
    """

    SYSTEM_PROMPT = """你是心脏科 ECG 专家。

你有以下工具可用：
{tool_prompt}

分析步骤：
1. 调用 ecg_analyze 工具识别心律失常类型和置信度
2. 调用 ecg_interval_measure 工具测量 PR/QRS/QTc 间期，判断传导异常
3. 综合两个工具结果，结合患者年龄和主诉给出完整临床解读
4. 明确说明是否存在高危心律或传导异常

注意：两个工具都必须调用，分别提供心律分类和间期测量两个维度的信息。
"""

    def run(self, state: AgentState) -> AgentState:
        logger.info("=== ECGAgent 开始 ===")

        if not state.ecg_image_paths:
            logger.info("无 ECG 数据，跳过")
            state.ecg_result = ECGResult(summary="未提供 ECG 数据")
            state.stage = Stage.ECHO_ANALYSIS
            return state

        system = self.SYSTEM_PROMPT.format(tool_prompt=get_tool_prompt())
        user_msg = f"""
患者信息: {json.dumps(state.patient_info, ensure_ascii=False)}
ECG 图像路径: {state.ecg_image_paths[0]}
患者年龄: {state.patient_info.get('age', 50)}

请依次调用 ecg_analyze 和 ecg_interval_measure 两个工具，完成完整的 ECG 分析。
""".strip()

        # ReAct 循环：自动调用 ecg_analyze + ecg_interval_measure
        analysis = self._react_loop(
            system_prompt=system,
            user_message=user_msg,
            history=[],
            max_steps=5,
        )

        # 直接调用两个工具获取结构化结果
        from tools.medical_tools import ecg_analyze, ecg_interval_measure
        raw = ecg_analyze(
            image_path=state.ecg_image_paths[0],
            patient_age=state.patient_info.get("age", 50)
        )
        interval = ecg_interval_measure(
            image_path=state.ecg_image_paths[0],
            heart_rate=state.patient_info.get("heart_rate", 75)
        )

        state.ecg_result = ECGResult(
            arrhythmia_labels=raw.get("arrhythmia_labels", []),
            confidence_scores=raw.get("confidence_scores", {}),
            max_confidence=raw.get("max_confidence", 0.0),
            is_high_risk=raw.get("is_high_risk", False) or bool(interval.get("abnormal_intervals")),
            summary=analysis,
        )

        # 间期异常写入 state 供 DiagnosisAgent 参考
        if interval.get("abnormal_intervals"):
            state.patient_info["_ecg_interval_abnormal"] = interval["abnormal_intervals"]

        if raw.get("max_confidence", 1.0) < cfg.rag.confidence_threshold or state.ecg_result.is_high_risk:
            state.trigger_rag = True

        state.stage = Stage.ECHO_ANALYSIS
        logger.info(f"ECG 分析完成: {raw.get('arrhythmia_labels')}, "
                    f"间期异常: {interval.get('abnormal_intervals')}")
        return state


# ── Agent 3：Echo 专科 ────────────────────────────────────────────────────────
class EchoAgent(BaseAgent):
    """
    Echo 超声专科 Agent。
    专属工具：echo_analyze（整体评估）、echo_wall_motion_detail（逐节段评分）

    两个工具分工：
      echo_analyze            → 整体 LVEF + 结构异常
      echo_wall_motion_detail → AHA 17节段逐节段评分，推断受累冠脉
    """

    SYSTEM_PROMPT = """你是心脏超声（Echo）专家。

你有以下工具可用：
{tool_prompt}

分析步骤：
1. 调用 echo_analyze 工具获取整体 LVEF 和结构异常
2. 调用 echo_wall_motion_detail 工具进行 AHA 17节段逐节段室壁运动评分
3. 根据异常节段推断受累冠脉（前壁→LAD，下壁→RCA，侧壁→LCX）
4. 结合 ECG 结果综合评估，给出 NYHA 心功能分级建议

注意：两个工具都必须调用，节段信息对鉴别诊断至关重要。
"""

    def run(self, state: AgentState) -> AgentState:
        logger.info("=== EchoAgent 开始 ===")

        if state.skip_echo or not state.echo_paths:
            logger.info("无 Echo 数据或已跳过")
            state.echo_result = EchoResult(summary="未提供 Echo 数据")
            state.stage = Stage.RAG_RETRIEVAL
            return state

        system = self.SYSTEM_PROMPT.format(tool_prompt=get_tool_prompt())

        ecg_context = ""
        if state.ecg_result:
            ecg_context = f"\nECG 分析结果（供参考）：{state.ecg_result.summary}"
            if state.patient_info.get("_ecg_interval_abnormal"):
                ecg_context += f"\nECG 间期异常：{state.patient_info['_ecg_interval_abnormal']}"

        user_msg = f"""
患者信息: {json.dumps(state.patient_info, ensure_ascii=False)}
Echo 路径: {state.echo_paths[0]}
{ecg_context}

请依次调用 echo_analyze 和 echo_wall_motion_detail 两个工具，完成完整的超声分析。
""".strip()

        # ReAct 循环：自动调用 echo_analyze + echo_wall_motion_detail
        analysis = self._react_loop(
            system_prompt=system,
            user_message=user_msg,
            history=[],
            max_steps=5,
        )

        from tools.medical_tools import echo_analyze, echo_wall_motion_detail
        raw = echo_analyze(echo_path=state.echo_paths[0])
        wall_detail = echo_wall_motion_detail(echo_path=state.echo_paths[0])

        # 合并两个工具的室壁运动结果并去重
        wall_abnormal = list(dict.fromkeys(
            raw.get("wall_motion_abnormalities", []) +
            wall_detail.get("abnormal_segments", [])
        ))

        state.echo_result = EchoResult(
            lvef=raw.get("lvef"),
            wall_motion_abnormalities=wall_abnormal,
            structural_findings=raw.get("structural_findings", []),
            confidence=raw.get("lvef_confidence", 0.0),
            is_severe=raw.get("is_severe", False),
            summary=analysis,
        )

        # 受累冠脉写入 state 供 DiagnosisAgent 参考
        if wall_detail.get("suspected_culprit_vessels"):
            state.patient_info["_culprit_vessels"] = wall_detail["suspected_culprit_vessels"]
            logger.info(f"受累冠脉推断: {wall_detail['suspected_culprit_vessels']}")

        if raw.get("lvef", 1.0) < 0.35:
            state.trigger_rag = True
            logger.warning(f"LVEF={raw.get('lvef'):.0%}，严重降低，强制触发 RAG")

        state.stage = Stage.RAG_RETRIEVAL
        logger.info(f"Echo 分析完成: LVEF={raw.get('lvef')}, "
                    f"WMSI={wall_detail.get('wall_motion_score_index')}")
        return state


# ── Agent 4：诊断综合 + Self-Critique ─────────────────────────────────────────
class DiagnosisAgent(BaseAgent):
    """
    诊断综合 Agent。
    专属工具：guideline_query（指南查询）、drug_recommendation（用药建议+冲突检查）
    """

    DIAGNOSIS_SYSTEM = """你是高级心脏科主治医师，负责综合诊断。

请基于以下信息，进行系统性诊断推理（Chain-of-Thought）：

{patient_history}

你有以下工具可用：
{tool_prompt}

诊断步骤：
1. 整合 ECG 和 Echo 发现，识别核心异常
2. 结合患者基本信息、风险评分、危险因素分析
3. 调用 guideline_query 查询主诊断对应的指南推荐级别
4. 调用 drug_recommendation 生成用药建议并检查与当前用药的冲突
5. 评估整体诊断置信度

请以如下 JSON 格式输出：
```json
{{
  "primary_diagnosis": "主诊断（中文）",
  "differential_diagnosis": ["鉴别诊断1", "鉴别诊断2"],
  "evidence": ["支撑证据1", "支撑证据2"],
  "treatment_recommendations": ["建议1", "建议2"],
  "follow_up": "随访建议",
  "overall_confidence": 0.0~1.0,
  "reasoning": "推理过程简述"
}}
```
"""

    CRITIQUE_SYSTEM = """你是医疗质控专家，负责审核诊断报告的逻辑一致性。

检查以下问题：
1. 诊断与 ECG/Echo 证据是否一致？
2. 置信度评估是否合理？
3. 高危情况是否有对应的紧急处理建议？
4. 是否存在明显遗漏的鉴别诊断？
5. 用药建议是否与当前用药存在冲突？

如果诊断合理，输出：{{"passed": true, "notes": ""}}
如果发现问题，输出：{{"passed": false, "notes": "具体问题描述", "corrections": "建议修正方向"}}
"""

    def run(self, state: AgentState) -> AgentState:
        logger.info("=== DiagnosisAgent 开始 ===")

        # ── 1. 构建诊断输入 ──────────────────────────────────────────────────
        context_parts = [
            f"**患者基本信息**: {json.dumps(state.patient_info, ensure_ascii=False)}",
        ]

        if state.patient_info.get("_risk_score"):
            r = state.patient_info["_risk_score"]
            context_parts.append(
                f"**风险评分**: {r['score_type']} {r['score']}分 ({r['risk_level']})"
            )

        if state.ecg_result:
            context_parts.append(f"**ECG 分析结果**:\n{state.ecg_result.summary}")
            if state.ecg_result.arrhythmia_labels:
                context_parts.append(f"  检出心律: {', '.join(state.ecg_result.arrhythmia_labels)}")
            if state.patient_info.get("_ecg_interval_abnormal"):
                context_parts.append(f"  间期异常: {state.patient_info['_ecg_interval_abnormal']}")

        if state.echo_result:
            context_parts.append(f"**Echo 分析结果**:\n{state.echo_result.summary}")
            if state.echo_result.lvef is not None:
                context_parts.append(f"  LVEF: {state.echo_result.lvef:.0%}")
            if state.patient_info.get("_culprit_vessels"):
                context_parts.append(f"  推断受累冠脉: {state.patient_info['_culprit_vessels']}")

        if state.retrieved_docs:
            doc_text = "\n\n".join(
                f"[{doc.source}]\n{doc.content}" for doc in state.retrieved_docs
            )
            context_parts.append(f"**相关指南参考**:\n{doc_text}")

        full_context = "\n\n".join(context_parts)

        # ── 2. 主诊断推理（ReAct 循环，自动调用 guideline_query + drug_recommendation）
        system = self.DIAGNOSIS_SYSTEM.format(
            patient_history=state.patient_info.get("_history_summary", "无历史记录"),
            tool_prompt=get_tool_prompt(),
        )

        response = self._react_loop(
            system_prompt=system,
            user_message=full_context,
            history=[],
            max_steps=5,
        )

        parsed = self._parse_json_output(response)

        # 直接调用两个专属工具补充结构化结果
        from tools.medical_tools import guideline_query, drug_recommendation
        primary_dx = parsed.get("primary_diagnosis", "")
        if primary_dx:
            guideline = guideline_query(primary_dx)
            drug_rec = drug_recommendation(
                diagnosis=primary_dx,
                lvef=state.echo_result.lvef if state.echo_result else 0.55,
                current_meds=state.patient_info.get("medications", []),
            )
            logger.info(f"指南查询: found={guideline.get('found')}, "
                        f"用药冲突: {drug_rec.get('drug_conflicts')}")

            # 把用药建议和冲突警告追加到治疗推荐
            extra = drug_rec.get("recommendations", [])
            if drug_rec.get("drug_conflicts"):
                extra.append(f"⚠️ 用药冲突: {'; '.join(drug_rec['drug_conflicts'])}")
            parsed.setdefault("treatment_recommendations", []).extend(extra)

        if not parsed:
            logger.warning("诊断输出 JSON 解析失败，使用原始文本")
            parsed = {"primary_diagnosis": response[:200], "overall_confidence": 0.5}

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
        logger.info(f"诊断完成: {report.primary_diagnosis} "
                    f"(置信度: {report.overall_confidence:.0%})")
        return state

    def _self_critique(self, report: DiagnosisReport,
                       context: str, state: AgentState) -> DiagnosisReport:
        """Self-Critique 自检，最多两轮。"""
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

            corrections = critique_result.get("corrections", "")
            logger.info(f"Self-Critique 发现问题: {corrections[:100]}")
            report.critique_notes = critique_result.get("notes", "")

            revised_response = self.llm.generate(
                messages=[{
                    "role": "user",
                    "content": f"{context}\n\n## 质控意见（请据此修正）\n{corrections}"
                }],
                system_prompt=self.DIAGNOSIS_SYSTEM.format(
                    patient_history="", tool_prompt="",
                ),
            )
            revised = self._parse_json_output(revised_response)
            if revised:
                report.primary_diagnosis = revised.get("primary_diagnosis", report.primary_diagnosis)
                report.differential_diagnosis = revised.get("differential_diagnosis", report.differential_diagnosis)
                report.evidence = revised.get("evidence", report.evidence)
                report.treatment_recommendations = revised.get("treatment_recommendations", report.treatment_recommendations)
                report.overall_confidence = revised.get("overall_confidence", report.overall_confidence)
        else:
            report.critique_passed = False
            logger.warning(f"Self-Critique 达到最大轮次 {max_rounds}，保留当前结果")

        return report
