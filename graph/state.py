"""
graph/state.py —— 全局状态定义

所有 Agent 共享同一个 AgentState 对象，通过它传递数据。
这等价于 LangGraph 的 State，但我们手写，更透明。

数据流向：
  用户输入 → DispatcherAgent 填充基础字段
           → ECGAgent 填充 ecg_result
           → EchoAgent 填充 echo_result
           → RAGAgent 填充 retrieved_docs
           → DiagnosisAgent 填充 diagnosis / report
           → Self-Critique 校验后输出
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ── 流程阶段枚举 ──────────────────────────────────────────────────────────────
class Stage(Enum):
    """当前 Agent 流水线所处的阶段，调度器根据此决定下一步。"""
    INIT = auto()           # 初始化，刚收到用户输入
    ECG_ANALYSIS = auto()   # ECG Agent 处理中
    ECHO_ANALYSIS = auto()  # Echo Agent 处理中
    RAG_RETRIEVAL = auto()  # RAG 检索中
    DIAGNOSIS = auto()      # 诊断综合 Agent 处理中
    CRITIQUE = auto()       # Self-Critique 校验中
    DONE = auto()           # 流程结束，输出报告
    ERROR = auto()          # 发生错误


# ── ECG 分析结果 ──────────────────────────────────────────────────────────────
@dataclass
class ECGResult:
    """ECG 专科 Agent 的输出结构。"""
    # 心律失常分类标签，例如 ["正常窦性心律", "房颤"]
    arrhythmia_labels: list[str] = field(default_factory=list)
    # 各类别置信度，例如 {"房颤": 0.92, "正常": 0.05}
    confidence_scores: dict[str, float] = field(default_factory=dict)
    # 主要异常置信度（用于决定是否触发 RAG）
    max_confidence: float = 0.0
    # 模型输出的文字描述（喂给诊断 Agent 的自然语言摘要）
    summary: str = ""
    # 是否检测到高危心律
    is_high_risk: bool = False
    # 原始 logits（供 Self-Critique 二次验证）
    raw_logits: Optional[list[float]] = None


# ── Echo 分析结果 ─────────────────────────────────────────────────────────────
@dataclass
class EchoResult:
    """Echo 专科 Agent 的输出结构。"""
    # 左室射血分数（0~1）
    lvef: Optional[float] = None
    # 室壁运动异常区域，例如 ["前壁运动减弱", "下壁运动消失"]
    wall_motion_abnormalities: list[str] = field(default_factory=list)
    # 其他结构异常，例如 ["二尖瓣轻度反流"]
    structural_findings: list[str] = field(default_factory=list)
    # 置信度
    confidence: float = 0.0
    # 自然语言摘要
    summary: str = ""
    # 是否存在严重异常
    is_severe: bool = False


# ── RAG 检索结果 ──────────────────────────────────────────────────────────────
@dataclass
class RetrievedDoc:
    """单条检索到的指南片段。"""
    content: str = ""
    source: str = ""        # 来源，例如 "ACC/AHA 2023 心衰指南 第3章"
    score: float = 0.0      # 向量相似度得分


# ── 诊断报告 ──────────────────────────────────────────────────────────────────
@dataclass
class DiagnosisReport:
    """最终结构化诊断报告。"""
    # 主诊断
    primary_diagnosis: str = ""
    # 鉴别诊断列表
    differential_diagnosis: list[str] = field(default_factory=list)
    # 支撑证据（来自 ECG + Echo + 患者信息）
    evidence: list[str] = field(default_factory=list)
    # 治疗建议（来自 RAG 检索指南）
    treatment_recommendations: list[str] = field(default_factory=list)
    # 随访建议
    follow_up: str = ""
    # 整体置信度（0~1）
    overall_confidence: float = 0.0
    # Self-Critique 是否通过
    critique_passed: bool = False
    # Critique 的意见（如果有修改）
    critique_notes: str = ""


# ── 主状态对象 ────────────────────────────────────────────────────────────────
@dataclass
class AgentState:
    """
    贯穿整个 Agent 流水线的全局状态。

    调度器（Dispatcher）负责初始化，各专科 Agent 依次填充各自字段，
    最终 DiagnosisAgent 汇总生成报告。
    """

    # ── 输入 ─────────────────────────────────────────────────────────────────
    # 患者基本信息（年龄、性别、主诉、既往史等）
    patient_info: dict = field(default_factory=dict)
    # ECG 图像路径（可能有多张，取自不同导联）
    ecg_image_paths: list[str] = field(default_factory=list)
    # Echo 视频/帧路径
    echo_paths: list[str] = field(default_factory=list)
    # 用户原始问题（多轮对话中的当前问题）
    user_query: str = ""

    # ── 各 Agent 输出 ────────────────────────────────────────────────────────
    ecg_result: Optional[ECGResult] = None
    echo_result: Optional[EchoResult] = None
    retrieved_docs: list[RetrievedDoc] = field(default_factory=list)
    diagnosis_report: Optional[DiagnosisReport] = None

    # ── 流程控制 ─────────────────────────────────────────────────────────────
    stage: Stage = Stage.INIT
    error_message: str = ""
    # 当前 Critique 轮次
    critique_round: int = 0
    # 调度器决定跳过 Echo 分析（比如没有 Echo 数据时）
    skip_echo: bool = False
    # 是否触发 RAG（由置信度决定）
    trigger_rag: bool = False

    # ── 多轮对话追踪 ─────────────────────────────────────────────────────────
    # 本轮会话 ID
    session_id: str = ""
    # 追问列表（Agent 认为信息不足时生成）
    clarification_questions: list[str] = field(default_factory=list)
    # 是否需要用户补充信息
    needs_clarification: bool = False
