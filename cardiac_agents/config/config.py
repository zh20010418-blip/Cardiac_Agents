"""
config.py —— 全局配置
所有路径、超参数集中在这里，方便修改，不要在各模块里硬编码。
"""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent  # 项目根目录


@dataclass
class ModelConfig:
    # ── 主控 LLM（本地模型路径，替换成你实际下载的路径） ──────────────────
    llm_path: str = "./data/models/Qwen2-7B-Instruct"   # 主 LLM
    llm_device: str = "cuda"                               # cuda / cpu
    llm_dtype: str = "float16"                             # float16 / bfloat16 / float32
    llm_max_new_tokens: int = 1024
    llm_temperature: float = 0.1                           # 诊断任务用低温，减少随机性
    llm_do_sample: bool = False

    # ── ECG 专科模型 ─────────────────────────────────────────────────────────
    ecg_model_path: str = "./data/models/ecg-classifier"   # 训练的心律失常分类器 or 自己微调好的模型 or 现存的ECG识别模型
    ecg_device: str = "cuda"

    # ── Echo 专科模型 ────────────────────────────────────────────────────────
    echo_model_path: str = "./data/models/echo-lvef"        # LVEF 回归 / 室壁运动分类器 or 自己微调好的模型 or 现存的ECHO识别模型
    echo_device: str = "cuda"

    # ── Embedding 模型（用于 RAG 向量化） ────────────────────────────────────
    embed_model_path: str = "./data/models/bge-large-zh-v1.5"
    embed_device: str = "cuda"
    embed_dim: int = 1024                                   # bge-large 输出维度


@dataclass
class RAGConfig:
    # FAISS 索引存储路径
    index_path: str = str(ROOT / "data" / "faiss_index")
    # 医学指南文档目录
    guidelines_dir: str = str(ROOT / "data" / "guidelines")
    # 每次检索返回的 top-k 片段
    top_k: int = 5
    # 文档分块大小（字符数）
    chunk_size: int = 512
    chunk_overlap: int = 64
    # 置信度低于此阈值时才触发 RAG 检索
    confidence_threshold: float = 0.75


@dataclass
class AgentConfig:
    # 短期记忆保留的最近 N 轮对话
    short_term_window: int = 10
    # Self-Critique 最大重试次数
    max_critique_rounds: int = 2
    # 工具调用超时（秒）
    tool_timeout: int = 30
    # 高危疾病列表（触发额外校验）
    high_risk_conditions: list = field(default_factory=lambda: [
        "心室颤动", "室性心动过速", "三度房室传导阻滞",
        "急性心肌梗死", "主动脉夹层", "心脏骤停"
    ])


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    log_level: str = "INFO"
    log_dir: str = str(ROOT / "logs")


# 全局单例，各模块直接 import 使用
cfg = AppConfig()
