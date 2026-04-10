"""
memory/memory.py —— 分层记忆管理

两层记忆：
  1. ShortTermMemory（短期）：当前会话的对话历史，滑动窗口，直接喂给 LLM
  2. LongTermMemory（长期）：患者档案向量存储，跨会话持久化，用 FAISS 检索

设计原理：
  - 短期记忆：deque 实现滑动窗口，防止 context 过长
  - 长期记忆：把历史诊断报告向量化存入 FAISS，新会话开始时检索相似历史
"""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────
@dataclass
class Message:
    """单条对话消息。"""
    role: str       # "user" / "assistant" / "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_name: Optional[str] = None    # 如果是工具调用结果，记录工具名


@dataclass
class PatientRecord:
    """患者历史档案，用于长期记忆。"""
    patient_id: str
    session_id: str
    timestamp: float
    patient_info: dict
    diagnosis: str
    confidence: float
    key_findings: list[str]    # ECG/Echo 关键发现，用于向量化检索


# ── 短期记忆 ──────────────────────────────────────────────────────────────────
class ShortTermMemory:
    """
    当前会话的滑动窗口记忆。

    只保留最近 window_size 轮对话，防止 LLM context 超长。
    会话结束后可导出到长期记忆。
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        # maxlen 限制最大长度，自动淘汰旧消息
        self._history: deque[Message] = deque(maxlen=window_size * 2)  # *2 因为每轮有 user+assistant

    def add(self, role: str, content: str, tool_name: Optional[str] = None):
        """添加一条消息。"""
        self._history.append(Message(role=role, content=content, tool_name=tool_name))

    def get_messages(self) -> list[dict]:
        """
        返回 LLM 格式的消息列表（用于直接传入 model.generate）。
        工具结果用 role="tool" 标记，诊断 LLM 可以看到工具的 Observation。
        """
        messages = []
        for msg in self._history:
            if msg.role == "tool":
                # 工具结果作为 assistant 消息（兼容大多数模型格式）
                messages.append({
                    "role": "assistant",
                    "content": f"[工具: {msg.tool_name}]\n{msg.content}"
                })
            else:
                messages.append({"role": msg.role, "content": msg.content})
        return messages

    def get_summary_text(self) -> str:
        """将历史对话压缩成文本摘要（用于长期记忆存储）。"""
        lines = []
        for msg in self._history:
            prefix = {"user": "用户", "assistant": "医生", "tool": "工具"}.get(msg.role, msg.role)
            lines.append(f"{prefix}: {msg.content[:200]}")  # 截断防止过长
        return "\n".join(lines)

    def clear(self):
        self._history.clear()

    def __len__(self):
        return len(self._history)


# ── 长期记忆 ──────────────────────────────────────────────────────────────────
class LongTermMemory:
    """
    患者档案的持久化存储与向量检索。

    存储：JSON 文件（简单可靠，实际生产可换成数据库）
    检索：基于关键发现的文本相似度（可接 FAISS，这里先用关键词匹配）

    设计考虑：长期记忆主要用于：
      1. 查询同一患者的历史诊断（按 patient_id）
      2. 检索相似病例作为参考（按症状/发现相似度）
    """

    def __init__(self, storage_dir: str = "./data/patient_records"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[PatientRecord]] = {}  # patient_id -> records

    def _record_path(self, patient_id: str) -> Path:
        # 用患者 ID 的前两位作为子目录，避免单目录文件过多
        subdir = self.storage_dir / patient_id[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{patient_id}.json"

    def save(self, record: PatientRecord):
        """保存患者诊断记录。"""
        path = self._record_path(record.patient_id)

        # 读取已有记录
        existing = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)

        existing.append(asdict(record))

        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        # 更新缓存
        self._cache.setdefault(record.patient_id, []).append(record)
        logger.info(f"患者记录已保存: {record.patient_id} @ {record.session_id}")

    def load_patient_history(self, patient_id: str) -> list[PatientRecord]:
        """加载某患者的所有历史记录。"""
        if patient_id in self._cache:
            return self._cache[patient_id]

        path = self._record_path(patient_id)
        if not path.exists():
            return []

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        records = [PatientRecord(**r) for r in raw]
        self._cache[patient_id] = records
        return records

    def get_history_summary(self, patient_id: str) -> str:
        """
        将患者历史诊断格式化为 LLM 可读的摘要，
        注入 system prompt 帮助当前会话的诊断参考。
        """
        records = self.load_patient_history(patient_id)
        if not records:
            return ""

        # 只取最近 3 次，避免 context 过长
        recent = sorted(records, key=lambda r: r.timestamp, reverse=True)[:3]
        lines = ["## 患者历史诊断记录\n"]
        for r in recent:
            import datetime
            dt = datetime.datetime.fromtimestamp(r.timestamp).strftime("%Y-%m-%d")
            lines.append(f"**{dt}诊断**: {r.diagnosis}（置信度: {r.confidence:.0%}）")
            if r.key_findings:
                lines.append(f"  关键发现: {', '.join(r.key_findings)}")
        return "\n".join(lines)


# ── 统一记忆管理器 ────────────────────────────────────────────────────────────
class MemoryManager:
    """
    对外暴露的统一记忆接口，各 Agent 通过它读写记忆，
    不需要关心底层是短期还是长期存储。
    """

    def __init__(self, session_id: str, patient_id: str = "", window_size: int = 10):
        self.session_id = session_id
        self.patient_id = patient_id
        self.short_term = ShortTermMemory(window_size=window_size)
        self.long_term = LongTermMemory()

    def add_user_message(self, content: str):
        self.short_term.add("user", content)

    def add_assistant_message(self, content: str):
        self.short_term.add("assistant", content)

    def add_tool_result(self, tool_name: str, content: str):
        self.short_term.add("tool", content, tool_name=tool_name)

    def get_chat_history(self) -> list[dict]:
        """获取短期对话历史，直接传给 LLM。"""
        return self.short_term.get_messages()

    def get_patient_context(self) -> str:
        """获取长期患者历史摘要，注入 system prompt。"""
        if self.patient_id:
            return self.long_term.get_history_summary(self.patient_id)
        return ""

    def save_diagnosis(self, diagnosis: str, confidence: float,
                       patient_info: dict, key_findings: list[str]):
        """会话结束后，将诊断结果存入长期记忆。"""
        if not self.patient_id:
            return
        record = PatientRecord(
            patient_id=self.patient_id,
            session_id=self.session_id,
            timestamp=time.time(),
            patient_info=patient_info,
            diagnosis=diagnosis,
            confidence=confidence,
            key_findings=key_findings,
        )
        self.long_term.save(record)

    def needs_clarification_prompt(self, questions: list[str]) -> str:
        """生成追问文本（Agent 认为信息不足时调用）。"""
        if not questions:
            return ""
        lines = ["为了提供更准确的诊断，需要了解以下信息：\n"]
        for i, q in enumerate(questions, 1):
            lines.append(f"{i}. {q}")
        return "\n".join(lines)
