"""
rag/retriever.py —— FAISS 向量检索模块

功能：
  1. build_index()  : 读取医学指南文档 → 分块 → 向量化 → 建 FAISS 索引
  2. retrieve()     : 给定查询文本 → 向量化 → FAISS 检索 → 返回 top-k 片段

分块策略：H1→H2→H3 结构化原子切分（简历里提到的）
  - 不用固定字符数切分（会截断句子）
  - 按章节标题层级切分，保证每块语义完整
  - 每块注入全局元数据（来源、章节路径）作为语义边界
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    ST_AVAILABLE = False

from graph.state import RetrievedDoc

logger = logging.getLogger(__name__)


# ── 文档分块 ──────────────────────────────────────────────────────────────────
class HierarchicalChunker:
    """
    H1→H2→H3 层级化文档分块器。

    输入：Markdown 格式的医学指南文本
    输出：带层级路径元数据的文本块列表

    示例块：
        {
            "content": "房颤患者应使用 CHA2DS2-VASc 评分...",
            "metadata": {
                "source": "ACC/AHA 2023 房颤管理指南",
                "h1": "抗凝治疗",
                "h2": "卒中风险评估",
                "h3": "CHA2DS2-VASc 评分",
                "chunk_id": "atrial_fibrillation_guide::抗凝治疗::卒中风险评估::0"
            }
        }
    """

    def __init__(self, max_chunk_size: int = 512, overlap: int = 64):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk_markdown(self, text: str, source: str) -> list[dict]:
        """将 Markdown 文档按标题层级切分。"""
        chunks = []
        current_h1 = current_h2 = current_h3 = ""
        current_content = []

        def flush(level_hint: str = ""):
            """把当前积累的内容保存为一个 chunk。"""
            content = "\n".join(current_content).strip()
            if not content:
                return
            # 注入元数据前缀（帮助检索时匹配语义）
            header_path = " > ".join(filter(None, [current_h1, current_h2, current_h3]))
            prefixed = f"[{source} | {header_path}]\n{content}" if header_path else content
            chunks.append({
                "content": prefixed,
                "metadata": {
                    "source": source,
                    "h1": current_h1,
                    "h2": current_h2,
                    "h3": current_h3,
                    "chunk_id": f"{source}::{current_h1}::{current_h2}::{len(chunks)}",
                }
            })
            current_content.clear()

        for line in text.split("\n"):
            if line.startswith("### "):
                flush("h3")
                current_h3 = line[4:].strip()
            elif line.startswith("## "):
                flush("h2")
                current_h2 = line[3:].strip()
                current_h3 = ""
            elif line.startswith("# "):
                flush("h1")
                current_h1 = line[2:].strip()
                current_h2 = current_h3 = ""
            else:
                current_content.append(line)

            # 防止单块过大：超过限制则提前切割（带重叠）
            current_text = "\n".join(current_content)
            if len(current_text) > self.max_chunk_size:
                flush("overflow")
                # 保留最后 overlap 字符作为下一块的上下文
                overlap_text = current_text[-self.overlap:]
                current_content.clear()
                current_content.append(overlap_text)

        flush("end")
        return chunks


# ── FAISS 检索器 ──────────────────────────────────────────────────────────────
class FAISSRetriever:
    """
    基于 FAISS 的医学指南检索器。

    索引结构：
      - faiss_index/index.bin   : FAISS IVF 索引文件
      - faiss_index/chunks.json : 对应的文本块元数据
    """

    def __init__(
        self,
        embed_model_path: str,
        index_path: str,
        guidelines_dir: str,
        top_k: int = 5,
        embed_device: str = "cpu",
        confidence_threshold: float = 0.75,
    ):
        self.index_path = Path(index_path)
        self.guidelines_dir = Path(guidelines_dir)
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold

        logger.info(f"加载 Embedding 模型: {embed_model_path}")
        self.embedder = SentenceTransformer(embed_model_path, device=embed_device)
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()

        self.chunker = HierarchicalChunker()
        self.index: Optional[faiss.Index] = None
        self.chunks: list[dict] = []

        # 尝试加载已有索引
        if self._index_exists():
            self._load_index()
        else:
            logger.info("未找到 FAISS 索引，尝试自动构建...")
            if self.guidelines_dir.exists():
                self.build_index()

    def _index_exists(self) -> bool:
        return (self.index_path / "index.bin").exists()

    def _load_index(self):
        """从磁盘加载 FAISS 索引和文本块。"""
        index_file = self.index_path / "index.bin"
        chunks_file = self.index_path / "chunks.json"
        self.index = faiss.read_index(str(index_file))
        with open(chunks_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        logger.info(f"FAISS 索引加载完成，共 {len(self.chunks)} 个文本块")

    def build_index(self):
        """
        从 guidelines_dir 读取所有 .md / .txt 文档，
        切块 → 向量化 → 构建 FAISS 索引 → 保存到磁盘。
        """
        logger.info(f"构建 FAISS 索引，文档目录: {self.guidelines_dir}")
        all_chunks = []

        # 读取所有指南文档
        doc_files = list(self.guidelines_dir.glob("*.md")) + \
                    list(self.guidelines_dir.glob("*.txt"))

        if not doc_files:
            logger.warning(f"未找到文档，请将医学指南放入 {self.guidelines_dir}")
            self._build_empty_index()
            return

        for doc_path in doc_files:
            logger.info(f"处理文档: {doc_path.name}")
            with open(doc_path, "r", encoding="utf-8") as f:
                text = f.read()
            source = doc_path.stem
            chunks = self.chunker.chunk_markdown(text, source)
            all_chunks.extend(chunks)
            logger.info(f"  → {len(chunks)} 个文本块")

        if not all_chunks:
            self._build_empty_index()
            return

        # 批量向量化（batch_size=64 避免 OOM）
        logger.info(f"向量化 {len(all_chunks)} 个文本块...")
        texts = [c["content"] for c in all_chunks]
        embeddings = self.embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,   # 归一化后 IP = cosine similarity
        ).astype(np.float32)

        # 构建 FAISS IndexFlatIP（内积，等价于归一化后的余弦相似度）
        # 数据量大时换成 IndexIVFFlat 加速：
        #   quantizer = faiss.IndexFlatIP(self.embed_dim)
        #   index = faiss.IndexIVFFlat(quantizer, self.embed_dim, 100)
        #   index.train(embeddings)
        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.index.add(embeddings)
        self.chunks = all_chunks

        # 保存到磁盘
        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path / "index.bin"))
        with open(self.index_path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        logger.info(f"FAISS 索引构建完成，共 {len(all_chunks)} 个块，已保存到 {self.index_path}")

    def _build_empty_index(self):
        """构建空索引，防止后续检索崩溃。"""
        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.chunks = []

    def retrieve(
        self,
        query: str,
        confidence: Optional[float] = None,
        force: bool = False,
    ) -> list[RetrievedDoc]:
        """
        检索相关指南片段。

        Args:
            query: 查询文本（通常是诊断摘要或关键词）
            confidence: 当前诊断置信度；低于阈值才检索（自适应机制）
            force: True 时强制检索，不看置信度

        Returns:
            RetrievedDoc 列表
        """
        # 自适应检索：置信度高则跳过，节省延时
        if not force and confidence is not None:
            if confidence >= self.confidence_threshold:
                logger.info(f"置信度 {confidence:.2f} 高于阈值，跳过 RAG 检索")
                return []

        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS 索引为空，无法检索")
            return []

        # 查询向量化
        query_vec = self.embedder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        # FAISS 检索
        scores, indices = self.index.search(query_vec, self.top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS 返回 -1 表示不足 top_k
                continue
            chunk = self.chunks[idx]
            results.append(RetrievedDoc(
                content=chunk["content"],
                source=chunk["metadata"].get("source", "未知来源"),
                score=float(score),
            ))

        logger.info(f"RAG 检索完成，返回 {len(results)} 条结果，top1 score={scores[0][0]:.3f}")
        return results

    def format_context(self, docs: list[RetrievedDoc]) -> str:
        """将检索结果格式化为 LLM 可读的上下文字符串。"""
        if not docs:
            return ""
        lines = ["## 相关医学指南参考\n"]
        for i, doc in enumerate(docs, 1):
            lines.append(f"**[参考{i}] 来源: {doc.source}（相关度: {doc.score:.2f}）**")
            lines.append(doc.content)
            lines.append("")
        return "\n".join(lines)
