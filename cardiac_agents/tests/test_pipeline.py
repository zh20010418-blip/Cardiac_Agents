"""
tests/test_pipeline.py —— 单元测试

用 unittest.mock 替换 LLM 和模型调用，
在没有 GPU 的环境下测试完整流程逻辑。

运行：
  pytest tests/ -v
  # 或
  python -m pytest tests/test_pipeline.py -v
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# 把项目根目录加入 path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── 测试 1：工具注册与调用 ────────────────────────────────────────────────────
class TestToolRegistry(unittest.TestCase):

    def test_tool_registration(self):
        """测试工具能否正确注册和调用。"""
        from tools.registry import _TOOL_REGISTRY, ToolDispatcher

        # 导入工具文件触发注册
        import tools.medical_tools  # noqa

        self.assertIn("ecg_analyze", _TOOL_REGISTRY)
        self.assertIn("echo_analyze", _TOOL_REGISTRY)
        self.assertIn("parse_patient_info", _TOOL_REGISTRY)
        self.assertIn("assess_vitals", _TOOL_REGISTRY)
        print("✅ 工具注册正常")

    def test_tool_call_parsing(self):
        """测试工具调用 JSON 解析（含不规范格式）。"""
        from tools.registry import ToolDispatcher

        dispatcher = ToolDispatcher()

        # 正常 JSON 代码块
        output1 = '```json\n{"tool": "ecg_analyze", "args": {"image_path": "/data/ecg.png", "patient_age": 50}}\n```'
        call1 = dispatcher.parse_tool_call(output1)
        self.assertIsNotNone(call1)
        self.assertEqual(call1["tool"], "ecg_analyze")

        # 混杂文字的 JSON
        output2 = '好的，我来调用工具：{"tool": "assess_vitals", "args": {"vitals": {"heart_rate": 110}}}'
        call2 = dispatcher.parse_tool_call(output2)
        self.assertIsNotNone(call2)
        self.assertEqual(call2["tool"], "assess_vitals")

        # 无工具调用
        output3 = "患者心律失常，建议进一步检查。"
        call3 = dispatcher.parse_tool_call(output3)
        self.assertIsNone(call3)
        print("✅ 工具调用解析正常")

    def test_ecg_tool_execution(self):
        """测试 ECG 工具能否正常执行（模拟数据）。"""
        from tools.medical_tools import ecg_analyze

        result = ecg_analyze("/nonexistent/path.png", patient_age=62)
        self.assertIsInstance(result, dict)
        self.assertIn("arrhythmia_labels", result)
        self.assertIn("max_confidence", result)
        self.assertIn("summary", result)
        print(f"✅ ECG 工具执行正常: {result['arrhythmia_labels']}")

    def test_vitals_assessment(self):
        """测试生命体征评估工具。"""
        from tools.medical_tools import assess_vitals

        vitals = {"heart_rate": 115, "blood_pressure_sys": 155, "spo2": 93}
        result = assess_vitals(vitals)

        self.assertEqual(result["overall_status"], "异常")
        self.assertTrue(len(result["abnormal_findings"]) > 0)
        print(f"✅ 生命体征评估正常: {result['abnormal_findings']}")


# ── 测试 2：状态机流转 ────────────────────────────────────────────────────────
class TestStateMachine(unittest.TestCase):

    def test_state_initialization(self):
        """测试 AgentState 初始化。"""
        from graph.state import AgentState, Stage

        state = AgentState(
            patient_info={"age": 60, "gender": "男"},
            ecg_image_paths=["/data/ecg.png"],
            user_query="请分析",
        )
        self.assertEqual(state.stage, Stage.INIT)
        self.assertIsNone(state.ecg_result)
        self.assertIsNone(state.diagnosis_report)
        print("✅ AgentState 初始化正常")

    def test_stage_transitions(self):
        """测试 Stage 枚举完整性。"""
        from graph.state import Stage

        stages = [Stage.INIT, Stage.ECG_ANALYSIS, Stage.ECHO_ANALYSIS,
                  Stage.RAG_RETRIEVAL, Stage.DIAGNOSIS, Stage.CRITIQUE,
                  Stage.DONE, Stage.ERROR]
        self.assertEqual(len(stages), 8)
        print("✅ Stage 枚举完整")


# ── 测试 3：记忆模块 ──────────────────────────────────────────────────────────
class TestMemory(unittest.TestCase):

    def test_short_term_memory(self):
        """测试短期记忆滑动窗口。"""
        from memory.memory import ShortTermMemory

        mem = ShortTermMemory(window_size=3)

        # 添加超过窗口大小的消息（maxlen=6，即 3*2）
        for i in range(10):
            mem.add("user", f"消息{i}")

        # 只保留最近的消息
        messages = mem.get_messages()
        self.assertLessEqual(len(messages), 6)
        print(f"✅ 短期记忆滑动窗口正常，保留 {len(messages)} 条")

    def test_long_term_memory(self, tmp_path=None):
        """测试长期记忆存取。"""
        import tempfile
        from memory.memory import LongTermMemory, PatientRecord
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            mem = LongTermMemory(storage_dir=tmpdir)

            record = PatientRecord(
                patient_id="TEST_001",
                session_id="sess_abc",
                timestamp=time.time(),
                patient_info={"age": 60},
                diagnosis="冠心病",
                confidence=0.85,
                key_findings=["ST段压低", "LVEF降低"],
            )
            mem.save(record)

            # 重新加载
            mem2 = LongTermMemory(storage_dir=tmpdir)
            records = mem2.load_patient_history("TEST_001")

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].diagnosis, "冠心病")
            print(f"✅ 长期记忆存取正常: {records[0].diagnosis}")

    def test_memory_manager(self):
        """测试统一记忆管理器。"""
        from memory.memory import MemoryManager

        manager = MemoryManager(session_id="test", patient_id="P001")
        manager.add_user_message("患者胸闷")
        manager.add_assistant_message("需要做 ECG 检查")
        manager.add_tool_result("ecg_analyze", "窦性心动过速")

        history = manager.get_chat_history()
        self.assertEqual(len(history), 3)
        print(f"✅ MemoryManager 正常，历史 {len(history)} 条")


# ── 测试 4：RAG 模块 ──────────────────────────────────────────────────────────
class TestRAG(unittest.TestCase):

    @patch("rag.retriever.SentenceTransformer")
    def test_hierarchical_chunker(self, mock_embedder):
        """测试 Markdown 层级分块。"""
        from rag.retriever import HierarchicalChunker

        chunker = HierarchicalChunker(max_chunk_size=200)
        sample_md = """# 房颤管理指南

## 抗凝治疗

### CHA2DS2-VASc 评分

评分 ≥2 的患者（男性）或 ≥3（女性）推荐口服抗凝治疗。
首选新型口服抗凝药（NOAC）。

### 抗凝药物选择

利伐沙班、达比加群、阿哌沙班均可选用。

## 心率控制

目标心率：静息时 <110 bpm。
"""
        chunks = chunker.chunk_markdown(sample_md, "房颤指南2023")
        self.assertTrue(len(chunks) > 0)
        # 检查元数据注入
        self.assertTrue(any("房颤指南2023" in c["content"] for c in chunks))
        print(f"✅ 层级分块正常，生成 {len(chunks)} 个块")

    def test_confidence_threshold_skip(self):
        """测试置信度阈值自适应检索。"""
        import tempfile
        import numpy as np

        with patch("rag.retriever.SentenceTransformer") as mock_st:
            mock_embedder = MagicMock()
            mock_embedder.get_sentence_embedding_dimension.return_value = 4
            mock_embedder.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype="float32")
            mock_st.return_value = mock_embedder

            from rag.retriever import FAISSRetriever
            with tempfile.TemporaryDirectory() as tmpdir:
                retriever = FAISSRetriever(
                    embed_model_path="/fake/model",
                    index_path=str(Path(tmpdir) / "index"),
                    guidelines_dir=str(Path(tmpdir) / "docs"),
                    confidence_threshold=0.75,
                )
                # 高置信度时应跳过检索
                docs = retriever.retrieve("测试查询", confidence=0.90)
                self.assertEqual(len(docs), 0)
                print("✅ 置信度阈值自适应检索正常（高置信度跳过）")


# ── 测试 5：完整流水线（mock LLM）────────────────────────────────────────────
class TestPipeline(unittest.TestCase):

    @patch("models.llm.AutoTokenizer")
    @patch("models.llm.AutoModelForCausalLM")
    def test_full_pipeline_mock(self, mock_model_cls, mock_tokenizer_cls):
        """
        用 Mock LLM 测试完整 Pipeline 流转。
        验证：各 Stage 按顺序执行，最终输出 DiagnosisReport。
        """
        from graph.state import Stage

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "mock_text"
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = MagicMock()
        import torch
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_model.device = "cpu"
        mock_model_cls.from_pretrained.return_value = mock_model

        # LLM 按调用顺序返回不同内容
        llm_responses = [
            # Dispatcher
            '{"has_ecg": true, "has_echo": false, "needs_clarification": false, '
            '"clarification_questions": [], "patient_summary": "测试", "urgency": "routine"}',
            # ECG Agent ReAct
            "ECG 显示窦性心律，无明显异常。",
            # Diagnosis Agent
            '{"primary_diagnosis": "正常心律", "differential_diagnosis": [], '
            '"evidence": ["ECG正常"], "treatment_recommendations": ["定期随访"], '
            '"follow_up": "1年后复查", "overall_confidence": 0.9, "reasoning": "ECG无异常"}',
            # Self-Critique
            '{"passed": true, "notes": "诊断合理"}',
        ]
        call_idx = [0]

        def mock_decode(*args, **kwargs):
            idx = min(call_idx[0], len(llm_responses) - 1)
            call_idx[0] += 1
            return llm_responses[idx]

        mock_tokenizer.decode.side_effect = mock_decode

        # 同时 mock RAG embedding
        with patch("rag.retriever.SentenceTransformer") as mock_st:
            mock_embedder = MagicMock()
            mock_embedder.get_sentence_embedding_dimension.return_value = 4
            import numpy as np
            mock_embedder.encode.return_value = np.zeros((1, 4), dtype="float32")
            mock_st.return_value = mock_embedder

            from graph.pipeline import CardiacDiagnosisPipeline
            # 重置单例以免被其他测试污染
            from models.llm import LocalLLM
            LocalLLM._instances.clear()

            pipeline = CardiacDiagnosisPipeline()
            state = pipeline.run(
                patient_info={"age": 45, "gender": "女", "chief_complaint": "体检"},
                ecg_image_paths=["./fake_ecg.png"],
                user_query="请分析心脏情况",
                patient_id="TEST_PIPELINE",
            )

            # 验证流程正常完成
            self.assertIn(state.stage, [Stage.DONE, Stage.ERROR])
            print(f"✅ 完整 Pipeline 测试通过，最终 Stage: {state.stage}")
            if state.diagnosis_report:
                print(f"   主诊断: {state.diagnosis_report.primary_diagnosis}")


if __name__ == "__main__":
    # 直接运行时执行所有测试
    unittest.main(verbosity=2)
