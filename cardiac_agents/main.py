"""
main.py —— 程序入口

提供两种运行模式：
  1. 单次诊断（命令行参数传入数据）
  2. 多轮交互对话（模拟真实诊疗场景）

运行示例：
  # 单次诊断
  python main.py --mode single --ecg /data/ecg/001.png --age 62

  # 多轮对话
  python main.py --mode chat
"""

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

# ── 日志配置 ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# 关闭 transformers 的冗余日志
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def run_single_diagnosis(args):
    """单次诊断模式。"""
    from graph.pipeline import CardiacDiagnosisPipeline

    pipeline = CardiacDiagnosisPipeline()

    patient_info = {
        "age": args.age,
        "gender": args.gender,
        "chief_complaint": args.complaint,
        "history": args.history.split(",") if args.history else [],
    }

    ecg_paths = [args.ecg] if args.ecg else []
    echo_paths = [args.echo] if args.echo else []

    print("\n正在分析，请稍候...\n")
    state = pipeline.run(
        patient_info=patient_info,
        ecg_image_paths=ecg_paths,
        echo_paths=echo_paths,
        user_query=args.query or "请分析该患者的心脏情况",
        patient_id=args.patient_id or "demo_patient",
    )

    print(pipeline.format_report(state))

def run_chat_mode():
    """
    多轮交互对话模式。

    模拟真实诊疗流程：
      1. 用户描述患者情况
      2. Agent 发现信息不足时追问
      3. 用户补充后，Agent 进入分析
      4. 输出诊断报告
      5. 用户可继续提问（随访、用药等）
    """
    from graph.pipeline import CardiacDiagnosisPipeline
    from memory.memory import MemoryManager

    pipeline = CardiacDiagnosisPipeline()
    session_id = str(uuid.uuid4())[:8]
    patient_id = input("请输入患者ID（留空则自动生成）: ").strip() or f"P{session_id}"
    memory = MemoryManager(session_id=session_id, patient_id=patient_id)

    print(f"\n{'='*60}")
    print("  心脏病智能诊疗助手（多轮对话模式）")
    print(f"  会话ID: {session_id} | 患者ID: {patient_id}")
    print(f"{'='*60}")
    print("输入患者信息开始诊断，输入 'quit' 退出\n")

    # 检查长期记忆中是否有该患者历史
    history_summary = memory.get_patient_context()
    if history_summary:
        print(f"[系统] 检测到该患者历史记录：\n{history_summary}\n")

    # 收集患者基本信息
    print("请描述患者情况（年龄、性别、主诉、既往史等）：")
    patient_desc = input("> ").strip()
    if patient_desc.lower() == "quit":
        return

    memory.add_user_message(patient_desc)

    # 询问是否有 ECG/Echo 数据
    print("\n是否有 ECG 图像？输入路径或直接回车跳过：")
    ecg_path = input("> ").strip()
    print("是否有 Echo 数据？输入路径或直接回车跳过：")
    echo_path = input("> ").strip()

    # 简单解析患者描述（真实系统这里会调用 parse_patient_info 工具）
    patient_info = {
        "description": patient_desc,
        "age": _extract_age(patient_desc),
        "gender": "男" if "男" in patient_desc else "女" if "女" in patient_desc else "未知",
        "chief_complaint": patient_desc,
        "_history_summary": history_summary,
    }

    print("\n[系统] 正在分析，请稍候...\n")

    state = pipeline.run(
        patient_info=patient_info,
        ecg_image_paths=[ecg_path] if ecg_path else [],
        echo_paths=[echo_path] if echo_path else [],
        user_query=patient_desc,
        patient_id=patient_id,
        session_id=session_id,
    )

    report_text = pipeline.format_report(state)
    print(report_text)
    memory.add_assistant_message(report_text)

    # ── 后续多轮问答 ──────────────────────────────────────────────────────────
    print("\n诊断完成。您可以继续提问（如用药调整、随访安排等），输入 'quit' 退出：\n")

    while True:
        user_input = input("您: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("会话结束，诊断记录已保存。")
            break
        if not user_input:
            continue

        memory.add_user_message(user_input)

        # 后续问答：用 LLM 直接回答（带完整对话历史）
        history = memory.get_chat_history()
        system_prompt = f"""你是心脏科医生，正在为患者 {patient_id} 进行随访问答。
已完成的诊断：{state.diagnosis_report.primary_diagnosis if state.diagnosis_report else '未知'}
请基于上述诊断和对话历史，回答患者问题。回答要专业、简洁。"""

        response = pipeline.llm.generate(
            messages=history + [{"role": "user", "content": user_input}],
            system_prompt=system_prompt,
        )

        print(f"\n医生: {response}\n")
        memory.add_assistant_message(response)

def _extract_age(text: str) -> int:
    """从文本中简单提取年龄数字。"""
    import re
    m = re.search(r"(\d{1,3})\s*[岁年]", text)
    return int(m.group(1)) if m else 50

def demo_mode():
    """
    演示模式：不需要真实模型，用模拟数据跑通完整流程。
    用于没有 GPU 时验证代码逻辑。
    """
    print("=" * 60)
    print("  演示模式（使用模拟数据，无需 GPU）")
    print("=" * 60)

    # 直接 mock LLM，用固定字符串替代模型输出
    from unittest.mock import MagicMock, patch

    mock_llm_outputs = [
        # DispatcherAgent 输出
        '```json\n{"has_ecg": true, "has_echo": true, "needs_clarification": false, '
        '"clarification_questions": [], "patient_summary": "62岁男性，胸闷气短", "urgency": "routine"}\n```',

        # ECGAgent 输出（ReAct 分析）
        "ECG 显示窦性心动过速，V4-V6 ST 段轻度压低，结合患者胸闷症状，提示心肌缺血可能，建议进一步检查。",

        # EchoAgent 输出
        "超声提示 LVEF 48%，轻度降低，前壁心尖段室壁运动减弱，支持局部心肌缺血诊断。",

        # DiagnosisAgent 主诊断输出
        '```json\n{"primary_diagnosis": "冠状动脉粥样硬化性心脏病（心绞痛型）", '
        '"differential_diagnosis": ["急性心肌梗死（NSTEMI）", "心肌病"], '
        '"evidence": ["ECG ST段压低", "LVEF轻度降低48%", "前壁室壁运动减弱", "高血压糖尿病危险因素"], '
        '"treatment_recommendations": ["启动抗血小板治疗（阿司匹林+氯吡格雷）", "他汀类药物调脂", '
        '"冠脉造影评估介入指征"], "follow_up": "1周后复查心肌酶学，1个月后超声随访", '
        '"overall_confidence": 0.82, "reasoning": "ECG+Echo+危险因素三方面证据支持"}\n```',

        # Self-Critique 输出
        '{"passed": true, "notes": "诊断证据充分，治疗方案符合指南"}'
    ]

    call_count = [0]

    def mock_generate(*args, **kwargs):
        idx = min(call_count[0], len(mock_llm_outputs) - 1)
        call_count[0] += 1
        return mock_llm_outputs[idx]

    # 用 mock 替换 LLM 推理，其他模块正常运行
    with patch("models.llm.LocalLLM._load_model"), \
         patch.object(__import__("models.llm", fromlist=["LocalLLM"]).LocalLLM,
                      "generate", mock_generate):

        from graph.pipeline import CardiacDiagnosisPipeline

        pipeline = CardiacDiagnosisPipeline()

        state = pipeline.run(
            patient_info={
                "age": 62,
                "gender": "男",
                "chief_complaint": "胸闷、气短 3 天，活动后加重",
                "history": ["高血压 10 年", "2型糖尿病 5 年"],
                "medications": ["氨氯地平 5mg", "二甲双胍 0.5g"],
                "smoking": True,
            },
            ecg_image_paths=["./data/sample_ecg.png"],
            echo_paths=["./data/sample_echo.mp4"],
            user_query="请帮我分析这位患者的心脏情况",
            patient_id="DEMO_001",
        )

        print(pipeline.format_report(state))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="心脏病智能诊疗 Agent 系统")
    parser.add_argument("--mode", choices=["single", "chat", "demo"], default="demo",
                        help="运行模式: single=单次诊断, chat=多轮对话, demo=演示（无需GPU）")
    parser.add_argument("--ecg", type=str, help="ECG 图像路径")
    parser.add_argument("--echo", type=str, help="Echo 数据路径")
    parser.add_argument("--age", type=int, default=50, help="患者年龄")
    parser.add_argument("--gender", type=str, default="男", help="患者性别")
    parser.add_argument("--complaint", type=str, default="胸闷气短", help="主诉")
    parser.add_argument("--history", type=str, default="", help="既往史，逗号分隔")
    parser.add_argument("--query", type=str, help="诊断问题")
    parser.add_argument("--patient-id", type=str, default="", help="患者ID")

    args = parser.parse_args()

    if args.mode == "demo":
        demo_mode()
    elif args.mode == "single":
        run_single_diagnosis(args)
    elif args.mode == "chat":
        run_chat_mode()
