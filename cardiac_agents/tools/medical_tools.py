"""
tools/medical_tools.py —— 具体医疗工具实现

每个函数对应一个临床分析工具。
@tool 装饰器把它注册到全局工具表，LLM 可以通过 Function Calling 调用。

替换说明：
  ecg_analyze()  和 echo_analyze() 内部各有一段注释标注的替换区域，
  无论你用的是传统分类器、还是微调好的模型，
  只要让你的模型返回注释里说明的字典格式即可，其他代码不需要动。
"""

import logging
from pathlib import Path

from tools.registry import tool

logger = logging.getLogger(__name__)


# ── 工具 1：ECG 心律失常分析 ──────────────────────────────────────────────────
@tool(
    name="ecg_analyze",
    description="分析 ECG 图像，检测心律失常类型并给出置信度。输入 ECG 图像路径，返回心律失常标签和置信度分布。",
    parameters={
        "image_path": {
            "type": "string",
            "description": "ECG 图像的本地路径，例如 /data/ecg/patient_001.png"
        },
        "patient_age": {
            "type": "integer",
            "description": "患者年龄，用于调整正常范围参考值"
        }
    }
)
def ecg_analyze(image_path: str, patient_age: int = 50) -> dict:
    """
    ECG 心律失常检测。

    ── 替换区域 ──────────────────────────────────────────────────────────────
    把下面这段换成你自己的模型调用，返回格式保持一致即可：

        your_model = load_your_model(...)
        result = your_model.predict(image_path)
        return {
            "arrhythmia_labels": ["窦性心动过速"],      # list[str]
            "confidence_scores":  {"窦性心动过速": 0.89}, # dict[str, float]
            "max_confidence":     0.89,                  # float
            "is_high_risk":       False,                 # bool
            "summary":            "...",                 # str，自然语言描述
            "pr_interval_ms":     168,                   # int，可选
            "qrs_duration_ms":    92,                    # int，可选
            "qt_interval_ms":     380,                   # int，可选
        }
    ── 替换区域结束 ───────────────────────────────────────────────────────────
    """
    logger.info(f"ECG 分析: {image_path}, 患者年龄: {patient_age}")

    if not Path(image_path).exists():
        logger.warning(f"ECG 图像不存在: {image_path}，使用模拟数据")
    
    return {
        "arrhythmia_labels": ["窦性心动过速", "ST段压低"],
        "confidence_scores": {
            "窦性心动过速": 0.89,
            "ST段压低": 0.76,
            "正常窦性心律": 0.05,
        },
        "max_confidence": 0.89,
        "is_high_risk": False,
        "summary": (
            "ECG 显示窦性心动过速（心率约 105 bpm），"
            "V4-V6 导联可见 ST 段轻度压低（约 0.5mm），"
            "提示可能存在心肌缺血，建议结合临床症状进一步评估。"
        ),
        "pr_interval_ms": 168,
        "qrs_duration_ms": 92,
        "qt_interval_ms": 380,
    }


# ── 工具 2：Echo 心脏超声分析 ─────────────────────────────────────────────────
@tool(
    name="echo_analyze",
    description="分析心脏超声（Echo）图像或视频，计算 LVEF 并检测室壁运动异常、瓣膜病变等结构异常。",
    parameters={
        "echo_path": {
            "type": "string",
            "description": "Echo 图像或视频的本地路径"
        },
        "view": {
            "type": "string",
            "description": "超声切面，可选 'A4C'（心尖四腔）/ 'PLAX'（胸骨旁长轴）/ 'auto'（自动识别）",
        }
    }
)
def echo_analyze(echo_path: str, view: str = "auto") -> dict:
    """
    Echo 超声分析：LVEF 回归 + 室壁运动分类。

    ── 替换区域 ──────────────────────────────────────────────────────────────
    把下面这段换成你自己的模型调用，返回格式保持一致即可：

        your_model = load_your_model(...)
        result = your_model.predict(echo_path, view)
        return {
            "lvef":                      0.48,               # float，0~1
            "lvef_confidence":           0.82,               # float
            "wall_motion_abnormalities": ["前壁心尖段运动减弱"], # list[str]
            "structural_findings":       ["左室轻度扩大"],    # list[str]
            "is_severe":                 False,              # bool
            "summary":                   "...",              # str，自然语言描述
        }
    ── 替换区域结束 ───────────────────────────────────────────────────────────
    """
    logger.info(f"Echo 分析: {echo_path}, 切面: {view}")

    return {
        "lvef": 0.48,
        "lvef_confidence": 0.82,
        "wall_motion_abnormalities": ["前壁心尖段运动减弱"],
        "structural_findings": ["左室轻度扩大", "二尖瓣轻度反流"],
        "is_severe": False,
        "summary": (
            "左室射血分数 48%，轻度降低。"
            "可见前壁心尖段室壁运动减弱，提示局部心肌缺血可能。"
            "左室轻度扩大（LVEDD 约 56mm），二尖瓣轻度反流。"
        ),
    }


# ── 工具 3：患者信息解析 ──────────────────────────────────────────────────────
@tool(
    name="parse_patient_info",
    description="从自然语言描述中提取结构化患者信息，包括年龄、性别、主诉、既往病史、用药史等。",
    parameters={
        "raw_text": {
            "type": "string",
            "description": "用户输入的患者描述文字"
        }
    }
)
def parse_patient_info(raw_text: str) -> dict:
    logger.info(f"解析患者信息: {raw_text[:50]}...")
    return {
        "age": 62,
        "gender": "男",
        "chief_complaint": "胸闷、气短 3 天，活动后加重",
        "history": ["高血压 10 年", "2型糖尿病 5 年"],
        "medications": ["氨氯地平 5mg qd", "二甲双胍 0.5g bid"],
        "allergies": [],
        "smoking": True,
        "family_history": ["父亲有冠心病史"],
    }


# ── 工具 4：生命体征评估 ──────────────────────────────────────────────────────
@tool(
    name="assess_vitals",
    description="评估患者生命体征是否在正常范围，标记异常指标。",
    parameters={
        "vitals": {
            "type": "object",
            "description": "生命体征字典，包括 heart_rate、blood_pressure_sys/dia、spo2、temperature 等"
        }
    }
)
def assess_vitals(vitals: dict) -> dict:
    abnormal = []
    warnings = []

    hr = vitals.get("heart_rate")
    if hr:
        if hr > 100:
            abnormal.append(f"心率偏快: {hr} bpm（正常 60-100）")
            warnings.append("心动过速，注意排查感染、心衰、甲亢等")
        elif hr < 60:
            abnormal.append(f"心率偏慢: {hr} bpm")

    sys_bp = vitals.get("blood_pressure_sys")
    if sys_bp and sys_bp > 140:
        abnormal.append(f"收缩压偏高: {sys_bp} mmHg")
        warnings.append("高血压，结合用药史评估")

    spo2 = vitals.get("spo2")
    if spo2 and spo2 < 95:
        abnormal.append(f"血氧饱和度偏低: {spo2}%（正常 ≥95%）")
        warnings.append("低氧血症，注意心衰或肺部疾病")

    return {
        "abnormal_findings": abnormal,
        "clinical_warnings": warnings,
        "overall_status": "异常" if abnormal else "正常",
    }