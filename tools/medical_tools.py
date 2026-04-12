"""
tools/medical_tools.py —— 医疗工具实现

工具按 Agent 归属分组：
  DispatcherAgent 专属：parse_patient_info、assess_vitals、patient_risk_score
  ECGAgent        专属：ecg_analyze、ecg_interval_measure
  EchoAgent       专属：echo_analyze、echo_wall_motion_detail
  DiagnosisAgent  专属：guideline_query、drug_recommendation

多Agent的意义：每类Agent只调用自己领域的工具，分工明确，
相比单模型端到端更易替换、调试和扩展。
"""

import logging
import random
from pathlib import Path

from tools.registry import tool

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DispatcherAgent 专属工具
# 职责：患者信息提取、生命体征评估、分诊优先级判断
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="parse_patient_info",
    description="从自然语言描述中提取结构化患者信息，包括年龄、性别、主诉、既往病史、用药史等。供 DispatcherAgent 使用。",
    parameters={
        "raw_text": {"type": "string", "description": "用户输入的患者描述文字"}
    }
)
def parse_patient_info(raw_text: str) -> dict:
    """提取结构化患者信息，替换时接入 NER 模型或 LLM 结构化提取。"""
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


@tool(
    name="assess_vitals",
    description="评估患者生命体征是否在正常范围，标记异常指标。供 DispatcherAgent 使用。",
    parameters={
        "vitals": {
            "type": "object",
            "description": "生命体征字典，包括 heart_rate、blood_pressure_sys/dia、spo2 等"
        }
    }
)
def assess_vitals(vitals: dict) -> dict:
    """生命体征异常判断，纯规则逻辑，无需替换。"""
    abnormal, warnings = [], []

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


@tool(
    name="patient_risk_score",
    description="基于患者基本信息计算心血管风险评分（GRACE/CHA2DS2-VASc），决定分诊优先级。供 DispatcherAgent 使用。",
    parameters={
        "patient_info": {"type": "object", "description": "患者结构化信息字典"},
        "score_type": {
            "type": "string",
            "description": "评分类型：'GRACE'（ACS风险）或 'CHA2DS2'（房颤卒中风险）"
        }
    }
)
def patient_risk_score(patient_info: dict, score_type: str = "GRACE") -> dict:
    """
    心血管风险评分。
    GRACE 评分用于 ACS 患者危险分层，决定是否需要紧急介入。
    CHA2DS2-VASc 用于房颤患者卒中风险评估，决定是否需要抗凝。

    DispatcherAgent 根据评分结果决定：
      - 高危 → 跳过追问，立即进入分析流程
      - 低危 → 允许追问补充信息
    """
    age = patient_info.get("age", 50)
    history = patient_info.get("history", [])
    has_diabetes = any("糖尿病" in h for h in history)
    has_hypertension = any("高血压" in h for h in history)

    if score_type == "CHA2DS2":
        score = 0
        if age >= 75: score += 2
        elif age >= 65: score += 1
        if has_hypertension: score += 1
        if has_diabetes: score += 1
        if patient_info.get("gender") == "女": score += 1
        risk_level = "高危" if score >= 2 else "中危" if score == 1 else "低危"
        recommendation = "推荐口服抗凝治疗" if score >= 2 else "根据临床判断决定是否抗凝"
        return {
            "score_type": "CHA2DS2-VASc",
            "score": score,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "urgent": score >= 3,
        }
    else:  # GRACE
        # 模拟 GRACE 评分（真实版需要肌酐、心率、收缩压等）
        base = 80
        if age > 65: base += 30
        if has_diabetes: base += 15
        if has_hypertension: base += 10
        score = base + random.randint(-10, 10)
        risk_level = "高危" if score > 140 else "中危" if score > 108 else "低危"
        return {
            "score_type": "GRACE",
            "score": score,
            "risk_level": risk_level,
            "recommendation": "建议24小时内行冠脉造影" if score > 140 else "建议72小时内行冠脉造影",
            "urgent": score > 140,
        }


# ══════════════════════════════════════════════════════════════════════════════
# ECGAgent 专属工具
# 职责：心律失常识别 + 关键间期测量，两个工具分工明确
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="ecg_analyze",
    description="分析 ECG 图像，识别心律失常类型并给出置信度分布。供 ECGAgent 使用。",
    parameters={
        "image_path": {"type": "string", "description": "ECG 图像本地路径"},
        "patient_age": {"type": "integer", "description": "患者年龄，用于调整正常范围"}
    }
)
def ecg_analyze(image_path: str, patient_age: int = 50) -> dict:
    """
    ECG 心律失常分类。
    ── 替换区域：换成你的模型，返回格式保持一致 ──────────────────────────────
        return {
            "arrhythmia_labels": ["窦性心动过速"],   # list[str]
            "confidence_scores":  {"窦性心动过速": 0.89},
            "max_confidence":     0.89,
            "is_high_risk":       False,
            "summary":            "...",
        }
    ── 替换区域结束 ───────────────────────────────────────────────────────────
    """
    logger.info(f"ECG 心律失常分析: {image_path}")
    from data.models.mock_models import MockECGModel
    return MockECGModel().predict(image_path, patient_age)


@tool(
    name="ecg_interval_measure",
    description="测量 ECG 关键间期指标：PR间期、QRS时限、QT/QTc间期，判断传导异常。供 ECGAgent 使用。",
    parameters={
        "image_path": {"type": "string", "description": "ECG 图像本地路径"},
        "heart_rate": {"type": "integer", "description": "心率（bpm），用于计算 QTc"}
    }
)
def ecg_interval_measure(image_path: str, heart_rate: int = 75) -> dict:
    """
    ECG 间期测量，与 ecg_analyze 分工：
      - ecg_analyze   → 识别"是什么心律"
      - ecg_interval_measure → 测量"各间期是否正常"

    两个工具配合让 ECGAgent 的分析更完整，
    单个多模态模型很难同时做好分类和精确测量。

    ── 替换区域：接入你的间期测量模型 ───────────────────────────────────────
    """
    logger.info(f"ECG 间期测量: {image_path}, 心率: {heart_rate}")

    pr  = random.randint(120, 220)
    qrs = random.randint(80, 130)
    qt  = random.randint(350, 480)
    # QTc = QT / sqrt(RR)，RR = 60000/heart_rate ms
    import math
    rr_ms = 60000 / heart_rate
    qtc = round(qt / math.sqrt(rr_ms / 1000))

    abnormal = []
    if pr > 200:
        abnormal.append(f"PR间期延长（{pr}ms > 200ms），提示一度房室传导阻滞")
    if qrs > 120:
        abnormal.append(f"QRS时限增宽（{qrs}ms > 120ms），提示束支传导阻滞或室性起搏")
    if qtc > 450:
        abnormal.append(f"QTc延长（{qtc}ms > 450ms），注意尖端扭转型室速风险")

    return {
        "pr_interval_ms":  pr,
        "qrs_duration_ms": qrs,
        "qt_interval_ms":  qt,
        "qtc_interval_ms": qtc,
        "abnormal_intervals": abnormal,
        "summary": (
            f"PR {pr}ms，QRS {qrs}ms，QTc {qtc}ms。"
            + ("传导间期异常：" + "；".join(abnormal) if abnormal else "各间期均在正常范围内。")
        )
    }


# ══════════════════════════════════════════════════════════════════════════════
# EchoAgent 专属工具
# 职责：LVEF 计算 + 室壁运动逐节段分析，两个工具分工明确
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="echo_analyze",
    description="分析心脏超声影像，计算 LVEF 并识别结构异常（瓣膜病变、心腔扩大等）。供 EchoAgent 使用。",
    parameters={
        "echo_path": {"type": "string", "description": "Echo 图像或视频本地路径"},
        "view": {
            "type": "string",
            "description": "超声切面：'A4C'（心尖四腔）/ 'PLAX'（胸骨旁长轴）/ 'auto'"
        }
    }
)
def echo_analyze(echo_path: str, view: str = "auto") -> dict:
    """
    Echo 整体评估：LVEF + 结构异常。
    ── 替换区域：换成你的模型，返回格式保持一致 ──────────────────────────────
        return {
            "lvef":                      0.48,
            "lvef_confidence":           0.82,
            "wall_motion_abnormalities": ["前壁心尖段运动减弱"],
            "structural_findings":       ["左室轻度扩大"],
            "is_severe":                 False,
            "summary":                   "...",
        }
    ── 替换区域结束 ───────────────────────────────────────────────────────────
    """
    logger.info(f"Echo 整体分析: {echo_path}, 切面: {view}")
    from data.models.mock_models import MockEchoModel
    return MockEchoModel().predict(echo_path, view)

@tool(
    name="echo_wall_motion_detail",
    description="对心脏超声进行逐节段室壁运动评分（AHA 17节段模型），定位缺血区域和冠脉供血范围。供 EchoAgent 使用。",
    parameters={
        "echo_path": {"type": "string", "description": "Echo 视频本地路径"},
        "segments":  {
            "type": "array",
            "description": "需要评估的节段列表，默认评估全部17节段，例如 ['前壁基底段', '前间隔中间段']"
        }
    }
)
def echo_wall_motion_detail(echo_path: str, segments: list = None) -> dict:
    """
    逐节段室壁运动评分，与 echo_analyze 分工：
      - echo_analyze          → 整体 LVEF + 有没有异常
      - echo_wall_motion_detail → 哪个节段异常、对应哪条冠脉、严重程度几级

    这个工具的输出直接影响 DiagnosisAgent 的鉴别诊断：
      前壁+前间隔异常 → 提示 LAD（左前降支）病变
      下壁+后壁异常   → 提示 RCA（右冠状动脉）病变

    ── 替换区域：接入你的节段分析模型 ───────────────────────────────────────
    """
    logger.info(f"Echo 逐节段室壁运动评分: {echo_path}")

    # AHA 17节段模型（模拟输出）
    all_segments = [
        "前壁基底段", "前间隔基底段", "下间隔基底段", "下壁基底段", "下侧壁基底段", "前侧壁基底段",
        "前壁中间段", "前间隔中间段", "下间隔中间段", "下壁中间段", "下侧壁中间段", "前侧壁中间段",
        "前壁心尖段", "间隔心尖段", "下壁心尖段", "侧壁心尖段", "心尖帽"
    ]
    target = segments if segments else all_segments

    # 运动评分：1=正常，2=运动减弱，3=无运动，4=矛盾运动
    score_map = {seg: random.choice([1, 1, 1, 2, 3]) for seg in target}
    # 模拟前壁相关节段异常
    for seg in ["前壁心尖段", "前间隔中间段"]:
        if seg in score_map:
            score_map[seg] = 2

    abnormal_segs = {seg: s for seg, s in score_map.items() if s > 1}
    score_labels = {1: "正常", 2: "运动减弱", 3: "无运动", 4: "矛盾运动"}

    # 根据异常节段推断受累冠脉
    lad_segs = {"前壁基底段", "前壁中间段", "前壁心尖段", "前间隔基底段", "前间隔中间段", "间隔心尖段"}
    rca_segs = {"下壁基底段", "下壁中间段", "下壁心尖段", "下间隔基底段", "下间隔中间段"}
    lcx_segs = {"前侧壁基底段", "前侧壁中间段", "下侧壁基底段", "下侧壁中间段", "侧壁心尖段"}

    involved = set(abnormal_segs.keys())
    culprit = []
    if involved & lad_segs: culprit.append("LAD（左前降支）")
    if involved & rca_segs: culprit.append("RCA（右冠状动脉）")
    if involved & lcx_segs: culprit.append("LCX（左回旋支）")

    wmsi = round(sum(score_map.values()) / len(score_map), 2)  # 室壁运动指数

    return {
        "segment_scores": {seg: {"score": s, "label": score_labels[s]}
                           for seg, s in score_map.items()},
        "abnormal_segments": [f"{seg}（{score_labels[s]}）" for seg, s in abnormal_segs.items()],
        "wall_motion_score_index": wmsi,       # >1 提示整体功能受损
        "suspected_culprit_vessels": culprit,  # 推断受累冠脉
        "summary": (
            f"室壁运动指数 WMSI={wmsi}（正常=1.0）。"
            + (f"异常节段：{'、'.join(abnormal_segs.keys())}，提示 {'/'.join(culprit)} 供血区域缺血。"
               if abnormal_segs else "各节段室壁运动未见明显异常。")
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DiagnosisAgent 专属工具
# 职责：指南查询 + 用药建议，基于 ECG/Echo 结果做决策支持
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="guideline_query",
    description="根据诊断结果查询相关临床指南的治疗推荐级别（ACC/AHA/ESC）。供 DiagnosisAgent 使用。",
    parameters={
        "diagnosis": {"type": "string", "description": "主诊断名称，例如 '心房颤动' 或 'NSTEMI'"},
        "clinical_scenario": {"type": "string", "description": "临床场景描述，例如 'LVEF<40%合并心衰'"}
    }
)
def guideline_query(diagnosis: str, clinical_scenario: str = "") -> dict:
    """
    临床指南查询，返回对应诊断的推荐治疗方案和证据级别。
    生产环境接入 RAG 检索（已在 pipeline 中实现），
    这里提供规则兜底，确保即使 RAG 为空也能返回基础推荐。
    """
    logger.info(f"指南查询: {diagnosis} | {clinical_scenario}")

    # 简化的规则库（生产环境由 RAG 替代）
    guidelines_db = {
        "心房颤动": {
            "recommendation": "CHA2DS2-VASc≥2分推荐口服抗凝（I类，A级证据）；优先选择NOAC",
            "rate_control": "目标静息心率<110bpm；首选β受体阻滞剂或非二氢吡啶类CCB",
            "source": "ACC/AHA 2023 房颤管理指南",
        },
        "NSTEMI": {
            "recommendation": "高危患者24h内行冠脉造影（I类，A级证据）",
            "antiplatelet": "阿司匹林300mg+替格瑞洛180mg负荷，随后维持双联抗血小板",
            "source": "ESC 2023 ACS指南",
        },
        "心力衰竭": {
            "recommendation": "HFrEF推荐ARNI+β受体阻滞剂+MRA+SGLT2i四联治疗（I类，A级证据）",
            "lvef_threshold": "LVEF<40%启动规范化药物治疗",
            "source": "ACC/AHA 2022 心衰指南",
        },
    }

    # 模糊匹配
    matched = None
    for key in guidelines_db:
        if key in diagnosis or diagnosis in key:
            matched = guidelines_db[key]
            break

    if matched:
        return {"diagnosis": diagnosis, "guideline": matched, "found": True}
    return {
        "diagnosis": diagnosis,
        "guideline": {"recommendation": "请参考最新相关专科指南，结合临床判断"},
        "found": False,
    }
@tool(
    name="drug_recommendation",
    description="根据诊断、LVEF、肾功能等信息给出用药建议，并检查与当前用药的潜在冲突。供 DiagnosisAgent 使用。",
    parameters={
        "diagnosis":        {"type": "string",  "description": "主诊断"},
        "lvef":             {"type": "number",  "description": "左室射血分数（0~1）"},
        "current_meds":     {"type": "array",   "description": "当前用药列表"},
        "contraindications":{"type": "array",   "description": "禁忌症列表，例如 ['严重肾功能不全']"}
    }
)
def drug_recommendation(
    diagnosis: str,
    lvef: float = 0.55,
    current_meds: list = None,
    contraindications: list = None,
) -> dict:
    """
    用药建议 + 冲突检查。
    DiagnosisAgent 在生成诊断报告后调用此工具，
    确保治疗建议与患者当前用药无冲突，提升安全性。

    生产环境可接入药物数据库 API（如 DrugBank）做精确冲突检测。
    """
    logger.info(f"用药建议: {diagnosis}, LVEF={lvef:.0%}")
    current_meds = current_meds or []
    contraindications = contraindications or []

    recommendations = []
    conflicts = []
    warnings = []

    if "心力衰竭" in diagnosis or lvef < 0.40:
        recommendations.append("ARNI（沙库巴曲缬沙坦）：起始剂量 50mg bid，逐步上调")
        recommendations.append("β受体阻滞剂：卡维地洛 3.125mg bid 起始")
        recommendations.append("SGLT2抑制剂：达格列净 10mg qd（无论是否合并糖尿病）")
        if "严重肾功能不全" in contraindications:
            warnings.append("SGLT2i 在 eGFR<20 时需谨慎，请评估肾功能")

    if "心房颤动" in diagnosis:
        recommendations.append("利伐沙班 20mg qd 或 达比加群 150mg bid（抗凝治疗）")
        # 检查冲突：已在用抗血小板药
        if any("阿司匹林" in m for m in current_meds):
            conflicts.append("已在使用阿司匹林，联合抗凝治疗显著增加出血风险，需评估利弊")

    if "冠心病" in diagnosis or "心绞痛" in diagnosis:
        recommendations.append("阿托伐他汀 40mg qn（高强度他汀降脂）")
        recommendations.append("阿司匹林 100mg qd + 氯吡格雷 75mg qd（双联抗血小板）")

    if not recommendations:
        recommendations.append("请结合完整检查结果和临床判断制定个体化方案")

    return {
        "recommendations": recommendations,
        "drug_conflicts":  conflicts,
        "warnings":        warnings,
        "summary": (
            f"针对{diagnosis}的用药建议：{'；'.join(recommendations[:2])}等。"
            + (f" 注意冲突：{'；'.join(conflicts)}" if conflicts else "")
        ),
    }
