"""
tools/registry.py —— 工具注册与调用系统

这是手写 Function Calling 的核心。
设计思路：
  1. 用装饰器 @tool 注册工具，自动提取函数签名作为 schema
  2. LLM 收到工具描述后，输出 JSON 格式的调用指令
  3. ToolDispatcher 解析 JSON，路由到对应函数执行
  4. 结果以自然语言摘要返回给 LLM 继续推理

这就是 ReAct（Reasoning + Acting）的 Acting 部分。
"""

import json
import logging
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional
from functools import wraps

logger = logging.getLogger(__name__)


# ── 工具定义 ──────────────────────────────────────────────────────────────────
@dataclass
class ToolSchema:
    """描述一个工具的结构，用于构造 LLM 的 system prompt。"""
    name: str
    description: str
    parameters: dict       # JSON Schema 格式的参数描述
    func: Callable         # 实际执行的函数


# 全局工具注册表：name -> ToolSchema
_TOOL_REGISTRY: dict[str, ToolSchema] = {}


def tool(name: str, description: str, parameters: dict):
    """
    工具注册装饰器。

    用法：
        @tool(
            name="ecg_analyze",
            description="分析 ECG 图像，检测心律失常",
            parameters={
                "image_path": {"type": "string", "description": "ECG 图像路径"},
            }
        )
        def ecg_analyze(image_path: str) -> dict:
            ...
    """
    def decorator(func: Callable):
        _TOOL_REGISTRY[name] = ToolSchema(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
        )
        logger.debug(f"注册工具: {name}")

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_all_tools() -> list[ToolSchema]:
    """返回所有已注册工具。"""
    return list(_TOOL_REGISTRY.values())


def get_tool_prompt() -> str:
    """
    生成工具描述的 system prompt 片段。
    注入到 LLM 的 system prompt 里，让 LLM 知道有哪些工具可用。
    """
    lines = ["## 可用工具\n"]
    lines.append("当你需要调用工具时，输出以下格式的 JSON（不要输出其他内容）：")
    lines.append('```json\n{"tool": "工具名", "args": {"参数名": "参数值"}}\n```\n')
    lines.append("### 工具列表\n")

    for schema in _TOOL_REGISTRY.values():
        lines.append(f"**{schema.name}**: {schema.description}")
        lines.append(f"参数: {json.dumps(schema.parameters, ensure_ascii=False)}\n")

    return "\n".join(lines)


# ── 工具调用解析器 ────────────────────────────────────────────────────────────
class ToolDispatcher:
    """
    解析 LLM 输出中的工具调用指令，路由执行，返回结果。

    ReAct 循环：
        Thought → Action（工具调用 JSON）→ Observation（工具结果）→ 继续推理
    """

    def parse_tool_call(self, llm_output: str) -> Optional[dict]:
        """
        从 LLM 输出中提取工具调用 JSON。
        LLM 可能在 JSON 前后输出一些思考文字，需要鲁棒地提取。

        Returns:
            {"tool": "...", "args": {...}} 或 None（没有工具调用）
        """
        # 尝试提取 ```json ... ``` 代码块
        if "```json" in llm_output:
            start = llm_output.find("```json") + 7
            end = llm_output.find("```", start)
            json_str = llm_output[start:end].strip()
        # 尝试直接提取 { } 块
        elif "{" in llm_output and "}" in llm_output:
            start = llm_output.find("{")
            end = llm_output.rfind("}") + 1
            json_str = llm_output[start:end]
        else:
            return None  # 没有工具调用

        try:
            call = json.loads(json_str)
            if "tool" in call and "args" in call:
                return call
        except json.JSONDecodeError as e:
            logger.warning(f"工具调用 JSON 解析失败: {e}\n原文: {json_str}")
        return None

    def execute(self, tool_name: str, args: dict) -> str:
        """
        执行工具并返回字符串结果（作为 Observation 喂回 LLM）。

        Returns:
            工具执行结果的字符串描述
        """
        if tool_name not in _TOOL_REGISTRY:
            return f"[错误] 未知工具: {tool_name}，可用工具: {list(_TOOL_REGISTRY.keys())}"

        schema = _TOOL_REGISTRY[tool_name]
        logger.info(f"执行工具: {tool_name}，参数: {args}")

        try:
            result = schema.func(**args)
            # 结果统一转成字符串
            if isinstance(result, dict):
                return json.dumps(result, ensure_ascii=False, indent=2)
            return str(result)
        except Exception as e:
            logger.error(f"工具 {tool_name} 执行失败: {e}", exc_info=True)
            return f"[错误] 工具执行失败: {e}"

    def run(self, llm_output: str) -> tuple[bool, str]:
        """
        完整的一次工具调用流程。

        Returns:
            (has_tool_call, observation_text)
            - has_tool_call: 是否检测到工具调用
            - observation_text: 工具执行结果（格式化为 Observation）
        """
        call = self.parse_tool_call(llm_output)
        if call is None:
            return False, ""

        result = self.execute(call["tool"], call.get("args", {}))
        observation = f"[工具执行结果 - {call['tool']}]\n{result}"
        return True, observation
