"""
models/llm.py —— 本地 LLM 统一封装

把 transformers 的加载和推理细节封装在这里。
各 Agent 只调用 generate() 方法，不关心底层实现。

设计要点：
  1. 单例模式：模型只加载一次，避免 OOM
  2. 统一的 chat template 处理（兼容 Qwen / LLaMA / ChatGLM）
  3. 支持流式输出
"""

import logging
import torch
from threading import Lock
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    GenerationConfig,
)
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


class LocalLLM:
    """
    本地 LLM 推理封装。

    用法：
        llm = LocalLLM("/data/models/Qwen2.5-7B-Instruct")
        response = llm.generate(messages=[
            {"role": "system", "content": "你是心脏科医生"},
            {"role": "user",   "content": "患者 ECG 显示 QRS 增宽..."}
        ])
    """

    _instances: dict = {}   # 路径 -> 实例，实现按路径单例
    _lock = Lock()

    def __new__(cls, model_path: str, **kwargs):
        """单例：同一路径的模型只初始化一次。"""
        with cls._lock:
            if model_path not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[model_path] = instance
            return cls._instances[model_path]

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "float16",
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        do_sample: bool = False,
    ):
        if self._initialized:
            return  # 单例：已初始化则跳过

        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        # dtype 映射
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(dtype, torch.float16)

        logger.info(f"加载本地 LLM: {model_path}")
        self._load_model()
        self._initialized = True

    def _load_model(self):
        """加载 tokenizer 和模型。"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map="auto",          # 自动分配多卡，单卡也兼容
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info(f"LLM 加载完成，设备: {self.device}")

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        同步生成。

        Args:
            messages: [{"role": "user/assistant/system", "content": "..."}]
            system_prompt: 如果提供，插入到 messages 最前面作为 system
            max_new_tokens: 覆盖默认值
            temperature: 覆盖默认值

        Returns:
            模型输出的字符串
        """
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        # 使用模型自带的 chat template 处理消息格式
        # Qwen2.5 / LLaMA3 / ChatGLM4 都支持 apply_chat_template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                temperature=temperature or self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 只取新生成的 token（去掉 prompt 部分）
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return response.strip()

    def stream_generate(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
    ) -> Iterator[str]:
        """
        流式生成（用于实时展示诊断过程）。

        用法：
            for chunk in llm.stream_generate(messages):
                print(chunk, end="", flush=True)
        """
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # 在后台线程里跑生成，主线程通过 streamer 迭代获取 token
        from threading import Thread
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for chunk in streamer:
            yield chunk

        thread.join()
