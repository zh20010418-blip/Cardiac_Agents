# 心脏病智能诊疗 Agent 系统

基于多智能体协作的心脏病辅助诊断框架

## 整体架构

```
用户输入（患者信息 + ECG图 + Echo视频）
        │
        ▼
┌──────────────────────────────────────────┐
│        DispatcherAgent（主控调度）        │
│  意图解析 / 数据完整性检查 / 追问生成     │
└────┬──────────────┬─────────────┬────────┘
     ▼              ▼             │
 ECGAgent       EchoAgent        │
 ReAct循环      ReAct循环        │
 工具调用       工具调用          │
 临床解读       临床解读          │
     └──────┬────────┘            │
            ▼                    │
     RAG检索（自适应）            │
     置信度低才触发               │
     FAISS向量检索                │
     H1→H2→H3层级分块            │
            ▼                    │
┌──────────────────────────────────────────┐
│       DiagnosisAgent（综合诊断）          │
│  CoT推理 + 结构化报告 + Self-Critique     │
└──────────────────────────────────────────┘
            │
            ▼
     结构化诊断报告 + 长期记忆存储
```

## 项目结构

```
cardiac_agent/
├── config/config.py          # 全局配置（模型路径、超参数）
├── graph/
│   ├── state.py              # AgentState：贯穿流水线的共享状态
│   └── pipeline.py           # 状态机调度器（核心）
├── agents/agents.py          # 四类Agent实现
├── tools/
│   ├── registry.py           # 工具注册 + ToolDispatcher（手写Function Calling）
│   └── medical_tools.py      # 具体工具实现
├── rag/retriever.py          # FAISS检索器 + H1→H2→H3层级分块器
├── memory/memory.py          # 短期(deque滑窗) + 长期(JSON持久化)记忆
├── models/llm.py             # 本地LLM封装（单例+流式输出）
├── data/guidelines/          # 医学指南Markdown文档, 放入你的医学指南 .md 文档，首次运行时自动构建 FAISS 索引
├── tests/test_pipeline.py    # 单元测试（Mock LLM，无需GPU）
├── main.py                   # 程序入口（single/chat/demo三种模式）
└── requirements.txt
```

## 环境要求

- Python 3.9 / 3.10 / 3.11
- PyTorch 2.1.x – 2.3.x（CUDA 11.8 或 12.1）
- 推荐组合：Python 3.10 + PyTorch 2.2.2 + CUDA 12.1
- 其余依赖见 `requirements.txt`

# 安装依赖的包
pip install -r requirements.txt

# 确保本地存在 对应的模型（提前使用ms或者hf下载好模型）
/data/models/Qwen2-7B-Instruct     LLM模型
/data/models/bge-large-zh-v1.5     Embedding模型
/data/models/ecg-classifier        ECG分析模型
/data/models/echo-lvef             ECHO分析模型

## 快速开始

# 演示模式（无需GPU，Mock LLM跑通完整流程）
python main.py --mode demo

# 单次诊断
python main.py --mode single --ecg /data/ecg/001.png --age 62 --complaint "胸闷气短"

# 多轮交互对话
python main.py --mode chat

# 运行测试
pytest tests/ -v
```

## 配置模型路径

编辑 `config/config.py`：

```python
llm_path: str = "/你的路径/Qwen2.5-7B-Instruct"
ecg_model_path: str = "/你的路径/ecg-classifier"
embed_model_path: str = "/你的路径/bge-large-zh-v1.5"
```

## 接入自己的 ECG/Echo 模型

当前 `tools/medical_tools.py` 使用 `MockECGModel` / `MockEchoModel`（见 `/data/models/mock_models.py`）作为占位，用于验证 pipeline 流程。

接入真实模型时，只需替换 `ecg_analyze()` 和 `echo_analyze()` 内的替换区域，返回格式保持一致即可，其他代码不需要改动。


## 核心设计要点

| 模块 | 技术方案 | 说明 |
|------|---------|------|
| Agent调度 | 手写状态机（Stage枚举） | 透明可控，不依赖框架 |
| Function Calling | @tool装饰器 + JSON解析 | 自动注册，鲁棒解析 |
| ReAct循环 | 思考→工具→观察→继续 | BaseAgent._react_loop() |
| RAG检索 | FAISS + 置信度阈值 | 低置信度才触发，减少延时 |
| 文档分块 | H1→H2→H3层级切分 | 保持语义完整，注入元数据 |
| 短期记忆 | deque(maxlen=N) | 滑动窗口，自动淘汰旧消息 |
| 长期记忆 | JSON文件持久化 | 跨会话患者档案 |
| 幻觉抑制 | Self-Critique（最多2轮） | LLM扮演质控专家审核 |
| LLM加载 | 单例模式 | 避免重复加载OOM |
