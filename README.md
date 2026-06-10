# CN-FC-Eval: Chinese Function Calling Evaluation

中文社区第一个公开的 Agent Function Calling 评估集。

## 概述

100 个中文测试用例，覆盖 5 类场景，评估 LLM 在中文语境下的工具调用能力。

| 类别 | 数量 | 说明 |
|------|------|------|
| 简单调用 | 20 | 单一工具、参数明确 |
| 多步调用 | 20 | 需先查后算、先后依赖 |
| 并行调用 | 20 | 多个独立调用同时进行 |
| 嵌套参数 | 20 | 复杂 JSON 嵌套结构 |
| 工具选择推理 | 20 | 多工具中选出正确的 |

## 中文特色

- **口语化表达**: "帮我瞅瞅" "你看能不能" "那个啥" "搞一下"
- **敬语处理**: "麻烦您" "请问能否帮"
- **中文语境**: 微信/钉钉/飞书、中国银行、滴滴/高德、饿了么/美团
- **歧义消解**: "打钱" → 转账而非赚钱, "下单" → 购买而非下线

## 快速开始

```bash
# 安装
pip install cn-fc-eval

# 运行评估（需要 DeepSeek API Key）
export ANTHROPIC_AUTH_TOKEN=your-deepseek-key
python -c "
from cn_fc_eval import load_test_cases, Evaluator, DeepSeekClient

cases = load_test_cases()
client = DeepSeekClient(api_key='your-key')
evaluator = Evaluator(client)
results = evaluator.run(cases)
evaluator.print_report(results)
"
```

## 本地开发

```bash
git clone https://github.com/fafaismygod/cn-fc-eval
cd cn-fc-eval
pip install -e .
python3 run_eval.py               # 完整 100 用例
python3 run_eval.py --quick        # 快速测试 10 用例
python3 run_eval.py --compare      # 对比多个模型
python3 run_eval.py --category nested  # 只测嵌套参数
```

## 评测指标

- **工具名称准确率 (Tool Name Accuracy)**: 选择正确工具的百分比
- **参数匹配准确率 (Argument Accuracy)**: 参数键值匹配的百分比（支持嵌套对比）

## 添加新模型

支持任何 OpenAI-compatible API：

```python
from cn_fc_eval.models import AnyOpenAICompatibleClient

client = AnyOpenAICompatibleClient(
    name="MyModel",
    api_key="...",
    base_url="https://api.example.com/v1",
    model="my-model",
)
```

预配置的国内模型工厂函数：

```python
from cn_fc_eval.models import make_qwen_client, make_glm_client

qwen = make_qwen_client(api_key="your-dashscope-key")
glm = make_glm_client(api_key="your-zhipu-key")
```

## 测试用例格式

```json
{
  "id": "simple-001",
  "category": "simple",
  "query": "帮我查一下北京的天气",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "查询指定城市的实时天气信息",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "城市名称"}
          },
          "required": ["city"]
        }
      }
    }
  ],
  "expected": {
    "name": "get_weather",
    "arguments": {"city": "北京"}
  }
}
```

## 引用

如果这个评估集对你的研究有帮助，请引用：

```bibtex
@misc{cn-fc-eval-2026,
  title={CN-FC-Eval: A Chinese Function Calling Benchmark},
  author={fafa},
  year={2026},
  howpublished={\\url{https://github.com/fafaismygod/cn-fc-eval}},
}
```

## License

MIT
