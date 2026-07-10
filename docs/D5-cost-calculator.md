# Spec 5: 重构 cost_calculator.py — chat 走原路径，embedding 走简化分支

> 交付物：重构 `cost_calculator.py`：chat 模型保留 tiered/flat pricing 路径，embedding 模型走简化分支（text + image token 直乘，无 tiered/cache/output）
> 前置依赖：D1 + D4（已完成）

---

## 核心结论

**DashScope 多模态 embedding 计费单位是 token，不是"张"或"秒"。** 图片/视频经过 token 化后按 `image_tokens` 计费，部分模型的图片/视频 token 与文本 token 单价不同。

| 模型 | 文本单价 | 图片/视频 token 单价 |
|------|---------|-------------------|
| qwen3-vl-embedding | 0.7元/百万token | 1.8元/百万token（更贵） |
| qwen2.5-vl-embedding | 0.7元/百万token | 1.8元/百万token（更贵） |
| multimodal-embedding-v1 | 0.7元/百万token | 0.9元/百万token（更贵） |
| tongyi-embedding-vision-*** | 0.5元/百万token | 同左（统一价） |

### 架构决策：chat 与 embedding 路径分离

chat 模型（`qwen-max`、`qwen-flash` 等）与 embedding 模型（`multimodal-embedding-v1` 等）的计费结构完全不同：

| 维度 | chat 模型 | embedding 模型 |
|------|----------|---------------|
| 定价结构 | tiered 或 flat | 仅 flat |
| output cost | 有（`output_cost_per_token`） | 无（恒为 0） |
| cache cost | 有（`cache_read_input_token_cost`） | 无 |
| reasoning cost | 有（`output_cost_per_reasoning_token`） | 无 |
| image token 独立单价 | 无 | 有（`input_cost_per_image_token`） |

因此 `cost_per_token()` 按 `model_info["mode"]` 分支，两条路径互不干扰。

---

## 验证标准

```bash
# 1. 纯文本输入：只收文本 token 费
pytest tests/test_litellm/llms/dashscope/test_dashscope_cost_calculator.py::TestDashscopeCostCalculator::test_multimodal_text_only -v
# → PASSED

# 2. 图片输入：文本 + 图片 token 分别计费（不同价）
pytest tests/test_litellm/llms/dashscope/test_dashscope_cost_calculator.py::TestDashscopeCostCalculator::test_multimodal_image_tokens -v
# → PASSED

# 3. tongyi-embedding-vision-plus：图片和文本同价
pytest tests/test_litellm/llms/dashscope/test_dashscope_cost_calculator.py::TestDashscopeCostCalculator::test_tongyi_image_tokens_same_price -v
# → PASSED

# 4. 顶层 image_tokens 字段（qwen3-vl-embedding）
pytest tests/test_litellm/llms/dashscope/test_dashscope_cost_calculator.py::TestDashscopeCostCalculator::test_top_level_image_tokens -v
# → PASSED

# 5. 模型无 pricing entry → 不抛异常
pytest tests/test_litellm/llms/dashscope/test_dashscope_cost_calculator.py::TestDashscopeCostCalculator::test_missing_pricing_entry -v
# → PASSED

# 6. 全部计费测试通过
pytest tests/test_litellm/llms/dashscope/test_dashscope_cost_calculator.py -v
# → 全部 PASSED
```

---

## 文件结构

```
litellm/llms/dashscope/
├── cost_calculator.py                  # 重构：chat 保留原路径，embedding 走简化分支
└── embed/
    ├── __init__.py
    ├── transformation.py
    └── transformation_multimodal.py    # 修改：填充 prompt_tokens_details.image_tokens + text_tokens

tests/test_litellm/llms/dashscope/
└── test_dashscope_cost_calculator.py   # 追加：多模态计费测试用例
```

---

## 核心代码

### `cost_calculator.py` — 重构

```python
"""
Cost calculator for Dashscope models.

Chat path: tiered or flat pricing, with cache and reasoning token support.
Embedding path: flat text + image token pricing (no tiered, no cache, no output).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from litellm.types.utils import ModelInfo, Usage
from litellm.utils import get_model_info


@dataclass
class TokenBreakdown:
    text_tokens: int
    cached_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    image_tokens: int  # 新增


def _extract_token_breakdown(usage: Usage) -> TokenBreakdown:
    cached_tokens = 0
    image_tokens = 0
    if usage.prompt_tokens_details:
        if hasattr(usage.prompt_tokens_details, "cached_tokens"):
            cached_tokens = usage.prompt_tokens_details.cached_tokens or 0
        if hasattr(usage.prompt_tokens_details, "image_tokens"):
            image_tokens = usage.prompt_tokens_details.image_tokens or 0

    text_tokens = max(0, usage.prompt_tokens - cached_tokens - image_tokens)

    reasoning_tokens = 0
    if (
        hasattr(usage, "completion_tokens_details")
        and usage.completion_tokens_details
        and hasattr(usage.completion_tokens_details, "reasoning_tokens")
    ):
        reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0

    completion_tokens = (usage.completion_tokens or 0) - reasoning_tokens

    return TokenBreakdown(text_tokens, cached_tokens, completion_tokens, reasoning_tokens, image_tokens)


# ---- tiered cost helper（原样保留） ----
def _calculate_tiered_cost(
    tokens: int,
    tiered_pricing: List[dict],
    cost_key: str,
    fallback_cost_key: Optional[str] = None,
) -> float:
    # ... 原样 ...


# ---- chat 路径（原 _calculate_prompt_cost，重命名） ----
def _calculate_prompt_cost_chat(
    breakdown: TokenBreakdown,
    model_info: ModelInfo,
    tiered_pricing: Optional[List[dict]],
) -> float:
    if tiered_pricing:
        text_cost = _calculate_tiered_cost(
            tokens=breakdown.text_tokens,
            tiered_pricing=tiered_pricing,
            cost_key="input_cost_per_token",
        )
        cache_cost = _calculate_tiered_cost(
            tokens=breakdown.cached_tokens,
            tiered_pricing=tiered_pricing,
            cost_key="cache_read_input_token_cost",
            fallback_cost_key="input_cost_per_token",
        )
        return text_cost + cache_cost

    input_cost = float(model_info.get("input_cost_per_token") or 0.0)
    cache_cost_val = model_info.get("cache_read_input_token_cost")
    if cache_cost_val is None:
        cache_cost = input_cost
    else:
        cache_cost = float(cache_cost_val)

    return (breakdown.text_tokens * input_cost) + (breakdown.cached_tokens * cache_cost)


# ---- chat 路径 completion（原样保留） ----
def _calculate_completion_cost_chat(
    breakdown: TokenBreakdown,
    model_info: ModelInfo,
    tiered_pricing: Optional[List[dict]],
) -> float:
    # ... 原样 ...


# ---- embedding 路径（新增） ----
def _calculate_prompt_cost_embedding(
    breakdown: TokenBreakdown, model_info: ModelInfo
) -> float:
    text_unit_price = float(model_info.get("input_cost_per_token") or 0.0)
    image_token_price = model_info.get("input_cost_per_image_token")
    if image_token_price is not None and breakdown.image_tokens:
        image_unit_price = float(image_token_price)
    else:
        image_unit_price = text_unit_price

    return breakdown.text_tokens * text_unit_price + breakdown.image_tokens * image_unit_price


# ---- 入口函数 ----
def cost_per_token(model: str, usage: Usage) -> Tuple[float, float]:
    try:
        model_info = get_model_info(model=model, custom_llm_provider="dashscope")
    except Exception:
        import logging
        logging.getLogger(__name__).warning(
            "No pricing entry found for dashscope model=%s; returning 0 cost", model
        )
        return 0.0, 0.0

    breakdown = _extract_token_breakdown(usage)
    mode = model_info.get("mode")

    if mode == "embedding":
        prompt_cost = _calculate_prompt_cost_embedding(breakdown, model_info)
        return prompt_cost, 0.0

    # chat 路径
    tiered_pricing = (
        model_info.get("tiered_pricing")
        if isinstance(model_info.get("tiered_pricing"), list)
        else None
    )
    prompt_cost = _calculate_prompt_cost_chat(breakdown, model_info, tiered_pricing)
    completion_cost = _calculate_completion_cost_chat(breakdown, model_info, tiered_pricing)
    return prompt_cost, completion_cost
```

### `transformation_multimodal.py` — 修改 `transform_embedding_response()`

在构造 `Usage` 时填充 `prompt_tokens_details.image_tokens` 和 `prompt_tokens_details.text_tokens`：

```python
def transform_embedding_response(self, model: str,
                                 raw_response: httpx.Response,
                                 model_response: EmbeddingResponse,
                                 logging_obj: LiteLLMLoggingObj,
                                 api_key: Optional[str],
                                 request_data: dict,
                                 optional_params: dict,
                                 litellm_params: dict) -> EmbeddingResponse:
    # ... 前面的 JSON 解析和错误处理不变 ...

    usage = response_json.get("usage") or {}
    input_tokens = usage.get("input_tokens", 0)
    total_tokens = usage.get("total_tokens", input_tokens)

    # 提取 image_tokens 和 text_tokens
    # - tongyi-embedding-vision-*: 嵌套在 input_tokens_details
    # - qwen3-vl-embedding / multimodal-embedding-v1: 顶层 image_tokens 字段
    text_tokens = None
    image_tokens = None
    if "input_tokens_details" in usage:
        image_tokens = usage["input_tokens_details"].get("image_tokens")
        text_tokens = usage["input_tokens_details"].get("text_tokens")
    elif "image_tokens" in usage:
        image_tokens = usage["image_tokens"]

    prompt_tokens_details = None
    if image_tokens is not None or text_tokens is not None:
        from litellm.types.utils import PromptTokensDetailsWrapper
        prompt_tokens_details = PromptTokensDetailsWrapper(
            image_tokens=image_tokens,
            text_tokens=text_tokens,
        )

    setattr(
        model_response,
        "usage",
        Usage(
            prompt_tokens=input_tokens,
            completion_tokens=0,
            total_tokens=total_tokens,
            prompt_tokens_details=prompt_tokens_details,
        ),
    )
    # ... 后续不变 ...
```

### 测试用例 — 追加到 `test_dashscope_cost_calculator.py`

```python
def test_multimodal_text_only(self):
    """纯文本输入：只收文本 token 费（image_tokens=0）"""
    usage = Usage(
        prompt_tokens=100, completion_tokens=0,
        prompt_tokens_details=PromptTokensDetailsWrapper(image_tokens=0)
    )
    prompt_cost, _ = dashscope_cost_per_token(
        model="multimodal-embedding-v1", usage=usage
    )
    model_info = litellm.get_model_info("dashscope/multimodal-embedding-v1")
    expected = 100 * model_info["input_cost_per_token"]
    assert math.isclose(prompt_cost, expected, rel_tol=1e-10)

def test_multimodal_image_tokens(self):
    """图片输入：文本 + 图片 token 分别计费（不同价）

    multimodal-embedding-v1: 文本 0.0007元, 图片 0.0009元
    100 文本 token + 896 图片 token
    = 100 * 7e-7 + 896 * 9e-7
    """
    usage = Usage(
        prompt_tokens=996, completion_tokens=0,
        prompt_tokens_details=PromptTokensDetailsWrapper(image_tokens=896)
    )
    prompt_cost, _ = dashscope_cost_per_token(
        model="multimodal-embedding-v1", usage=usage
    )
    model_info = litellm.get_model_info("dashscope/multimodal-embedding-v1")
    text_cost = 100 * model_info["input_cost_per_token"]
    image_cost = 896 * model_info["input_cost_per_image_token"]
    expected = text_cost + image_cost
    assert math.isclose(prompt_cost, expected, rel_tol=1e-10)

def test_tongyi_image_tokens_same_price(self):
    """tongyi-embedding-vision-plus：图片和文本同价（无 image_token 单价字段）

    即使 usage 中有 image_tokens，因为模型无 input_cost_per_image_token 字段，
    _calculate_prompt_cost_embedding 回退到 input_cost_per_token。
    """
    usage = Usage(
        prompt_tokens=100, completion_tokens=0,
        prompt_tokens_details=PromptTokensDetailsWrapper(image_tokens=50)
    )
    prompt_cost, _ = dashscope_cost_per_token(
        model="tongyi-embedding-vision-plus", usage=usage
    )
    model_info = litellm.get_model_info("dashscope/tongyi-embedding-vision-plus")
    expected = 100 * model_info["input_cost_per_token"]
    assert math.isclose(prompt_cost, expected, rel_tol=1e-10)

def test_top_level_image_tokens(self):
    """qwen3-vl-embedding：顶层 image_tokens 字段"""
    usage = Usage(
        prompt_tokens=7, completion_tokens=0,
        prompt_tokens_details=PromptTokensDetailsWrapper(image_tokens=896)
    )
    prompt_cost, _ = dashscope_cost_per_token(
        model="qwen3-vl-embedding", usage=usage
    )
    model_info = litellm.get_model_info("dashscope/qwen3-vl-embedding")
    text_cost = 7 * model_info["input_cost_per_token"]
    image_cost = 896 * model_info["input_cost_per_image_token"]
    expected = text_cost + image_cost
    assert math.isclose(prompt_cost, expected, rel_tol=1e-10)

def test_missing_pricing_entry(self):
    """模型无 pricing entry（如 unknown-model）→ 不抛异常，返回 (0, 0)"""
    usage = Usage(prompt_tokens=100, completion_tokens=0)
    prompt_cost, completion_cost = dashscope_cost_per_token(
        model="unknown-model", usage=usage
    )
    assert prompt_cost == 0.0
    assert completion_cost == 0.0

def test_no_prompt_tokens_details(self):
    """usage.prompt_tokens_details 为 None：不抛异常"""
    usage = Usage(prompt_tokens=100, completion_tokens=0)
    prompt_cost, _ = dashscope_cost_per_token(
        model="multimodal-embedding-v1", usage=usage
    )
    model_info = litellm.get_model_info("dashscope/multimodal-embedding-v1")
    expected = 100 * model_info["input_cost_per_token"]
    assert math.isclose(prompt_cost, expected, rel_tol=1e-10)
```

---

## 设计说明

### 为什么 chat 和 embedding 拆成两条路径

DashScope 的 chat 模型和 embedding 模型的计费结构完全不同：

| 特性 | chat 模型 | embedding 模型 |
|------|----------|---------------|
| 定价结构 | tiered 或 flat | 仅 flat |
| output cost | 有 | 无（恒为 0） |
| cache cost | 有 | 无 |
| reasoning cost | 有 | 无 |
| image token 独立单价 | 无 | 有（`input_cost_per_image_token`） |

将两条路径放在同一个 `_calculate_prompt_cost` 里用条件判断处理，会引入大量不必要的分支。按 `mode` 分支后，embedding 路径只需 3 行纯乘法，chat 路径完全不变。

### 为什么 `text_tokens` 要排除 `image_tokens`

`_extract_token_breakdown` 中 `text_tokens = prompt_tokens - cached_tokens - image_tokens` 匹配 litellm 通用 cost calculator 的模式（`llm_cost_calc/utils.py:725`）。如果不排除，`_calculate_prompt_cost` 会把 image token 也按文本单价计费，然后需要额外的"撤销"逻辑。排除后，embedding 路径只需直接加 `image_tokens * image_unit_price`，无需任何修正。

### 为什么 embedding 路径的 image token 单价 fallback 到 text 单价

tongyi-embedding-vision-*** 系列没有 `input_cost_per_image_token` 字段（图/文本同价）。当该字段缺失时，`_calculate_prompt_cost_embedding` 回退到 `input_cost_per_token`，等价于 `(text_tokens + image_tokens) * text_unit_price`，与 tongyi 的计费行为一致。

### 为什么修改 `transformation_multimodal.py` 而非在 cost calculator 中解析原始 usage

DashScope 多模态 API 返回的 usage 格式因模型而异：

| 模型 | usage 格式 |
|------|-----------|
| `tongyi-embedding-vision-*` | `input_tokens` 含图片，嵌套 `input_tokens_details{image_tokens, text_tokens}` |
| `qwen3-vl-embedding` | `input_tokens` 仅文本，顶层 `image_tokens` |
| `qwen2.5-vl-embedding` | `input_tokens` 仅文本，顶层 `image_tokens` |
| `multimodal-embedding-v1` | `input_tokens` 仅文本，顶层 `image_tokens` |

在 transformation 层统一归一化到 `PromptTokensDetailsWrapper`，cost calculator 只需读取标准字段，不关心具体模型。

### 边界情况

| 场景 | 处理方式 |
|------|----------|
| 模型无 pricing entry（如 unknown-model） | try/except 捕获异常，log warning，返回 (0, 0) |
| embedding 模型有 `input_cost_per_image_token` 但 `image_tokens=0` | 只收 text 费 |
| embedding 模型无 `input_cost_per_image_token` 字段（tongyi 系列） | image 走 text 单价 |
| `prompt_tokens_details` 为 None | `image_tokens=0`，安全处理 |
| `image_tokens` 缺失 | 视为 0 |
| `input_cost_per_token` 为 0（免费模型） | `image_tokens * image_token_price`（仍按图片价收费） |

---

## 数据模型

### Usage 字段流转

```
DashScope API response usage
  ├── input_tokens: int              → Usage.prompt_tokens
  ├── input_tokens_details:
  │     ├── image_tokens: int        → Usage.prompt_tokens_details.image_tokens
  │     └── text_tokens: int         → Usage.prompt_tokens_details.text_tokens
  └── image_tokens: int              → Usage.prompt_tokens_details.image_tokens (qwen3/multimodal)

model_prices_and_context_window.json
  ├── input_cost_per_token: float            → 文本 token 单价
  └── input_cost_per_image_token: float      → 图片/视频 token 单价（可选）
```

### 成本计算公式

**chat 路径（tiered 或 flat）：**

```
prompt_cost = text_tokens * input_cost_per_token
            + cached_tokens * cache_read_input_token_cost
completion_cost = completion_tokens * output_cost_per_token
                + reasoning_tokens * output_cost_per_reasoning_token
```

**embedding 路径（flat）：**

```
prompt_cost = text_tokens * input_cost_per_token
            + image_tokens * input_cost_per_image_token
            (若无 input_cost_per_image_token，回退到 input_cost_per_token)
completion_cost = 0
```
