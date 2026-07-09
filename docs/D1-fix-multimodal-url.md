# Spec 1: 新建 DashScopeMultimodalEmbeddingConfig

> 交付物：新建 `transformation_multimodal.py` + 修改 `__init__.py` + 修改 `utils.py` 路由
> 前置依赖：D0（已完成）

---

## 验证标准

```bash
# 1. 默认无 api_base → 返回 DEFAULT_API_BASE
pytest tests/test_litellm/llms/dashscope/test_dashscope_multimodal_embedding_transformation.py::test_get_complete_url_default -v
# → PASSED

# 2. 传入自定义 api_base（VPC 端点）→ 追加 /services/embeddings/multimodal-embedding
pytest tests/test_litellm/llms/dashscope/test_dashscope_multimodal_embedding_transformation.py::test_get_complete_url_custom_base -v
# → PASSED

# 3. 传入含 /compatible-mode/ 的 api_base → 回退到 DEFAULT_API_BASE
pytest tests/test_litellm/llms/dashscope/test_dashscope_multimodal_embedding_transformation.py::test_get_complete_url_compatible_mode -v
# → PASSED

# 4. 传入 DASHSCOPE_API_BASE 环境变量 → 直接返回环境变量值
pytest tests/test_litellm/llms/dashscope/test_dashscope_multimodal_embedding_transformation.py::test_get_complete_url_env_var -v
# → PASSED

# 5. 传入的 api_base 已以 /multimodal-embedding 结尾 → 直接返回
pytest tests/test_litellm/llms/dashscope/test_dashscope_multimodal_embedding_transformation.py::test_get_complete_url_already_has_path -v
# → PASSED

# 6. 模型识别：多模态模型返回 True
pytest tests/test_litellm/llms/dashscope/test_dashscope_multimodal_embedding_transformation.py::test_is_multimodal_embedding_vision_flash -v
# → PASSED

# 7. 输入归一化：string → {"text": "..."}
pytest tests/test_litellm/llms/dashscope/test_dashscope_multimodal_embedding_transformation.py::test_normalize_input_item_string -v
# → PASSED

# 8. 输入归一化：text + image content blocks → content_list
pytest tests/test_litellm/llms/dashscope/test_dashscope_multimodal_embedding_transformation.py::test_normalize_input_item_mixed -v
# → PASSED

# 9. 请求转换：完整请求体
pytest tests/test_litellm/llms/dashscope/test_dashscope_multimodal_embedding_transformation.py::test_transform_embedding_request_mixed_input -v
# → PASSED

# 10. 响应解析：成功响应
pytest tests/test_litellm/llms/dashscope/test_dashscope_multimodal_embedding_transformation.py::test_transform_embedding_response_success -v
# → PASSED

# 11. 响应解析：错误响应（code 字段）
pytest tests/test_litellm/llms/dashscope/test_dashscope_multimodal_embedding_transformation.py::test_transform_embedding_response_error -v
# → PASSED

# 12. 环境验证：缺少 API Key 抛 ValueError
pytest tests/test_litellm/llms/dashscope/test_dashscope_multimodal_embedding_transformation.py::test_validate_environment_missing_key -v
# → PASSED
```

---

## 文件结构

```
litellm/llms/dashscope/embed/
├── __init__.py                          # 修改：追加导出 DashScopeMultimodalEmbeddingConfig
├── transformation.py                    # 不变
└── transformation_multimodal.py         # 新建

litellm/utils.py                         # 修改：追加多模态路由逻辑
```

新建 `transformation_multimodal.py` 而非修改 `transformation.py`，因为多模态走原生 API，与标准文本的 OpenAI 兼容模式在 URL、请求格式、响应格式、错误处理上完全不同。

---

## 核心代码

### `transformation_multimodal.py` — 完整类

```python
"""
Transform request/response for DashScope multimodal embeddings.
Reference: https://help.aliyun.com/zh/model-studio/multimodal-embedding-api-reference
"""

from typing import List, Optional, Union
import httpx
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage
from ..common_utils import DashScopeError

DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"

DASHSCOPE_MULTIMODAL_EMBEDDING_MODELS = {
    "tongyi-embedding-vision-flash",
    "tongyi-embedding-vision-flash-2026-03-06",
    "tongyi-embedding-vision-plus",
    "tongyi-embedding-vision-plus-2026-03-06",
    "qwen3-vl-embedding",
    "qwen2.5-vl-embedding",
}


class DashScopeMultimodalEmbeddingConfig(BaseEmbeddingConfig):

    @staticmethod
    def is_multimodal_embedding(model: str) -> bool:
        """Check if model name matches multimodal embedding prefix."""
        base = model.split("/")[-1] if "/" in model else model
        return any(
            base.startswith(prefix)
            for prefix in DASHSCOPE_MULTIMODAL_EMBEDDING_MODELS | {"multimodal-embedding"}
        )

    def get_supported_openai_params(self, model: str) -> List[str]:
        return ["dimensions"]

    def map_openai_params(self, non_default_params: dict, optional_params: dict,
                          model: str, drop_params: bool = False) -> dict:
        if "dimensions" in non_default_params:
            optional_params["dimension"] = non_default_params["dimensions"]
        return optional_params

    def validate_environment(self, headers: dict, model: str,
                             messages: List[AllMessageValues],
                             optional_params: dict, litellm_params: dict,
                             api_key: Optional[str] = None,
                             api_base: Optional[str] = None) -> dict:
        if api_key is None:
            api_key = get_secret_str("DASHSCOPE_API_KEY")
        if api_key is None:
            raise ValueError("DashScope API key is required. "
                             "Set 'DASHSCOPE_API_KEY' env var or pass api_key explicitly.")
        return {"Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}", **headers}

    def get_complete_url(self, api_base: Optional[str], api_key: Optional[str],
                         model: str, optional_params: dict, litellm_params: dict,
                         stream: Optional[bool] = None) -> str:
        user_base = get_secret_str("DASHSCOPE_API_BASE")
        if user_base:
            return user_base
        if api_base and "/compatible-mode/" in api_base:
            return DEFAULT_API_BASE
        if api_base:
            base = api_base.rstrip("/")
            if base.endswith("/multimodal-embedding/multimodal-embedding"):
                return base
            if base.endswith("/multimodal-embedding"):
                return f"{base}/multimodal-embedding"
            return f"{base}/services/embeddings/multimodal-embedding/multimodal-embedding"
        return DEFAULT_API_BASE

    def _normalize_content_blocks(self, content: list) -> list:
        """Convert OpenAI content blocks to DashScope native format."""
        result = []
        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                result.append({"text": block.get("text", "")})
            elif block_type == "image_url":
                image_url = block.get("image_url", {})
                if isinstance(image_url, dict):
                    image_url = image_url.get("url", "")
                result.append({"image": image_url})
            else:
                result.append(block)
        return result

    def _normalize_input_item(self, item: Union[str, list, dict]) -> dict:
        if isinstance(item, str):
            return {"text": item}
        if isinstance(item, list):
            blocks = self._normalize_content_blocks(item)
            if len(blocks) == 1:
                return blocks[0]
            return {"content_list": blocks}
        return item

    def transform_embedding_request(self, model: str,
                                    input: AllEmbeddingInputValues,
                                    optional_params: dict,
                                    headers: dict) -> dict:
        inputs = input if isinstance(input, list) else [input]
        contents = [self._normalize_input_item(item) for item in inputs]
        body: dict = {"model": model, "input": {"contents": contents}}
        params = {}
        if "dimension" in optional_params:
            params["dimension"] = optional_params["dimension"]
        if "output_type" in optional_params:
            params["output_type"] = optional_params["output_type"]
        if params:
            body["parameters"] = params
        return body

    def transform_embedding_response(self, model: str,
                                     raw_response: httpx.Response,
                                     model_response: EmbeddingResponse,
                                     logging_obj: LiteLLMLoggingObj,
                                     api_key: Optional[str],
                                     request_data: dict,
                                     optional_params: dict,
                                     litellm_params: dict) -> EmbeddingResponse:
        try:
            response_json = raw_response.json()
        except Exception as e:
            raise DashScopeError(status_code=raw_response.status_code,
                                 message=f"Failed to parse DashScope response as JSON: {str(e)}")
        logging_obj.post_call(input=request_data.get("input"), api_key=api_key,
                              additional_args={"complete_input_dict": request_data},
                              original_response=response_json)
        if "code" in response_json:
            raise DashScopeError(status_code=raw_response.status_code,
                                 message=response_json.get("message", str(response_json)))
        output = response_json.get("output", {})
        embeddings = output.get("embeddings", [])
        model_response.object = "list"
        model_response.data = [
            {"object": "embedding", "index": emb.get("index", i),
             "embedding": emb.get("embedding", [])}
            for i, emb in enumerate(embeddings)
        ]
        model_response.model = model
        usage = response_json.get("usage") or {}
        input_tokens = usage.get("input_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens)
        setattr(model_response, "usage",
                Usage(prompt_tokens=input_tokens, completion_tokens=0, total_tokens=total_tokens))
        if "request_id" in response_json:
            setattr(model_response, "id", response_json["request_id"])
        return model_response

    def get_error_class(self, error_message: str, status_code: int,
                        headers: Union[dict, httpx.Headers]) -> BaseLLMException:
        if isinstance(headers, dict):
            headers = httpx.Headers(headers)
        return DashScopeError(status_code=status_code, message=error_message, headers=headers)
```

### `__init__.py` — 追加导出

```python
from .transformation import DashScopeEmbeddingConfig
from .transformation_multimodal import DashScopeMultimodalEmbeddingConfig

__all__ = ["DashScopeEmbeddingConfig", "DashScopeMultimodalEmbeddingConfig"]
```

### `utils.py` — 路由逻辑

```python
elif litellm.LlmProviders.DASHSCOPE == provider:
    from litellm.llms.dashscope.embed.transformation import DashScopeEmbeddingConfig
    from litellm.llms.dashscope.embed.transformation_multimodal import DashScopeMultimodalEmbeddingConfig
    if DashScopeMultimodalEmbeddingConfig.is_multimodal_embedding(model):
        return DashScopeMultimodalEmbeddingConfig()
    return DashScopeEmbeddingConfig()
```

---

## 设计说明

### 为什么新建文件而非修改现有文件

`transformation.py` 实现的是标准文本 embedding（OpenAI 兼容模式），而多模态 embedding 走的是 DashScope 原生 API，两者在 URL、请求格式、响应格式、错误处理上完全不同。新建文件保持关注点分离，与 Voyage 等多模态 provider 的模式一致。

### URL 拼接决策树

```
get_complete_url(api_base)
  ├─ DASHSCOPE_API_BASE 环境变量有值? → 直接返回
  ├─ api_base 含 /compatible-mode/? → 返回 DEFAULT_API_BASE
  ├─ api_base 非空?
   │   ├─ 已以 /multimodal-embedding/multimodal-embedding 结尾? → 直接返回
   │   ├─ 已以 /multimodal-embedding 结尾? → 追加 /multimodal-embedding
   │   └─ 否 → api_base + /services/embeddings/multimodal-embedding/multimodal-embedding
  └─ 默认 → 返回 DEFAULT_API_BASE
```

### 输入归一化流程

```
input item
  ├─ str → {"text": item}
  └─ list → 遍历 content blocks
       ├─ type="text" → {"text": block.text}
       ├─ type="image_url" → {"image": url}
       └─ 其他 → 原样保留
       └─ 只有 1 个 block? → 直接返回该 block
       └─ 多个 block → {"content_list": [blocks]}
```

### 边界情况

| 场景 | 处理方式 |
|------|----------|
| `api_base=None` | 返回 `DEFAULT_API_BASE` |
| `api_base` 含 `/compatible-mode/` | 多模态不走兼容模式，回退到默认 |
| `api_base` 是 VPC 端点（如 `*.maas.aliyuncs.com/api/v1`） | 追加 `/services/embeddings/multimodal-embedding/multimodal-embedding` |
| `api_base` 已以 `/multimodal-embedding/multimodal-embedding` 结尾 | 直接返回，不重复拼接 |
| `api_base` 已以 `/multimodal-embedding` 结尾（但缺一级） | 追加 `/multimodal-embedding` |
| 设置了 `DASHSCOPE_API_BASE` 环境变量 | 直接返回环境变量值，不做任何拼接 |
| `api_base` 尾部有 `/` | 先 strip 再判断 |
| input 是纯字符串 | 转为 `{"text": "..."}` |
| input 是单个 content block | 直接返回该 block，不包装 content_list |
| input 是多个 content block | 包装为 `{"content_list": [...]}` |
| 响应含 `code` 字段 | 抛出 DashScopeError |
| 响应 JSON 解析失败 | 抛出 DashScopeError |
| 缺少 API Key | 抛出 ValueError |
| 模型名带 `/` 前缀（如 `gateway/xxx`） | `is_multimodal_embedding` 取 `/` 后部分判断 |

---

## 数据模型

### 请求体

```json
{
    "model": "tongyi-embedding-vision-flash",
    "input": {
        "contents": [
            {"text": "纯文本"},
            {"content_list": [{"text": "描述"}, {"image": "https://..."}]}
        ]
    },
    "parameters": {"dimension": 512}
}
```

### DashScope 原生响应

```json
{
    "output": {
        "embeddings": [
            {"index": 0, "embedding": [0.1, 0.2, ...], "type": "text"}
        ]
    },
    "usage": {"input_tokens": 10, "total_tokens": 10},
    "request_id": "xxx"
}
```

### 转换后 OpenAI 兼容格式

```python
EmbeddingResponse(
    object="list",
    data=[{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
    model="tongyi-embedding-vision-flash",
    usage=Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10),
    id="xxx"
)
```

### 错误响应

```json
{
    "code": "InvalidParameter",
    "message": "dimension must be between 1 and 2048",
    "request_id": "xxx"
}
```
