# Spec 2: 多模态 Embedding 单元测试

> 交付物：`test_dashscope_multimodal_embedding_transformation.py` 单元测试文件
> 前置依赖：D1（已完成）

---

## 验证标准

```bash
# 跑全部多模态 embedding 单元测试
pytest tests/test_litellm/llms/dashscope/test_dashscope_multimodal_embedding_transformation.py -v
# → 全部 PASSED
```

---

## 文件结构

```
tests/test_litellm/llms/dashscope/
├── test_dashscope_embedding_transformation.py           # 已有（标准文本）
├── test_dashscope_multimodal_embedding_transformation.py  # 新建
└── test_dashscope_multimodal_embedding_integration.py     # D3
```

新建文件而非追加到已有文件，因为多模态与标准文本是两条独立路径，分开维护更清晰。

---

## 核心代码

### 测试分组

```
1. is_multimodal_embedding — 模型识别
2. get_complete_url — URL 拼接（含环境变量 mock）
3. _normalize_input_item / _normalize_content_blocks — 输入归一化
4. transform_embedding_request — 请求转换
5. transform_embedding_response — 响应解析（成功 + 错误）
6. map_openai_params — 参数映射
7. validate_environment — 环境验证
```

### 测试用例清单

```python
# === 1. 模型识别 ===
def test_is_multimodal_embedding_vision_flash():
    # tongyi-embedding-vision-flash -> True
def test_is_multimodal_embedding_vision_plus():
    # tongyi-embedding-vision-plus -> True
def test_is_multimodal_embedding_text():
    # text-embedding-v4 -> False
def test_is_multimodal_embedding_with_prefix():
    # host/tongyi-embedding-vision-flash -> True

# === 2. URL 拼接 ===
def test_get_complete_url_default():
    # api_base=None -> DEFAULT_API_BASE
def test_get_complete_url_custom_base():
    # api_base="https://xxx.maas.aliyuncs.com/api/v1"
    # -> "https://xxx.maas.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding"
def test_get_complete_url_compatible_mode():
    # api_base 含 /compatible-mode/ -> DEFAULT_API_BASE
def test_get_complete_url_env_var():
    # 设置 DASHSCOPE_API_BASE -> 返回环境变量值
def test_get_complete_url_already_has_path():
    # api_base 已以 /multimodal-embedding 结尾 -> 直接返回

# === 3. 输入归一化 ===
def test_normalize_input_item_string():
    # "hello" -> {"text": "hello"}
def test_normalize_input_item_dict_text():
    # [{"type": "text", "text": "hi"}] -> {"text": "hi"}
def test_normalize_input_item_dict_image():
    # [{"type": "image_url", "image_url": {"url": "https://img.jpg"}}]
    # -> {"image": "https://img.jpg"}
def test_normalize_input_item_mixed():
    # text + image -> {"content_list": [{"text": ...}, {"image": ...}]}

# === 4. 请求转换 ===
def test_transform_embedding_request_string_input():
    # input=["hello"] -> {"model": ..., "input": {"contents": [{"text": "hello"}]}}
def test_transform_embedding_request_mixed_input():
    # input=[text_block, image_block] -> 正确转换
def test_transform_embedding_request_with_dimension():
    # optional_params={"dimensions": 512} -> parameters.dimension=512

# === 5. 响应解析 ===
def test_transform_embedding_response_success():
    # 正常响应 -> EmbeddingResponse 正确填充
def test_transform_embedding_response_error():
    # 含 code 字段 -> 抛出 DashScopeError
def test_transform_embedding_response_non_200():
    # HTTP 非 200 -> 抛出 DashScopeError

# === 6. 参数映射 ===
def test_map_openai_params_dimensions():
    # dimensions -> dimension

# === 7. 环境验证 ===
def test_validate_environment():
    # api_key -> Authorization: Bearer
def test_validate_environment_missing_key():
    # 无 api_key -> ValueError
```

---

## 设计说明

### 测试策略

- 所有测试使用 `unittest.mock` 模拟外部依赖（`get_secret_str`）
- 不发起真实 HTTP 请求
- 每个测试只测一个方法的一个行为
- 测试数据使用真实模型名（`tongyi-embedding-vision-flash`）

### 边界情况

| 场景 | 处理方式 |
|------|----------|
| 模型名带 `/` 前缀（如 `gateway/tongyi-embedding-vision-flash`） | `is_multimodal_embedding` 取 `/` 后部分判断 |
| 空 input 列表 | `transform_embedding_request` 生成空 `contents` |
| 响应中无 `output.embeddings` | 返回空 data 列表 |
| 响应中 usage 字段缺失 | prompt_tokens/total_tokens 默认为 0 |
| 响应 JSON 解析失败 | 抛出 DashScopeError |

---

## 数据模型

无数据模型变更。测试数据直接使用 `EmbeddingResponse` 和 `httpx.Response` 构造。
