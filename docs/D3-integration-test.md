# Spec 3: 多模态 Embedding 集成测试

> 交付物：`test_dashscope_multimodal_embedding_integration.py` 集成测试文件
> 前置依赖：D1 + D2（已完成）

---

## 验证标准

```bash
# 跑全部多模态 embedding 集成测试
pytest tests/test_litellm/llms/dashscope/test_dashscope_multimodal_embedding_integration.py -v
# → 全部 PASSED
```

---

## 文件结构

```
tests/test_litellm/llms/dashscope/
├── test_dashscope_embedding_transformation.py              # 已有
├── test_dashscope_multimodal_embedding_transformation.py   # D2 新建
└── test_dashscope_multimodal_embedding_integration.py      # 新建
```

---

## 核心代码

### 测试分组

```python
# 1. embedding_sync — 同步 embedding 完整流程
# 2. embedding_async — 异步 embedding 完整流程
# 3. embedding_error — 错误响应处理
# 4. embedding_custom_base — 自定义 api_base 场景
```

### 测试用例清单

```python
# === 1. 同步完整流程 ===
def test_sync_multimodal_embedding():
    # mock httpx.Client.post
    # 调用 BaseLLMHTTPHandler.embedding()
    # 验证:
    #   - POST URL 包含 /multimodal-embedding
    #   - 请求体格式正确
    #   - 返回 EmbeddingResponse 数据正确

# === 2. 异步完整流程 ===
@pytest.mark.asyncio
async def test_async_multimodal_embedding():
    # mock httpx.AsyncClient.post
    # 调用 BaseLLMHTTPHandler.aembedding()
    # 验证同上

# === 3. 错误响应 ===
def test_multimodal_embedding_api_error():
    # mock 返回含 code 的错误响应
    # 验证抛出 DashScopeError

# === 4. 自定义 api_base ===
def test_multimodal_embedding_custom_api_base():
    # 传入自定义 api_base（VPC 端点）
    # 验证 POST URL 正确拼接
```

---

## 设计说明

### 为什么需要集成测试

单元测试验证的是每个方法独立行为，集成测试验证的是整个调用链路的协作正确性：

1. `BaseLLMHTTPHandler.embedding()` 是否正确调用了 `get_complete_url()`
2. URL 拼接结果是否真正体现在 HTTP 请求中
3. 请求体和响应体的格式转换是否在完整链路中正确工作

### Mock 策略

- 使用 `unittest.mock.patch` 拦截 `httpx.Client.post` / `httpx.AsyncClient.post`
- 不 mock `DashScopeMultimodalEmbeddingConfig` 的任何方法——让真实代码跑完整链路
- 只 mock 网络层，业务逻辑全部真实执行

### 边界情况

| 场景 | 处理方式 |
|------|----------|
| 网络超时 | mock 抛出 httpx.TimeoutException，验证被正确包装为 DashScopeError |
| API 返回非 JSON | mock 返回非 JSON body，验证 JSON 解析错误处理 |
| 空 embedding 列表 | mock 返回 `output.embeddings: []`，验证空列表处理 |

---

## 数据模型

### Mock 响应模板

```json
{
    "output": {
        "embeddings": [
            {"embedding": [0.1, 0.2, 0.3]}
        ]
    },
    "usage": {"input_tokens": 10, "total_tokens": 10},
    "request_id": "mock-request-id"
}
```

### 错误响应模板

```json
{
    "code": "InvalidParameter",
    "message": "dimension must be between 1 and 2048",
    "request_id": "mock-request-id"
}
```
