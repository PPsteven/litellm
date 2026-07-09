# Architecture

> 项目基础、技术决策、开发环境。实现 Spec 前的必读文档。

---

## 项目基础

DashScope Embedding Bugfix — 为 litellm 中 DashScope 多模态 embedding 新建完整的 `DashScopeMultimodalEmbeddingConfig` 实现。

**目标：**
- 新建 `transformation_multimodal.py`，实现 `DashScopeMultimodalEmbeddingConfig`（URL 拼接、输入归一化、请求/响应转换、参数映射、错误处理）
- 在 `utils.py` 路由中追加多模态模型判断
- 补充多模态 embedding 的单元测试和集成测试，防止回归

**范围：**
- 新建 `litellm/llms/dashscope/embed/transformation_multimodal.py`
- 修改 `litellm/llms/dashscope/embed/__init__.py`（追加导出）
- 修改 `litellm/utils.py` 路由逻辑（追加多模态判断）
- 不修改 `transformation.py`（标准文本 embedding 逻辑正确）
- 不修改 `BaseEmbeddingConfig` 基类契约

---

## 架构

### 当前状态

```
main.py: embedding()
  → BaseLLMHTTPHandler.embedding()
    → ProviderConfigManager.get_provider_embedding_config()
      → DashScopeEmbeddingConfig（唯一实现，所有 dashscope 模型都走此路径）
        → /compatible-mode/v1/embeddings（OpenAI 兼容）
```

### 目标状态（D1 完成后）

```
main.py: embedding()
  → BaseLLMHTTPHandler.embedding()
    → ProviderConfigManager.get_provider_embedding_config()
      ├─ 非多模态模型 → DashScopeEmbeddingConfig → /compatible-mode/v1/embeddings
      └─ 多模态模型 → DashScopeMultimodalEmbeddingConfig（新建）
                      → /api/v1/services/embeddings/multimodal-embedding/multimodal-embedding（原生 API）
```

### 关键路径

```
用户调用 embedding(model="tongyi-embedding-vision-flash", ...)
  → main.py 解析 DASHSCOPE_API_KEY
  → BaseLLMHTTPHandler.embedding()
  → ProviderConfigManager.get_provider_embedding_config()
  → 路由到 DashScopeMultimodalEmbeddingConfig  ← 新建
  → get_complete_url()                          ← 新建
  → transform_embedding_request()               ← 新建
  → POST httpx 请求到原生 API
  → transform_embedding_response()              ← 新建
  → 返回 EmbeddingResponse
```

---

## 技术决策

| 决策 | 理由 |
|------|------|
| 新建 `transformation_multimodal.py` 而非修改 `transformation.py` | 多模态走原生 API，与标准文本的 OpenAI 兼容模式在 URL、请求格式、响应格式、错误处理上完全不同。保持关注点分离，与 Voyage 等多模态 provider 的模式一致 |
| 不改 `BaseEmbeddingConfig` 基类 | 基类契约已满足需求，无需修改 |
| 不改 `transformation.py`（标准文本） | 标准文本逻辑正确，不需要动 |
| 在 `utils.py` 追加多模态路由 | 路由逻辑需要根据模型名判断走哪条路径 |
| 新建测试文件而非追加到已有文件 | 多模态与标准文本是两条独立路径，分开维护更清晰 |
| 模型识别用 `startswith` 而非精确匹配 | 兼容未来可能的新模型名（如 `tongyi-embedding-vision-xxx`） |
| URL 拼接兼容 VPC 端点 | 阿里云 VPC 端点形如 `*.maas.aliyuncs.com/api/v1`，需要追加路径 |

---

## 技术栈

| 层 | 技术 |
|----|------|
| 语言 | Python 3.8+ |
| 测试框架 | pytest |
| HTTP 客户端 | httpx |
| 模拟 | unittest.mock |

## 开发环境

- macOS, Python 3.8+
- pytest, httpx

## 平台约束

- 所有代码与 litellm 现有风格一致（无额外注释）
- 测试不依赖真实网络请求，全部 mock
