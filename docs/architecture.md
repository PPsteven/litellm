# Architecture

> 项目基础、技术决策、开发环境。实现 Spec 前的必读文档。

---

## 项目基础

DashScope Embedding Bugfix — 为 litellm 中 DashScope 多模态 embedding 新建完整的 `DashScopeMultimodalEmbeddingConfig` 实现。

**目标：**
- 新建 `transformation_multimodal.py`，实现 `DashScopeMultimodalEmbeddingConfig`（URL 拼接、输入归一化、请求/响应转换、参数映射、错误处理）
- 在 `utils.py` 路由中追加多模态模型判断
- 补充多模态 embedding 的单元测试和集成测试，防止回归
- **修复 cost_calculator.py 漏算图片/视频成本的问题**
- **补充缺失的 pricing entry（text-embedding-v3/v4）**

**范围：**
- 新建 `litellm/llms/dashscope/embed/transformation_multimodal.py`
- 修改 `litellm/llms/dashscope/embed/__init__.py`（追加导出）
- 修改 `litellm/utils.py` 路由逻辑（追加多模态判断）
- 修改 `litellm/llms/dashscope/cost_calculator.py`（增加图片/视频成本计算）
- 补充 `model_prices_and_context_window.json` 中缺失的 pricing entry
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
| 在 `dashscope_cost_per_token` 中增加多模态 token 成本计算 | 保持 DashScope 专用计算器的统一入口，不绕到 generic_cost_per_token；避免调用方感知多模态 vs 文本差异 |
| 使用 `input_cost_per_image_token` 字段（非 `input_cost_per_image` / `input_cost_per_video_per_second`） | 官方定价单位是 token（每千输入Token），不是"张"或"秒"。多模态 embedding 的 image_tokens 字段是图片/视频经过 token 化后的数量 |
| 从 usage 中提取 `image_tokens` | 多模态 embedding API 返回的 usage 中包含 `image_tokens`（tongyi 系列在 `input_tokens_details` 嵌套字段；qwen3/multimodal-embedding-v1 在顶层），用于按 token 计费 |
| tongyi-embedding-vision-*** 系列不设 `input_cost_per_image_token` 字段 | 该系列图/文本同价，统一用 `input_cost_per_token` 计费，无需额外字段。cost calculator 检测到字段缺失时跳过调整 |
| `cost_per_token` 对无 pricing entry 的模型返回 (0, 0) 而非抛 KeyError | `text-embedding-v3/v4` 等模型无 pricing entry。汇率变动时不应让计费失败，warning 后按 0 计费更安全 |
| 在 `transformation_multimodal.py` 中归一化 `image_tokens` 到 `prompt_tokens_details` | 各模型 usage 格式不同（嵌套 vs 顶层），在 transformation 层统一归一化，cost calculator 只读标准字段 |
| `cost_per_token` 按 `mode` 分支成 chat/embedding 两条路径 | chat 模型有 tiered/output/cache 计费结构，embedding 模型只有 text + image token 简单乘法。两者计费结构差异大，独立路径更清晰，删除 `_apply_image_token_cost` 启发式 |
| `_extract_token_breakdown` 中 `text_tokens` 排除 `image_tokens` | 匹配 litellm 通用 cost calculator 模式（`llm_cost_calc/utils.py:725`），避免 embedding 路径中 text 和 image token 计费互相干扰 |
| 使用 `input_cost_per_image_token` 而非 `input_cost_per_image` / `input_cost_per_video_per_second` | 官方定价单位是 token（每千输入Token），不是"张"或"秒"。D4 修正了 `qwen3-vl-embedding`、`multimodal-embedding-v1` 的错误字段名 |
| `qwen2.5-vl-embedding` 价格从 0.0 修正为 7e-07 | 该模型之前被标记为免费（0.0），实际官方定价为 0.7元/百万token |
| `text-embedding-v3/v4` 新增 pricing entry（9.6e-08） | 之前完全缺失，cost calculator 无法查到价格 |
| 在 `transformation_multimodal.py` 中对 qwen3 系列模型合并 `input_tokens + image_tokens` 为 `prompt_tokens` | qwen3-vl-embedding 等模型的 `input_tokens` 只含文本 token，不含图片 token。cost calculator 的 `text_tokens = prompt_tokens - image_tokens` 公式需要 `prompt_tokens` 为总数 |

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
