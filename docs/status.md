# Status

> 进度跟踪。每次会话结束时更新。

## 当前焦点

全部 Spec 已完成。项目交付物已实现并验证。

## 进度

- [x] 创建 AGENTS.md
- [x] 创建 docs/architecture.md
- [x] 创建 docs/status.md
- [x] 创建 docs/D0-reference.md（已更新至阿里云最新 API）
- [x] 创建 docs/D1-fix-multimodal-url.md（新建 transformation_multimodal.py）
- [x] 创建 docs/D2-unit-tests.md
- [x] 创建 docs/D3-integration-test.md
- [x] 创建 docs/D4-pricing-mapping.md
- [x] 创建 docs/D5-cost-calculator.md
- [x] **D1**: 新建 `transformation_multimodal.py` + 修改 `__init__.py` + 修改 `utils.py` 路由
- [x] **D2**: 实现多模态 embedding 单元测试并验证（22 tests passed）
- [x] **D3**: 实现多模态 embedding 集成测试并验证（4 tests passed）
- [x] **D4**: 修正 `model_prices_and_context_window.json` 中多模态 embedding 定价字段（`input_cost_per_image` → `input_cost_per_image_token`，删除 `input_cost_per_video_per_second`），修正 `qwen2.5-vl-embedding` 价格（0.0 → 7e-07），新增 `text-embedding-v3/v4` 条目
- [x] **D5**: 重构 `cost_calculator.py`（chat/embedding 路径分离），更新 `transformation_multimodal.py`（归一化 `image_tokens` 到 `prompt_tokens_details`），追加 6 个多模态计费测试用例（11 tests passed）

## 已知问题

- （无）

## 下一步

- 无。所有 Spec 已完成，可提 PR。
