# Spec 4: 梳理所有 Multimodal Embedding 定价条目

> 交付物：确认 `model_prices_and_context_window.json` 中所有 multimodal embedding 模型的定价条目完整且正确，修正错误的字段命名
> 前置依赖：D0 + D1（已完成）

---

## 验证标准

```bash
# 1. 确认所有 multimodal embedding 模型有 pricing entry
python3 -c "
import json
with open('model_prices_and_context_window.json') as f:
    data = json.load(f)
models = [k for k in data if k.startswith('dashscope/') and data[k].get('mode') == 'embedding']
for m in sorted(models):
    info = data[m]
    print(f'{m}: input_cost_per_token={info.get(\"input_cost_per_token\")}, '
          f'input_cost_per_image_token={info.get(\"input_cost_per_image_token\")}')
"
# → 所有模型都有 input_cost_per_token；多模态模型有 input_cost_per_image_token
```

---

## 核心结论

**DashScope 多模态 embedding 的定价单位是 token，不是"张"或"秒"。** 图片和视频经过 token 化后，按各自的 token 单价计费：

- `input_tokens_details.image_tokens`（或顶层 `image_tokens`）= 图片/视频的 token 数量
- `input_tokens_details.text_tokens`（或顶层 `input_tokens`）= 文本的 token 数量

---

## 最终定价条目

### 所有 dashscope embedding 模型（9 个）

| 模型 Key | 文本单价 (USD/token) | 图片/视频单价 (USD/token) | output_vector_size | max_input_tokens |
|---|---|---|---|---|
| `dashscope/qwen3-vl-embedding` | 7e-07 | 1.8e-06 | 2560 | 32000 |
| `dashscope/qwen2.5-vl-embedding` | 7e-07 | 1.8e-06 | 1024 | — |
| `dashscope/tongyi-embedding-vision-plus-2026-03-06` | 5e-07 | 同左（图/文本同价） | 1152 | 1024 |
| `dashscope/tongyi-embedding-vision-flash-2026-03-06` | 1.5e-07 | 同左 | 768 | 1024 |
| `dashscope/tongyi-embedding-vision-plus` | 5e-07 | 同左 | 1152 | 1024 |
| `dashscope/tongyi-embedding-vision-flash` | 1.5e-07 | 同左 | 768 | 1024 |
| `dashscope/multimodal-embedding-v1` | 7e-07 | 9e-07 | 1024 | 512 |
| `dashscope/text-embedding-v3` | 9.6e-08 | —（纯文本） | 1024 | — |
| `dashscope/text-embedding-v4` | 9.6e-08 | —（纯文本） | 1024 | — |

### 官方定价表（人民币）

| 模型 | 文本单价 | 图片/视频单价 | 关系 |
|------|---------|-------------|------|
| qwen3-vl-embedding | 0.7元/百万token | 1.8元/百万token | **图片/视频更贵** |
| qwen2.5-vl-embedding | 0.7元/百万token | 1.8元/百万token | **图片/视频更贵** |
| multimodal-embedding-v1 | 0.7元/百万token | 0.9元/百万token | **图片/视频更贵** |
| tongyi-embedding-vision-plus | 0.5元/百万token | 同左 | **统一价** |
| tongyi-embedding-vision-flash | 0.15元/百万token | 同左 | **统一价** |
| text-embedding-v3/v4 | 0.7元/百万token | — | 纯文本 |

---

## 设计说明

### 字段语义

- `input_cost_per_token` — 文本 token 单价（USD/token）
- `input_cost_per_image_token` — 图片/视频 token 单价（USD/token）。缺失时默认回退到 `input_cost_per_token`（即 tongyi 系列的行为）

### 修正内容

| 模型 | 修正前 | 修正后 |
|------|--------|--------|
| `qwen3-vl-embedding` | `input_cost_per_image: 1.8e-06`（语义错误） | `input_cost_per_image_token: 1.8e-06` |
| `qwen3-vl-embedding` | `input_cost_per_video_per_second: 1.8e-06`（语义错误） | **删除** |
| `qwen2.5-vl-embedding` | `input_cost_per_token: 0.0`（免费，错误） | `input_cost_per_token: 7e-07`, `input_cost_per_image_token: 1.8e-06` |
| `multimodal-embedding-v1` | `input_cost_per_image: 9e-07`（语义错误） | `input_cost_per_image_token: 9e-07` |
| `multimodal-embedding-v1` | `input_cost_per_video_per_second: 9e-07`（语义错误） | **删除** |
| `text-embedding-v3` | 无 pricing entry | 新增 `input_cost_per_token: 9.6e-08` |
| `text-embedding-v4` | 无 pricing entry | 新增 `input_cost_per_token: 9.6e-08` |

### 踩坑提醒：同步 backup 文件

`model_prices_and_context_window.json` 修改后，必须同步修改 `litellm/model_prices_and_context_window_backup.json`（包内备份文件）。`LITELLM_LOCAL_MODEL_COST_MAP=True` 模式下读取的是备份文件，否则测试/运行时找不到新增模型。

---

## 边界情况

| 场景 | 处理方式 |
|------|----------|
| `input_tokens_details.image_tokens` 缺失 | 图片成本为 0 |
| `input_cost_per_image_token` 字段缺失 | cost calculator 回退到 `input_cost_per_token`（与 tongyi 系列一致） |
| 模型无 pricing entry | D5 中处理：warning + 返回 (0, 0) |

---

## 数据模型

### 修正后的 pricing entry（示例）

```json
{
    "dashscope/qwen3-vl-embedding": {
        "input_cost_per_token": 7e-07,
        "input_cost_per_image_token": 1.8e-06,
        "litellm_provider": "dashscope",
        "max_input_tokens": 32000,
        "mode": "embedding",
        "output_cost_per_token": 0.0,
        "output_vector_size": 2560,
        "source": "https://help.aliyun.com/zh/model-studio/multimodal-embedding-api-reference",
        "supports_embedding_image_input": true,
        "supports_image_input": true,
        "supports_video_input": true
    }
}
```

### Usage 中的多模态计费字段

```json
// tongyi-embedding-vision-* 系列
{
    "input_tokens": 903,
    "input_tokens_details": {"image_tokens": 896, "text_tokens": 7},
    "output_tokens": 3,
    "total_tokens": 906
}
// 计费: 7 * input_cost_per_token + 896 * input_cost_per_token（同价）
//       = 903 * input_cost_per_token

// qwen3-vl-embedding / qwen2.5-vl-embedding / multimodal-embedding-v1
{
    "input_tokens": 7,
    "image_tokens": 896
}
// 计费: 7 * input_cost_per_token + 896 * input_cost_per_image_token（不同价）
```
