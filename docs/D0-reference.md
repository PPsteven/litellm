# DashScope 多模态 Embedding API 参考（只读）

> 本文件是对标参考，不修改。开发时对标此文件确认兼容性。
> 来源：阿里云百炼官方文档 https://help.aliyun.com/zh/model-studio/multimodal-embedding-api-reference

---

## 核心功能

- **跨模态检索**：以文搜图、以图搜视频、以图搜图等
- **语义相似度计算**：统一向量空间中衡量不同模态内容的语义相似性
- **内容分类与聚类**：基于语义向量进行智能分组、打标和聚类分析

所有模态（文本、图片、视频）的向量均位于同一语义空间，可通过余弦相似度直接进行跨模态匹配。

---

## 接口规范

### 端点

```
POST https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding
```

### 请求头

| 头 | 值 |
|----|-----|
| Authorization | `Bearer {DASHSCOPE_API_KEY}` |
| Content-Type | `application/json` |

### 请求体

```json
{
    "model": "tongyi-embedding-vision-plus",
    "input": {
        "contents": [
            {"text": "描述文本"},
            {"image": "https://example.com/img.jpg"},
            {"video": "https://example.com/video.mp4"},
            {"multi_images": ["https://example.com/img1.png", "https://example.com/img2.png"]}
        ]
    },
    "parameters": {
        "dimension": 1152,
        "output_type": "dense",
        "fps": 1.0,
        "instruct": "custom instruction",
        "enable_fusion": false,
        "res_level": 1,
        "max_video_frames": 8
    }
}
```

### contents 元素类型

| 类型 | key | value 格式 | 说明 |
|------|-----|-----------|------|
| 文本 | `text` | 字符串 | 也可不通过 dict 直接传入字符串 |
| 图片 | `image` | URL 或 Base64 Data URI | Base64 格式: `data:image/{format};base64,{data}` |
| 多图片 | `multi_images` | 图片 URL 列表 | 仅 plus/flash 系列支持 |
| 视频 | `video` | 公开可访问的 URL | — |

### 融合向量（2026-03-06 版本）

将 text、image、video 放在同一个 content 对象中实现融合：

```json
{
    "model": "tongyi-embedding-vision-plus-2026-03-06",
    "input": {
        "contents": [
            {"text": "描述文本", "image": "https://example.com/img.jpg", "video": "https://example.com/video.mp4"}
        ]
    },
    "parameters": {"dimension": 1152}
}
```

### 参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | string | 是 | 模型名 |
| `input.contents` | array | 是 | 输入内容列表 |
| `input.contents[].text` | string | 否 | 文本内容 |
| `input.contents[].image` | string | 否 | 图片 URL 或 Base64 |
| `input.contents[].video` | string | 否 | 视频 URL |
| `input.contents[].multi_images` | array | 否 | 多图 URL 列表 |
| `parameters.dimension` | int | 否 | 输出向量维度（各模型支持不同范围） |
| `parameters.output_type` | string | 否 | 输出类型，目前仅支持 `dense` |
| `parameters.fps` | float | 否 | 视频帧率比例 [0,1]，默认 1.0 |
| `parameters.instruct` | string | 否 | 自定义任务说明，建议英文，可提升 1%-5% 效果 |
| `parameters.enable_fusion` | bool | 否 | 是否生成融合向量，仅 qwen3-vl-embedding |
| `parameters.res_level` | int | 否 | 分辨率档位 0/1/2/3，默认 1（仅 2026-03-06 版本） |
| `parameters.max_video_frames` | int | 否 | 视频最大采样帧数，默认 8，最大 64（仅 2026-03-06 版本） |

### 成功响应

```json
{
    "output": {
        "embeddings": [
            {"index": 0, "embedding": [-0.0266, ...], "type": "text"},
            {"index": 1, "embedding": [0.0515, ...], "type": "image"}
        ]
    },
    "usage": {
        "input_tokens": 10,
        "input_tokens_details": {"image_tokens": 896, "text_tokens": 7},
        "output_tokens": 3,
        "total_tokens": 906
    },
    "request_id": "1fff9502-..."
}
```

**type 字段取值**：`text` / `image` / `video` / `multi_images`（独立向量），`fused`（2026-03-06 融合），`fusion`（qwen3-vl-embedding/qwen2.5-vl-embedding 融合），`vl`（qwen3-vl-embedding 独立）

### usage 字段差异

| 模型 | 返回字段 |
|------|---------|
| tongyi-embedding-vision-* | input_tokens(含图片), input_tokens_details{image_tokens, text_tokens}, output_tokens, total_tokens |
| qwen3-vl-embedding | input_tokens(仅文本), image_tokens, total_tokens(=input_tokens+image_tokens) |
| qwen2.5-vl-embedding | input_tokens, image_tokens（无 total_tokens） |
| multimodal-embedding-v1 | input_tokens, image_tokens, image_count, duration |

### 错误响应

```json
{
    "code": "InvalidApiKey",
    "message": "Invalid API-key provided.",
    "request_id": "fb53c4ec-..."
}
```

错误通过 `code` 字段标识，而非 OpenAI 风格的 `error` 字段。

---

## 支持模型

| 模型 | 默认维度 | 向量类型 | 支持的输入 | 说明 |
|------|---------|---------|-----------|------|
| qwen3-vl-embedding | 2560 | 独立/融合 | text, image, video, 多个 image | `enable_fusion` 参数开启融合 |
| qwen2.5-vl-embedding | 1024 | 仅融合 | text, image, video | 始终返回 1 个融合向量，不支持多图 |
| tongyi-embedding-vision-plus-2026-03-06 | 1152 | 独立/融合 | text, image, video, multi_images | Qwen3 底座，多分辨率，30+ 语言 |
| tongyi-embedding-vision-flash-2026-03-06 | 768 | 独立/融合 | 同上 | 同上 |
| tongyi-embedding-vision-plus | 1152 | 仅独立 | 支持 multi_images（最多 8 张） | — |
| tongyi-embedding-vision-flash | 768 | 仅独立 | 同上 | — |
| multimodal-embedding-v1 | 1024 | — | text, image, video | 不支持 dimension 参数 |

---

## 限制

| 模型 | 文本长度 | 图片大小 | 视频大小 | 单次请求条数 |
|------|---------|---------|---------|-------------|
| qwen3-vl-embedding | 32,000 Token | 单张 ≤10 MB | ≤50 MB | 元素≤20，图片≤5，视频≤1 |
| qwen2.5-vl-embedding | — | 单张 ≤5 MB | — | 每种类型最多 1 次 |
| tongyi-embedding-vision-plus-2026-03-06 | 1,024 Token | 建议≤5 MB，最大 10 MB，最多 64 张 | ≤50 MB，H.264/H.265 | 元素≤20，图片≤64，视频≤8 |
| tongyi-embedding-vision-flash-2026-03-06 | 1,024 Token | 同上 | 同上 | 同上 |
| tongyi-embedding-vision-plus | 1,024 Token | 单张≤3 MB，最多 8 张 | ≤10 MB | 无明确限制 |
| tongyi-embedding-vision-flash | 1,024 Token | 同上 | 同上 | 同上 |
| multimodal-embedding-v1 | 512 Token | 单张≤3 MB | ≤10 MB | 元素≤20，图片/视频各≤1，文本≤20 |

### 图片格式

JPEG, PNG, WEBP, BMP, TIFF, ICO, DIB, ICNS, SGI（URL 或 Base64）

### 视频格式

MP4, MPEG, MOV, MPG, WEBM, AVI, FLV, MKV（仅 URL）
