# Multimodal-Embedding API 详情（阿里云百炼官方文档）

> 来源：https://help.aliyun.com/zh/model-studio/multimodal-embedding-api-reference
> 保存日期：2026-07-09
> 用途：设计参考，与 D0-reference.md 互补

---

## 核心能力

- **跨模态检索**：以文搜图、以图搜视频、以图搜图等跨模态语义搜索。
- **语义相似度计算**：在统一向量空间中衡量不同模态内容之间的语义相似性。
- **内容分类与聚类**：基于语义向量进行智能分组、打标和聚类分析。

> 所有模态（文本、图片、视频）的向量均位于同一语义空间，可通过余弦相似度等方法直接进行跨模态匹配与比较。

## 向量类型说明

多模态向量模型支持两种向量生成方式：

- **多模态独立向量**：为 contents 中的每个输入（文本、图片、视频、多图）分别生成独立向量。适用于逐项对比不同内容的场景。
- **多模态融合向量**：将 contents 中的所有输入融合为 1 个向量。适用于需要整体理解多模态内容的场景。
  - `qwen3-vl-embedding`：通过设置 `enable_fusion=true` 开启融合模式
  - `tongyi-embedding-vision-plus-2026-03-06` / `tongyi-embedding-vision-flash-2026-03-06`：通过将 text、image、video 放在同一个 content 对象中实现融合
  - `qwen2.5-vl-embedding`：仅支持融合向量，不支持独立向量

## 模型概览

### 北京

| 模型 | 向量维度 | 文本长度限制 | 图片大小限制 | 视频大小限制 | 单价（每千输入Token） |
|------|---------|-------------|-------------|-------------|-------------------|
| qwen3-vl-embedding | 2560(默认),2048,1536,1024,768,512,256 | 32,000 Token | 单张 ≤10 MB | ≤50 MB | 图片/视频：0.0018元；文本：0.0007元 |
| qwen2.5-vl-embedding | 2048,1024(默认),768,512 | - | 单张 ≤5 MB | - | - |
| tongyi-embedding-vision-plus-2026-03-06 | 1152(默认),1024,512,256,128,64 | 1,024 Token | 建议≤5 MB，最大10 MB，最多64张 | ≤50 MB，H.264/H.265 | 0.0005元 |
| tongyi-embedding-vision-flash-2026-03-06 | 768(默认),512,256,128,64 | 1,024 Token | 同上 | 同上 | 0.00015元 |
| tongyi-embedding-vision-plus | 1152（固定） | 1,024 Token | 单张≤3 MB，最多8张 | ≤10 MB | 0.0005元 |
| tongyi-embedding-vision-flash | 768（固定） | 1,024 Token | 同上 | 同上 | 0.00015元 |
| multimodal-embedding-v1 | 1,024（固定） | 512 Token | 单张≤3 MB | ≤10 MB | 图片/视频：0.0009元；文本：0.0007元 |

### 模型能力对照

| 模型 | 默认维度 | 向量类型 | 支持的输入 | 说明 |
|------|---------|---------|-----------|------|
| qwen3-vl-embedding | 2560 | 独立/融合 | text, image, video, 多个image | 通过 `enable_fusion` 参数开启融合模式 |
| qwen2.5-vl-embedding | 1024 | 仅融合 | text, image, video | 始终返回1个融合向量，不支持独立向量/多图 |
| tongyi-embedding-vision-plus-2026-03-06 | 1152 | 独立/融合 | text, image, video, multi_images | 基于Qwen3底座，多分辨率，30+语言 |
| tongyi-embedding-vision-flash-2026-03-06 | 768 | 独立/融合 | 同上 | 同上 |
| tongyi-embedding-vision-plus | 1152 | 仅独立 | 支持 multi_images（最多8张） | - |
| tongyi-embedding-vision-flash | 768 | 仅独立 | 同上 | - |
| multimodal-embedding-v1 | 1024 | - | text, image, video | 不支持 dimension 参数 |

## HTTP 调用

```
POST https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding
```

### 请求头

| 头 | 值 |
|---|-----|
| Content-Type | `application/json`（必选） |
| Authorization | `Bearer {API_KEY}`（必选） |

### 请求体

```json
{
    "model": "tongyi-embedding-vision-plus",
    "input": {
        "contents": [
            {"text": "多模态向量模型"},
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

### 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| model | string | 是 | 模型名称 |
| input.contents | array | 是 | 待处理的内容列表，每个元素为 {"模态类型": "值"} |
| parameters.dimension | integer | 否 | 输出向量维度（不同模型支持不同范围） |
| parameters.output_type | string | 否 | 输出向量格式，目前仅支持 `dense` |
| parameters.fps | float | 否 | 视频帧率比例 [0,1]，默认1.0 |
| parameters.instruct | string | 否 | 自定义任务说明，建议英文，可提升1%-5%效果 |
| parameters.enable_fusion | bool | 否 | 是否生成融合向量，仅 qwen3-vl-embedding 支持 |
| parameters.res_level | integer | 否 | 分辨率档位 0/1/2/3，默认1（仅2026-03-06版本） |
| parameters.max_video_frames | integer | 否 | 视频最大采样帧数，默认8，最大64（仅2026-03-06版本） |

### contents 元素类型

| 类型 | key | value 格式 | 说明 |
|------|-----|-----------|------|
| 文本 | `text` | 字符串 | 也可不通过dict直接传入字符串 |
| 图片 | `image` | URL 或 Base64 Data URI | Base64格式: `data:image/{format};base64,{data}` |
| 多图片 | `multi_images` | 图片URL列表 | 仅 plus/flash 系列支持 |
| 视频 | `video` | 公开可访问的URL | - |

### 融合向量（2026-03-06 版本）

将 text、image、video 放在同一个 content 对象中：

```json
{
    "model": "tongyi-embedding-vision-plus-2026-03-06",
    "input": {
        "contents": [
            {
                "text": "描述文本",
                "image": "https://example.com/img.jpg",
                "video": "https://example.com/video.mp4"
            }
        ]
    },
    "parameters": {
        "dimension": 1152
    }
}
```

### 成功响应

```json
{
    "output": {
        "embeddings": [
            {
                "index": 0,
                "embedding": [-0.0266, -0.0165, ...],
                "type": "text"
            },
            {
                "index": 1,
                "embedding": [0.0515, 0.0077, ...],
                "type": "image"
            }
        ]
    },
    "usage": {
        "input_tokens": 10,
        "input_tokens_details": {
            "image_tokens": 896,
            "text_tokens": 7
        },
        "output_tokens": 3,
        "total_tokens": 906
    },
    "request_id": "1fff9502-..."
}
```

**type 字段取值**：
- `text` / `image` / `video` / `multi_images` — 对应各模态独立向量
- `fused` — 2026-03-06 版本融合向量
- `fusion` — qwen3-vl-embedding / qwen2.5-vl-embedding 融合向量
- `vl` — qwen3-vl-embedding 独立向量

### usage 字段差异

| 模型 | 返回字段 |
|------|---------|
| tongyi-embedding-vision-* | input_tokens(含图片), input_tokens_details{image_tokens, text_tokens}, output_tokens, total_tokens |
| qwen3-vl-embedding | input_tokens(仅文本), image_tokens, total_tokens(=input_tokens+image_tokens) |
| qwen2.5-vl-embedding | input_tokens, image_tokens（无 total_tokens） |
| multimodal-embedding-v1 | input_tokens, image_tokens, image_count, duration |

### 异常响应

```json
{
    "code": "InvalidApiKey",
    "message": "Invalid API-key provided.",
    "request_id": "fb53c4ec-..."
}
```

## SDK 使用

DashScope Python SDK 通过 `dashscope.MultiModalEmbedding.call()` 调用：

```python
import dashscope
from http import HTTPStatus

resp = dashscope.MultiModalEmbedding.call(
    model="tongyi-embedding-vision-plus",
    input=[{"image": "https://example.com/img.jpg"}],
    # 可选参数
    dimension=1152,
    enable_fusion=True,  # 仅 qwen3-vl-embedding
    res_level=1,         # 仅 2026-03-06 版本
    max_video_frames=64  # 仅 2026-03-06 版本
)
```

> SDK 的 `input` 参数对应 HTTP 请求体中的 `input.contents`，两者结构**不一致**。

## 输入格式与语种限制

| 模型 | 文本 | 图片 | 视频 | 单次请求条数 |
|------|------|------|------|-------------|
| qwen3-vl-embedding | 33种主流语言 | JPEG,PNG,WEBP,BMP,TIFF,ICO,DIB,ICNS,SGI(URL或Base64) | MP4,AVI,MOV(仅URL) | 内容元素≤20，图片≤5，视频≤1 |
| qwen2.5-vl-embedding | 11种主流语言 | 同上 | 同上 | 每种类型最多出现1次 |
| tongyi-embedding-vision-plus-2026-03-06 | 30+种语言 | 同上 | MP4,MPEG,MOV,MPG,WEBM,AVI,FLV,MKV(仅URL) | 元素≤20，图片≤64，视频≤8 |
| tongyi-embedding-vision-flash-2026-03-06 | 同上 | 同上 | 同上 | 同上 |
| tongyi-embedding-vision-plus | 中英文 | JPG,PNG,BMP(URL或Base64) | 同上 | 无明确限制 |
| tongyi-embedding-vision-flash | 中英文 | 同上 | 同上 | 同上 |
| multimodal-embedding-v1 | 中英文 | 同上 | 同上 | 元素≤20，图片/视频各≤1，文本≤20 |
