# Troubleshoot

> 跨会话踩坑记录。每次实现完成后追加。

---

## [D1/D2/D3] pytest-asyncio 未安装导致 async 测试无法收集

**Problem description:**
集成测试中写了 `@pytest.mark.asyncio async def test_xxx()` 的异步测试，运行时报：
```
async def functions are not natively supported.
You need to install a suitable plugin...
```
pyproject.toml 中虽然配置了 `asyncio_mode = "auto"`，但 pytest 报 `Unknown config option: asyncio_mode`。

**Root cause:**
当前 venv（`/Users/ppsteven/.venvs/base`）未安装 `pytest-asyncio`，所以 `asyncio_mode` 配置无效、`@pytest.mark.asyncio` 无法识别。`anyio` 已安装但不足以单独驱动 `@pytest.mark.asyncio`。

**Solution / workaround:**
将 async 集成测试改为用 `asyncio.run()` 包裹：
```python
def test_async_multimodal_embedding():
    async def _run():
        ...  # 原 async 逻辑
    result = asyncio.run(_run())
    assert ...
```
这样不依赖任何 async 测试插件，pytest 直接作为同步测试收集执行。

---

## [本地构建] Docker Desktop 过旧导致 `cryptography` / Rust 扩展 SIGILL

**Problem description:**
在 Apple Silicon Mac 上用本地源码构建的 `litellm:local` 镜像，`docker run` 后容器立刻以 exit code 132 退出，无任何日志输出。手动进容器执行：
```
/app/.venv/bin/python3 -c "import litellm"
Illegal instruction
```
连 `from cryptography.hazmat.bindings._rust import x509` 也同样崩溃。

**Root cause:**
两个叠加问题：

1. **Docker Desktop 版本过旧（4.27.2，linuxkit 6.6.12）**：`cryptography 44+` 开始在 AArch64 共享库中使用 BTI（Branch Target Identification）指令，而旧版 linuxkit 内核不完整支持该特性，导致 dlopen 时触发 SIGILL 信号直接 kill 进程，不抛 Python 异常。

2. **`litellm/rust_bridge/loader.py` 只捕获 `ImportError`**：`_native.so` 引发的 SIGILL 是操作系统信号而非 Python 异常，`except ImportError` 无法捕获，进程被 kill。

**Solution:**

- **根治**：升级 Docker Desktop 到 4.30+（linuxkit 6.10+）。可通过 Homebrew 升级：
  ```bash
  brew install --cask --force docker-desktop
  ```
  如果遇到 `/usr/local/bin/docker-credential-ecr-login` 需要 sudo 的报错，先手动挂载 DMG 并替换 `/Applications/Docker.app`：
  ```bash
  hdiutil attach ~/Library/Caches/Homebrew/downloads/<docker>.dmg -nobrowse
  rm -rf /Applications/Docker.app
  cp -R /Volumes/Docker/Docker.app /Applications/Docker.app
  hdiutil detach /Volumes/Docker
  ```

- **代码层修复**（已合入）：`litellm/rust_bridge/loader.py` 的 `get_native_bridge()` 改为用 subprocess 探针预检测 `.so` 是否能被正常 import，防止 SIGILL 杀死主进程：
  ```python
  def _probe_native_bridge() -> bool:
      result = subprocess.run(
          [sys.executable, "-c", "from litellm.rust_bridge import _native"],
          timeout=10, capture_output=True,
      )
      return result.returncode == 0
  ```

**验证：**
```bash
docker run --rm python:3.13-slim sh -c \
  'pip install "cryptography==48.0.1" -q && python -c \
   "from cryptography.hazmat.bindings._rust import x509; print(\"ok\")"'
# 输出: ok  （升级前输出 Illegal instruction）
```

---

## [本地构建] Docker 虚拟磁盘满导致 PostgreSQL 无法启动

**Problem description:**
`docker compose up -d db` 后容器反复 Restart，日志：
```
FATAL:  could not write lock file "postmaster.pid": No space left on device
```

**Root cause:**
Docker Desktop 的虚拟磁盘（`Docker.raw`）被历史镜像占满。本例中 3 个 `nccl-tests` GPU 镜像共占用约 40GB，加上构建缓存 8.6GB，导致磁盘耗尽。

**Solution:**
查看磁盘使用情况，清理不再需要的镜像和缓存：
```bash
docker system df                  # 查看各类占用
docker image prune -f             # 清理 dangling 镜像
docker buildx prune -f            # 清理构建缓存
docker rmi <大镜像 ID...>          # 手动删除不用的大镜像
```

---

## [本地构建] docker-compose.yml 使用的是远端镜像 tag，需手动指定本地镜像

**Problem description:**
`docker-compose.yml` 中 `litellm` 服务的 `image` 字段写的是
`docker.litellm.ai/berriai/litellm:main-stable`（远端 tag），直接
`docker compose up` 会拉远端而非使用本地构建的镜像。

**Solution:**
用本地镜像启动时，绕开 compose 直接 `docker run`，挂到同一网络：
```bash
# 先启动数据库
docker compose up -d db

# 用本地镜像启动 proxy
docker run -d \
  --name litellm_proxy \
  --network litellm_default \
  -p 4000:4000 \
  -e DATABASE_URL="postgresql://llmproxy:dbpassword9090@litellm_db:5432/litellm" \
  -e STORE_MODEL_IN_DB="True" \
  -e LITELLM_MASTER_KEY="sk-1234" \
  litellm:local
```

或修改 `docker-compose.yml`，将 `image:` 行改为本地 tag：
```yaml
litellm:
  image: litellm:local   # 改为本地构建的 tag
  # build: 注释掉或保留
```

**健康检查：**
```bash
curl http://localhost:4000/health/liveliness   # "I'm alive!"
curl http://localhost:4000/health/readiness    # {"status":"healthy","db":"connected"}
```

---

## [D1] `utils.py` 路由需要在函数体内 import，不能在模块顶层 import

**Problem description:**
`utils.py` 中 provider config 路由逻辑都在 `get_provider_embedding_config()` 函数内部用局部 import，而不是文件顶部 import。

**Root cause:**
litellm 项目规模很大，避免循环依赖和启动时间，provider-specific 模块在路由函数中按需 import。

**Solution / workaround:**
新增路由时，在 `elif litellm.LlmProviders.DASHSCOPE == provider:` 块内部做局部 import：
```python
elif litellm.LlmProviders.DASHSCOPE == provider:
    from litellm.llms.dashscope.embed.transformation import DashScopeEmbeddingConfig
    from litellm.llms.dashscope.embed.transformation_multimodal import DashScopeMultimodalEmbeddingConfig
    if DashScopeMultimodalEmbeddingConfig.is_multimodal_embedding(model):
        return DashScopeMultimodalEmbeddingConfig()
    return DashScopeEmbeddingConfig()
```

---

## [D4/D5] qwen3-vl-embedding 的 usage.input_tokens 只含文本 token，不含图片 token

**Problem description:**
`qwen3-vl-embedding` 和 `qwen2.5-vl-embedding` 的 API 返回中，`usage.input_tokens` 只包含文本 token 数（如 7），图片 token 在顶层 `usage.image_tokens` 字段（如 896）。而 `tongyi-embedding-vision-*` 系列的 `input_tokens` 是总数（文本 + 图片），图片 token 嵌套在 `input_tokens_details.image_tokens`。

**Root cause:**
DashScope 多模态 embedding API 的 usage 格式因模型而异，没有统一规范。cost calculator 的 `_extract_token_breakdown` 使用 `text_tokens = prompt_tokens - image_tokens` 公式，要求 `prompt_tokens` 为总数。

**Solution / workaround:**
在 `transformation_multimodal.py` 的 `transform_embedding_response` 中，检测到顶层 `image_tokens` 字段时，将 `input_tokens` 修正为 `input_tokens + image_tokens`，使 `prompt_tokens` 成为总数。`tongyi` 系列走 `input_tokens_details` 分支，`input_tokens` 本身已是总数，无需修正。

---

## [D4/D5] backup JSON 文件缺失 dashscope embedding 条目

**Problem description:**
`litellm/model_prices_and_context_window_backup.json` 中完全没有 `dashscope/qwen3-vl-embedding` 等 7 个 embedding 模型的 pricing entry。修改主文件后，如果不同步 backup，`LITELLM_LOCAL_MODEL_COST_MAP=True` 模式下找不到这些模型。

**Root cause:**
backup 文件是主文件的独立副本，需要手动同步。主文件中有这些条目但 backup 中没有，可能是历史同步遗漏。

**Solution:**
修改主文件后，必须同步修改 backup 文件。直接复制主文件中对应的 JSON 块到 backup 的相同位置（按字母序插入）。
```
