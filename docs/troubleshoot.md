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
