---
title: 开源身份认证与授权系统Authentik使用体验
tags: [LLM]
categories: coding 
date: 2025-10-8
---

[Authentik](https://goauthentik.io)是目前在自托管领域非常受欢迎的开源身份认证与授权系统，它是很多现代应用（包括FastAPI、Vue、Kubernetes Dashboard、Grafana 等）常用的统一身份认证（SSO）解决方案。
它可以实现的主要功能清单如下：

| 类别            | 功能描述                                       |
| ------------- | ------------------------------------------ |
| 🧑 用户管理    | 用户注册、密码找回、邮箱验证、分组、属性扩展                     |
| 🔐 登录方式       | 用户名+密码、MFA（TOTP、短信、WebAuthn）、社交登录          |
| 🔄 单点登录 (SSO) | 完整支持 OIDC、OAuth2、SAML 2.0                  |
| 🎟️ 授权        | 内置 RBAC（角色、组、策略），支持自定义规则（Python 表达式）       |
| 🧾 应用管理       | 可以为任意外部应用注册 client_id/client_secret 实现统一登录 |
| 🧰 集成接口       | 支持 LDAP、SCIM、Webhook、API Token 等           |
| 🧩 多租户        | 可通过 namespace / directory 隔离不同组织的身份系统      |
| 🪪 用户门户       | 用户可自行修改资料、重置密码、查看活跃会话、解绑 MFA               |
| 🧠 审计与监控      | 所有操作事件均可审计；支持 Prometheus metrics 导出        |

# 准备
下载`docker-compose.yml`文件：
```sh
curl -O https://docs.goauthentik.io/docker-compose.yml
# or use wget
# wget https://docs.goauthentik.io/docker-compose.yml
```
产生密码和Secret Key：
```sh
echo "PG_PASS=$(openssl rand -base64 36 | tr -d '\n')" >> .env
echo "AUTHENTIK_SECRET_KEY=$(openssl rand -base64 60 | tr -d '\n')" >> .env
```
开启email通知：（在`.env`文件中添加如下信息）
```sh
# SMTP Host Emails are sent to
AUTHENTIK_EMAIL__HOST=localhost
AUTHENTIK_EMAIL__PORT=25
# Optionally authenticate (don't add quotation marks to your password)
AUTHENTIK_EMAIL__USERNAME=
AUTHENTIK_EMAIL__PASSWORD=
# Use StartTLS
AUTHENTIK_EMAIL__USE_TLS=false
# Use SSL
AUTHENTIK_EMAIL__USE_SSL=false
AUTHENTIK_EMAIL__TIMEOUT=10
# Email address authentik will send from, should have a correct @domain
AUTHENTIK_EMAIL__FROM=authentik@localhost
```

Authentik默认监听HTTP的9000端口和HTTPS的9443端口，可以指定自定义端口：
```sh
COMPOSE_PORT_HTTP=80
COMPOSE_PORT_HTTPS=443
```

# 启动
```sh
docker compose pull
docker compose up -d
```
然后通过浏览器访问如下链接为`akadmin`用户设置邮箱和密码（注意不要忘了最后的`/`符号）：
```sh
http://<your server's IP or hostname>:9000/if/flow/initial-setup/.
```

# 核心概念理解
在开始动手之前，先搞清几个核心概念：

| 名称                         | 含义 / 角色                                                                                                        |
| -------------------------- | ---------------------------------------------------------------------------------- |
| Authentik                  | 一个开源的身份提供者 (Identity Provider, IdP)，支持 OAuth2 / OIDC / SAML / LDAP 等协议   |
| Provider                   | 在 Authentik 中，表示一个你希望让用户用来登录或认证你的应用的协议实体。    |
| Application                | 你的目标应用（客户端），它会让用户通过 Authentik 登录访问。            |
| Flow / Stage / Policy      | 用于定制登录、注册、恢复等流程：Flow 是多个 Stage 的组合，Stage 是一步操作（如用户名密码、2FA 等），Policy 用来判断是否执行某个 Stage 或流程。 |
| Directory / Users / Groups | 用户目录、用户与用户组概念。          |

所以Authentik的大致工作机制就是：用户访问你的应用 → 应用重定向到 Authentik 认证 → 用户登录 → Authentik 发一个 token / assertion 给应用 → 应用允许访问。

# HelloWorld入门例子
## 前置条件
- 你已在本地启动了 Authentik，地址：`http://localhost:9000/`
- 你可以登录 Authentik 管理后台（默认路径：`http://localhost:9000/if/admin/`）。
- 本示例默认 FastAPI 运行在 `http://localhost:8000/`。


## 在 Authentik 中创建 OIDC Provider 与 Application
创建 Provider（OIDC）:
1. 登录 Authentik 管理后台。
2. 进入：Applications → Providers → Create（或通过 Applications 创建并绑定 Provider）。
3. Provider 类型选择：`OpenID Connect`（授权模式使用 Authorization Code）。
4. 在 Redirect URIs 中添加：`http://localhost:8000/auth/callback`
5. 可选：Post-Logout Redirect URI 添加：`http://localhost:8000/`
6. Scopes 保留/包含：`openid profile email`。
7. 保存后，记录下：`Client ID` 与 `Client Secret`（如果secret找不到了，可以通过edit再次查看）。

创建 Application 并绑定 Provider
1. 进入：`Applications → Applications → Create`。
2. 填写 `Name` 与 `Slug`（例如：`fastapi-demo-app`）。
3. `Launch URL` 填写：`http://localhost:8000/`（从门户点击应用时跳转到此地址）。
4. 在 `Provider` 字段选择上面创建的 OIDC Provider（例如：`fastapi-demo`）。
5. 可选：为 Application 配置访问策略（Policies），例如仅允许某些用户或组访问。
6. 保存后，在用户门户 `http://localhost:9000/if/` 可以看到该应用卡片；点击卡片将跳转到 `Launch URL` 并触发 OIDC 登录。


将两者关联后，需要重新查看Provider，获得Issuer信息，比如：`http://localhost:9000/application/o/fastapi-demo`。


## 配置本地应用
将如下环境变量替换为Authentik中的数值：
```sh
AUTHENTIK_ISSUER=http://localhost:9000/application/o/fastapi-demo
OIDC_CLIENT_ID=demo-fastapi-client-id
OIDC_CLIENT_SECRET=demo-fastapi-client-secret
```

## 本地应用示例
该示例可以实现如下功能：
访问首页：`http://localhost:8000/`
   - 点击 “Login with Authentik” 会跳转到 Authentik 登录并返回。
   - 登录完成后看到欢迎页与 “Logout” 链接。
   - 访问受保护的接口：`http://localhost:8000/api/hello`。


```python
import os
from fastapi import FastAPI, Request, HTTPException
from starlette.responses import HTMLResponse, RedirectResponse, PlainTextResponse
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth, OAuthError
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

app = FastAPI()

# Session secret for signing cookies
SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-secret-change-me")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

# Authentik OIDC configuration from environment
AUTHENTIK_ISSUER = os.getenv("AUTHENTIK_ISSUER")  # e.g. http://localhost:9000/application/o/fastapi-demo
OIDC_CLIENT_ID = os.getenv("OIDC_CLIENT_ID")
OIDC_CLIENT_SECRET = os.getenv("OIDC_CLIENT_SECRET")

if not AUTHENTIK_ISSUER:
    print("[WARN] AUTHENTIK_ISSUER is not set. Set it to your provider base URL.")
if not OIDC_CLIENT_ID or not OIDC_CLIENT_SECRET:
    print("[WARN] OIDC_CLIENT_ID / OIDC_CLIENT_SECRET are not set. Login will fail until set.")

# Configure OAuth client with Authlib

def _build_metadata_url(issuer: str | None):
    if not issuer:
        return None
    issuer_clean = issuer.rstrip('/')
    well_known = '/.well-known/openid-configuration'
    if issuer_clean.endswith(well_known):
        return issuer_clean
    return f"{issuer_clean}{well_known}"

oauth = OAuth()
oauth.register(
    name="authentik",
    client_id=OIDC_CLIENT_ID,
    client_secret=OIDC_CLIENT_SECRET,
    server_metadata_url=_build_metadata_url(AUTHENTIK_ISSUER),
    client_kwargs={"scope": "openid profile email"},
)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user = request.session.get("user")
    if user:
        display_name = user.get("name") or user.get("preferred_username") or user.get("email") or "User"
        return HTMLResponse(
            f"""
            <h1>Hello, {display_name}</h1>
            <p>You are signed in via Authentik.</p>
            <p><a href=\"/logout\">Logout</a></p>
            <p><a href=\"/api/hello\">Call protected API</a></p>
            """
        )
    return HTMLResponse(
        """
        <h1>Hello, FastAPI + Authentik</h1>
        <p><a href=\"/login\">Login with Authentik</a></p>
        """
    )

@app.get("/login")
async def login(request: Request):
    if not AUTHENTIK_ISSUER or not OIDC_CLIENT_ID or not OIDC_CLIENT_SECRET:
        return PlainTextResponse("OIDC is not configured. Set AUTHENTIK_ISSUER, OIDC_CLIENT_ID, OIDC_CLIENT_SECRET.", status_code=500)
    redirect_uri = request.url_for("auth_callback")
    return await oauth.authentik.authorize_redirect(request, redirect_uri)

@app.get("/auth/callback")
async def auth_callback(request: Request):
    try:
        token = await oauth.authentik.authorize_access_token(request)
    except OAuthError as err:
        return PlainTextResponse(f"OAuth error: {err.error}", status_code=400)

    # Prefer ID Token; fallback to UserInfo when id_token is missing
    user = None
    try:
        if isinstance(token, dict) and token.get("id_token"):
            user = await oauth.authentik.parse_id_token(request, token)
    except Exception:
        user = None

    if user is None:
        try:
            user = await oauth.authentik.userinfo(token=token)
        except Exception as e:
            return PlainTextResponse(f"Failed to fetch user info: {e}", status_code=400)

    request.session["user"] = user
    return RedirectResponse(url="/")

@app.get("/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    return RedirectResponse(url="/")

# Simple protected API example
@app.get("/api/hello")
async def api_hello(request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    who = user.get("preferred_username") or user.get("email") or "user"
    return {"message": f"Hello, {who}! This is a protected endpoint."}
```


# 参考教程
- [Up主ecwuuu的Authentik视频教程](https://space.bilibili.com/509461/lists/2694404?type=season)
