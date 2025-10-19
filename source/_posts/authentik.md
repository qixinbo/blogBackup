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
SESSION_SECRET=demo-secret-9b8e2d8e3a4c5f6d7e8f9a0b1c2d3e4  # 这个随意设置
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
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "dev-secret"))

AUTHENTIK_ISSUER = os.getenv("AUTHENTIK_ISSUER")
OIDC_CLIENT_ID = os.getenv("OIDC_CLIENT_ID")
OIDC_CLIENT_SECRET = os.getenv("OIDC_CLIENT_SECRET")

oauth = OAuth()
oauth.register(
    name="authentik",
    client_id=OIDC_CLIENT_ID,
    client_secret=OIDC_CLIENT_SECRET,
    server_metadata_url=f"{AUTHENTIK_ISSUER.rstrip('/')}/.well-known/openid-configuration" if AUTHENTIK_ISSUER else None,
    client_kwargs={"scope": "openid profile email"},
)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user = request.session.get("user")
    if user:
        name = user.get("name") or user.get("preferred_username") or user.get("email") or "User"
        return HTMLResponse(
            f"<h1>Hello, {name}</h1>\n"
            f"<p><a href=\"/api/hello\">Protected API</a></p>\n"
            f"<p><a href=\"/logout\">Logout</a></p>"
        )
    return HTMLResponse(
        "<h1>FastAPI + Authentik</h1>\n"
        "<p><a href=\"/login\">Login with Authentik</a></p>"
    )

@app.get("/login")
async def login(request: Request):
    return await oauth.authentik.authorize_redirect(request, request.url_for("auth_callback"))

@app.get("/auth/callback")
async def auth_callback(request: Request):
    token = await oauth.authentik.authorize_access_token(request)
    user = await oauth.authentik.userinfo(token=token)
    request.session["user"] = user
    return RedirectResponse("/")

@app.get("/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    return RedirectResponse("/")

@app.get("/api/hello")
async def api_hello(request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    who = user.get("preferred_username") or user.get("email") or "user"
    return {"message": f"Hello, {who}!"}
```

# 允许用户注册
默认Authentik没有允许用户自己注册账号，所以在登录页面没有signup这个选项。
这里配置开启该功能，核心思路是：我们告诉主登录流程，“如果遇到不认识的用户，就把他们引导到这个专门的注册流程去”。
第1步：确认存在一个“注册流程”(Enrollment Flow)
Authentik 通常会内置一个默认的注册流程，我们先检查它是否存在且配置正确。
1、在左侧导航栏，展开 Flows & Stages -> Flows。
2、在流程列表中，寻找一个名为 default-enrollment-flow 的流程。
（1）如果存在，点击进入，检查一下它是否绑定了必要的 Stage，通常默认的配置是可用的。
（2）如果不存在，从[这里](https://docs.goauthentik.io/add-secure-apps/flows-stages/flow/examples/flows/#enrollment-2-stage)下载示例文件，并在页面导入。
第2步：将注册流程关联到主登录流程
这是最关键的一步。
1、回到 Flows & Stages -> Flows 列表。
2、点击打开您的主登录流程，即 default-authentication-flow。
3、点击 Stage Bindings 标签页。
4、点击编辑那个 Identification Stage (默认名为 default-authentication-identification-stage)。
5、现在，在这个 Stage 的编辑页面中，您会看到一个名为 Enrollment 的板块。
6、找到 Enrollment flow 字段。它默认是空的。
7、点击下拉菜单，选择您在第一步中确认或创建的 default-enrollment-flow。
8、(可选) 在 Recovery flow 字段中，您可以类似地关联一个 default-recovery-flow，这样登录页面就会同时出现“忘记密码”的选项。
9、点击页面底部的 Update 保存更改。

现在请您刷新登录页面，应该就可以看到 "Sign up" 的链接了。如果点击该链接，就会进入一个引导您设置用户名、密码等信息的流程。
此时再次使用上面的HelloWorld应用，就可以新建用户了。

注意，如果用新建的用户访问一些权限高的Authentik链接，就会报错，比如`"Interface can only be accessed by internal users"`。
这是因为Authentik 出于安全考虑，默认会保护一些敏感的应用，比如它自己的管理后台 (authentik Embedded Outpost)。它的默认策略是：只允许“内部用户”(Internal users) 访问。
内部用户 (Internal user): 通常指由管理员在后台直接创建的用户。
外部用户 (External user): 通过开放注册、社交登录（如Google/GitHub）或邀请链接创建的用户，默认都属于“外部用户”。

# 参考教程
- [Up主ecwuuu的Authentik视频教程](https://space.bilibili.com/509461/lists/2694404?type=season)
