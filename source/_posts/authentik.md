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
