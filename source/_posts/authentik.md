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

