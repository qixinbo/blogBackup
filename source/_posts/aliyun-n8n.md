---
title: 在阿里云上部署n8n工作流
tags: [Aliyun]
categories: coding 
date: 2025-9-14
---
这是记录一下在阿里云服务器部署n8n的过程。。

# 购买
云服务器推荐购买至少4GB内存，这样服务器不容易卡死。

# 部署n8n
使用`npm`部署n8n未成功，此处使用docker进行部署。

## 创建卷
```bash
docker volume create n8n_data
```

## 运行
```bash
docker run -it -d --restart unless-stopped --name n8n -p 5678:5678 -e N8N_SECURE_COOKIE=false -e N8N_HOST=http://8.130.100.xx -e N8N_PORT=5678 -e N8N_PROTOCOL=http -e N8N_EDITOR_BASE_URL=http://8.130.100.xx:5678/ -e N8N_WEBHOOK_URL=http://8.130.100.xx:5678/ -v n8n_data:/home/node/.n8n  n8nio/n8n
```
这里与官方命令有几个不同：
1、加上了`-d`参数，默认后台运行
2、将镜像地址由`docker.n8n.io/n8nio/n8n`改成`n8nio/n8n`，因为国内连接不稳定。
此时注意将docker镜像源改成国内地址。
3、加上了`N8N_HOST`、`N8N_EDITOR_BASE_URL`、`N8N_WEBHOOK_URL`等地址，因为n8n默认绑定到`localhost`，在使用表单时会无法弹出。
