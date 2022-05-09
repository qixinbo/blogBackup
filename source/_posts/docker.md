---
title: Docker知识点
tags: [Docker]
categories: coding 
date: 2022-4-20
---

参考文献：
[Docker 入门教程](https://www.ruanyifeng.com/blog/2018/02/docker-tutorial.html)
[Docker —— 从入门到实践](https://yeasy.gitbook.io/docker_practice/)
[【狂神说Java】Docker最新超详细版教程通俗易懂](https://www.bilibili.com/video/BV1og4y1q7M4)

# 为什么要用Docker
Docker解决的问题是将软件连带其环境一起安装。
## 虚拟机与容器
### 虚拟机
虚拟机（virtual machine）就是带环境安装的一种解决方案。它可以在一种操作系统里面运行另一种操作系统，比如在 Windows 系统里面运行 Linux 系统。应用程序对此毫无感知，因为虚拟机看上去跟真实系统一模一样，而对于底层系统来说，虚拟机就是一个普通文件，不需要了就删掉，对其他部分毫无影响。
虽然用户可以通过虚拟机还原软件的原始环境。但是，这个方案有几个缺点。
（1）资源占用多
虚拟机会独占一部分内存和硬盘空间。它运行的时候，其他程序就不能使用这些资源了。哪怕虚拟机里面的应用程序，真正使用的内存只有 1MB，虚拟机依然需要几百 MB 的内存才能运行。
（2）冗余步骤多
虚拟机是完整的操作系统，一些系统级别的操作步骤，往往无法跳过，比如用户登录。
（3）启动慢
启动操作系统需要多久，启动虚拟机就需要多久。可能要等几分钟，应用程序才能真正运行。

### Linux容器
由于虚拟机存在这些缺点，Linux 发展出了另一种虚拟化技术：Linux 容器（Linux Containers，缩写为 LXC）。
Linux 容器不是模拟一个完整的操作系统，而是对进程进行隔离。或者说，在正常进程的外面套了一个保护层。对于容器里面的进程来说，它接触到的各种资源都是虚拟的，从而实现与底层系统的隔离。
由于容器是进程级别的，相比虚拟机有很多优势。
（1）启动快
容器里面的应用，直接就是底层系统的一个进程，而不是虚拟机内部的进程。所以，启动容器相当于启动本机的一个进程，而不是启动一个操作系统，速度就快很多。
（2）资源占用少
容器只占用需要的资源，不占用那些没有用到的资源；虚拟机由于是完整的操作系统，不可避免要占用所有资源。另外，多个容器可以共享资源，虚拟机都是独享资源。
（3）体积小
容器只要包含用到的组件即可，而虚拟机是整个操作系统的打包，所以容器文件比虚拟机文件要小很多。
总之，容器有点像轻量级的虚拟机，能够提供虚拟化的环境，但是成本开销小得多。

## Docker
Docker 属于 Linux 容器的一种封装，提供简单易用的容器使用接口。它是目前最流行的 Linux 容器解决方案。
Docker 将应用程序与该程序的依赖，打包在一个文件里面。运行这个文件，就会生成一个虚拟容器。程序在这个虚拟容器里运行，就好像在真实的物理机上运行一样。有了 Docker，就不用担心环境问题。
总体来说，Docker 的接口相当简单，用户可以方便地创建和使用容器，把自己的应用放入容器。容器还可以进行版本管理、复制、分享、修改，就像管理普通的代码一样。


# 什么是Docker
## 安装Docker
首先安装一下Docker，然后实操起来看看什么是Docker。
参考官方教程，在[这里](https://docs.docker.com/engine/install/)或[该教程](https://yeasy.gitbook.io/docker_practice/install)。
以ubuntu为例：
（1）卸载旧版本
```python
sudo apt-get remove docker docker-engine docker.io containerd runc
```
（2）安装依赖包
```python
 sudo apt-get update
 sudo apt-get install ca-certificates curl gnupg lsb-release
```
（3）添加官方的key
```python
curl -fsSLhttps://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor-o/usr/share/keyrings/docker-archive-keyring.gpg
```
（4）配置仓库
```python
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu  $(lsb_release -cs) stable"| sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
```
注意这个地方是特用于ubuntu的。如果是基于ubuntu的再次发行版，比如Linux Mint，需要将自动探测版本那块的代码改成确定的ubuntu的版本，否则会报找不到包的错误。
比如改成：
```python
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu  bionic stable"| sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
```
（5）安装Docker Engine
```python
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

安装完成后，使用下面命令验证是否安装成功：
```python
docker version
```

## 配置权限
Docker 是服务器----客户端架构。命令行运行docker命令的时候，需要本机有 Docker 服务。如果这项服务没有启动，可以用下面其中一个命令启动：
```python
sudo service docker start
或
sudo systemctl start docker
```

Docker 需要用户具有 sudo 权限，为了避免每次命令都输入sudo，可以把用户加入 Docker 用户组：
```python
sudo usermod -aG docker $USER
```
添加后需要重新启动系统。


## 镜像文件
Docker 把应用程序及其依赖，打包在 image 镜像文件里面。只有通过这个文件，才能生成 Docker 容器。image 文件可以看作是容器的模板。Docker 根据 image 文件生成容器的实例。同一个 image 文件，可以生成多个同时运行的容器实例。
Docker 镜像是一个特殊的文件系统，除了提供容器运行时所需的程序、库、资源、配置等文件外，还包含了一些为运行时准备的一些配置参数（如匿名卷、环境变量、用户等）。镜像不包含任何动态数据，其内容在构建之后也不会被改变。
image 是二进制文件。实际开发中，一个 image 文件往往通过继承另一个 image 文件，加上一些个性化设置而生成。举例来说，你可以在 Ubuntu 的 image 基础上，往里面加入 Apache 服务器，形成你的 image。
```python
# 列出本机的所有 image 文件。
$ docker image ls
# 删除 image 文件
$ docker image rm [imageName]
```
image 文件是通用的，一台机器的 image 文件拷贝到另一台机器，照样可以使用。一般来说，为了节省时间，我们应该尽量使用别人制作好的 image 文件，而不是自己制作。即使要定制，也应该基于别人的 image 文件进行加工，而不是从零开始制作。

## 仓库
为了方便共享，image 文件制作完成后，可以上传到网上的仓库，用于集中的存储、分发镜像，Docker Registry 就是这样的服务。
一个 Docker Registry 中可以包含多个 仓库（Repository）；每个仓库可以包含多个 标签（Tag）；每个标签对应一个镜像。
通常，一个仓库会包含同一个软件不同版本的镜像，而标签就常用于对应该软件的各个版本。我们可以通过 <仓库名>:<标签> 的格式来指定具体是这个软件哪个版本的镜像。如果不给出标签，将以 latest 作为默认标签。
仓库名经常以 两段式路径 形式出现，比如 jwilder/nginx-proxy，前者往往意味着 Docker Registry 多用户环境下的用户名，后者则往往是对应的软件名。但这并非绝对，取决于所使用的具体 Docker Registry 的软件或服务。
最常使用的 Registry 公开服务是官方的 Docker Hub，这也是默认的 Registry，并拥有大量的高质量的官方镜像。
如果在使用过程中发现拉取 Docker 镜像十分缓慢，可以配置 Docker[国内镜像加速](https://yeasy.gitbook.io/docker_practice/install/mirror)。
从docker官方拉取hello-world镜像：
```python
docker pull hello-world
```

## 容器文件
镜像（Image）和容器（Container）的关系，就像是面向对象程序设计中的 类 和 实例 一样，镜像是静态的定义，容器是镜像运行时的实体。容器可以被创建、启动、停止、删除、暂停等。
容器的实质是进程，但与直接在宿主执行的进程不同，容器进程运行于属于自己的独立的 命名空间。因此容器可以拥有自己的 root 文件系统、自己的网络配置、自己的进程空间，甚至自己的用户 ID 空间。容器内的进程是运行在一个隔离的环境里，使用起来，就好像是在一个独立于宿主的系统下操作一样。这种特性使得容器封装的应用比直接在宿主运行更加安全。

启动容器有两种方式，一种是基于镜像新建一个容器并启动，另外一个是将在终止状态（stopped）的容器重新启动。
因为 Docker 的容器实在太轻量级了，很多时候用户都是随时删除和新创建容器。

生成容器：
```python
docker run IMAGE
```
run命令是新建容器，每运行一次，就会新建一个容器。同样的命令运行两次，就会生成两个一模一样的容器文件。
当利用 docker run 来创建容器时，Docker 在后台运行的标准操作包括：
	- 检查本地是否存在指定的镜像，不存在就从公有仓库下载
	- 利用镜像创建并启动一个容器
	- 分配一个文件系统，并在只读的镜像层外面挂载一层可读写层
	- 从宿主主机配置的网桥接口中桥接一个虚拟接口到容器中去
	- 从地址池配置一个 ip 地址给容器
	- 执行用户指定的应用程序
	- 执行完毕后容器被终止

如果希望重复使用容器，就要使用start命令，它用来启动已经生成、已经停止运行的容器文件：
```python
docker start IMAGE
```
image 文件生成的容器实例，本身也是一个文件，称为容器文件。也就是说，一旦容器生成，就会同时存在两个文件： image 文件和容器文件。而且关闭容器并不会删除容器文件，只是容器停止运行而已。
```python
# 列出本机正在运行的容器
$ docker container ls
# 列出本机所有容器，包括终止运行的容器
$ docker container ls --all
```

终止运行的容器文件，依然会占据硬盘空间，可以使用以下命令删除。
```python
$ docker rm [containerID]
```

# 镜像相关命令
列出镜像：`docker images`，REPOSITORY/TAG指明了一个具体镜像
查看镜像详细信息：`docker inspect`
删除镜像：`docker rmi` 镜像名或ID，如果是镜像名，可能只删除了某个tag，如果是ID，则将这个镜像完全删除。

搜索镜像：`docker search`，也可以通过官网搜索，在[https://hub.docker.com/search](https://hub.docker.com/search)。
拉取镜像：`docker pull`，可以修改`/etc/default/docker`文件，添加镜像源。
推送镜像：`docker push`

构建镜像的作用：
（1）保存对容器的修改，并再次使用；
（2）自定义镜像的能力；
（3）以软件的形式打包并分发服务及其运行环境
构建镜像的方式：
（1）通过容器构建：`docker commit`
（2）通过Dockerfile文件构建：`docker build`

Dockerfile指令格式：INSTRUCTION argument。常用指令的有趣解释如下：
![dockerfile](https://user-images.githubusercontent.com/6218739/164161578-bf4e791f-8351-4f38-b353-986324ceabb9.png)

`FROM <image>`或`FROM <image>:<tag>`
`MAINTAINER <name>`指定镜像的作者信息，包含镜像的所有者和联系信息
`RUN <command>` 或 `RUN ["executable", "param1", "param2"]`指定当前构建过程中运行的命令，前者是shell模式，后者是exec模式。
`EXPOSE <port> [<port>…]` 指定运行该镜像的容器使用的端口
`CMD command param1 param2` 或 `CMD ["executable", "param1", "param2"]` 或 `CMD ["param1", "param2"]` 指定容器运行时的默认行为，即如果`docker run`时指定了命令，它会将这里的命令覆盖。第一种是shell模式，第二种是exec模式，第三种是作为EXTRYPOINT指令的默认参数。
`ENTRYPOINT command param1 param2` 或 `ENTRYPOINT ["exectable", "param1", "param2"]` 与CMD指令类似，但ENTRYPOINT指令不会被`docker run`中的命令所覆盖，只能使用`docker run --entrypoint`来覆盖。
`ADD <src> … <dst>` 或 `ADD ["<src>"… "<dst>"]` 复制文件，且ADD包含类似tar的解压功能，来源路径是构建路径中的相对路径，目标路径必须是镜像中的绝对路径，后者适用于文件路径中有空格的情况
`COPY <src> … <dst>` 或 `COPY ["<src>"… "<dst>"]` 单纯复制文件推荐使用COPY指令
`WORKDIR /path/to/workdir` 指定工作目录，一般为绝对路径，若为相对路径，则路径会传递。
`ENV <key> <value>`或`ENV <key>=<value>` 指定环境变量，构建过程中或容器运行中都有效
`USER user` 指定镜像以什么用户运行，若不指定，则默认使用root运行。
`ONBUILD [INSTRUCTION]` 为镜像添加触发器，当该镜像被其他镜像作为基础镜像时运行

Dockerfile构建过程（docker build会删除中间的容器，但不会删除中间的镜像，所以可以利用中间层镜像进行调试）：
（1）从基础镜像运行一个容器；
（2）执行一条指令，对容器做出修改；
（3）执行类似docker commit的操作，提交一个新的镜像层；
（4）再基于刚提交的镜像运行一个新容器；
（5）指定Dockerfile中的下一条指令，直至所有指令执行完毕。

`docker history` 查看镜像构建过程。

# 容器相关命令
启动容器：`docker run IMAGE [command] [args]`
启动交互式容器： `docker run -i -t IMAGE [command] [args]` （-i --interactive -t --tty），比如：
```sh
docker run -it ubuntu /bin/bash
```
退出容器就是`exit`。

自定义容器名：加上`--name`选项
查看容器：
（1）`docker ps [-a] [-l]` （列出当前正在运行的容器，加上`-a`就是列出所有）
（2）`docker inspect NameOfContainer`查看容器的元数据，包括主机配置、ip地址等。
重新启动已经停止的容器：`docker start [-i] NameOfContainer`
删除已经停止的容器：`docker rm` （加上`-f`可以强制删除正在运行的容器）

守护式容器：（1）能够长期运行；（2）没有交互式会话；（3）适合运行应用程序和服务
启动守护式容器的两种方式：
（1）将容器以交互式方式启动后，`Ctrl+P Ctrl+Q`即可进入守护模式，然后附加到正在运行的容器：`docker attach`
（2）`docker run -d`
查看容器中的日志：`docker logs [-f] [-t] [--tail] NameOfContainer` （-f --follow -t --timestamps）
查看容器中的进程：`docker top`
在运行中的容器内启动新进程：`decker exec [-d] [-i] [-t] 容器名 [COMMAND] [args]`，比如进入这个容器：
```sh
docker exec -it ubuntu /bin/bash
```

停止守护式容器：
`docker stop` 发送一个停止信号，等待停止； 
`docker kill` 直接杀死容器

设置容器的端口映射：
`docker run [-P] [-p]`
-P --publish-all=true | false，为容器暴露的所有端口进行映射
-p 指定特定的端口，可以单独指定容器端口、宿主机端口和容器端口、ip地址+容器端口、ip地址+宿主机端口和容器端口

对于端口映射这块，额外补充个知识点。因为我主机是Windows系统，然后通过Virtualbox虚拟了一个Linux Mint系统，而docker是放在Linux虚拟机中的，所以需要外面的Windows系统能访问到Linux系统，此时可以通过在Virtualbox中设置端口转发，来建立两者之间的联系，这样就是涉及了三个端口，一个是主机windows的端口，一个是linux的端口，一个是docker容器的端口，三者之间要建立好映射。
一篇很好的教程见：
[VirtualBox主机和虚拟机互相通信](https://www.cnblogs.com/Reyzal/p/7743747.html)

`-v 主机目录:容器内目录`：容器数据卷，实现容器数据的持久化和同步。

`docker port` 查看容器到宿主机的端口映射
`docker cp` 拷贝容器内的文件到主机

常用命令图谱：
![cmd](https://user-images.githubusercontent.com/6218739/164161697-fdf5da26-5f65-406d-9cc1-687c32e7b76a.png)


# Docker Compose
Compose 项目是 Docker 官方的开源项目，负责实现对 Docker 容器集群的快速编排。
在日常工作中，经常会碰到需要多个容器相互配合来完成某项任务的情况。例如要实现一个 Web 项目，除了 Web 服务容器本身，往往还需要再加上后端的数据库服务容器，甚至还包括负载均衡容器等。
Compose 恰好满足了这样的需求。它允许用户通过一个单独的`docker-compose.yml`模板文件（YAML 格式）来定义一组相关联的应用容器为一个项目（project）。
Compose 中有两个重要的概念：
（1）服务 (service)：一个应用的容器，实际上可以包括若干运行相同镜像的容器实例。
（2）项目 (project)：由一组关联的应用容器组成的一个完整业务单元，在 docker-compose.yml 文件中定义。
Compose 的默认管理对象是项目，通过子命令对项目中的一组容器进行便捷地生命周期管理。

## 安装
（1）下载安装包
```sh
sudo curl -L"https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)"-o/usr/local/bin/docker-compose
```
（2）对二进制包添加权限
```sh
sudo ln -s/usr/local/bin/docker-compose /usr/bin/docker-compose
```
测试是否安装成功：
```sh
docker-compose --version
```

