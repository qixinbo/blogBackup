---
title: PostgreSQL简明教程
tags: [Web Crawler]
categories: data analysis
date: 2023-3-12
---

# 介绍
本篇介绍一下PostgreSQL的用法。先看一下该数据库的特点（以下来自ChatGPT）：
> 1.  开源免费：PostgreSQL是一款开源的关系型数据库管理系统，用户可以免费使用和修改。
> 2.  高度可扩展性：PostgreSQL支持水平和垂直扩展，可以满足不同规模应用的需求。
> 3.  ACID兼容：PostgreSQL保证了数据的原子性、一致性、隔离性和持久性，确保了数据的完整性和可靠性。  
> 4.  多版本并发控制：PostgreSQL采用了多版本并发控制（MVCC）技术，可以支持高并发读写操作。 
> 5.  支持复杂数据类型：PostgreSQL支持各种复杂数据类型，如数组、JSON、XML等，可以满足各种应用的需求。
> 6.  大数据处理能力：PostgreSQL支持大数据处理，可以处理数百万甚至数十亿条数据。  
> 7.  可扩展的存储引擎：PostgreSQL支持多种存储引擎，如B-tree、哈希表、GiST、SP-GiST、GIN、BRIN等，可以满足不同的应用场景。
> 8.  可编程性：PostgreSQL支持多种编程语言，如SQL、PL/pgSQL、PL/Python、PL/Perl等，可以方便地进行开发和扩展。

# 安装
postgreSQL的下载地址在[这里](https://www.postgresql.org/download/)。
对于`Windows`和`macOS`，直接使用`EDB`网站上的安装器最简单了，见[这里](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads)。
下载后双击安装，中间安装过程中配置一下默认自带的`postgres`数据库的密码即可。
pgSQL默认会创建：
1.  数据库：PostgreSQL会默认创建一个名为“postgres”的数据库，作为系统默认的数据库。
2.  用户：PostgreSQL会默认创建一个名为“postgres”的超级用户，该用户具有所有权限。
然后也可以通过pgAdmin或者命令行快速的创建自定义的用户/角色和数据库，并且将两者关联起来。
（在 PostgreSQL 中，角色和用户是相同的概念。在其他数据库管理系统中，可能会将角色和用户分开，角色用于管理权限和访问控制，而用户只用于身份验证和授权。但在 PostgreSQL 中，角色可以扮演这两个角色，既可以作为一个用户登录数据库，也可以作为一个授权角色授予其他用户权限。因此，pgsql中的角色和用户没有明显的区别。一个小区别是CREATE USER is now an alias for [CREATE ROLE](https://www.postgresql.org/docs/9.5/sql-createrole.html). The only difference is that when the command is spelled CREATE USER, LOGIN is assumed by default, whereas NOLOGIN is assumed when the command is spelled CREATE ROLE.）

对于`Linux`系统，我从chatGPT上也问了一下教程如下（谨慎服用）。
1.  打开终端。
2.  运行以下命令更新软件包列表：
```
sudo apt-get update
```
3.  运行以下命令安装PostgreSQL：
```
sudo apt-get install postgresql postgresql-contrib
```
4.  安装完成后，PostgreSQL服务将自动启动。您可以使用以下命令检查它是否正在运行：
```
sudo systemctl status postgresql
```
 如果PostgreSQL正在运行，您将看到“Active: active (running)”的消息。
5.  创建一个新的PostgreSQL用户。默认情况下，PostgreSQL使用名为“postgres”的超级用户。您可以使用以下命令创建新用户：
```
sudo -u postgres createuser --interactive
```
根据提示输入新用户的名称和是否为超级用户。
6.  创建一个新的数据库。您可以使用以下命令创建新数据库：
```
sudo -u postgres createdb dbname
```
将“dbname”替换为您要创建的数据库的名称。
7.  使用以下命令登录到PostgreSQL：
```
sudo -u postgres psql
```
您将看到一个以“postgres=#”开头的命令行提示符。
8.  在PostgreSQL中创建一个新用户并授予其对新数据库的访问权限。使用以下命令创建新用户：
```
CREATE USER username WITH PASSWORD 'password';
```
将“username”替换为您要创建的新用户的名称，“password”替换为该用户的密码。
9.  授予新用户对新数据库的访问权限。使用以下命令授予访问权限：
 ```
 GRANT ALL PRIVILEGES ON DATABASE dbname TO username;
 ```
 将“dbname”替换为您要授予访问权限的数据库的名称，“username”替换为您要授予访问权限的用户的名称。
10.  退出PostgreSQL命令行界面。使用以下命令退出：
 ```
 \q
 ```
11.  您现在已经成功安装了PostgreSQL并创建了新用户和数据库。

# 远程访问
1.    修改pg_hba.conf文件
在PostgreSQL的data目录下找到pg_hba.conf文件，打开并修改，添加以下一行：
```
host    all             all             0.0.0.0/0               md5
```
其中，0.0.0.0/0表示允许所有IP访问，md5表示使用密码验证方式。
2.  修改postgresql.conf文件
在PostgreSQL的data目录下找到postgresql.conf文件，打开并修改，将以下一行的注释去掉：
```
listen_addresses = '*'
```
该配置项表示监听所有IP地址。
3.  重启PostgreSQL服务
修改完以上两个文件后，需要重启PostgreSQL服务才能生效。
4.  防火墙设置
如果使用了防火墙，需要开放PostgreSQL的端口号，默认为5432。
5.  测试远程访问
使用psql工具连接远程PostgreSQL服务器，例如：
```
psql -h 192.168.1.100 -p 5432 -U postgres
```
其中，192.168.1.100为远程服务器IP地址，5432为PostgreSQL的默认端口号，postgres为连接的用户名。如果连接成功，则表示配置远程访问成功。

# 一个简单的例子
（以下也全部来自chatgpt）
假设我们有一个学生信息表格，包含以下字段：学生ID、姓名、性别、年龄、所在班级。
我们可以使用pgsql来创建这个表格，并插入一些样本数据。
首先，我们需要在pgsql中创建一个新的数据库，并连接到该数据库：
```
CREATE DATABASE mydatabase;
\c mydatabase;
```
接下来，我们可以使用以下命令创建学生信息表格：
```
CREATE TABLE students (
    id serial PRIMARY KEY,
    name varchar(50) NOT NULL,
    gender varchar(10),
    age int,
    class varchar(50)
);
```
这将创建一个名为“students”的表格，并定义了一个自增的主键“id”以及四个其他字段。
现在，我们可以向表格中插入一些样本数据：
```
INSERT INTO students (name, gender, age, class)
VALUES ('张三', '男', 18, '一班'),
       ('李四', '女', 19, '二班'),
       ('王五', '男', 20, '三班');
```
这将向学生信息表格中插入三个新的学生记录。
我们可以使用以下命令来查询学生信息表格中的所有记录：
```
SELECT * FROM students;
```
这将返回一个包含所有学生信息的表格。
我们也可以使用其他查询命令来获取特定的数据，例如：
```
-- 获取所有年龄大于18岁的学生
SELECT * FROM students WHERE age > 18;

-- 获取所有所在班级为“二班”的学生
SELECT * FROM students WHERE class = '二班';

-- 获取所有男生的姓名和年龄
SELECT name, age FROM students WHERE gender = '男';
```
这些命令将返回符合特定条件的学生信息。
以上就是一个简单的pgsql用法示例，展示了如何创建表格、插入数据和查询数据。

# 创建外键
在 PostgreSQL 中，创建外键需要以下步骤：
1.  创建主表和从表。
```
CREATE TABLE main_table (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50)
);

CREATE TABLE sub_table (
    id SERIAL PRIMARY KEY,
    main_id INTEGER,
    sub_name VARCHAR(50),
    FOREIGN KEY (main_id) REFERENCES main_table(id)
);
```

2.  在从表中创建外键。
```
ALTER TABLE sub_table ADD CONSTRAINT sub_table_main_id_fkey FOREIGN KEY (main_id) REFERENCES main_table(id);
```
3.  测试外键是否生效。
```
INSERT INTO main_table (name) VALUES ('Main 1');
INSERT INTO sub_table (main_id, sub_name) VALUES (1, 'Sub 1');
```
如果外键设置正确，则在从表中插入的数据必须与主表中的数据匹配。如果外键设置不正确，则会出现错误。

# python接口
在 Python 中，可以通过 psycopg2 模块来连接 PostgreSQL 数据库。以下是连接 PostgreSQL 数据库的基本步骤：
1.  安装 psycopg2 模块
可以使用 pip 命令来安装 psycopg2 模块：
```
pip install psycopg2
```
2.  导入 psycopg2 模块
```
import psycopg2
```
3.  连接 PostgreSQL 数据库
```
conn = psycopg2.connect(
    host="localhost",
    database="mydatabase",
    user="myusername",
    password="mypassword"
)
```
其中，host 是数据库主机名或 IP 地址，database 是数据库名称，user 和 password 是连接数据库的用户名和密码。
4.  创建游标
```
cur = conn.cursor()
```
5.  执行 SQL 语句
```
cur.execute("SELECT * FROM mytable")
```
6.  获取查询结果
```
rows = cur.fetchall()
for row in rows:
    print(row)
```
7.  关闭游标和连接
```
cur.close()
conn.close()
```
完整的示例代码：
```
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="mydatabase",
    user="myusername",
    password="mypassword"
)

cur = conn.cursor()
cur.execute("SELECT * FROM mytable")
rows = cur.fetchall()
for row in rows:
    print(row)

cur.close()
conn.close()
```
以上是连接 PostgreSQL 数据库的基本步骤，具体操作可以根据需求进行调整。
