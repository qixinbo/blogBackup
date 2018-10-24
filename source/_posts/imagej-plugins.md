---
title: ImageJ的插件开发
tags: [ImageJ]
categories: programming
date: 2018-10-6
---

# 开篇说明
前面介绍过开发ImageJ的Python脚本的过程（[在这里](http://qixinbo.info/2018/09/15/imagej-python/)），这里介绍怎样开发ImageJ的插件。
插件相对于脚本来说，一方面，它的功能更加强大，可以认为是寄生在ImageJ上面的一个完备的小程序；另一方面，它使用Java开发，格式可以采用编译好的class文件，有利于保护代码不被泄露。
ImageJ由于历史原因，存在着ImageJ1和ImageJ2两个版本，且两者的API是迥然不同的，底层架构更是不同，导致两者的开发套路有很大差别。在写这篇tutorial时，深感两者在网上的文档资源相互交叉，论坛里的答案在不同版本间有时不适用。所以这里尝试从零开始一步步地记录ImageJ2的插件开发过程。
所基于的Fiji/ImageJ版本为：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvylcrof71j30bg095wff.jpg)
参考资源有：
- [Writing plugins](https://imagej.net/Writing_plugins)
- [Developing Fiji](https://imagej.net/Developing_Fiji)

ImageJ1的开发的参考资源有：
- [Developing Plugins for ImageJ 1.x](https://imagej.net/Developing_Plugins_for_ImageJ_1.x)
- [A simple Maven project implementing an ImageJ 1.x plugin](https://github.com/imagej/example-legacy-plugin/)

# 上手例子
下载ImageJ的一个简单事例（在[这里](https://github.com/imagej/example-imagej-command)），看看插件的源代码是什么样子以及怎样编译和安装。
## 编译
这个例子是一个Maven工程（Maven是Java的一种项目管理软件），即遵循了Maven的代码约定：有一个关键的配置文件pom.xml，以及源代码放在src文件夹下。pom.xml文件中指明了src目录下的java源码的依赖，如果直接使用传统的javac编译该源码，就会报“找不到ij包”的错误，所以这里要使用Maven进行编译（Maven的下载、安装和配置不在这里赘述）。
```shell
mvn package
```
上述命令就是使用Maven进行打包，结果就是生成target文件夹及下面的“GaussFiltering-0.1.0-SNAPSHOT.jar”这个jar包。
（Attention!!：这里生成的jar包中并没有下划线，这是ImageJ2允许的，它可以通过在源码的Plugin注解中指明menuPath即可在ImageJ对应菜单下显示，而ImageJ1则不允许，其明确要求jar包中必须有下划线，且它自定义在菜单中的显示是通过plugins.config配置文件指定，说明在[这里](https://imagej.net/Description_of_ImageJ%27s_plugin_architecture)）
## 安装
第二步是将该jar包“安装”到Fiji目录中，这里就是直接复制过去就行：
```shell
copy target/GaussFiltering-0.1.0-SNAPSHOT.jar /Path/to/Fiji.app/plugins/
```
然后启动Fiji（如果Fiji本来就运行着，则退出重启）。
## 运行
那么，这个插件怎么在Fiji中调用呢？有两种方法：
- 直接使用命令查找功能，即Ctrl+l（小写的字母L），然后输入该插件的名字Gauss Filtering，这是最快捷的一种方式。
- 第二种方式是可以在Plugins菜单下找到Gauss Filtering该插件。默认情形下插件是安装在Plugins菜单下，但也可以在java源码中自定义，即在Plugin注解中指定参数menuPath，前面已提到过）。

## 使用IDE编译
上面是使用Maven的命令行进行编译，实际上Maven已经与常用的Java集成开发环境IDE进行了集成，同时IDE还方便对代码的编辑，所以推荐使用IDE进行源码编写和编译。
常用的Java开发的IDE有Eclipse、NetBeans和IDEA等，以下是三种软件打开Maven工程的方法：
- Eclipse：File > Import > Existing Maven Projects
- NetBeans：File > Open Project
- IDEA：File > Open Project (Select pom.xml)

这里使用IDEA-2018-2进行操作，将pom.xml作为Project打开后，IDEA会根据pom中指定的依赖dependencies联网下载所依赖的jar包。然后点击左侧的Maven Projects，选择LifeCycle下面的Package，双击即可进行打包操作，如下：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvyiyh836ej30lb0g9my8.jpg)

# ImageJ插件开发的“Hello World”
学习任何一名编程语言，第一个例子肯定是向新世界打招呼——“Hello, World!”。基于ImageJ开发的插件也不例外。从[这里](https://github.com/imagej/tutorials/)下载ImageJ的入门例子，其中就包含了“Hello, World!”的书写方式。
## 编译运行
依然是用IDEA-2018-2来打开该项目的pom.xml文件，点击窗口左侧的Maven Projects，找到"Simple ImageJ Commands"这个项目，然后在LifeCycle下双击package命令进行打包，如图：
![](https://ws1.sinaimg.cn/large/0072Lfvtly1fvym64ybm2j31hc0suk0s.jpg)
生成jar包后，复制到Fiji的plugins文件夹下，重启Fiji，然后可以在Help菜单下发现"Hello, World!"命令。
## 代码解析
下面是一步步地分析HelloWorld的代码。
### 插件声明
```java
@Plugin(type = Command.class, headless = true, menuPath = "Help>Hello, World!")
```
技术上讲，ImageJ插件是一个名为Plugin的注解（ImageJ基于SciJava插件系统），这样注解的类可以在运行时自动查找到并索引。有很多种插件类型，HelloWorld用的是Command类型的插件，这种类型的插件是应用最广泛的：Command接收输入，然后产生输出。Command插件最常用的就是与菜单绑定，这里使用menuPath指定在哪个菜单下显示及其显示名称。
关于插件，多说两句：
（1）插件类型
除了上面的Command类型的插件，还有一些基于Service的插件。显式地指定插件type后，可以有效地找到某种类型的插件。比如：
```java
@Plugin(type=Service.class)
public class MyService implements Service { }

@Plugin(type=SpecialService.class)
public class SpecialService implements Service { }
```
那么问题来了，上述代码中，如果软件上下文想要的插件是Service，那么将会找到哪个插件？
答案是MyService和SpecialService都会返回，因为SpecialService是Service的Service的一个子类；而如果想要的是SpecialService插件，那么就只返回SpecialService类。
（2）插件优先级
当匹配的插件有很多时，插件会按类的priority的顺序来呈现，这个变量是double类型的数值，也可以使用已规定好的静态常量来表示：
```java
@Plugin(priority=Priority.HIGH_PRIORITY)
public class MyService implements Service { }

@Plugin(priority=224)
public class SpecialService implements Service { }
```
那么问题是，当寻找Service插件时，上面哪个插件会排在前面？
答案是SpecialService，因为从[SciJava-Common的源码](https://github.com/scijava/scijava-common/blob/scijava-common-2.47.0/src/main/java/org/scijava/Priority.java)上可以看出HIGH_PRIORITY的数值是100。
更好的一种方式是使用相对优先级，比如：
```java
@Plugin(priority=Priority.HIGH_PRIORITY+124)
public class SpecialService implements Service { }
```
### 该命令的输入输出，两者都是使用Parameter注解。
```java
public class HelloWorld implements Command {

	@Parameter(label = "What is your name?")
	private String name = "J. Doe";

	@Parameter(type = ItemIO.OUTPUT)
	private String greeting;
```
从command类中派生出helloWorld类。每个command类都有input和output，所以这里使用两个成员变量来承载，两者都要用Parameter来注解，所不同的是output需要Parameter的type明确指定为"ItemIO.OUTPUT"。
即，使用Parameter来注解的变量都是该命令的输入或输出，其中非Service类的变量会明确显示出来，而Service类的变量是用来隐式地调用。

### 命令的运行函数
```java
	@Override
	public void run() {
		greeting = "Hello, " + name + "!";
	}
```
上面的run函数是ImageJ的每个command的入口点，当用户点击菜单上的该命令时，先让用户输入input值，然后就会执行该函数。

### 命令的调试
```java
	public static void main(final String... args) {
		// Launch ImageJ as usual.
		final ImageJ ij = new ImageJ();
		ij.launch(args);

		// Launch our "Hello World" command right away.
		ij.command().run(HelloWorld.class, true);
	}

}
```
上面的代码是用来在IDE中调试所用。通过创建main函数，使得IDE创建一个ImageJ的上下文环境，然后调用该插件。
更一般的调用方式是：
```java
ij.command().run(MyPlugin.class, "inputImage", myImage)}
```

这就是最基本的ImageJ插件。
上面的代码中除了HelloWorld，还包含了更多ImageJ插件开发的入门例子，自行探索吧！
