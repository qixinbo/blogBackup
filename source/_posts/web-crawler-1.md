---
title: 52讲轻松搞定网络爬虫笔记1
tags: [Web Crawler]
categories: data analysis
date: 2023-1-8
---

# 资料
[52讲轻松搞定网络爬虫](https://kaiwu.lagou.com/course/courseInfo.htm?courseId=46#/sale)

# HTTP基本原理
## URI 和 URL

首先，我们来了解一下 URI 和 URL，URI 的全称为 Uniform Resource Identifier，即统一资源标志符，URL 的全称为 Universal Resource Locator，即统一资源定位符。
举例来说，<a href="https://github.com/favicon.ico">https://github.com/favicon.ico</a>，它是一个 URL，也是一个 URI。即有这样的一个图标资源，我们用 URL/URI 来唯一指定了它的访问方式，这其中包括了访问协议 HTTPS、访问路径（即根目录）和资源名称 favicon.ico。通过这样一个链接，我们便可以从互联网上找到这个资源，这就是 URL/URI。
URL 是 URI 的子集，也就是说每个 URL 都是 URI，但不是每个 URI 都是 URL。那么，什么样的 URI 不是 URL 呢？URI 还包括一个子类叫作 URN，它的全称为 Universal Resource Name，即统一资源名称。
URN 只命名资源而不指定如何定位资源，比如 urn:isbn:0451450523 指定了一本书的 ISBN，可以唯一标识这本书，但是没有指定到哪里定位这本书，这就是 URN。URL、URN 和 URI 的关系可以用图表示。
<img src="https://s0.lgstatic.com/i/image3/M01/6B/32/CgpOIF5XRxOAT9vMAAAwHI1kHMc253.jpg" alt="">
但是在目前的互联网，URN 的使用非常少，几乎所有的 URI 都是 URL，所以一般的网页链接我们可以称之为 URL，也可以称之为 URI。
## 超文本

接下来，我们再了解一个概念 —— 超文本，其英文名称叫作 Hypertext，我们在浏览器里看到的网页就是超文本解析而成的，其网页源代码是一系列 HTML 代码，里面包含了一系列标签，比如 img 显示图片，p 指定显示段落等。浏览器解析这些标签后，便形成了我们平常看到的网页，而网页的源代码 HTML 就可以称作超文本。
例如，我们在 Chrome 浏览器里面打开任意一个页面，如淘宝首页，右击任一地方并选择 “检查” 项（或者直接按快捷键 F12），即可打开浏览器的开发者工具，这时在 Elements 选项卡即可看到当前网页的源代码，这些源代码都是超文本，如图所示。
<img src="https://s0.lgstatic.com/i/image3/M01/6B/33/Cgq2xl5XRxSAciN7AAvjJD9Z3yA461.png" alt="">
## HTTP 和 HTTPS

在淘宝的首页&nbsp;<a href="https://www.taobao.com/">https://www.taobao.com/</a>中，URL 的开头会有 http 或 https，这个就是访问资源需要的协议类型，有时我们还会看到 ftp、sftp、smb 开头的 URL，那么这里的 ftp、sftp、smb 都是指的协议类型。在爬虫中，我们抓取的页面通常就是 http 或 https 协议的，我们在这里首先来了解一下这两个协议的含义。
HTTP 的全称是 Hyper Text Transfer Protocol，中文名叫作超文本传输协议，HTTP 协议是用于从网络传输超文本数据到本地浏览器的传送协议，它能保证高效而准确地传送超文本文档。HTTP 由万维网协会（World Wide Web Consortium）和 Internet 工作小组 IETF（Internet Engineering Task Force）共同合作制定的规范，目前广泛使用的是 HTTP 1.1 版本。
HTTPS 的全称是 Hyper Text Transfer Protocol over Secure Socket Layer，是以安全为目标的 HTTP 通道，简单讲是 HTTP 的安全版，即 HTTP 下加入 SSL 层，简称为 HTTPS。
HTTPS 的安全基础是 SSL，因此通过它传输的内容都是经过 SSL 加密的，它的主要作用可以分为两种：
<ul>
<li>建立一个信息安全通道，来保证数据传输的安全。</li>
<li>确认网站的真实性，凡是使用了 HTTPS 的网站，都可以通过点击浏览器地址栏的锁头标志来查看网站认证之后的真实信息，也可以通过 CA 机构颁发的安全签章来查询。</li>
</ul>
现在越来越多的网站和 App 都已经向 HTTPS 方向发展。例如：
<ul>
<li>苹果公司强制所有 iOS App 在 2017 年 1 月 1 日 前全部改为使用 HTTPS 加密，否则 App 就无法在应用商店上架。</li>
<li>谷歌从 2017 年 1 月推出的 Chrome 56 开始，对未进行 HTTPS 加密的网址链接亮出风险提示，即在地址栏的显著位置提醒用户 “此网页不安全”。</li>
<li>腾讯微信小程序的官方需求文档要求后台使用 HTTPS 请求进行网络通信，不满足条件的域名和协议无法请求。</li>
</ul>
因此，HTTPS 已经是大势所趋。
## HTTP 请求过程

我们在浏览器中输入一个 URL，回车之后便可以在浏览器中观察到页面内容。实际上，这个过程是浏览器向网站所在的服务器发送了一个请求，网站服务器接收到这个请求后进行处理和解析，然后返回对应的响应，接着传回给浏览器。响应里包含了页面的源代码等内容，浏览器再对其进行解析，便将网页呈现了出来，传输模型如图所示。
<img src="https://s0.lgstatic.com/i/image3/M01/6B/32/CgpOIF5XRxSANG0_AACovJPtMYQ497.jpg" alt="">
此处客户端即代表我们自己的 PC 或手机浏览器，服务器即要访问的网站所在的服务器。
为了更直观地说明这个过程，这里用 Chrome 浏览器的开发者模式下的 Network 监听组件来做下演示，它可以显示访问当前请求网页时发生的所有网络请求和响应。
打开 Chrome 浏览器，右击并选择 “检查” 项，即可打开浏览器的开发者工具。这里访问百度&nbsp;<a href="http://www.baidu.com/">http://www.baidu.com/</a>，输入该 URL 后回车，观察这个过程中发生了怎样的网络请求。可以看到，在 Network 页面下方出现了一个个的条目，其中一个条目就代表一次发送请求和接收响应的过程，如图所示。
<img src="https://s0.lgstatic.com/i/image3/M01/6B/33/Cgq2xl5XRxSANsceAAS_kKedSt0173.png" alt="">
我们先观察第一个网络请求，即&nbsp;<a href="www.baidu.com">www.baidu.com</a>，其中各列的含义如下。
<ul>
<li>第一列 Name：请求的名称，一般会将 URL 的最后一部分内容当作名称。</li>
<li>第二列 Status：响应的状态码，这里显示为 200，代表响应是正常的。通过状态码，我们可以判断发送了请求之后是否得到了正常的响应。</li>
<li>第三列 Type：请求的文档类型。这里为 document，代表我们这次请求的是一个 HTML 文档，内容就是一些 HTML 代码。</li>
<li>第四列 Initiator：请求源。用来标记请求是由哪个对象或进程发起的。</li>
<li>第五列 Size：从服务器下载的文件和请求的资源大小。如果是从缓存中取得的资源，则该列会显示 from cache。</li>
<li>第六列 Time：发起请求到获取响应所用的总时间。</li>
<li>第七列 Waterfall：网络请求的可视化瀑布流。</li>
</ul>
我们点击这个条目即可看到其更详细的信息，如图所示。
<img src="https://s0.lgstatic.com/i/image3/M01/6B/32/CgpOIF5XRxSAXezkAAJwMNsqow0849.jpg" alt="">
首先是 General 部分，Request URL 为请求的 URL，Request Method 为请求的方法，Status Code 为响应状态码，Remote Address 为远程服务器的地址和端口，Referrer Policy 为 Referrer 判别策略。
再继续往下，可以看到，有 Response Headers 和 Request Headers，这分别代表响应头和请求头。请求头里带有许多请求信息，例如浏览器标识、Cookies、Host 等信息，这是请求的一部分，服务器会根据请求头内的信息判断请求是否合法，进而作出对应的响应。图中看到的 Response Headers 就是响应的一部分，例如其中包含了服务器的类型、文档类型、日期等信息，浏览器接受到响应后，会解析响应内容，进而呈现网页内容。
下面我们分别来介绍一下请求和响应都包含哪些内容。
## 请求

请求，由客户端向服务端发出，可以分为 4 部分内容：请求方法（Request Method）、请求的网址（Request URL）、请求头（Request Headers）、请求体（Request Body）。
<h4>请求方法</h4>
常见的请求方法有两种：GET 和 POST。
在浏览器中直接输入 URL 并回车，这便发起了一个 GET 请求，请求的参数会直接包含到 URL 里。例如，在百度中搜索 Python，这就是一个 GET 请求，链接为&nbsp;<a href="https://www.baidu.com/s?wd=Python">https://www.baidu.com/s?wd=Python</a>，其中 URL 中包含了请求的参数信息，这里参数 wd 表示要搜寻的关键字。POST 请求大多在表单提交时发起。比如，对于一个登录表单，输入用户名和密码后，点击 “登录” 按钮，这通常会发起一个 POST 请求，其数据通常以表单的形式传输，而不会体现在 URL 中。
GET 和 POST 请求方法有如下区别。
<ul>
<li>GET 请求中的参数包含在 URL 里面，数据可以在 URL 中看到，而 POST 请求的 URL 不会包含这些数据，数据都是通过表单形式传输的，会包含在请求体中。</li>
<li>GET 请求提交的数据最多只有 1024 字节，而 POST 请求没有限制。</li>
</ul>
一般来说，登录时，需要提交用户名和密码，其中包含了敏感信息，使用 GET 方式请求的话，密码就会暴露在 URL 里面，造成密码泄露，所以这里最好以 POST 方式发送。上传文件时，由于文件内容比较大，也会选用 POST 方式。
我们平常遇到的绝大部分请求都是 GET 或 POST 请求，另外还有一些请求方法，如 HEAD、PUT、DELETE、OPTIONS、CONNECT、TRACE 等，我们简单将其总结为下表。
<img src="https://s0.lgstatic.com/i/image3/M01/6B/36/Cgq2xl5XTMeAFYgXAADGW57EU8s068.png" alt="">
请求的网址本表参考：<a href="http://www.runoob.com/http/http-methods.html">http://www.runoob.com/http/http-methods.html</a>。
请求的网址，即统一资源定位符 URL，它可以唯一确定我们想请求的资源。
<h4>请求头</h4>
请求头，用来说明服务器要使用的附加信息，比较重要的信息有 Cookie、Referer、User-Agent 等。下面简要说明一些常用的头信息。
<ul>
<li>Accept：请求报头域，用于指定客户端可接受哪些类型的信息。</li>
<li>Accept-Language：指定客户端可接受的语言类型。</li>
<li>Accept-Encoding：指定客户端可接受的内容编码。</li>
<li>Host：用于指定请求资源的主机 IP 和端口号，其内容为请求 URL 的原始服务器或网关的位置。从 HTTP 1.1 版本开始，请求必须包含此内容。</li>
<li>Cookie：也常用复数形式 Cookies，这是网站为了辨别用户进行会话跟踪而存储在用户本地的数据。它的主要功能是维持当前访问会话。例如，我们输入用户名和密码成功登录某个网站后，服务器会用会话保存登录状态信息，后面我们每次刷新或请求该站点的其他页面时，会发现都是登录状态，这就是 Cookies 的功劳。Cookies 里有信息标识了我们所对应的服务器的会话，每次浏览器在请求该站点的页面时，都会在请求头中加上 Cookies 并将其发送给服务器，服务器通过 Cookies 识别出是我们自己，并且查出当前状态是登录状态，所以返回结果就是登录之后才能看到的网页内容。</li>
<li>Referer：此内容用来标识这个请求是从哪个页面发过来的，服务器可以拿到这一信息并做相应的处理，如做来源统计、防盗链处理等。</li>
<li>User-Agent：简称 UA，它是一个特殊的字符串头，可以使服务器识别客户使用的操作系统及版本、浏览器及版本等信息。在做爬虫时加上此信息，可以伪装为浏览器；如果不加，很可能会被识别出为爬虫。</li>
<li>Content-Type：也叫互联网媒体类型（Internet Media Type）或者 MIME 类型，在 HTTP 协议消息头中，它用来表示具体请求中的媒体类型信息。例如，text/html 代表 HTML 格式，image/gif 代表 GIF 图片，application/json 代表 JSON 类型，更多对应关系可以查看此对照表：<a href="http://tool.oschina.net/commons">http://tool.oschina.net/commons</a>。</li>
</ul>
因此，请求头是请求的重要组成部分，在写爬虫时，大部分情况下都需要设定请求头。
<h4>请求体</h4>
请求体一般承载的内容是 POST 请求中的表单数据，而对于 GET 请求，请求体则为空。
例如，这里我登录 GitHub 时捕获到的请求和响应如图所示。
<img src="https://s0.lgstatic.com/i/image3/M01/6B/33/Cgq2xl5XRxWADwdYAARQqRIV11o167.jpg" alt="">
登录之前，我们填写了用户名和密码信息，提交时这些内容就会以表单数据的形式提交给服务器，此时需要注意 Request Headers 中指定 Content-Type 为 application/x-www-form-urlencoded。只有设置 Content-Type 为 application/x-www-form-urlencoded，才会以表单数据的形式提交。另外，我们也可以将 Content-Type 设置为 application/json 来提交 JSON 数据，或者设置为 multipart/form-data 来上传文件。
表格中列出了 Content-Type 和 POST 提交数据方式的关系。
<img src="https://s0.lgstatic.com/i/image3/M01/6B/35/CgpOIF5XTOWAPlCWAABOQqmuH1c663.png" alt="">
在爬虫中，如果要构造 POST 请求，需要使用正确的 Content-Type，并了解各种请求库的各个参数设置时使用的是哪种 Content-Type，不然可能会导致 POST 提交后无法正常响应。
## 响应

响应，由服务端返回给客户端，可以分为三部分：响应状态码（Response Status Code）、响应头（Response Headers）和响应体（Response Body）。
<h4>响应状态码</h4>
响应状态码表示服务器的响应状态，如 200 代表服务器正常响应，404 代表页面未找到，500 代表服务器内部发生错误。在爬虫中，我们可以根据状态码来判断服务器响应状态，如状态码为 200，则证明成功返回数据，再进行进一步的处理，否则直接忽略。下表列出了常见的错误代码及错误原因。
<img src="https://s0.lgstatic.com/i/image3/M01/6B/36/Cgq2xl5XTQSAfWsUAAa-jFIsTTw064.png" alt="">
响应头包含了服务器对请求的应答信息，如 Content-Type、Server、Set-Cookie 等。下面简要说明一些常用的响应头信息。
<ul>
<li>Date：标识响应产生的时间。</li>
<li>Last-Modified：指定资源的最后修改时间。</li>
<li>Content-Encoding：指定响应内容的编码。</li>
<li>Server：包含服务器的信息，比如名称、版本号等。</li>
<li>Content-Type：文档类型，指定返回的数据类型是什么，如 text/html 代表返回 HTML 文档，application/x-javascript 则代表返回 JavaScript 文件，image/jpeg 则代表返回图片。</li>
<li>Set-Cookie：设置 Cookies。响应头中的 Set-Cookie 告诉浏览器需要将此内容放在 Cookies 中，下次请求携带 Cookies 请求。</li>
<li>Expires：指定响应的过期时间，可以使代理服务器或浏览器将加载的内容更新到缓存中。如果再次访问时，就可以直接从缓存中加载，降低服务器负载，缩短加载时间。</li>
</ul>
<h4>响应体</h4>
最重要的当属响应体的内容了。响应的正文数据都在响应体中，比如请求网页时，它的响应体就是网页的 HTML 代码；请求一张图片时，它的响应体就是图片的二进制数据。我们做爬虫请求网页后，要解析的内容就是响应体，如图所示。
<img src="https://s0.lgstatic.com/i/image3/M01/6B/32/CgpOIF5XRxWASx82AAJGjt0ry0I419.jpg" alt="">
在浏览器开发者工具中点击 Preview，就可以看到网页的源代码，也就是响应体的内容，它是解析的目标。
在做爬虫时，我们主要通过响应体得到网页的源代码、JSON 数据等，然后从中做相应内容的提取。

# Web网页基础
当我们用浏览器访问网站时，页面各不相同，那么你有没有想过它为何会呈现成这个样子呢？本部分讲解网页的基本组成、结构和节点等内容。
## 网页的组成

首先，我们来了解网页的基本组成，网页可以分为三大部分：HTML、CSS 和 JavaScript。
如果把网页比作一个人的话，HTML 相当于骨架，JavaScript 相当于肌肉，CSS 相当于皮肤，三者结合起来才能形成一个完整的网页。下面我们来分别介绍一下这三部分的功能。
<h4>HTML</h4>
HTML 是用来描述网页的一种语言，其全称叫作 Hyper Text Markup Language，即超文本标记语言。
我们浏览的网页包括文字、按钮、图片和视频等各种复杂的元素，其基础架构就是 HTML。不同类型的元素通过不同类型的标签来表示，如图片用 img 标签表示，视频用 video 标签表示，段落用 p 标签表示，它们之间的布局又常通过布局标签 div 嵌套组合而成，各种标签通过不同的排列和嵌套就可以形成网页的框架。
我们在 Chrome 浏览器中打开百度，右击并选择 “检查” 项（或按 F12 键），打开开发者模式，这时在 Elements 选项卡中即可看到网页的源代码，如图所示。
<img src="https://s0.lgstatic.com/i/image3/M01/6B/9D/CgpOIF5YedSAA-wbAAcAZryA2fc271.png" alt="">
这就是 HTML，整个网页就是由各种标签嵌套组合而成的。这些标签定义的节点元素相互嵌套和组合形成了复杂的层次关系，就形成了网页的架构。
<h4>CSS</h4>
虽然 HTML 定义了网页的结构，但是只有 HTML 页面的布局并不美观，可能只是简单的节点元素的排列，为了让网页看起来更好看一些，这里就需要借助 CSS 了。
CSS，全称叫作 Cascading Style Sheets，即层叠样式表。“层叠” 是指当在 HTML 中引用了数个样式文件，并且样式发生冲突时，浏览器能依据层叠顺序处理。“样式” 指网页中文字大小、颜色、元素间距、排列等格式。
CSS 是目前唯一的网页页面排版样式标准，有了它的帮助，页面才会变得更为美观。
图的右侧即为 CSS，例如：
<pre><code data-language="python" class="lang-python"><span class="hljs-comment">#head_wrapper.s-ps-islite&nbsp;.s-p-top&nbsp;{</span>

&nbsp;&nbsp;&nbsp;position:&nbsp;absolute;

&nbsp;&nbsp;&nbsp;bottom:&nbsp;<span class="hljs-number">40</span>px;

&nbsp;&nbsp;&nbsp;width:&nbsp;<span class="hljs-number">100</span>%;

&nbsp;&nbsp;&nbsp;height:&nbsp;<span class="hljs-number">181</span>px;
</code></pre>
这就是一个 CSS 样式。大括号前面是一个 CSS 选择器。此选择器的作用是首先选中 id 为 head_wrapper 且 class 为 s-ps-islite 的节点，然后再选中其内部的 class 为 s-p-top 的节点。
大括号内部写的就是一条条样式规则，例如 position 指定了这个元素的布局方式为绝对布局，bottom 指定元素的下边距为 40 像素，width 指定了宽度为 100% 占满父元素，height 则指定了元素的高度。
也就是说，我们将位置、宽度、高度等样式配置统一写成这样的形式，然后用大括号括起来，接着在开头再加上 CSS 选择器，这就代表这个样式对 CSS 选择器选中的元素生效，元素就会根据此样式来展示了。
在网页中，一般会统一定义整个网页的样式规则，并写入 CSS 文件中（其后缀为 css）。在 HTML 中，只需要用 link 标签即可引入写好的 CSS 文件，这样整个页面就会变得美观、优雅。
<h4>JavaScript</h4>
JavaScript，简称 JS，是一种脚本语言。HTML 和 CSS 配合使用，提供给用户的只是一种静态信息，缺乏交互性。我们在网页里可能会看到一些交互和动画效果，如下载进度条、提示框、轮播图等，这通常就是 JavaScript 的功劳。它的出现使得用户与信息之间不只是一种浏览与显示的关系，而是实现了一种实时、动态、交互的页面功能。
JavaScript 通常也是以单独的文件形式加载的，后缀为 js，在 HTML 中通过 script 标签即可引入，例如：
<pre><code data-language="html" class="lang-html"><span class="hljs-tag">&lt;<span class="hljs-name">script</span> <span class="hljs-attr">src</span>=<span class="hljs-string">"jquery-2.1.0.js"</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">script</span>&gt;</span>
</code></pre>
综上所述，HTML 定义了网页的内容和结构，CSS 描述了网页的布局，JavaScript 定义了网页的行为。
## 网页的结构

了解了网页的基本组成，我们再用一个例子来感受下 HTML 的基本结构。新建一个文本文件，名称可以自取，后缀为 html，内容如下：
<pre><code data-language="html" class="lang-html"><span class="hljs-meta">&lt;!DOCTYPE <span class="hljs-meta-keyword">html</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">html</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">head</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">meta</span> <span class="hljs-attr">charset</span>=<span class="hljs-string">"UTF-8"</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">title</span>&gt;</span>This is a Demo<span class="hljs-tag">&lt;/<span class="hljs-name">title</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">head</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">body</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">id</span>=<span class="hljs-string">"container"</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"wrapper"</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">h2</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"title"</span>&gt;</span>Hello World<span class="hljs-tag">&lt;/<span class="hljs-name">h2</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">p</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"text"</span>&gt;</span>Hello, this is a paragraph.<span class="hljs-tag">&lt;/<span class="hljs-name">p</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">body</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">html</span>&gt;</span>
</code></pre>
这就是一个最简单的 HTML 实例。开头用 DOCTYPE 定义了文档类型，其次最外层是 html 标签，最后还有对应的结束标签来表示闭合，其内部是 head 标签和 body 标签，分别代表网页头和网页体，它们也需要结束标签。
head 标签内定义了一些页面的配置和引用，如：&lt;meta charset="UTF-8"&gt;<span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;">，它指定了网页的编码为 UTF-8。title 标签则定义了网页的标题，会显示在网页的选项卡中，不会显示在正文中。body 标签内则是在网页正文中显示的内容。</span></span>
div 标签定义了网页中的区块，它的 id 是 container，这是一个非常常用的属性，且 id 的内容在网页中是唯一的，我们可以通过它来获取这个区块。然后在此区块内又有一个 div 标签，它的 class 为 wrapper，这也是一个非常常用的属性，经常与 CSS 配合使用来设定样式。
然后此区块内部又有一个 h2 标签，这代表一个二级标题。另外，还有一个 p 标签，这代表一个段落。在这两者中直接写入相应的内容即可在网页中呈现出来，它们也有各自的 class 属性。
将代码保存后，在浏览器中打开该文件，可以看到如图所示的内容。
<img src="https://s0.lgstatic.com/i/image3/M01/6B/9E/Cgq2xl5YedWAYhjqAAB1ma8qoXU278.png" alt="">
可以看到，在选项卡上显示了 This is a Demo 字样，这是我们在 head 中的 title 里定义的文字。而网页正文是 body 标签内部定义的各个元素生成的，可以看到这里显示了二级标题和段落。
这个实例便是网页的一般结构。一个网页的标准形式是 html 标签内嵌套 head 和 body 标签，head 内定义网页的配置和引用，body 内定义网页的正文。
## 节点树及节点间的关系

在 HTML 中，所有标签定义的内容都是节点，它们构成了一个 HTML DOM 树。
我们先看下什么是 DOM。DOM 是 W3C（万维网联盟）的标准，其英文全称 Document Object Model，即文档对象模型。它定义了访问 HTML 和 XML 文档的标准：
<blockquote>
W3C 文档对象模型（DOM）是中立于平台和语言的接口，它允许程序和脚本动态地访问和更新文档的内容、结构和样式。
</blockquote>
W3C DOM 标准被分为 3 个不同的部分：
<ul>
<li>核心 DOM - 针对任何结构化文档的标准模型</li>
<li>XML DOM - 针对 XML 文档的标准模型</li>
<li>HTML DOM - 针对 HTML 文档的标准模型</li>
</ul>
根据 W3C 的 HTML DOM 标准，HTML 文档中的所有内容都是节点：
<ul>
<li>整个文档是一个文档节点</li>
<li>每个 HTML 元素是元素节点</li>
<li>HTML 元素内的文本是文本节点</li>
<li>每个 HTML 属性是属性节点</li>
<li>注释是注释节点</li>
</ul>
HTML DOM 将 HTML 文档视作树结构，这种结构被称为节点树，如图所示。
<img src="https://s0.lgstatic.com/i/image3/M01/6B/9D/CgpOIF5YedWAYM8VAAA48tvLktY497.jpg" alt="">
通过 HTML DOM，树中的所有节点均可通过 JavaScript 访问，所有 HTML 节点元素均可被修改，也可以被创建或删除。
节点树中的节点彼此拥有层级关系。我们常用父（parent）、子（child）和兄弟（sibling）等术语描述这些关系。父节点拥有子节点，同级的子节点被称为兄弟节点。
在节点树中，顶端节点称为根（root）。除了根节点之外，每个节点都有父节点，同时可拥有任意数量的子节点或兄弟节点。图中展示了节点树以及节点之间的关系。
<img src="https://s0.lgstatic.com/i/image3/M01/6B/9E/Cgq2xl5YedWAMswxAAAoJRVTTes621.jpg" alt="">
本段参考 W3SCHOOL，链接：<a href="http://www.w3school.com.cn/htmldom/dom_nodes.asp">http://www.w3school.com.cn/htmldom/dom_nodes.asp</a>。
## 选择器

我们知道网页由一个个节点组成，CSS 选择器会根据不同的节点设置不同的样式规则，那么怎样来定位节点呢？
在 CSS 中，我们使用 CSS 选择器来定位节点。例如，上例中 div 节点的 id 为 container，那么就可以表示为 #container，其中 # 开头代表选择 id，其后紧跟 id 的名称。
另外，如果我们想选择 class 为 wrapper 的节点，便可以使用 .wrapper，这里以点“.”开头代表选择 class，其后紧跟 class 的名称。另外，还有一种选择方式，那就是根据标签名筛选，例如想选择二级标题，直接用 h2 即可。这是最常用的 3 种表示，分别是根据 id、class、标签名筛选，请牢记它们的写法。
另外，CSS 选择器还支持嵌套选择，各个选择器之间加上空格分隔开便可以代表嵌套关系，如 #container .wrapper p 则代表先选择 id 为 container 的节点，然后选中其内部的 class 为 wrapper 的节点，然后再进一步选中其内部的 p 节点。
另外，如果不加空格，则代表并列关系，如 div#container .wrapper p.text 代表先选择 id 为 container 的 div 节点，然后选中其内部的 class 为 wrapper 的节点，再进一步选中其内部的 class 为 text 的 p 节点。这就是 CSS 选择器，其筛选功能还是非常强大的。
另外，CSS 选择器还有一些其他语法规则，具体如表所示。因为表中的内容非常的多，我就不在一一介绍，课下你可以参考文字内容详细理解掌握这部分知识。
<table>
<thead>
<tr>
<th>选　择　器</th>
<th>例　　子</th>
<th>例子描述</th>
</tr>
</thead>
<tbody>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">.class</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">.intro</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择 class="intro" 的所有节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">#id</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">#firstname</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择 id="firstname" 的所有节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">*</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">*</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择所有节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">element</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">p</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择所有 p 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">element,element</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">div,p</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择所有 div 节点和所有 p 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">element element</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">div p</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择 div 节点内部的所有 p 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">element&gt;element</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">div&gt;p</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择父节点为 div 节点的所有 p 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">element+element</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">div+p</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择紧接在 div 节点之后的所有 p 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">[attribute]</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">[target]</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择带有 target 属性的所有节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">[attribute=value]</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">[target=blank]</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择 target="blank" 的所有节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">[attribute~=value]</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">[title~=flower]</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择 title 属性包含单词 flower 的所有节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:link</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">a:link</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择所有未被访问的链接</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:visited</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">a:visited</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择所有已被访问的链接</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:active</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">a:active</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择活动链接</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:hover</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">a:hover</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择鼠标指针位于其上的链接</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:focus</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">input:focus</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择获得焦点的 input 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:first-letter</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">p:first-letter</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择每个 p 节点的首字母</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:first-line</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">p:first-line</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择每个 p 节点的首行</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:first-child</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">p:first-child</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择属于父节点的第一个子节点的所有 p 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:before</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">p:before</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">在每个 p 节点的内容之前插入内容</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:after</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">p:after</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">在每个 p 节点的内容之后插入内容</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:lang(language)</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">p:lang</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择带有以 it 开头的 lang 属性值的所有 p 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">element1~element2</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">p~ul</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择前面有 p 节点的所有 ul 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">[attribute^=value]</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">a[src^="https"]</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择其 src 属性值以 https 开头的所有 a 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">[attribute$=value]</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">a[src$=".pdf"]</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择其 src 属性以.pdf 结尾的所有 a 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">[attribute*=value]</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">a[src*="abc"]</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择其 src 属性中包含 abc 子串的所有 a 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:first-of-type</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">p:first-of-type</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择属于其父节点的首个 p 节点的所有 p 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:last-of-type</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">p:last-of-type</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择属于其父节点的最后 p 节点的所有 p 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:only-of-type</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">p:only-of-type</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择属于其父节点唯一的 p 节点的所有 p 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:only-child</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">p:only-child</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择属于其父节点的唯一子节点的所有 p 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:nth-child(n)</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">p:nth-child</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择属于其父节点的第二个子节点的所有 p 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:nth-last-child(n)</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">p:nth-last-child</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">同上，从最后一个子节点开始计数</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:nth-of-type(n)</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">p:nth-of-type</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择属于其父节点第二个 p 节点的所有 p 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:nth-last-of-type(n)</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">p:nth-last-of-type</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">同上，但是从最后一个子节点开始计数</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:last-child</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">p:last-child</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择属于其父节点最后一个子节点的所有 p 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:root</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:root</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择文档的根节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:empty</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">p:empty</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择没有子节点的所有 p 节点（包括文本节点）</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:target</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">#news:target</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择当前活动的 #news 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:enabled</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">input:enabled</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择每个启用的 input 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:disabled</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">input:disabled</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择每个禁用的 input 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:checked</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">input:checked</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择每个被选中的 input 节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:not(selector)</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">:not</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择非 p 节点的所有节点</span></span></span></td>
</tr>
<tr>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">::selection</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">::selection</span></span></span></td>
<td><span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">选择被用户选取的节点部分</span></span></span></td>
</tr>
</tbody>
</table>
<span class="colour" style="color:rgb(63, 63, 63)"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px"></span></span></span>
另外，还有一种比较常用的选择器是 XPath，这种选择方式后面会详细介绍。
# 爬虫基本原理
我们可以把互联网比作一张大网，而爬虫（即网络爬虫）便是在网上爬行的蜘蛛。如果把网的节点比作一个个网页，爬虫爬到这就相当于访问了该页面，获取了其信息。可以把节点间的连线比作网页与网页之间的链接关系，这样蜘蛛通过一个节点后，可以顺着节点连线继续爬行到达下一个节点，即通过一个网页继续获取后续的网页，这样整个网的节点便可以被蜘蛛全部爬行到，网站的数据就可以被抓取下来了。
## 爬虫概述

简单来说，爬虫就是获取网页并提取和保存信息的自动化程序，下面概要介绍一下。
<h4>获取网页</h4>
爬虫首先要做的工作就是获取网页，这里就是获取网页的源代码。
源代码里包含了网页的部分有用信息，所以只要把源代码获取下来，就可以从中提取想要的信息了。
前面讲了请求和响应的概念，向网站的服务器发送一个请求，返回的响应体便是网页源代码。所以，最关键的部分就是构造一个请求并发送给服务器，然后接收到响应并将其解析出来，那么这个流程怎样实现呢？总不能手工去截取网页源码吧？
不用担心，Python 提供了许多库来帮助我们实现这个操作，如 urllib、requests 等。我们可以用这些库来帮助我们实现 HTTP 请求操作，请求和响应都可以用类库提供的数据结构来表示，得到响应之后只需要解析数据结构中的 Body 部分即可，即得到网页的源代码，这样我们可以用程序来实现获取网页的过程了。
<h4>提取信息</h4>
获取网页源代码后，接下来就是分析网页源代码，从中提取我们想要的数据。首先，最通用的方法便是采用正则表达式提取，这是一个万能的方法，但是在构造正则表达式时比较复杂且容易出错。
另外，由于网页的结构有一定的规则，所以还有一些根据网页节点属性、CSS 选择器或 XPath 来提取网页信息的库，如 Beautiful Soup、pyquery、lxml 等。使用这些库，我们可以高效快速地从中提取网页信息，如节点的属性、文本值等。
提取信息是爬虫非常重要的部分，它可以使杂乱的数据变得条理清晰，以便我们后续处理和分析数据。
<h4>保存数据</h4>
提取信息后，我们一般会将提取到的数据保存到某处以便后续使用。这里保存形式有多种多样，如可以简单保存为 TXT 文本或 JSON 文本，也可以保存到数据库，如 MySQL 和 MongoDB 等，还可保存至远程服务器，如借助 SFTP 进行操作等。
<h4>自动化程序</h4>
说到自动化程序，意思是说爬虫可以代替人来完成这些操作。首先，我们手工当然可以提取这些信息，但是当量特别大或者想快速获取大量数据的话，肯定还是要借助程序。爬虫就是代替我们来完成这份爬取工作的自动化程序，它可以在抓取过程中进行各种异常处理、错误重试等操作，确保爬取持续高效地运行。
## 能抓怎样的数据

在网页中我们能看到各种各样的信息，最常见的便是常规网页，它们对应着 HTML 代码，而最常抓取的便是 HTML 源代码。
另外，可能有些网页返回的不是 HTML 代码，而是一个 JSON 字符串（其中 API 接口大多采用这样的形式），这种格式的数据方便传输和解析，它们同样可以抓取，而且数据提取更加方便。
此外，我们还可以看到各种二进制数据，如图片、视频和音频等。利用爬虫，我们可以将这些二进制数据抓取下来，然后保存成对应的文件名。
另外，还可以看到各种扩展名的文件，如 CSS、JavaScript 和配置文件等，这些其实也是最普通的文件，只要在浏览器里面可以访问到，就可以将其抓取下来。
上述内容其实都对应各自的 URL，是基于 HTTP 或 HTTPS 协议的，只要是这种数据，爬虫都可以抓取。
## JavaScript 渲染页面

有时候，我们在用 urllib 或 requests 抓取网页时，得到的源代码实际和浏览器中看到的不一样。
这是一个非常常见的问题。现在网页越来越多地采用 Ajax、前端模块化工具来构建，整个网页可能都是由 JavaScript 渲染出来的，也就是说原始的 HTML 代码就是一个空壳，例如：
<pre><code data-language="html" class="lang-html"><span class="hljs-meta">&lt;!DOCTYPE <span class="hljs-meta-keyword">html</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">html</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">head</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">meta</span> <span class="hljs-attr">charset</span>=<span class="hljs-string">"UTF-8"</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">title</span>&gt;</span>This is a Demo<span class="hljs-tag">&lt;/<span class="hljs-name">title</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">head</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">body</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">id</span>=<span class="hljs-string">"container"</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">body</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">script</span> <span class="hljs-attr">src</span>=<span class="hljs-string">"app.js"</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">script</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">html</span>&gt;</span>
</code></pre>
body 节点里面只有一个 id 为 container 的节点，但是需要注意在 body 节点后引入了 app.js，它便负责整个网站的渲染。
在浏览器中打开这个页面时，首先会加载这个 HTML 内容，接着浏览器会发现其中引入了一个 app.js 文件，然后便会接着去请求这个文件，获取到该文件后，便会执行其中的 JavaScript 代码，而 JavaScript 则会改变 HTML 中的节点，向其添加内容，最后得到完整的页面。
但是在用 urllib 或 requests 等库请求当前页面时，我们得到的只是这个 HTML 代码，它不会帮助我们去继续加载这个 JavaScript 文件，这样也就看不到浏览器中的内容了。
这也解释了为什么有时我们得到的源代码和浏览器中看到的不一样。
因此，使用基本 HTTP 请求库得到的源代码可能跟浏览器中的页面源代码不太一样。对于这样的情况，我们可以分析其后台 Ajax 接口，也可使用 Selenium、Splash 这样的库来实现模拟 JavaScript 渲染。
后面，我们会详细介绍如何采集 JavaScript 渲染的网页。本节介绍了爬虫的一些基本原理，这可以帮助我们在后面编写爬虫时更加得心应手。

# Session与Cookie
还有一些网站，在打开浏览器时就自动登录了，而且很长时间都不会失效，这种情况又是为什么？其实这里面涉及 Session 和 Cookies 的相关知识，本节就来揭开它们的神秘面纱。
## 静态网页和动态网页

在开始介绍它们之前，我们需要先了解一下静态网页和动态网页的概念。这里还是前面的示例代码，内容如下：
<pre><code data-language="html" class="lang-html"><span class="hljs-meta">&lt;!DOCTYPE <span class="hljs-meta-keyword">html</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">html</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">head</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">meta</span> <span class="hljs-attr">charset</span>=<span class="hljs-string">"UTF-8"</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">title</span>&gt;</span>This is a Demo<span class="hljs-tag">&lt;/<span class="hljs-name">title</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">head</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">body</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">id</span>=<span class="hljs-string">"container"</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"wrapper"</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">h2</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"title"</span>&gt;</span>Hello World<span class="hljs-tag">&lt;/<span class="hljs-name">h2</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">p</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"text"</span>&gt;</span>Hello, this is a paragraph.<span class="hljs-tag">&lt;/<span class="hljs-name">p</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">body</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">html</span>&gt;</span>
</code></pre>
这是最基本的 HTML 代码，我们将其保存为一个 .html 文件，然后把它放在某台具有固定公网 IP 的主机上，主机上装上 Apache 或 Nginx 等服务器，这样这台主机就可以作为服务器了，其他人便可以通过访问服务器看到这个页面，这就搭建了一个最简单的网站。
这种网页的内容是 HTML 代码编写的，文字、图片等内容均通过写好的 HTML 代码来指定，这种页面叫作静态网页。它加载速度快，编写简单，但是存在很大的缺陷，如可维护性差，不能根据 URL 灵活多变地显示内容等。例如，我们想要给这个网页的 URL 传入一个 name 参数，让其在网页中显示出来，是无法做到的。
因此，动态网页应运而生，它可以动态解析 URL 中参数的变化，关联数据库并动态呈现不同的页面内容，非常灵活多变。我们现在遇到的大多数网站都是动态网站，它们不再是一个简单的 HTML，而是可能由 JSP、PHP、Python 等语言编写的，其功能比静态网页强大和丰富太多了。
此外，动态网站还可以实现用户登录和注册的功能。再回到开头来看提到的问题，很多页面是需要登录之后才可以查看的。按照一般的逻辑来说，输入用户名和密码登录之后，肯定是拿到了一种类似凭证的东西，有了它，我们才能保持登录状态，才能访问登录之后才能看到的页面。
那么，这种神秘的凭证到底是什么呢？其实它就是 Session 和 Cookies 共同产生的结果，下面我们来一探究竟。
## 无状态 HTTP

在了解 Session 和 Cookies 之前，我们还需要了解 HTTP 的一个特点，叫作无状态。
HTTP 的无状态是指 HTTP 协议对事务处理是没有记忆能力的，也就是说服务器不知道客户端是什么状态。
当我们向服务器发送请求后，服务器解析此请求，然后返回对应的响应，服务器负责完成这个过程，而且这个过程是完全独立的，服务器不会记录前后状态的变化，也就是缺少状态记录。
这意味着如果后续需要处理前面的信息，则必须重传，这也导致需要额外传递一些前面的重复请求，才能获取后续响应，然而这种效果显然不是我们想要的。为了保持前后状态，我们肯定不能将前面的请求全部重传一次，这太浪费资源了，对于这种需要用户登录的页面来说，更是棘手。
这时两个用于保持 HTTP 连接状态的技术就出现了，它们分别是 Session 和 Cookies。Session 在服务端，也就是网站的服务器，用来保存用户的 Session 信息；Cookies 在客户端，也可以理解为浏览器端，有了 Cookies，浏览器在下次访问网页时会自动附带上它发送给服务器，服务器通过识别 Cookies 并鉴定出是哪个用户，然后再判断用户是否是登录状态，进而返回对应的响应。
我们可以理解为 Cookies 里面保存了登录的凭证，有了它，只需要在下次请求携带 Cookies 发送请求而不必重新输入用户名、密码等信息重新登录了。
因此在爬虫中，有时候处理需要登录才能访问的页面时，我们一般会直接将登录成功后获取的 Cookies 放在请求头里面直接请求，而不必重新模拟登录。
好了，了解 Session 和 Cookies 的概念之后，我们再来详细剖析它们的原理。
## Session

Session，中文称之为会话，其本身的含义是指有始有终的一系列动作 / 消息。比如，打电话时，从拿起电话拨号到挂断电话这中间的一系列过程可以称为一个 Session。
而在 Web 中，Session 对象用来存储特定用户 Session 所需的属性及配置信息。这样，当用户在应用程序的 Web 页之间跳转时，存储在 Session 对象中的变量将不会丢失，而是在整个用户 Session 中一直存在下去。当用户请求来自应用程序的 Web 页时，如果该用户还没有 Session，则 Web 服务器将自动创建一个 Session 对象。当 Session 过期或被放弃后，服务器将终止该 Session。
## Cookies

Cookies 指某些网站为了辨别用户身份、进行 Session 跟踪而存储在用户本地终端上的数据。
<h4>Session 维持</h4>
那么，我们怎样利用 Cookies 保持状态呢？当客户端第一次请求服务器时，服务器会返回一个响应头中带有 Set-Cookie 字段的响应给客户端，用来标记是哪一个用户，客户端浏览器会把 Cookies 保存起来。当浏览器下一次再请求该网站时，浏览器会把此 Cookies 放到请求头一起提交给服务器，Cookies 携带了 Session ID 信息，服务器检查该 Cookies 即可找到对应的 Session 是什么，然后再判断 Session 来以此来辨认用户状态。
在成功登录某个网站时，服务器会告诉客户端设置哪些 Cookies 信息，在后续访问页面时客户端会把 Cookies 发送给服务器，服务器再找到对应的 Session 加以判断。如果 Session 中的某些设置登录状态的变量是有效的，那就证明用户处于登录状态，此时返回登录之后才可以查看的网页内容，浏览器再进行解析便可以看到了。
反之，如果传给服务器的 Cookies 是无效的，或者 Session 已经过期了，我们将不能继续访问页面，此时可能会收到错误的响应或者跳转到登录页面重新登录。
所以，Cookies 和 Session 需要配合，一个处于客户端，一个处于服务端，二者共同协作，就实现了登录 Session 控制。
<h4>属性结构</h4>
接下来，我们来看看 Cookies 都有哪些内容。这里以知乎为例，在浏览器开发者工具中打开 Application 选项卡，然后在左侧会有一个 Storage 部分，最后一项即为 Cookies，将其点开，如图所示，这些就是 Cookies。
<img src="https://s0.lgstatic.com/i/image3/M01/6E/54/CgpOIF5fSryAMaivAANBRdQDiCI200.jpg" alt="">
可以看到，这里有很多条目，其中每个条目可以称为 Cookie。它有如下几个属性。
<ul>
<li>Name，即该 Cookie 的名称。Cookie 一旦创建，名称便不可更改。</li>
<li>Value，即该 Cookie 的值。如果值为 Unicode 字符，需要为字符编码。如果值为二进制数据，则需要使用 BASE64 编码。</li>
<li>Max Age，即该 Cookie 失效的时间，单位秒，也常和 Expires 一起使用，通过它可以计算出其有效时间。Max Age 如果为正数，则该 Cookie 在 Max Age 秒之后失效。如果为负数，则关闭浏览器时 Cookie 即失效，浏览器也不会以任何形式保存该 Cookie。</li>
<li>Path，即该 Cookie 的使用路径。如果设置为 /path/，则只有路径为 /path/ 的页面可以访问该 Cookie。如果设置为 /，则本域名下的所有页面都可以访问该 Cookie。</li>
<li>Domain，即可以访问该 Cookie 的域名。例如如果设置为 .zhihu.com，则所有以 zhihu.com，结尾的域名都可以访问该 Cookie。</li>
<li>Size 字段，即此 Cookie 的大小。</li>
<li>Http 字段，即 Cookie 的 httponly 属性。若此属性为 true，则只有在 HTTP Headers 中会带有此 Cookie 的信息，而不能通过 document.cookie 来访问此 Cookie。</li>
<li>Secure，即该 Cookie 是否仅被使用安全协议传输。安全协议。安全协议有 HTTPS、SSL 等，在网络上传输数据之前先将数据加密。默认为 false。</li>
</ul>
<h4>会话 Cookie 和持久 Cookie</h4>
从表面意思来说，会话 Cookie 就是把 Cookie 放在浏览器内存里，浏览器在关闭之后该 Cookie 即失效；持久 Cookie 则会保存到客户端的硬盘中，下次还可以继续使用，用于长久保持用户登录状态。
其实严格来说，没有会话 Cookie 和持久 Cookie 之 分，只是由 Cookie 的 Max Age 或 Expires 字段决定了过期的时间。
因此，一些持久化登录的网站其实就是把 Cookie 的有效时间和 Session 有效期设置得比较长，下次我们再访问页面时仍然携带之前的 Cookie，就可以直接保持登录状态。
## 常见误区

在谈论 Session 机制的时候，常常听到这样一种误解 ——“只要关闭浏览器，Session 就消失了”。可以想象一下会员卡的例子，除非顾客主动对店家提出销卡，否则店家绝对不会轻易删除顾客的资料。对 Session 来说，也是一样，除非程序通知服务器删除一个 Session，否则服务器会一直保留。比如，程序一般都是在我们做注销操作时才去删除 Session。
但是当我们关闭浏览器时，浏览器不会主动在关闭之前通知服务器它将要关闭，所以服务器根本不会有机会知道浏览器已经关闭。之所以会有这种错觉，是因为大部分网站都使用会话 Cookie 来保存 Session ID 信息，而关闭浏览器后 Cookies 就消失了，再次连接服务器时，也就无法找到原来的 Session 了。如果服务器设置的 Cookies 保存到硬盘上，或者使用某种手段改写浏览器发出的 HTTP 请求头，把原来的 Cookies 发送给服务器，则再次打开浏览器，仍然能够找到原来的 Session ID，依旧还是可以保持登录状态的。
而且恰恰是由于关闭浏览器不会导致 Session 被删除，这就需要服务器为 Session 设置一个失效时间，当距离客户端上一次使用 Session 的时间超过这个失效时间时，服务器就可以认为客户端已经停止了活动，才会把 Session 删除以节省存储空间。

# 多进程基本原理（1）
我们知道，在一台计算机中，我们可以同时打开许多软件，比如同时浏览网页、听音乐、打字等等，看似非常正常。但仔细想想，为什么计算机可以做到这么多软件同时运行呢？这就涉及到计算机中的两个重要概念：多进程和多线程了。 
同样，在编写爬虫程序的时候，为了提高爬取效率，我们可能想同时运行多个爬虫任务。这里同样需要涉及多进程和多线程的知识。 
本课时，我们就先来了解一下多线程的基本原理，以及在 Python 中如何实现多线程。 

## 多线程的含义
 说起多线程，就不得不先说什么是线程。然而想要弄明白什么是线程，又不得不先说什么是进程。 
进程我们可以理解为是一个可以独立运行的程序单位，比如打开一个浏览器，这就开启了一个浏览器进程；打开一个文本编辑器，这就开启了一个文本编辑器进程。但一个进程中是可以同时处理很多事情的，比如在浏览器中，我们可以在多个选项卡中打开多个页面，有的页面在播放音乐，有的页面在播放视频，有的网页在播放动画，它们可以同时运行，互不干扰。为什么能同时做到同时运行这么多的任务呢？这里就需要引出线程的概念了，其实这一个个任务，实际上就对应着一个个线程的执行。 
而进程呢？它就是线程的集合，进程就是由一个或多个线程构成的，线程是操作系统进行运算调度的最小单位，是进程中的一个最小运行单元。比如上面所说的浏览器进程，其中的播放音乐就是一个线程，播放视频也是一个线程，当然其中还有很多其他的线程在同时运行，这些线程的并发或并行执行最后使得整个浏览器可以同时运行这么多的任务。 
了解了线程的概念，多线程就很容易理解了，多线程就是一个进程中同时执行多个线程，前面所说的浏览器的情景就是典型的多线程执行。 

## 并发和并行
 说到多进程和多线程，这里就需要再讲解两个概念，那就是并发和并行。我们知道，一个程序在计算机中运行，其底层是处理器通过运行一条条的指令来实现的。 
<strong>并发</strong>，英文叫作 concurrency。它是指同一时刻只能有一条指令执行，但是多个线程的对应的指令被快速轮换地执行。比如一个处理器，它先执行线程 A 的指令一段时间，再执行线程 B 的指令一段时间，再切回到线程 A 执行一段时间。 
由于处理器执行指令的速度和切换的速度非常非常快，人完全感知不到计算机在这个过程中有多个线程切换上下文执行的操作，这就使得宏观上看起来多个线程在同时运行。但微观上只是这个处理器在连续不断地在多个线程之间切换和执行，每个线程的执行一定会占用这个处理器一个时间片段，同一时刻，其实只有一个线程在执行。 
<strong>并行</strong>，英文叫作 parallel。它是指同一时刻，有多条指令在多个处理器上同时执行，并行必须要依赖于多个处理器。不论是从宏观上还是微观上，多个线程都是在同一时刻一起执行的。 
并行只能在多处理器系统中存在，如果我们的计算机处理器只有一个核，那就不可能实现并行。而并发在单处理器和多处理器系统中都是可以存在的，因为仅靠一个核，就可以实现并发。 
举个例子，比如系统处理器需要同时运行多个线程。如果系统处理器只有一个核，那它只能通过并发的方式来运行这些线程。如果系统处理器有多个核，当一个核在执行一个线程时，另一个核可以执行另一个线程，这样这两个线程就实现了并行执行，当然其他的线程也可能和另外的线程处在同一个核上执行，它们之间就是并发执行。具体的执行方式，就取决于操作系统的调度了。 

##多线程适用场景
 在一个程序进程中，有一些操作是比较耗时或者需要等待的，比如等待数据库的查询结果的返回，等待网页结果的响应。如果使用单线程，处理器必须要等到这些操作完成之后才能继续往下执行其他操作，而这个线程在等待的过程中，处理器明显是可以来执行其他的操作的。如果使用多线程，处理器就可以在某个线程等待的时候，去执行其他的线程，从而从整体上提高执行效率。 
像上述场景，线程在执行过程中很多情况下是需要等待的。比如网络爬虫就是一个非常典型的例子，爬虫在向服务器发起请求之后，有一段时间必须要等待服务器的响应返回，这种任务就属于 IO 密集型任务。对于这种任务，如果我们启用多线程，处理器就可以在某个线程等待的过程中去处理其他的任务，从而提高整体的爬取效率。 
但并不是所有的任务都是 IO 密集型任务，还有一种任务叫作计算密集型任务，也可以称之为 CPU 密集型任务。顾名思义，就是任务的运行一直需要处理器的参与。此时如果我们开启了多线程，一个处理器从一个计算密集型任务切换到切换到另一个计算密集型任务上去，处理器依然不会停下来，始终会忙于计算，这样并不会节省总体的时间，因为需要处理的任务的计算总量是不变的。如果线程数目过多，反而还会在线程切换的过程中多耗费一些时间，整体效率会变低。 
所以，如果任务不全是计算密集型任务，我们可以使用多线程来提高程序整体的执行效率。尤其对于网络爬虫这种 IO 密集型任务来说，使用多线程会大大提高程序整体的爬取效率。 
##Python 实现多线程
 在 Python 中，实现多线程的模块叫作 threading，是 Python 自带的模块。下面我们来了解下使用 threading 实现多线程的方法。 <h4>Thread 直接创建子线程</h4> 
首先，我们可以使用 Thread 类来创建一个线程，创建时需要指定 target 参数为运行的方法名称，如果被调用的方法需要传入额外的参数，则可以通过 Thread 的 args 参数来指定。示例如下：
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span> threading
<span class="hljs-keyword">import</span> time

<span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">target</span><span class="hljs-params">(second)</span>:</span>
    print(<span class="hljs-string">f'Threading <span class="hljs-subst">{threading.current_thread().name}</span> is running'</span>)
    print(<span class="hljs-string">f'Threading <span class="hljs-subst">{threading.current_thread().name}</span> sleep <span class="hljs-subst">{second}</span>s'</span>)
    time.sleep(second)
    print(<span class="hljs-string">f'Threading <span class="hljs-subst">{threading.current_thread().name}</span> is ended'</span>)

print(<span class="hljs-string">f'Threading <span class="hljs-subst">{threading.current_thread().name}</span> is running'</span>)
<span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> [<span class="hljs-number">1</span>, <span class="hljs-number">5</span>]:
    thread = threading.Thread(target=target, args=[i])
    thread.start()
print(<span class="hljs-string">f'Threading <span class="hljs-subst">{threading.current_thread().name}</span> is ended'</span>)
</code></pre>
运行结果如下：
<pre><code data-language="python" class="lang-python">Threading MainThread <span class="hljs-keyword">is</span> running
Threading Thread<span class="hljs-number">-1</span> <span class="hljs-keyword">is</span> running
Threading Thread<span class="hljs-number">-1</span> sleep <span class="hljs-number">1</span>s
Threading Thread<span class="hljs-number">-2</span> <span class="hljs-keyword">is</span> running
Threading Thread<span class="hljs-number">-2</span> sleep <span class="hljs-number">5</span>s
Threading MainThread <span class="hljs-keyword">is</span> ended
Threading Thread<span class="hljs-number">-1</span> <span class="hljs-keyword">is</span> ended
Threading Thread<span class="hljs-number">-2</span> <span class="hljs-keyword">is</span> ended
</code></pre>
在这里我们首先声明了一个方法，叫作 target，它接收一个参数为 second，通过方法的实现可以发现，这个方法其实就是执行了一个 time.sleep 休眠操作，second 参数就是休眠秒数，其前后都 print 了一些内容，其中线程的名字我们通过 `threading.current_thread().name` 来获取出来，如果是主线程的话，其值就是 MainThread，如果是子线程的话，其值就是 `Thread-*`。 
然后我们通过 Thead 类新建了两个线程，target 参数就是刚才我们所定义的方法名，args 以列表的形式传递。两次循环中，这里 i 分别就是 1 和 5，这样两个线程就分别休眠 1 秒和 5 秒，声明完成之后，我们调用 start 方法即可开始线程的运行。 
观察结果我们可以发现，这里一共产生了三个线程，分别是主线程 MainThread 和两个子线程 Thread-1、Thread-2。另外我们观察到，主线程首先运行结束，紧接着 Thread-1、Thread-2 才接连运行结束，分别间隔了 1 秒和 4 秒。这说明主线程并没有等待子线程运行完毕才结束运行，而是直接退出了，有点不符合常理。 
如果我们想要主线程等待子线程运行完毕之后才退出，可以让每个子线程对象都调用下 join 方法，实现如下：
<pre><code data-language="python" class="lang-python">threads = []
<span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> [<span class="hljs-number">1</span>, <span class="hljs-number">5</span>]:
    thread = threading.Thread(target=target, args=[i])
    threads.append(thread)
    thread.start()
<span class="hljs-keyword">for</span> thread <span class="hljs-keyword">in</span> threads:
    thread.join()
</code></pre>
运行结果如下：
<pre><code data-language="python" class="lang-python">Threading MainThread <span class="hljs-keyword">is</span> running
Threading Thread<span class="hljs-number">-1</span> <span class="hljs-keyword">is</span> running
Threading Thread<span class="hljs-number">-1</span> sleep <span class="hljs-number">1</span>s
Threading Thread<span class="hljs-number">-2</span> <span class="hljs-keyword">is</span> running
Threading Thread<span class="hljs-number">-2</span> sleep <span class="hljs-number">5</span>s
Threading Thread<span class="hljs-number">-1</span> <span class="hljs-keyword">is</span> ended
Threading Thread<span class="hljs-number">-2</span> <span class="hljs-keyword">is</span> ended
Threading MainThread <span class="hljs-keyword">is</span> ended
</code></pre>
这样，主线程必须等待子线程都运行结束，主线程才继续运行并结束。 <h4>继承 Thread 类创建子线程</h4> 
另外，我们也可以通过继承 Thread 类的方式创建一个线程，该线程需要执行的方法写在类的 run 方法里面即可。上面的例子的等价改写为：
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span> threading
<span class="hljs-keyword">import</span> time


<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">MyThread</span><span class="hljs-params">(threading.Thread)</span>:</span>
    <span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">__init__</span><span class="hljs-params">(self, second)</span>:</span>
        threading.Thread.__init__(self)
        self.second = second
    
    <span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">run</span><span class="hljs-params">(self)</span>:</span>
        print(<span class="hljs-string">f'Threading <span class="hljs-subst">{threading.current_thread().name}</span> is running'</span>)
        print(<span class="hljs-string">f'Threading <span class="hljs-subst">{threading.current_thread().name}</span> sleep <span class="hljs-subst">{self.second}</span>s'</span>)
        time.sleep(self.second)
        print(<span class="hljs-string">f'Threading <span class="hljs-subst">{threading.current_thread().name}</span> is ended'</span>)


print(<span class="hljs-string">f'Threading <span class="hljs-subst">{threading.current_thread().name}</span> is running'</span>)
threads = []
<span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> [<span class="hljs-number">1</span>, <span class="hljs-number">5</span>]:
    thread = MyThread(i)
    threads.append(thread)
    thread.start()
<span class="hljs-keyword">for</span> thread <span class="hljs-keyword">in</span> threads:
    thread.join()
print(<span class="hljs-string">f'Threading <span class="hljs-subst">{threading.current_thread().name}</span> is ended'</span>)
</code></pre>
运行结果如下： <pre><code data-language="python" class="lang-python">Threading MainThread <span class="hljs-keyword">is</span> running Threading Thread<span class="hljs-number">-1</span> <span class="hljs-keyword">is</span> running Threading Thread<span class="hljs-number">-1</span> sleep <span class="hljs-number">1</span>s Threading Thread<span class="hljs-number">-2</span> <span class="hljs-keyword">is</span> running Threading Thread<span class="hljs-number">-2</span> sleep <span class="hljs-number">5</span>s Threading Thread<span class="hljs-number">-1</span> <span class="hljs-keyword">is</span> ended Threading Thread<span class="hljs-number">-2</span> <span class="hljs-keyword">is</span> ended Threading MainThread <span class="hljs-keyword">is</span> ended </code></pre> 
可以看到，两种实现方式，其运行效果是相同的。 <h4>守护线程</h4> 
在线程中有一个叫作守护线程的概念，如果一个线程被设置为守护线程，那么意味着这个线程是“不重要”的，这意味着，如果主线程结束了而该守护线程还没有运行完，那么它将会被强制结束。在 Python 中我们可以通过 setDaemon 方法来将某个线程设置为守护线程。 
示例如下：
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span> threading
<span class="hljs-keyword">import</span> time


<span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">target</span><span class="hljs-params">(second)</span>:</span>
    print(<span class="hljs-string">f'Threading <span class="hljs-subst">{threading.current_thread().name}</span> is running'</span>)
    print(<span class="hljs-string">f'Threading <span class="hljs-subst">{threading.current_thread().name}</span> sleep <span class="hljs-subst">{second}</span>s'</span>)
    time.sleep(second)
    print(<span class="hljs-string">f'Threading <span class="hljs-subst">{threading.current_thread().name}</span> is ended'</span>)


print(<span class="hljs-string">f'Threading <span class="hljs-subst">{threading.current_thread().name}</span> is running'</span>)
t1 = threading.Thread(target=target, args=[<span class="hljs-number">2</span>])
t1.start()
t2 = threading.Thread(target=target, args=[<span class="hljs-number">5</span>])
t2.setDaemon(<span class="hljs-literal">True</span>)
t2.start()
print(<span class="hljs-string">f'Threading <span class="hljs-subst">{threading.current_thread().name}</span> is ended'</span>)
</code></pre>
在这里我们通过 setDaemon 方法将 t2 设置为了守护线程，这样主线程在运行完毕时，t2 线程会随着线程的结束而结束。 
运行结果如下： <pre><code data-language="python" class="lang-python">Threading MainThread <span class="hljs-keyword">is</span> running Threading Thread<span class="hljs-number">-1</span> <span class="hljs-keyword">is</span> running Threading Thread<span class="hljs-number">-1</span> sleep <span class="hljs-number">2</span>s Threading Thread<span class="hljs-number">-2</span> <span class="hljs-keyword">is</span> running Threading Thread<span class="hljs-number">-2</span> sleep <span class="hljs-number">5</span>s Threading MainThread <span class="hljs-keyword">is</span> ended Threading Thread<span class="hljs-number">-1</span> <span class="hljs-keyword">is</span> ended </code></pre> 
可以看到，我们没有看到 Thread-2 打印退出的消息，Thread-2 随着主线程的退出而退出了。 
不过细心的你可能会发现，这里并没有调用 join 方法，如果我们让 t1 和 t2 都调用 join 方法，主线程就会仍然等待各个子线程执行完毕再退出，不论其是否是守护线程。 
##互斥锁
 在一个进程中的多个线程是共享资源的，比如在一个进程中，有一个全局变量 count 用来计数，现在我们声明多个线程，每个线程运行时都给 count 加 1，让我们来看看效果如何，代码实现如下：
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span> threading
<span class="hljs-keyword">import</span> time

count = <span class="hljs-number">0</span>

<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">MyThread</span><span class="hljs-params">(threading.Thread)</span>:</span>
    <span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">__init__</span><span class="hljs-params">(self)</span>:</span>
        threading.Thread.__init__(self)

    <span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">run</span><span class="hljs-params">(self)</span>:</span>
        <span class="hljs-keyword">global</span> count
        temp = count + <span class="hljs-number">1</span>
        time.sleep(<span class="hljs-number">0.001</span>)
        count = temp

threads = []
<span class="hljs-keyword">for</span> _ <span class="hljs-keyword">in</span> range(<span class="hljs-number">1000</span>):
    thread = MyThread()
    thread.start()
    threads.append(thread)

<span class="hljs-keyword">for</span> thread <span class="hljs-keyword">in</span> threads:
    thread.join()
print(<span class="hljs-string">f'Final count: <span class="hljs-subst">{count}</span>'</span>)
</code></pre>
在这里，我们声明了 1000 个线程，每个线程都是现取到当前的全局变量 count 值，然后休眠一小段时间，然后对 count 赋予新的值。 
那这样，按照常理来说，最终的 count 值应该为 1000。但其实不然，我们来运行一下看看。 
运行结果如下： <pre><code data-language="python" class="lang-python">Final count: <span class="hljs-number">69</span> </code></pre> 
最后的结果居然只有 69，而且多次运行或者换个环境运行结果是不同的。 
这是为什么呢？因为 count 这个值是共享的，每个线程都可以在执行 temp = count 这行代码时拿到当前 count 的值，但是这些线程中的一些线程可能是并发或者并行执行的，这就导致不同的线程拿到的可能是同一个 count 值，最后导致有些线程的 count 的加 1 操作并没有生效，导致最后的结果偏小。 
所以，如果多个线程同时对某个数据进行读取或修改，就会出现不可预料的结果。为了避免这种情况，我们需要对多个线程进行同步，要实现同步，我们可以对需要操作的数据进行加锁保护，这里就需要用到 threading.Lock 了。 
加锁保护是什么意思呢？就是说，某个线程在对数据进行操作前，需要先加锁，这样其他的线程发现被加锁了之后，就无法继续向下执行，会一直等待锁被释放，只有加锁的线程把锁释放了，其他的线程才能继续加锁并对数据做修改，修改完了再释放锁。这样可以确保同一时间只有一个线程操作数据，多个线程不会再同时读取和修改同一个数据，这样最后的运行结果就是对的了。 
我们可以将代码修改为如下内容：
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span> threading
<span class="hljs-keyword">import</span> time

count = <span class="hljs-number">0</span>

<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">MyThread</span><span class="hljs-params">(threading.Thread)</span>:</span>
    <span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">__init__</span><span class="hljs-params">(self)</span>:</span>
        threading.Thread.__init__(self)

    <span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">run</span><span class="hljs-params">(self)</span>:</span>
        <span class="hljs-keyword">global</span> count
        lock.acquire()
        temp = count + <span class="hljs-number">1</span>
        time.sleep(<span class="hljs-number">0.001</span>)
        count = temp
        lock.release()

lock = threading.Lock()
threads = []
<span class="hljs-keyword">for</span> _ <span class="hljs-keyword">in</span> range(<span class="hljs-number">1000</span>):
    thread = MyThread()
    thread.start()
    threads.append(thread)

<span class="hljs-keyword">for</span> thread <span class="hljs-keyword">in</span> threads:
    thread.join()
print(<span class="hljs-string">f'Final count: <span class="hljs-subst">{count}</span>'</span>)
</code></pre>
在这里我们声明了一个 lock 对象，其实就是 threading.Lock 的一个实例，然后在 run 方法里面，获取 count 前先加锁，修改完 count 之后再释放锁，这样多个线程就不会同时获取和修改 count 的值了。 
运行结果如下： <pre><code data-language="python" class="lang-python">Final count: <span class="hljs-number">1000</span> </code></pre> 
这样运行结果就正常了。 
关于 Python 多线程的内容，这里暂且先介绍这些，关于 theading 更多的使用方法，如信号量、队列等，可以参考官方文档：<a href="https://docs.python.org/zh-cn/3.7/library/threading.html#module-threading">https://docs.python.org/zh-cn/3.7/library/threading.html#module-threading</a>。 
##Python 多线程的问题
 由于 Python 中 GIL 的限制，导致不论是在单核还是多核条件下，在同一时刻只能运行一个线程，导致 Python 多线程无法发挥多核并行的优势。 
GIL 全称为 Global Interpreter Lock，中文翻译为全局解释器锁，其最初设计是出于数据安全而考虑的。 
在 Python 多线程下，每个线程的执行方式如下： <ul> <li>获取 GIL</li> <li>执行对应线程的代码</li> <li>释放 GIL</li> </ul> 
可见，某个线程想要执行，必须先拿到 GIL，我们可以把 GIL 看作是通行证，并且在一个 Python 进程中，GIL 只有一个。拿不到通行证的线程，就不允许执行。这样就会导致，即使是多核条件下，一个 Python 进程下的多个线程，同一时刻也只能执行一个线程。 
不过对于爬虫这种 IO 密集型任务来说，这个问题影响并不大。而对于计算密集型任务来说，由于 GIL 的存在，多线程总体的运行效率相比可能反而比单线程更低。

# 多进程基本原理（2）
在上一课时我们了解了多线程的基本概念，同时我们也提到，Python 中的多线程是不能很好发挥多核优势的，如果想要发挥多核优势，最好还是使用多进程。
那么本课时我们就来了解下多进程的基本概念和用 Python 实现多进程的方法。

##多进程的含义

进程（Process）是具有一定独立功能的程序关于某个数据集合上的一次运行活动，是系统进行资源分配和调度的一个独立单位。
顾名思义，多进程就是启用多个进程同时运行。由于进程是线程的集合，而且进程是由一个或多个线程构成的，所以多进程的运行意味着有大于或等于进程数量的线程在运行。

##Python 多进程的优势

通过上一课时我们知道，由于进程中 GIL 的存在，Python 中的多线程并不能很好地发挥多核优势，一个进程中的多个线程，在同一时刻只能有一个线程运行。
而对于多进程来说，每个进程都有属于自己的 GIL，所以，在多核处理器下，多进程的运行是不会受 GIL 的影响的。因此，多进程能更好地发挥多核的优势。
当然，对于爬虫这种 IO 密集型任务来说，多线程和多进程影响差别并不大。对于计算密集型任务来说，Python 的多进程相比多线程，其多核运行效率会有成倍的提升。
总的来说，Python 的多进程整体来看是比多线程更有优势的。所以，在条件允许的情况下，能用多进程就尽量用多进程。
不过值得注意的是，由于进程是系统进行资源分配和调度的一个独立单位，所以各个进程之间的数据是无法共享的，如多个进程无法共享一个全局变量，进程之间的数据共享需要有单独的机制来实现，这在后面也会讲到。

##多进程的实现

在 Python 中也有内置的库来实现多进程，它就是 multiprocessing。
multiprocessing 提供了一系列的组件，如 Process（进程）、Queue（队列）、Semaphore（信号量）、Pipe（管道）、Lock（锁）、Pool（进程池）等，接下来让我们来了解下它们的使用方法。
<h4>直接使用 Process 类</h4>
在 multiprocessing 中，每一个进程都用一个 Process 类来表示。它的 API 调用如下：
<pre><code data-language="python" class="lang-python">Process([group [, target [, name [, args [, kwargs]]]]])
</code></pre>
<ul>
<li>target 表示调用对象，你可以传入方法的名字。</li>
<li>args 表示被调用对象的位置参数元组，比如 target 是函数 func，他有两个参数 m，n，那么 args 就传入 [m, n] 即可。</li>
<li>kwargs 表示调用对象的字典。</li>
<li>name 是别名，相当于给这个进程取一个名字。</li>
<li>group 分组。</li>
</ul>
我们先用一个实例来感受一下：
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span> multiprocessing

<span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">process</span><span class="hljs-params">(index)</span>:</span>
    print(<span class="hljs-string">f'Process: <span class="hljs-subst">{index}</span>'</span>)

<span class="hljs-keyword">if</span> __name__ == <span class="hljs-string">'__main__'</span>:
    <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> range(<span class="hljs-number">5</span>):
        p = multiprocessing.Process(target=process, args=(i,))
        p.start()
</code></pre>
这是一个实现多进程最基础的方式：通过创建 Process 来新建一个子进程，其中 target 参数传入方法名，args 是方法的参数，是以元组的形式传入，其和被调用的方法 process 的参数是一一对应的。
<blockquote>
注意：这里 args 必须要是一个元组，如果只有一个参数，那也要在元组第一个元素后面加一个逗号，如果没有逗号则和单个元素本身没有区别，无法构成元组，导致参数传递出现问题。
</blockquote>
创建完进程之后，我们通过调用 start 方法即可启动进程了。运行结果如下：
<pre><code data-language="python" class="lang-python">Process:&nbsp;<span class="hljs-number">0</span>
Process:&nbsp;<span class="hljs-number">1</span>
Process:&nbsp;<span class="hljs-number">2</span>
Process:&nbsp;<span class="hljs-number">3</span>
Process:&nbsp;<span class="hljs-number">4</span>
</code></pre>
可以看到，我们运行了 5 个子进程，每个进程都调用了 process 方法。process 方法的 index 参数通过 Process 的 args 传入，分别是 0~4 这 5 个序号，最后打印出来，5 个子进程运行结束。
由于进程是 Python 中最小的资源分配单元，因此这些进程和线程不同，各个进程之间的数据是不会共享的，每启动一个进程，都会独立分配资源。
另外，在当前 CPU 核数足够的情况下，这些不同的进程会分配给不同的 CPU 核来运行，实现真正的并行执行。
multiprocessing 还提供了几个比较有用的方法，如我们可以通过 cpu_count 的方法来获取当前机器 CPU 的核心数量，通过 active_children 方法获取当前还在运行的所有进程。
下面通过一个实例来看一下：
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span> multiprocessing
<span class="hljs-keyword">import</span> time

<span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">process</span><span class="hljs-params">(index)</span>:</span>
    time.sleep(index)
    print(<span class="hljs-string">f'Process: <span class="hljs-subst">{index}</span>'</span>)

<span class="hljs-keyword">if</span> __name__ == <span class="hljs-string">'__main__'</span>:
    <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> range(<span class="hljs-number">5</span>):
        p = multiprocessing.Process(target=process, args=[i])
        p.start()
    print(<span class="hljs-string">f'CPU number: <span class="hljs-subst">{multiprocessing.cpu_count()}</span>'</span>)
    <span class="hljs-keyword">for</span> p <span class="hljs-keyword">in</span> multiprocessing.active_children():
        print(<span class="hljs-string">f'Child process name: <span class="hljs-subst">{p.name}</span> id: <span class="hljs-subst">{p.pid}</span>'</span>)
    print(<span class="hljs-string">'Process Ended'</span>)
</code></pre>
运行结果如下：
<pre><code data-language="python" class="lang-python">Process: <span class="hljs-number">0</span>
CPU number: <span class="hljs-number">8</span>
Child process name: Process<span class="hljs-number">-5</span> id: <span class="hljs-number">73595</span>
Child process name: Process<span class="hljs-number">-2</span> id: <span class="hljs-number">73592</span>
Child process name: Process<span class="hljs-number">-3</span> id: <span class="hljs-number">73593</span>
Child process name: Process<span class="hljs-number">-4</span> id: <span class="hljs-number">73594</span>
Process Ended
Process: <span class="hljs-number">1</span>
Process: <span class="hljs-number">2</span>
Process: <span class="hljs-number">3</span>
Process: <span class="hljs-number">4</span>
</code></pre>
在上面的例子中我们通过 cpu_count 成功获取了 CPU 核心的数量：8 个，当然不同的机器结果可能不同。
另外我们还通过 active_children 获取到了当前正在活跃运行的进程列表。然后我们遍历了每个进程，并将它们的名称和进程号打印出来了，这里进程号直接使用 pid 属性即可获取，进程名称直接通过 name 属性即可获取。
以上我们就完成了多进程的创建和一些基本信息的获取。
<h4>继承 Process 类</h4>
在上面的例子中，我们创建进程是直接使用 Process 这个类来创建的，这是一种创建进程的方式。不过，创建进程的方式不止这一种，同样，我们也可以像线程 Thread 一样来通过继承的方式创建一个进程类，进程的基本操作我们在子类的 run 方法中实现即可。
通过一个实例来看一下：
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span> multiprocessing <span class="hljs-keyword">import</span> Process
<span class="hljs-keyword">import</span> time

<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">MyProcess</span><span class="hljs-params">(Process)</span>:</span>
    <span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">__init__</span><span class="hljs-params">(self, loop)</span>:</span>
        Process.__init__(self)
        self.loop = loop

    <span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">run</span><span class="hljs-params">(self)</span>:</span>
        <span class="hljs-keyword">for</span> count <span class="hljs-keyword">in</span> range(self.loop):
            time.sleep(<span class="hljs-number">1</span>)
            print(<span class="hljs-string">f'Pid: <span class="hljs-subst">{self.pid}</span> LoopCount: <span class="hljs-subst">{count}</span>'</span>)

<span class="hljs-keyword">if</span> __name__ == <span class="hljs-string">'__main__'</span>:
    <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> range(<span class="hljs-number">2</span>, <span class="hljs-number">5</span>):
        p = MyProcess(i)
        p.start()
</code></pre>
我们首先声明了一个构造方法，这个方法接收一个 loop 参数，代表循环次数，并将其设置为全局变量。在 run 方法中，又使用这个 loop 变量循环了 loop 次并打印了当前的进程号和循环次数。
在调用时，我们用 range 方法得到了 2、3、4 三个数字，并把它们分别初始化了 MyProcess 进程，然后调用 start 方法将进程启动起来。
<blockquote>
注意：这里进程的执行逻辑需要在 run 方法中实现，启动进程需要调用 start 方法，调用之后 run 方法便会执行。
</blockquote>
运行结果如下：
<pre><code data-language="python" class="lang-python">Pid: <span class="hljs-number">73667</span> LoopCount: <span class="hljs-number">0</span>
Pid: <span class="hljs-number">73668</span> LoopCount: <span class="hljs-number">0</span>
Pid: <span class="hljs-number">73669</span> LoopCount: <span class="hljs-number">0</span>
Pid: <span class="hljs-number">73667</span> LoopCount: <span class="hljs-number">1</span>
Pid: <span class="hljs-number">73668</span> LoopCount: <span class="hljs-number">1</span>
Pid: <span class="hljs-number">73669</span> LoopCount: <span class="hljs-number">1</span>
Pid: <span class="hljs-number">73668</span> LoopCount: <span class="hljs-number">2</span>
Pid: <span class="hljs-number">73669</span> LoopCount: <span class="hljs-number">2</span>
Pid: <span class="hljs-number">73669</span> LoopCount: <span class="hljs-number">3</span>
</code></pre>
可以看到，三个进程分别打印出了 2、3、4 条结果，即进程 73667 打印了 2 次 结果，进程 73668 打印了 3 次结果，进程 73669 打印了 4 次结果。
<blockquote>
注意，这里的进程 pid 代表进程号，不同机器、不同时刻运行结果可能不同。
</blockquote>
通过上面的方式，我们也非常方便地实现了一个进程的定义。为了复用方便，我们可以把一些方法写在每个进程类里封装好，在使用时直接初始化一个进程类运行即可。
<h4>守护进程</h4>
在多进程中，同样存在守护进程的概念，如果一个进程被设置为守护进程，当父进程结束后，子进程会自动被终止，我们可以通过设置 daemon 属性来控制是否为守护进程。
还是原来的例子，增加了 deamon 属性的设置：
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span> multiprocessing <span class="hljs-keyword">import</span> Process
<span class="hljs-keyword">import</span> time

<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">MyProcess</span><span class="hljs-params">(Process)</span>:</span>
    <span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">__init__</span><span class="hljs-params">(self, loop)</span>:</span>
        Process.__init__(self)
        self.loop = loop

    <span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">run</span><span class="hljs-params">(self)</span>:</span>
        <span class="hljs-keyword">for</span> count <span class="hljs-keyword">in</span> range(self.loop):
            time.sleep(<span class="hljs-number">1</span>)
            print(<span class="hljs-string">f'Pid: <span class="hljs-subst">{self.pid}</span> LoopCount: <span class="hljs-subst">{count}</span>'</span>)

<span class="hljs-keyword">if</span> __name__ == <span class="hljs-string">'__main__'</span>:
    <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> range(<span class="hljs-number">2</span>, <span class="hljs-number">5</span>):
        p = MyProcess(i)
        p.daemon = <span class="hljs-literal">True</span>
        p.start()

print(<span class="hljs-string">'Main Process ended'</span>)
</code></pre>
运行结果如下：
<pre><code>Main Process ended
</code></pre>
结果很简单，因为主进程没有做任何事情，直接输出一句话结束，所以在这时也直接终止了子进程的运行。
这样可以有效防止无控制地生成子进程。这样的写法可以让我们在主进程运行结束后无需额外担心子进程是否关闭，避免了独立子进程的运行。
<h4>进程等待</h4>
上面的运行效果其实不太符合我们预期：主进程运行结束时，子进程（守护进程）也都退出了，子进程什么都没来得及执行。
能不能让所有子进程都执行完了然后再结束呢？当然是可以的，只需要加入 join 方法即可，我们可以将代码改写如下：
<pre><code data-language="python" class="lang-python">processes = []
<span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> range(<span class="hljs-number">2</span>, <span class="hljs-number">5</span>):
    p = MyProcess(i)
    processes.append(p)
    p.daemon = <span class="hljs-literal">True</span>
    p.start()
<span class="hljs-keyword">for</span> p <span class="hljs-keyword">in</span> processes:
    p.join()
</code></pre>
运行结果如下：
<pre><code data-language="python" class="lang-python">Pid: <span class="hljs-number">40866</span> LoopCount: <span class="hljs-number">0</span>
Pid: <span class="hljs-number">40867</span> LoopCount: <span class="hljs-number">0</span>
Pid: <span class="hljs-number">40868</span> LoopCount: <span class="hljs-number">0</span>
Pid: <span class="hljs-number">40866</span> LoopCount: <span class="hljs-number">1</span>
Pid: <span class="hljs-number">40867</span> LoopCount: <span class="hljs-number">1</span>
Pid: <span class="hljs-number">40868</span> LoopCount: <span class="hljs-number">1</span>
Pid: <span class="hljs-number">40867</span> LoopCount: <span class="hljs-number">2</span>
Pid: <span class="hljs-number">40868</span> LoopCount: <span class="hljs-number">2</span>
Pid: <span class="hljs-number">40868</span> LoopCount: <span class="hljs-number">3</span>
Main Process ended
</code></pre>
在调用 start 和 join 方法后，父进程就可以等待所有子进程都执行完毕后，再打印出结束的结果。
默认情况下，join 是无限期的。也就是说，如果有子进程没有运行完毕，主进程会一直等待。这种情况下，如果子进程出现问题陷入了死循环，主进程也会无限等待下去。怎么解决这个问题呢？可以给 join 方法传递一个超时参数，代表最长等待秒数。如果子进程没有在这个指定秒数之内完成，会被强制返回，主进程不再会等待。也就是说这个参数设置了主进程等待该子进程的最长时间。
例如这里我们传入 1，代表最长等待 1 秒，代码改写如下：
<pre><code data-language="python" class="lang-python">processes = []
<span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> range(<span class="hljs-number">3</span>, <span class="hljs-number">5</span>):
    p = MyProcess(i)
    processes.append(p)
    p.daemon = <span class="hljs-literal">True</span>
    p.start()
<span class="hljs-keyword">for</span> p <span class="hljs-keyword">in</span> processes:
    p.join(<span class="hljs-number">1</span>)
</code></pre>
运行结果如下：
<pre><code data-language="python" class="lang-python">Pid: <span class="hljs-number">40970</span> LoopCount: <span class="hljs-number">0</span>
Pid: <span class="hljs-number">40971</span> LoopCount: <span class="hljs-number">0</span>
Pid: <span class="hljs-number">40970</span> LoopCount: <span class="hljs-number">1</span>
Pid: <span class="hljs-number">40971</span> LoopCount: <span class="hljs-number">1</span>
Main Process ended
</code></pre>
可以看到，有的子进程本来要运行 3 秒，结果运行 1 秒就被强制返回了，由于是守护进程，该子进程被终止了。
到这里，我们就了解了守护进程、进程等待和超时设置的用法。
<h4>终止进程</h4>
当然，终止进程不止有守护进程这一种做法，我们也可以通过 terminate 方法来终止某个子进程，另外我们还可以通过 is_alive 方法判断进程是否还在运行。
下面我们来看一个实例：
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span> multiprocessing
<span class="hljs-keyword">import</span> time

<span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">process</span><span class="hljs-params">()</span>:</span>
    print(<span class="hljs-string">'Starting'</span>)
    time.sleep(<span class="hljs-number">5</span>)
    print(<span class="hljs-string">'Finished'</span>)

<span class="hljs-keyword">if</span> __name__ == <span class="hljs-string">'__main__'</span>:
    p = multiprocessing.Process(target=process)
    print(<span class="hljs-string">'Before:'</span>, p, p.is_alive())

    p.start()
    print(<span class="hljs-string">'During:'</span>, p, p.is_alive())

    p.terminate()
    print(<span class="hljs-string">'Terminate:'</span>, p, p.is_alive())

    p.join()
    print(<span class="hljs-string">'Joined:'</span>, p, p.is_alive())
</code></pre>
在上面的例子中，我们用 Process 创建了一个进程，接着调用 start 方法启动这个进程，然后调用 terminate 方法将进程终止，最后调用 join 方法。
另外，在进程运行不同的阶段，我们还通过 is_alive 方法判断当前进程是否还在运行。
运行结果如下：
<pre><code data-language="python" class="lang-python">Before: &lt;Process(Process<span class="hljs-number">-1</span>, initial)&gt; <span class="hljs-literal">False</span>
During: &lt;Process(Process<span class="hljs-number">-1</span>, started)&gt; <span class="hljs-literal">True</span>
Terminate: &lt;Process(Process<span class="hljs-number">-1</span>, started)&gt; <span class="hljs-literal">True</span>
Joined: &lt;Process(Process<span class="hljs-number">-1</span>, stopped[SIGTERM])&gt; <span class="hljs-literal">False</span>
</code></pre>
这里有一个值得注意的地方，在调用 terminate 方法之后，我们用 `is_alive` 方法获取进程的状态发现依然还是运行状态。在调用 join 方法之后，`is_alive` 方法获取进程的运行状态才变为终止状态。
所以，在调用 terminate 方法之后，记得要调用一下 join 方法，这里调用 join 方法可以为进程提供时间来更新对象状态，用来反映出最终的进程终止效果。

##进程互斥锁

在上面的一些实例中，我们可能会遇到如下的运行结果：
<pre><code data-language="python" class="lang-python">Pid: <span class="hljs-number">73993</span> LoopCount: <span class="hljs-number">0</span>
Pid: <span class="hljs-number">73993</span> LoopCount: <span class="hljs-number">1</span>
Pid: <span class="hljs-number">73994</span> LoopCount: <span class="hljs-number">0</span>Pid: <span class="hljs-number">73994</span> LoopCount: <span class="hljs-number">1</span>

Pid: <span class="hljs-number">73994</span> LoopCount: <span class="hljs-number">2</span>
Pid: <span class="hljs-number">73995</span> LoopCount: <span class="hljs-number">0</span>
Pid: <span class="hljs-number">73995</span> LoopCount: <span class="hljs-number">1</span>
Pid: <span class="hljs-number">73995</span> LoopCount: <span class="hljs-number">2</span>
Pid: <span class="hljs-number">73995</span> LoopCount: <span class="hljs-number">3</span>
Main Process ended
</code></pre>
我们发现，有的输出结果没有换行。这是什么原因造成的呢？
这种情况是由多个进程并行执行导致的，两个进程同时进行了输出，结果第一个进程的换行没有来得及输出，第二个进程就输出了结果，导致最终输出没有换行。
那如何来避免这种问题？如果我们能保证，多个进程运行期间的任一时间，只能一个进程输出，其他进程等待，等刚才那个进程输出完毕之后，另一个进程再进行输出，这样就不会出现输出没有换行的现象了。
这种解决方案实际上就是实现了进程互斥，避免了多个进程同时抢占临界区（输出）资源。我们可以通过 multiprocessing 中的 Lock 来实现。Lock，即锁，在一个进程输出时，加锁，其他进程等待。等此进程执行结束后，释放锁，其他进程可以进行输出。
我们首先实现一个不加锁的实例，代码如下：
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span> multiprocessing <span class="hljs-keyword">import</span> Process, Lock
<span class="hljs-keyword">import</span> time

<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">MyProcess</span><span class="hljs-params">(Process)</span>:</span>
    <span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">__init__</span><span class="hljs-params">(self, loop, lock)</span>:</span>
        Process.__init__(self)
        self.loop = loop
        self.lock = lock

    <span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">run</span><span class="hljs-params">(self)</span>:</span>
        <span class="hljs-keyword">for</span> count <span class="hljs-keyword">in</span> range(self.loop):
            time.sleep(<span class="hljs-number">0.1</span>)
            <span class="hljs-comment"># self.lock.acquire()</span>
            print(<span class="hljs-string">f'Pid: <span class="hljs-subst">{self.pid}</span> LoopCount: <span class="hljs-subst">{count}</span>'</span>)
            <span class="hljs-comment"># self.lock.release()</span>

<span class="hljs-keyword">if</span> __name__ == <span class="hljs-string">'__main__'</span>:
    lock = Lock()
    <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> range(<span class="hljs-number">10</span>, <span class="hljs-number">15</span>):
        p = MyProcess(i, lock)
        p.start()
</code></pre>
运行结果如下：
<pre><code data-language="python" class="lang-python">Pid: <span class="hljs-number">74030</span> LoopCount: <span class="hljs-number">0</span>
Pid: <span class="hljs-number">74031</span> LoopCount: <span class="hljs-number">0</span>
Pid: <span class="hljs-number">74032</span> LoopCount: <span class="hljs-number">0</span>
Pid: <span class="hljs-number">74033</span> LoopCount: <span class="hljs-number">0</span>
Pid: <span class="hljs-number">74034</span> LoopCount: <span class="hljs-number">0</span>
Pid: <span class="hljs-number">74030</span> LoopCount: <span class="hljs-number">1</span>
Pid: <span class="hljs-number">74031</span> LoopCount: <span class="hljs-number">1</span>
Pid: <span class="hljs-number">74032</span> LoopCount: <span class="hljs-number">1</span>Pid: <span class="hljs-number">74033</span> LoopCount: <span class="hljs-number">1</span>

Pid: <span class="hljs-number">74034</span> LoopCount: <span class="hljs-number">1</span>
Pid: <span class="hljs-number">74030</span> LoopCount: <span class="hljs-number">2</span>
...
</code></pre>
可以看到运行结果中有些输出已经出现了不换行的问题。
我们对其加锁，取消掉刚才代码中的两行注释，重新运行，运行结果如下：
<pre><code data-language="python" class="lang-python">Pid: <span class="hljs-number">74061</span> LoopCount: <span class="hljs-number">0</span>
Pid: <span class="hljs-number">74062</span> LoopCount: <span class="hljs-number">0</span>
Pid: <span class="hljs-number">74063</span> LoopCount: <span class="hljs-number">0</span>
Pid: <span class="hljs-number">74064</span> LoopCount: <span class="hljs-number">0</span>
Pid: <span class="hljs-number">74065</span> LoopCount: <span class="hljs-number">0</span>
Pid: <span class="hljs-number">74061</span> LoopCount: <span class="hljs-number">1</span>
Pid: <span class="hljs-number">74062</span> LoopCount: <span class="hljs-number">1</span>
Pid: <span class="hljs-number">74063</span> LoopCount: <span class="hljs-number">1</span>
Pid: <span class="hljs-number">74064</span> LoopCount: <span class="hljs-number">1</span>
Pid: <span class="hljs-number">74065</span> LoopCount: <span class="hljs-number">1</span>
Pid: <span class="hljs-number">74061</span> LoopCount: <span class="hljs-number">2</span>
Pid: <span class="hljs-number">74062</span> LoopCount: <span class="hljs-number">2</span>
Pid: <span class="hljs-number">74064</span> LoopCount: <span class="hljs-number">2</span>
...
</code></pre>
这时输出效果就正常了。
所以，在访问一些临界区资源时，使用 Lock 可以有效避免进程同时占用资源而导致的一些问题。

##信号量

进程互斥锁可以使同一时刻只有一个进程能访问共享资源，如上面的例子所展示的那样，在同一时刻只能有一个进程输出结果。但有时候我们需要允许多个进程来访问共享资源，同时还需要限制能访问共享资源的进程的数量。
这种需求该如何实现呢？可以用信号量，信号量是进程同步过程中一个比较重要的角色。它可以控制临界资源的数量，实现多个进程同时访问共享资源，限制进程的并发量。
如果你学过操作系统，那么一定对这方面非常了解，如果你还不了解信号量是什么，可以先熟悉一下这个概念。
我们可以用 multiprocessing 库中的 Semaphore 来实现信号量。
那么接下来我们就用一个实例来演示一下进程之间利用 Semaphore 做到多个进程共享资源，同时又限制同时可访问的进程数量，代码如下：
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span> multiprocessing <span class="hljs-keyword">import</span> Process, Semaphore, Lock, Queue
<span class="hljs-keyword">import</span> time

buffer = Queue(<span class="hljs-number">10</span>)
empty = Semaphore(<span class="hljs-number">2</span>)
full = Semaphore(<span class="hljs-number">0</span>)
lock = Lock()

<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">Consumer</span><span class="hljs-params">(Process)</span>:</span>
    <span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">run</span><span class="hljs-params">(self)</span>:</span>
        <span class="hljs-keyword">global</span> buffer, empty, full, lock
        <span class="hljs-keyword">while</span> <span class="hljs-literal">True</span>:
            full.acquire()
            lock.acquire()
            buffer.get()
            print(<span class="hljs-string">'Consumer pop an element'</span>)
            time.sleep(<span class="hljs-number">1</span>)
            lock.release()
            empty.release()

<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">Producer</span><span class="hljs-params">(Process)</span>:</span>
    <span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">run</span><span class="hljs-params">(self)</span>:</span>
        <span class="hljs-keyword">global</span> buffer, empty, full, lock
        <span class="hljs-keyword">while</span> <span class="hljs-literal">True</span>:
            empty.acquire()
            lock.acquire()
            buffer.put(<span class="hljs-number">1</span>)
            print(<span class="hljs-string">'Producer append an element'</span>)
            time.sleep(<span class="hljs-number">1</span>)
            lock.release()
            full.release()

<span class="hljs-keyword">if</span> __name__ == <span class="hljs-string">'__main__'</span>:
    p = Producer()
    c = Consumer()
    p.daemon = c.daemon = <span class="hljs-literal">True</span>
    p.start()
    c.start()
    p.join()
    c.join()
    print(<span class="hljs-string">'Main Process Ended'</span>)
</code></pre>
如上代码实现了经典的生产者和消费者问题。它定义了两个进程类，一个是消费者，一个是生产者。
另外，这里使用 multiprocessing 中的 Queue 定义了一个共享队列，然后定义了两个信号量 Semaphore，一个代表缓冲区空余数，一个表示缓冲区占用数。
生产者 Producer 使用 acquire 方法来占用一个缓冲区位置，缓冲区空闲区大小减 1，接下来进行加锁，对缓冲区进行操作，然后释放锁，最后让代表占用的缓冲区位置数量加 1，消费者则相反。
运行结果如下：
<pre><code data-language="python" class="lang-python">Producer append an element
Producer append an element
Consumer pop an element
Consumer pop an element
Producer append an element
Producer append an element
Consumer pop an element
Consumer pop an element
Producer append an element
Producer append an element
Consumer pop an element
Consumer pop an element
Producer append an element
Producer append an element
</code></pre>
我们发现两个进程在交替运行，生产者先放入缓冲区物品，然后消费者取出，不停地进行循环。 你可以通过上面的例子来体会信号量 Semaphore 的用法，通过 Semaphore 我们很好地控制了进程对资源的并发访问数量。

##队列

在上面的例子中我们使用 Queue 作为进程通信的共享队列使用。
而如果我们把上面程序中的 Queue 换成普通的 list，是完全起不到效果的，因为进程和进程之间的资源是不共享的。即使在一个进程中改变了这个 list，在另一个进程也不能获取到这个 list 的状态，所以声明全局变量对多进程是没有用处的。
那进程如何共享数据呢？可以用 Queue，即队列。当然这里的队列指的是 multiprocessing 里面的 Queue。
依然用上面的例子，我们一个进程向队列中放入随机数据，然后另一个进程取出数据。
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span> multiprocessing <span class="hljs-keyword">import</span> Process, Semaphore, Lock, Queue
<span class="hljs-keyword">import</span> time
<span class="hljs-keyword">from</span> random <span class="hljs-keyword">import</span> random

buffer = Queue(<span class="hljs-number">10</span>)
empty = Semaphore(<span class="hljs-number">2</span>)
full = Semaphore(<span class="hljs-number">0</span>)
lock = Lock()

<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">Consumer</span><span class="hljs-params">(Process)</span>:</span>
    <span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">run</span><span class="hljs-params">(self)</span>:</span>
        <span class="hljs-keyword">global</span> buffer, empty, full, lock
        <span class="hljs-keyword">while</span> <span class="hljs-literal">True</span>:
            full.acquire()
            lock.acquire()
            print(<span class="hljs-string">f'Consumer get <span class="hljs-subst">{buffer.get()}</span>'</span>)
            time.sleep(<span class="hljs-number">1</span>)
            lock.release()
            empty.release()

<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">Producer</span><span class="hljs-params">(Process)</span>:</span>
    <span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">run</span><span class="hljs-params">(self)</span>:</span>
        <span class="hljs-keyword">global</span> buffer, empty, full, lock
        <span class="hljs-keyword">while</span> <span class="hljs-literal">True</span>:
            empty.acquire()
            lock.acquire()
            num = random()
            print(<span class="hljs-string">f'Producer put <span class="hljs-subst">{num}</span>'</span>)
            buffer.put(num)
            time.sleep(<span class="hljs-number">1</span>)
            lock.release()
            full.release()

<span class="hljs-keyword">if</span> __name__ == <span class="hljs-string">'__main__'</span>:
    p = Producer()
    c = Consumer()
    p.daemon = c.daemon = <span class="hljs-literal">True</span>
    p.start()
    c.start()
    p.join()
    c.join()
    print(<span class="hljs-string">'Main Process Ended'</span>)
</code></pre>
运行结果如下：
<pre><code data-language="python" class="lang-python">Producer put  <span class="hljs-number">0.719213647437</span>
Producer put  <span class="hljs-number">0.44287326683</span>
Consumer get <span class="hljs-number">0.719213647437</span>
Consumer get <span class="hljs-number">0.44287326683</span>
Producer put  <span class="hljs-number">0.722859424381</span>
Producer put  <span class="hljs-number">0.525321338921</span>
Consumer get <span class="hljs-number">0.722859424381</span>
Consumer get <span class="hljs-number">0.525321338921</span>
</code></pre>
在上面的例子中我们声明了两个进程，一个进程为生产者 Producer，另一个为消费者 Consumer，生产者不断向 Queue 里面添加随机数，消费者不断从队列里面取随机数。
生产者在放数据的时候调用了 Queue 的 put 方法，消费者在取的时候使用了 get 方法，这样我们就通过 Queue 实现两个进程的数据共享了。

##管道

刚才我们使用 Queue 实现了进程间的数据共享，那么进程之间直接通信，如收发信息，用什么比较好呢？可以用 Pipe，管道。
管道，我们可以把它理解为两个进程之间通信的通道。管道可以是单向的，即 half-duplex：一个进程负责发消息，另一个进程负责收消息；也可以是双向的 duplex，即互相收发消息。
默认声明 Pipe 对象是双向管道，如果要创建单向管道，可以在初始化的时候传入 deplex 参数为 False。
我们用一个实例来感受一下：
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span> multiprocessing <span class="hljs-keyword">import</span> Process, Pipe

<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">Consumer</span><span class="hljs-params">(Process)</span>:</span>
    <span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">__init__</span><span class="hljs-params">(self, pipe)</span>:</span>
        Process.__init__(self)
        self.pipe = pipe

    <span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">run</span><span class="hljs-params">(self)</span>:</span>
        self.pipe.send(<span class="hljs-string">'Consumer Words'</span>)
        print(<span class="hljs-string">f'Consumer Received: <span class="hljs-subst">{self.pipe.recv()}</span>'</span>)

<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">Producer</span><span class="hljs-params">(Process)</span>:</span>
    <span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">__init__</span><span class="hljs-params">(self, pipe)</span>:</span>
        Process.__init__(self)
        self.pipe = pipe

    <span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">run</span><span class="hljs-params">(self)</span>:</span>
        print(<span class="hljs-string">f'Producer Received: <span class="hljs-subst">{self.pipe.recv()}</span>'</span>)
        self.pipe.send(<span class="hljs-string">'Producer Words'</span>)

<span class="hljs-keyword">if</span> __name__ == <span class="hljs-string">'__main__'</span>:
    pipe = Pipe()
    p = Producer(pipe[<span class="hljs-number">0</span>])
    c = Consumer(pipe[<span class="hljs-number">1</span>])
    p.daemon = c.daemon = <span class="hljs-literal">True</span>
    p.start()
    c.start()
    p.join()
    c.join()
    print(<span class="hljs-string">'Main Process Ended'</span>)
</code></pre>
在这个例子里我们声明了一个默认为双向的管道，然后将管道的两端分别传给两个进程。两个进程互相收发。观察一下结果：
<pre><code data-language="python" class="lang-python">Producer Received: Consumer Words
Consumer Received: Producer Words
Main Process Ended
</code></pre>
管道 Pipe 就像进程之间搭建的桥梁，利用它我们就可以很方便地实现进程间通信了。

##进程池

在前面，我们讲了可以使用 Process 来创建进程，同时也讲了如何用 Semaphore 来控制进程的并发执行数量。
假如现在我们遇到这么一个问题，我有 10000 个任务，每个任务需要启动一个进程来执行，并且一个进程运行完毕之后要紧接着启动下一个进程，同时我还需要控制进程的并发数量，不能并发太高，不然 CPU 处理不过来（如果同时运行的进程能维持在一个最高恒定值当然利用率是最高的）。
那么我们该如何来实现这个需求呢？
用 Process 和 Semaphore 可以实现，但是实现起来比较我们可以用 Process 和 Semaphore 解决问题，但是实现起来比较烦琐。而这种需求在平时又是非常常见的。此时，我们就可以派上进程池了，即 multiprocessing 中的 Pool。
Pool 可以提供指定数量的进程，供用户调用，当有新的请求提交到 pool 中时，如果池还没有满，就会创建一个新的进程用来执行该请求；但如果池中的进程数已经达到规定最大值，那么该请求就会等待，直到池中有进程结束，才会创建新的进程来执行它。
我们用一个实例来实现一下，代码如下：
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span> multiprocessing <span class="hljs-keyword">import</span> Pool
<span class="hljs-keyword">import</span> time


<span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">function</span><span class="hljs-params">(index)</span>:</span>
    print(<span class="hljs-string">f'Start process: <span class="hljs-subst">{index}</span>'</span>)
    time.sleep(<span class="hljs-number">3</span>)
    print(<span class="hljs-string">f'End process <span class="hljs-subst">{index}</span>'</span>, )


<span class="hljs-keyword">if</span> __name__ == <span class="hljs-string">'__main__'</span>:
    pool = Pool(processes=<span class="hljs-number">3</span>)
    <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> range(<span class="hljs-number">4</span>):
        pool.apply_async(function, args=(i,))

    print(<span class="hljs-string">'Main Process started'</span>)
    pool.close()
    pool.join()
    print(<span class="hljs-string">'Main Process ended'</span>)
</code></pre>
在这个例子中我们声明了一个大小为 3 的进程池，通过 processes 参数来指定，如果不指定，那么会自动根据处理器内核来分配进程数。接着我们使用 apply_async 方法将进程添加进去，args 可以用来传递参数。
运行结果如下：
<pre><code data-language="python" class="lang-python">Main Process started
Start process: <span class="hljs-number">0</span>
Start process: <span class="hljs-number">1</span>
Start process: <span class="hljs-number">2</span>
End process <span class="hljs-number">0</span>
End process <span class="hljs-number">1</span>
End process <span class="hljs-number">2</span>
Start process: <span class="hljs-number">3</span>
End process <span class="hljs-number">3</span>
Main Process ended
</code></pre>
进程池大小为 3，所以最初可以看到有 3 个进程同时执行，第4个进程在等待，在有进程运行完毕之后，第4个进程马上跟着运行，出现了如上的运行效果。
最后，我们要记得调用 close 方法来关闭进程池，使其不再接受新的任务，然后调用 join 方法让主进程等待子进程的退出，等子进程运行完毕之后，主进程接着运行并结束。
不过上面的写法多少有些烦琐，这里再介绍进程池一个更好用的 map 方法，可以将上述写法简化很多。
map 方法是怎么用的呢？第一个参数就是要启动的进程对应的执行方法，第 2 个参数是一个可迭代对象，其中的每个元素会被传递给这个执行方法。
举个例子：现在我们有一个 list，里面包含了很多 URL，另外我们也定义了一个方法用来抓取每个 URL 内容并解析，那么我们可以直接在 map 的第一个参数传入方法名，第 2 个参数传入 URL 数组。
我们用一个实例来感受一下：
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span> multiprocessing <span class="hljs-keyword">import</span> Pool
<span class="hljs-keyword">import</span> urllib.request
<span class="hljs-keyword">import</span> urllib.error


<span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">scrape</span><span class="hljs-params">(url)</span>:</span>
    <span class="hljs-keyword">try</span>:
        urllib.request.urlopen(url)
        print(<span class="hljs-string">f'URL <span class="hljs-subst">{url}</span> Scraped'</span>)
    <span class="hljs-keyword">except</span> (urllib.error.HTTPError, urllib.error.URLError):
        print(<span class="hljs-string">f'URL <span class="hljs-subst">{url}</span> not Scraped'</span>)


<span class="hljs-keyword">if</span> __name__ == <span class="hljs-string">'__main__'</span>:
    pool = Pool(processes=<span class="hljs-number">3</span>)
    urls = [
        <span class="hljs-string">'https://www.baidu.com'</span>,
        <span class="hljs-string">'http://www.meituan.com/'</span>,
        <span class="hljs-string">'http://blog.csdn.net/'</span>,
        <span class="hljs-string">'http://xxxyxxx.net'</span>
    ]
    pool.map(scrape, urls)
    pool.close()
</code></pre>
这个例子中我们先定义了一个 scrape 方法，它接收一个参数 url，这里就是请求了一下这个链接，然后输出爬取成功的信息，如果发生错误，则会输出爬取失败的信息。
首先我们要初始化一个 Pool，指定进程数为 3。然后我们声明一个 urls 列表，接着我们调用了 map 方法，第 1 个参数就是进程对应的执行方法，第 2 个参数就是 urls 列表，map 方法会依次将 urls 的每个元素作为 scrape 的参数传递并启动一个新的进程，加到进程池中执行。
运行结果如下：
<pre><code data-language="python" class="lang-python">URL https://www.baidu.com Scraped
URL http://xxxyxxx.net <span class="hljs-keyword">not</span> Scraped
URL http://blog.csdn.net/ Scraped
URL http://www.meituan.com/ Scraped
</code></pre>
这样，我们就可以实现 3 个进程并行运行。不同的进程相互独立地输出了对应的爬取结果。
可以看到，我们利用 Pool 的 map 方法非常方便地实现了多进程的执行。后面我们也会在实战案例中结合进程池来实现数据的爬取。

