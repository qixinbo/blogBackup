---
title: 52讲轻松搞定网络爬虫笔记9
tags: [Web Crawler]
categories: data analysis
date: 2023-1-25
---

# 资料
[52讲轻松搞定网络爬虫](https://kaiwu.lagou.com/course/courseInfo.htm?courseId=46#/sale)


# 无所不知的Scrapy爬虫框架的介绍
<p data-nodeid="1911">在前面编写爬虫的时候，如果我们使用 requests、aiohttp 等库，需要从头至尾把爬虫完整地实现一遍，比如说异常处理、爬取调度等，如果写的多了，的确会比较麻烦。</p>

<p data-nodeid="1242">那么有没有什么办法可以提升我们编写爬虫的效率呢？当然是有的，那就是利用现有的爬虫框架。</p>
<p data-nodeid="1243">说到 Python 的爬虫框架，Scrapy 当之无愧是最流行最强大的框架了。本节我们就来初步认识一下 Scrapy，后面的课时我们会对 Scrapy 的功能模块进行详细介绍。</p>
<h3 data-nodeid="2045" class="">Scrapy 介绍</h3>

<p data-nodeid="1245">Scrapy 是一个基于 Twisted 的异步处理框架，是纯 Python 实现的爬虫框架，其架构清晰，模块之间的耦合程度低，可扩展性极强，可以灵活完成各种需求。我们只需要定制开发几个模块就可以轻松实现一个爬虫。</p>
<p data-nodeid="2567">首先我们来看下 Scrapy 框架的架构，如图所示：</p>
<p data-nodeid="2568" class=""><img src="https://s0.lgstatic.com/i/image/M00/29/B6/Ciqc1F77DzyAOptlAAJygBiwVD4072.png" alt="image (3).png" data-nodeid="2576"></p>


<p data-nodeid="1248">它可以分为如下的几个部分。</p>
<ul data-nodeid="1249">
<li data-nodeid="1250">
<p data-nodeid="1251">Engine（引擎）：用来处理整个系统的数据流处理、触发事务，是整个框架的核心。</p>
</li>
<li data-nodeid="1252">
<p data-nodeid="1253">Item（项目）：定义了爬取结果的数据结构，爬取的数据会被赋值成该对象。</p>
</li>
<li data-nodeid="1254">
<p data-nodeid="1255">Scheduler（调度器）：用来接受引擎发过来的请求并加入队列中，并在引擎再次请求的时候提供给引擎。</p>
</li>
<li data-nodeid="1256">
<p data-nodeid="1257">Downloader（下载器）：用于下载网页内容，并将网页内容返回给蜘蛛。</p>
</li>
<li data-nodeid="1258">
<p data-nodeid="1259">Spiders（蜘蛛）：其内定义了爬取的逻辑和网页的解析规则，它主要负责解析响应并生成提取结果和新的请求。</p>
</li>
<li data-nodeid="1260">
<p data-nodeid="1261">Item Pipeline（项目管道）：负责处理由蜘蛛从网页中抽取的项目，它的主要任务是清洗、验证和存储数据。</p>
</li>
<li data-nodeid="1262">
<p data-nodeid="1263">Downloader Middlewares（下载器中间件）：位于引擎和下载器之间的钩子框架，主要是处理引擎与下载器之间的请求及响应。</p>
</li>
<li data-nodeid="1264">
<p data-nodeid="1265">Spider Middlewares（蜘蛛中间件）：位于引擎和蜘蛛之间的钩子框架，主要工作是处理蜘蛛输入的响应和输出的结果及新的请求。</p>
</li>
</ul>
<p data-nodeid="1266">初看起来的确比较懵，不过不用担心，我们在后文会结合案例来对 Scrapy 的功能模块进行介绍，相信你会慢慢地理解各个模块的含义及功能。</p>
<h3 data-nodeid="2849" class="">数据流</h3>

<p data-nodeid="1268">了解了架构，下一步就是要了解它是怎样进行数据爬取和处理的，所以我们接下来需要了解 Scrapy 的数据流机制。</p>
<p data-nodeid="1269">Scrapy 中的数据流由引擎控制，其过程如下：</p>
<ul data-nodeid="1270">
<li data-nodeid="1271">
<p data-nodeid="1272">Engine 首先打开一个网站，找到处理该网站的 Spider 并向该 Spider 请求第一个要爬取的 URL。</p>
</li>
<li data-nodeid="1273">
<p data-nodeid="1274">Engine 从 Spider 中获取到第一个要爬取的 URL 并通过 Scheduler 以 Request 的形式调度。</p>
</li>
<li data-nodeid="1275">
<p data-nodeid="1276">Engine 向 Scheduler 请求下一个要爬取的 URL。</p>
</li>
<li data-nodeid="1277">
<p data-nodeid="1278">Scheduler 返回下一个要爬取的 URL 给 Engine，Engine 将 URL 通过 Downloader Middlewares 转发给 Downloader 下载。</p>
</li>
<li data-nodeid="1279">
<p data-nodeid="1280">一旦页面下载完毕， Downloader 生成一个该页面的 Response，并将其通过 Downloader Middlewares 发送给 Engine。</p>
</li>
<li data-nodeid="1281">
<p data-nodeid="1282">Engine 从下载器中接收到 Response 并通过 Spider Middlewares 发送给 Spider 处理。</p>
</li>
<li data-nodeid="1283">
<p data-nodeid="1284">Spider 处理 Response 并返回爬取到的 Item 及新的 Request 给 Engine。</p>
</li>
<li data-nodeid="1285">
<p data-nodeid="1286">Engine 将 Spider 返回的 Item 给 Item Pipeline，将新的 Request 给 Scheduler。</p>
</li>
<li data-nodeid="1287">
<p data-nodeid="1288">重复第二步到最后一步，直到  Scheduler 中没有更多的 Request，Engine 关闭该网站，爬取结束。</p>
</li>
</ul>
<p data-nodeid="1289">通过多个组件的相互协作、不同组件完成工作的不同、组件对异步处理的支持，Scrapy 最大限度地利用了网络带宽，大大提高了数据爬取和处理的效率。</p>
<h3 data-nodeid="3123" class="">安装</h3>

<p data-nodeid="1291">了解了 Scrapy 的基本情况之后，下一步让我们来动手安装一下吧。</p>
<p data-nodeid="1292">Scrapy 的安装方法当然首推官方文档，其地址为：<a href="https://docs.scrapy.org/en/latest/intro/install.html" data-nodeid="1354">https://docs.scrapy.org/en/latest/intro/install.html</a>，另外也可以参考 <a href="https://cuiqingcai.com/5421.html" data-nodeid="1358">https://cuiqingcai.com/5421.html</a>。</p>
<p data-nodeid="1293">安装完成之后，如果可以正常使用 scrapy 命令，那就是可以了。</p>
<h3 data-nodeid="3397" class="">项目结构</h3>

<p data-nodeid="1295">既然 Scrapy 是框架，那么 Scrapy 一定帮我们预先配置好了很多可用的组件和编写爬虫时所用的脚手架，也就是预生成一个项目框架，我们可以基于这个框架来快速编写爬虫。</p>
<p data-nodeid="1296">Scrapy 框架是通过命令行来创建项目的，创建项目的命令如下：</p>
<pre class="lang-java" data-nodeid="3808"><code data-language="java">scrapy startproject demo
</code></pre>


<p data-nodeid="1298">执行完成之后，在当前运行目录下便会出现一个文件夹，叫作 demo，这就是一个 Scrapy 项目框架，我们可以基于这个项目框架来编写爬虫。<br>
项目文件结构如下所示：</p>
<pre class="lang-yaml te-preview-highlight" data-nodeid="15684"><code data-language="yaml"><span class="hljs-string">scrapy.cfg</span>
<span class="hljs-string">project/</span>
 &nbsp;  <span class="hljs-string">__init__.py</span>
 &nbsp;  <span class="hljs-string">items.py</span>
 &nbsp;  <span class="hljs-string">pipelines.py</span>
 &nbsp;  <span class="hljs-string">settings.py</span>
 &nbsp;  <span class="hljs-string">middlewares.py</span>
 &nbsp;  <span class="hljs-string">spiders/</span>
 &nbsp; &nbsp; &nbsp;  <span class="hljs-string">__init__.py</span>
 &nbsp; &nbsp; &nbsp;  <span class="hljs-string">spider1.py</span>
 &nbsp; &nbsp; &nbsp;  <span class="hljs-string">spider2.py</span>
 &nbsp; &nbsp; &nbsp;  <span class="hljs-string">...</span>
</code></pre>












































<p data-nodeid="1300">在此要将各个文件的功能描述如下：</p>
<ul data-nodeid="1301">
<li data-nodeid="1302">
<p data-nodeid="1303">scrapy.cfg：它是 Scrapy 项目的配置文件，其内定义了项目的配置文件路径、部署相关信息等内容。</p>
</li>
<li data-nodeid="1304">
<p data-nodeid="1305">items.py：它定义 Item 数据结构，所有的 Item 的定义都可以放这里。</p>
</li>
<li data-nodeid="1306">
<p data-nodeid="1307">pipelines.py：它定义 Item Pipeline 的实现，所有的 Item Pipeline 的实现都可以放这里。</p>
</li>
<li data-nodeid="1308">
<p data-nodeid="1309">settings.py：它定义项目的全局配置。</p>
</li>
<li data-nodeid="1310">
<p data-nodeid="1311">middlewares.py：它定义 Spider Middlewares 和 Downloader Middlewares 的实现。</p>
</li>
<li data-nodeid="1312">
<p data-nodeid="1313">spiders：其内包含一个个 Spider 的实现，每个 Spider 都有一个文件。</p>
</li>
</ul>
<p data-nodeid="1314">好了，到现在为止我们就大体知道了 Scrapy 的基本架构并实操创建了一个 Scrapy 项目，后面我们会详细了解 Scrapy 的用法，感受它的强大，下节课见。</p>


# 初窥门路Scrapy的基本使用
<p data-nodeid="1060">接下来介绍一个简单的项目，完成一遍 Scrapy 抓取流程。通过这个过程，我们可以对 Scrapy 的基本用法和原理有大体了解。</p>



<h3 data-nodeid="1760" class="">本节目标</h3>

<p data-nodeid="5">本节要完成的任务如下。</p>
<ul data-nodeid="6">
<li data-nodeid="7">
<p data-nodeid="8">创建一个 Scrapy 项目。</p>
</li>
<li data-nodeid="9">
<p data-nodeid="10">创建一个 Spider 来抓取站点和处理数据。</p>
</li>
<li data-nodeid="11">
<p data-nodeid="12">通过命令行将抓取的内容导出。</p>
</li>
<li data-nodeid="13">
<p data-nodeid="14">将抓取的内容保存到 MongoDB 数据库。</p>
</li>
</ul>
<p data-nodeid="15">本节抓取的目标站点为 <a href="http://quotes.toscrape.com/" data-nodeid="162">http://quotes.toscrape.com/</a>。</p>
<h3 data-nodeid="2460" class="">准备工作</h3>

<p data-nodeid="17">我们需要安装好 Scrapy 框架、MongoDB 和 PyMongo 库。如果尚未安装，请参照之前几节的安装说明。</p>
<h3 data-nodeid="3160" class="">创建项目</h3>

<p data-nodeid="19">创建一个 Scrapy 项目，项目文件可以直接用 scrapy 命令生成，命令如下所示：</p>
<pre class="lang-java" data-nodeid="4210"><code data-language="java">scrapy startproject tutorial
</code></pre>


<p data-nodeid="21">这个命令可以在任意文件夹运行。如果提示权限问题，可以加 sudo 运行该命令。这个命令将会创建一个名为 tutorial 的文件夹，文件夹结构如下所示：</p>
<pre class="lang-sql" data-nodeid="15394"><code data-language="sql">scrapy.cfg &nbsp; &nbsp; <span class="hljs-comment"># Scrapy 部署时的配置文件</span>
tutorial &nbsp; &nbsp; &nbsp; &nbsp; <span class="hljs-comment"># 项目的模块，引入的时候需要从这里引入</span>
 &nbsp;  __init__.py &nbsp; &nbsp;
 &nbsp;  items.py &nbsp; &nbsp; <span class="hljs-comment"># Items 的定义，定义爬取的数据结构</span>
 &nbsp;  middlewares.py &nbsp; <span class="hljs-comment"># Middlewares 的定义，定义爬取时的中间件</span>
 &nbsp;  pipelines.py &nbsp; &nbsp; &nbsp; <span class="hljs-comment"># Pipelines 的定义，定义数据管道</span>
 &nbsp;  settings.py &nbsp; &nbsp; &nbsp; <span class="hljs-comment"># 配置文件</span>
 &nbsp;  spiders &nbsp; &nbsp; &nbsp; &nbsp; <span class="hljs-comment"># 放置 Spiders 的文件夹</span>
 &nbsp;  __init__.py
</code></pre>




<h3 data-nodeid="16093" class="">创建 Spider</h3>

<p data-nodeid="24">Spider 是自己定义的类，Scrapy 用它从网页里抓取内容，并解析抓取的结果。不过这个类必须继承 Scrapy 提供的 Spider 类 scrapy.Spider，还要定义 Spider 的名称和起始请求，以及怎样处理爬取后的结果的方法。</p>
<p data-nodeid="25">你也可以使用命令行创建一个 Spider。比如要生成 Quotes 这个 Spider，可以执行如下命令：</p>
<pre class="lang-java" data-nodeid="31472"><code data-language="java">cd tutorial
scrapy genspider quotes &nbsp; &nbsp; 
</code></pre>



<p data-nodeid="27">进入刚才创建的 tutorial 文件夹，然后执行 genspider 命令。第一个参数是 Spider 的名称，第二个参数是网站域名。执行完毕之后，spiders 文件夹中多了一个 quotes.py，它就是刚刚创建的 Spider，内容如下所示：</p>
<pre class="lang-dart" data-nodeid="41258"><code data-language="dart"><span class="hljs-keyword">import</span> scrapy
​
<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">QuotesSpider</span>(<span class="hljs-title">scrapy</span>.<span class="hljs-title">Spider</span>):
 &nbsp; &nbsp;<span class="hljs-title">name</span> = "<span class="hljs-title">quotes</span>"
 &nbsp; &nbsp;<span class="hljs-title">allowed_domains</span> = ["<span class="hljs-title">quotes</span>.<span class="hljs-title">toscrape</span>.<span class="hljs-title">com</span>"]
 &nbsp; &nbsp;<span class="hljs-title">start_urls</span> = ['<span class="hljs-title">http</span>://<span class="hljs-title">quotes</span>.<span class="hljs-title">toscrape</span>.<span class="hljs-title">com</span>/']
​
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">parse</span>(<span class="hljs-title">self</span>, <span class="hljs-title">response</span>):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">pass</span>
</span></code></pre>


<p data-nodeid="29">这里有三个属性——name、allowed_domains 和 start_urls，还有一个方法 parse。</p>
<ul data-nodeid="30">
<li data-nodeid="31">
<p data-nodeid="32">name：它是每个项目唯一的名字，用来区分不同的 Spider。</p>
</li>
<li data-nodeid="33">
<p data-nodeid="34">allowed_domains：它是允许爬取的域名，如果初始或后续的请求链接不是这个域名下的，则请求链接会被过滤掉。</p>
</li>
<li data-nodeid="35">
<p data-nodeid="36">start_urls：它包含了 Spider 在启动时爬取的 url 列表，初始请求是由它来定义的。</p>
</li>
<li data-nodeid="37">
<p data-nodeid="38">parse：它是 Spider 的一个方法。默认情况下，被调用时 start_urls 里面的链接构成的请求完成下载执行后，返回的响应就会作为唯一的参数传递给这个函数。该方法负责解析返回的响应、提取数据或者进一步生成要处理的请求。</p>
</li>
</ul>
<h3 data-nodeid="41957" class="">创建 Item</h3>

<p data-nodeid="40">Item 是保存爬取数据的容器，它的使用方法和字典类似。不过，相比字典，Item 多了额外的保护机制，可以避免拼写错误或者定义字段错误。</p>
<p data-nodeid="41">创建 Item 需要继承 scrapy.Item 类，并且定义类型为 scrapy.Field 的字段。观察目标网站，我们可以获取到的内容有 text、author、tags。</p>
<p data-nodeid="42">定义 Item，此时将 items.py 修改如下：</p>
<pre class="lang-dart" data-nodeid="48249"><code data-language="dart"><span class="hljs-keyword">import</span> scrapy
​
<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">QuoteItem</span>(<span class="hljs-title">scrapy</span>.<span class="hljs-title">Item</span>):
​
 &nbsp; &nbsp;<span class="hljs-title">text</span> = <span class="hljs-title">scrapy</span>.<span class="hljs-title">Field</span>()
 &nbsp; &nbsp;<span class="hljs-title">author</span> = <span class="hljs-title">scrapy</span>.<span class="hljs-title">Field</span>()
 &nbsp; &nbsp;<span class="hljs-title">tags</span> = <span class="hljs-title">scrapy</span>.<span class="hljs-title">Field</span>()
</span></code></pre>


<p data-nodeid="44">这里定义了三个字段，将类的名称修改为 QuoteItem，接下来爬取时我们会使用到这个 Item。</p>
<h3 data-nodeid="48948" class="">解析 Response</h3>

<p data-nodeid="46">前面我们看到，parse 方法的参数 response 是 start_urls 里面的链接爬取后的结果。所以在 parse 方法中，我们可以直接对 response 变量包含的内容进行解析，比如浏览请求结果的网页源代码，或者进一步分析源代码内容，或者找出结果中的链接而得到下一个请求。</p>
<p data-nodeid="47">我们可以看到网页中既有我们想要的结果，又有下一页的链接，这两部分内容我们都要进行处理。</p>
<p data-nodeid="50338">首先看看网页结构，如图所示。每一页都有多个 class 为 quote 的区块，每个区块内都包含 text、author、tags。那么我们先找出所有的 quote，然后提取每一个 quote 中的内容。</p>
<p data-nodeid="50339" class=""><img src="https://s0.lgstatic.com/i/image/M00/2B/8B/CgqCHl7-qgqALNV_AAWr2qeUbbw688.png" alt="image (5).png" data-nodeid="50347"></p>


<p data-nodeid="50">提取的方式可以是 CSS 选择器或 XPath 选择器。在这里我们使用 CSS 选择器进行选择，parse 方法的改写如下所示：</p>
<pre class="lang-java" data-nodeid="51054"><code data-language="java"><span class="hljs-function">def <span class="hljs-title">parse</span><span class="hljs-params">(self, response)</span>:
 &nbsp; &nbsp;quotes </span>= response.css(<span class="hljs-string">'.quote'</span>)
 &nbsp; &nbsp;<span class="hljs-keyword">for</span> quote in quotes:
 &nbsp; &nbsp; &nbsp; &nbsp;text = quote.css(<span class="hljs-string">'.text::text'</span>).extract_first()
 &nbsp; &nbsp; &nbsp; &nbsp;author = quote.css(<span class="hljs-string">'.author::text'</span>).extract_first()
 &nbsp; &nbsp; &nbsp; &nbsp;tags = quote.css(<span class="hljs-string">'.tags .tag::text'</span>).extract()
</code></pre>

<p data-nodeid="51761">这里首先利用选择器选取所有的 quote，并将其赋值为 quotes 变量，然后利用 for 循环对每个 quote 遍历，解析每个 quote 的内容。</p>
<p data-nodeid="51762">对 text 来说，观察到它的 class 为 text，所以可以用 .text 选择器来选取，这个结果实际上是整个带有标签的节点，要获取它的正文内容，可以加 ::text 来获取。这时的结果是长度为 1 的列表，所以还需要用 extract_first 方法来获取第一个元素。而对于 tags 来说，由于我们要获取所有的标签，所以用 extract 方法获取整个列表即可。</p>

<p data-nodeid="53">以第一个 quote 的结果为例，各个选择方法及结果的说明如下内容。</p>
<p data-nodeid="54">源码如下：</p>
<pre class="lang-dart" data-nodeid="55301"><code data-language="dart">&lt;div <span class="hljs-class"><span class="hljs-keyword">class</span>="<span class="hljs-title">quote</span>" <span class="hljs-title">itemscope</span>=""<span class="hljs-title">itemtype</span>="<span class="hljs-title">http</span>://<span class="hljs-title">schema</span>.<span class="hljs-title">org</span>/<span class="hljs-title">CreativeWork</span>"&gt;
 &nbsp; &nbsp; &nbsp; &nbsp;&lt;<span class="hljs-title">span</span> <span class="hljs-title">class</span>="<span class="hljs-title">text</span>" <span class="hljs-title">itemprop</span>="<span class="hljs-title">text</span>"&gt;“<span class="hljs-title">The</span> <span class="hljs-title">world</span> <span class="hljs-title">as</span> <span class="hljs-title">we</span> <span class="hljs-title">have</span> <span class="hljs-title">created</span> <span class="hljs-title">it</span> <span class="hljs-title">is</span> <span class="hljs-title">a</span> <span class="hljs-title">process</span> <span class="hljs-title">of</span> <span class="hljs-title">our</span> <span class="hljs-title">thinking</span>. <span class="hljs-title">It</span> <span class="hljs-title">cannot</span> <span class="hljs-title">be</span> <span class="hljs-title">changed</span> <span class="hljs-title">without</span> <span class="hljs-title">changing</span> <span class="hljs-title">our</span> <span class="hljs-title">thinking</span>.”&lt;/<span class="hljs-title">span</span>&gt;
 &nbsp; &nbsp; &nbsp; &nbsp;&lt;<span class="hljs-title">span</span>&gt;<span class="hljs-title">by</span> &lt;<span class="hljs-title">small</span> <span class="hljs-title">class</span>="<span class="hljs-title">author</span>" <span class="hljs-title">itemprop</span>="<span class="hljs-title">author</span>"&gt;<span class="hljs-title">Albert</span> <span class="hljs-title">Einstein</span>&lt;/<span class="hljs-title">small</span>&gt;
 &nbsp; &nbsp; &nbsp; &nbsp;&lt;<span class="hljs-title">a</span> <span class="hljs-title">href</span>="/<span class="hljs-title">author</span>/<span class="hljs-title">Albert</span>-<span class="hljs-title">Einstein</span>"&gt;(<span class="hljs-title">about</span>)&lt;/<span class="hljs-title">a</span>&gt;
 &nbsp; &nbsp; &nbsp; &nbsp;&lt;/<span class="hljs-title">span</span>&gt;
 &nbsp; &nbsp; &nbsp; &nbsp;&lt;<span class="hljs-title">div</span> <span class="hljs-title">class</span>="<span class="hljs-title">tags</span>"&gt;
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  <span class="hljs-title">Tags</span>:
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&lt;<span class="hljs-title">meta</span> <span class="hljs-title">class</span>="<span class="hljs-title">keywords</span>" <span class="hljs-title">itemprop</span>="<span class="hljs-title">keywords</span>" <span class="hljs-title">content</span>="<span class="hljs-title">change</span>,<span class="hljs-title">deep</span>-<span class="hljs-title">thoughts</span>,<span class="hljs-title">thinking</span>,<span class="hljs-title">world</span>"&gt; 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&lt;<span class="hljs-title">a</span> <span class="hljs-title">class</span>="<span class="hljs-title">tag</span>" <span class="hljs-title">href</span>="/<span class="hljs-title">tag</span>/<span class="hljs-title">change</span>/<span class="hljs-title">page</span>/1/"&gt;<span class="hljs-title">change</span>&lt;/<span class="hljs-title">a</span>&gt;
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&lt;<span class="hljs-title">a</span> <span class="hljs-title">class</span>="<span class="hljs-title">tag</span>" <span class="hljs-title">href</span>="/<span class="hljs-title">tag</span>/<span class="hljs-title">deep</span>-<span class="hljs-title">thoughts</span>/<span class="hljs-title">page</span>/1/"&gt;<span class="hljs-title">deep</span>-<span class="hljs-title">thoughts</span>&lt;/<span class="hljs-title">a</span>&gt;
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&lt;<span class="hljs-title">a</span> <span class="hljs-title">class</span>="<span class="hljs-title">tag</span>" <span class="hljs-title">href</span>="/<span class="hljs-title">tag</span>/<span class="hljs-title">thinking</span>/<span class="hljs-title">page</span>/1/"&gt;<span class="hljs-title">thinking</span>&lt;/<span class="hljs-title">a</span>&gt;
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&lt;<span class="hljs-title">a</span> <span class="hljs-title">class</span>="<span class="hljs-title">tag</span>" <span class="hljs-title">href</span>="/<span class="hljs-title">tag</span>/<span class="hljs-title">world</span>/<span class="hljs-title">page</span>/1/"&gt;<span class="hljs-title">world</span>&lt;/<span class="hljs-title">a</span>&gt;
 &nbsp; &nbsp; &nbsp; &nbsp;&lt;/<span class="hljs-title">div</span>&gt;
 &nbsp; &nbsp;&lt;/<span class="hljs-title">div</span>&gt;
</span></code></pre>





<p data-nodeid="56">不同选择器的返回结果如下。</p>
<h4 data-nodeid="57">quote.css('.text')</h4>
<pre class="lang-java" data-nodeid="56008"><code data-language="java">[&lt;Selector xpath=<span class="hljs-string">"descendant-or-self::*[@class and contains(concat(' ', normalize-space(@class), ' '), ' text ')]"</span>data=<span class="hljs-string">'&lt;span class="text"itemprop="text"&gt;“The '</span>&gt;]
</code></pre>

<h4 data-nodeid="59">quote.css('.text::text')</h4>
<pre class="lang-java" data-nodeid="56715"><code data-language="java">[&lt;Selector xpath=<span class="hljs-string">"descendant-or-self::*[@class and contains(concat(' ', normalize-space(@class), ' '), ' text ')]/text()"</span>data=<span class="hljs-string">'“The world as we have created it is a pr'</span>&gt;]
</code></pre>

<h4 data-nodeid="61">quote.css('.text').extract()</h4>
<pre class="lang-java" data-nodeid="57422"><code data-language="java">[<span class="hljs-string">'&lt;span class="text"itemprop="text"&gt;“The world as we have created it is a process of our thinking. It cannot be changed without changing our thinking.”&lt;/span&gt;'</span>]
</code></pre>

<h4 data-nodeid="63">quote.css('.text::text').extract()</h4>
<pre class="lang-java" data-nodeid="58129"><code data-language="java">[<span class="hljs-string">'“The world as we have created it is a process of our thinking. It cannot be changed without changing our thinking.”'</span>]
</code></pre>

<h4 data-nodeid="65">quote.css('.text::text').extract_first()</h4>
<pre class="lang-js" data-nodeid="59543"><code data-language="js">“The world <span class="hljs-keyword">as</span> we have created it is a process <span class="hljs-keyword">of</span> our thinking. It cannot be changed without changing our thinking.”
</code></pre>


<p data-nodeid="67">所以，对于 text，获取结果的第一个元素即可，所以使用 extract_first 方法，对于 tags，要获取所有结果组成的列表，所以使用 extract 方法。</p>
<h3 data-nodeid="60250" class="">使用 Item</h3>

<p data-nodeid="69">上文定义了 Item，接下来就要使用它了。Item 可以理解为一个字典，不过在声明的时候需要实例化。然后依次用刚才解析的结果赋值 Item 的每一个字段，最后将 Item 返回即可。</p>
<p data-nodeid="70">QuotesSpider 的改写如下所示：</p>
<pre class="lang-dart" data-nodeid="65907"><code data-language="dart"><span class="hljs-keyword">import</span> scrapy
from tutorial.items <span class="hljs-keyword">import</span> QuoteItem
​
<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">QuotesSpider</span>(<span class="hljs-title">scrapy</span>.<span class="hljs-title">Spider</span>):
 &nbsp; &nbsp;<span class="hljs-title">name</span> = "<span class="hljs-title">quotes</span>"
 &nbsp; &nbsp;<span class="hljs-title">allowed_domains</span> = ["<span class="hljs-title">quotes</span>.<span class="hljs-title">toscrape</span>.<span class="hljs-title">com</span>"]
 &nbsp; &nbsp;<span class="hljs-title">start_urls</span> = ['<span class="hljs-title">http</span>://<span class="hljs-title">quotes</span>.<span class="hljs-title">toscrape</span>.<span class="hljs-title">com</span>/']
​
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">parse</span>(<span class="hljs-title">self</span>, <span class="hljs-title">response</span>):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">quotes</span> = <span class="hljs-title">response</span>.<span class="hljs-title">css</span>('.<span class="hljs-title">quote</span>')
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">for</span> <span class="hljs-title">quote</span> <span class="hljs-title">in</span> <span class="hljs-title">quotes</span>:
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">item</span> = <span class="hljs-title">QuoteItem</span>()
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">item</span>['<span class="hljs-title">text</span>'] = <span class="hljs-title">quote</span>.<span class="hljs-title">css</span>('.<span class="hljs-title">text</span>::<span class="hljs-title">text</span>').<span class="hljs-title">extract_first</span>()
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">item</span>['<span class="hljs-title">author</span>'] = <span class="hljs-title">quote</span>.<span class="hljs-title">css</span>('.<span class="hljs-title">author</span>::<span class="hljs-title">text</span>').<span class="hljs-title">extract_first</span>()
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">item</span>['<span class="hljs-title">tags</span>'] = <span class="hljs-title">quote</span>.<span class="hljs-title">css</span>('.<span class="hljs-title">tags</span> .<span class="hljs-title">tag</span>::<span class="hljs-title">text</span>').<span class="hljs-title">extract</span>()
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">yield</span> <span class="hljs-title">item</span>
</span></code></pre>








<p data-nodeid="72">如此一来，首页的所有内容被解析出来，并被赋值成了一个个 QuoteItem。</p>
<h3 data-nodeid="66614" class="">后续 Request</h3>

<p data-nodeid="74">上面的操作实现了从初始页面抓取内容。那么，下一页的内容该如何抓取？这就需要我们从当前页面中找到信息来生成下一个请求，然后在下一个请求的页面里找到信息再构造下一个请求。这样循环往复迭代，从而实现整站的爬取。</p>
<p data-nodeid="68020">将刚才的页面拉到最底部，如图所示。</p>
<p data-nodeid="68021" class=""><img src="https://s0.lgstatic.com/i/image/M00/2B/8B/CgqCHl7-qmWAfK6lAAKdlFmjiVk937.png" alt="image (6).png" data-nodeid="68029"></p>


<p data-nodeid="77">有一个 Next 按钮，查看一下源代码，可以发现它的链接是 /page/2/，实际上全链接就是：<a href="http://quotes.toscrape.com/page/2" data-nodeid="254">http://quotes.toscrape.com/page/2</a>，通过这个链接我们就可以构造下一个请求。</p>
<p data-nodeid="78">构造请求时需要用到 scrapy.Request。这里我们传递两个参数——url 和 callback，这两个参数的说明如下。</p>
<ul data-nodeid="79">
<li data-nodeid="80">
<p data-nodeid="81">url：它是请求链接。</p>
</li>
<li data-nodeid="82">
<p data-nodeid="83">callback：它是回调函数。当指定了该回调函数的请求完成之后，获取到响应，引擎会将该响应作为参数传递给这个回调函数。回调函数进行解析或生成下一个请求，回调函数如上文的 parse() 所示。</p>
</li>
</ul>
<p data-nodeid="84">由于 parse 就是解析 text、author、tags 的方法，而下一页的结构和刚才已经解析的页面结构是一样的，所以我们可以再次使用 parse 方法来做页面解析。</p>
<p data-nodeid="85">接下来我们要做的就是利用选择器得到下一页链接并生成请求，在 parse 方法后追加如下的代码：</p>
<pre class="lang-java" data-nodeid="68744"><code data-language="java">next = response.css(<span class="hljs-string">'.pager .next a::attr(href)'</span>).extract_first()
url = response.urljoin(next)
yield scrapy.Request(url=url, callback=self.parse)
</code></pre>

<p data-nodeid="69459">第一句代码首先通过 CSS 选择器获取下一个页面的链接，即要获取 a 超链接中的 href 属性。这里用到了 ::attr(href) 操作。然后再调用 extract_first 方法获取内容。</p>
<p data-nodeid="69460">第二句代码调用了 urljoin 方法，urljoin() 方法可以将相对 URL 构造成一个绝对的 URL。例如，获取到的下一页地址是 /page/2，urljoin 方法处理后得到的结果就是：<a href="http://quotes.toscrape.com/page/2/" data-nodeid="69467">http://quotes.toscrape.com/page/2/</a>。</p>

<p data-nodeid="88">第三句代码通过 url 和 callback 变量构造了一个新的请求，回调函数 callback 依然使用 parse 方法。这个请求完成后，响应会重新经过 parse 方法处理，得到第二页的解析结果，然后生成第二页的下一页，也就是第三页的请求。这样爬虫就进入了一个循环，直到最后一页。</p>
<p data-nodeid="89">通过几行代码，我们就轻松实现了一个抓取循环，将每个页面的结果抓取下来了。现在，改写之后的整个 Spider 类如下所示：</p>
<pre class="lang-dart" data-nodeid="77333"><code data-language="dart"><span class="hljs-keyword">import</span> scrapy
from tutorial.items <span class="hljs-keyword">import</span> QuoteItem
​
<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">QuotesSpider</span>(<span class="hljs-title">scrapy</span>.<span class="hljs-title">Spider</span>):
 &nbsp; &nbsp;<span class="hljs-title">name</span> = "<span class="hljs-title">quotes</span>"
 &nbsp; &nbsp;<span class="hljs-title">allowed_domains</span> = ["<span class="hljs-title">quotes</span>.<span class="hljs-title">toscrape</span>.<span class="hljs-title">com</span>"]
 &nbsp; &nbsp;<span class="hljs-title">start_urls</span> = ['<span class="hljs-title">http</span>://<span class="hljs-title">quotes</span>.<span class="hljs-title">toscrape</span>.<span class="hljs-title">com</span>/']
​
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">parse</span>(<span class="hljs-title">self</span>, <span class="hljs-title">response</span>):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">quotes</span> = <span class="hljs-title">response</span>.<span class="hljs-title">css</span>('.<span class="hljs-title">quote</span>')
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">for</span> <span class="hljs-title">quote</span> <span class="hljs-title">in</span> <span class="hljs-title">quotes</span>:
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">item</span> = <span class="hljs-title">QuoteItem</span>()
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">item</span>['<span class="hljs-title">text</span>'] = <span class="hljs-title">quote</span>.<span class="hljs-title">css</span>('.<span class="hljs-title">text</span>::<span class="hljs-title">text</span>').<span class="hljs-title">extract_first</span>()
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">item</span>['<span class="hljs-title">author</span>'] = <span class="hljs-title">quote</span>.<span class="hljs-title">css</span>('.<span class="hljs-title">author</span>::<span class="hljs-title">text</span>').<span class="hljs-title">extract_first</span>()
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">item</span>['<span class="hljs-title">tags</span>'] = <span class="hljs-title">quote</span>.<span class="hljs-title">css</span>('.<span class="hljs-title">tags</span> .<span class="hljs-title">tag</span>::<span class="hljs-title">text</span>').<span class="hljs-title">extract</span>()
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">yield</span> <span class="hljs-title">item</span>
​
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">next</span> = <span class="hljs-title">response</span>.<span class="hljs-title">css</span>('.<span class="hljs-title">pager</span> .<span class="hljs-title">next</span> <span class="hljs-title">a</span>::<span class="hljs-title">attr</span>("<span class="hljs-title">href</span>")').<span class="hljs-title">extract_first</span>()
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">url</span> = <span class="hljs-title">response</span>.<span class="hljs-title">urljoin</span>(<span class="hljs-title">next</span>)
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">yield</span> <span class="hljs-title">scrapy</span>.<span class="hljs-title">Request</span>(<span class="hljs-title">url</span>=<span class="hljs-title">url</span>, <span class="hljs-title">callback</span>=<span class="hljs-title">self</span>.<span class="hljs-title">parse</span>)
</span></code></pre>











<h3 data-nodeid="78048" class="">运行</h3>

<p data-nodeid="92">接下来，进入目录，运行如下命令：</p>
<pre class="lang-java" data-nodeid="78764"><code data-language="java">scrapy crawl quotes
</code></pre>

<p data-nodeid="94">就可以看到 Scrapy 的运行结果了。</p>
<pre class="lang-java" data-nodeid="79479"><code data-language="java"><span class="hljs-number">2020</span>-<span class="hljs-number">02</span>-<span class="hljs-number">19</span> <span class="hljs-number">13</span>:<span class="hljs-number">37</span>:<span class="hljs-number">20</span> [scrapy.utils.log] INFO: Scrapy <span class="hljs-number">1.3</span>.<span class="hljs-number">0</span> started (bot: tutorial)
<span class="hljs-number">2020</span>-<span class="hljs-number">02</span>-<span class="hljs-number">19</span> <span class="hljs-number">13</span>:<span class="hljs-number">37</span>:<span class="hljs-number">20</span> [scrapy.utils.log] INFO: Overridden settings: {<span class="hljs-string">'NEWSPIDER_MODULE'</span>: <span class="hljs-string">'tutorial.spiders'</span>, <span class="hljs-string">'SPIDER_MODULES'</span>: [<span class="hljs-string">'tutorial.spiders'</span>], <span class="hljs-string">'ROBOTSTXT_OBEY'</span>: True, <span class="hljs-string">'BOT_NAME'</span>: <span class="hljs-string">'tutorial'</span>}
<span class="hljs-number">2020</span>-<span class="hljs-number">02</span>-<span class="hljs-number">19</span> <span class="hljs-number">13</span>:<span class="hljs-number">37</span>:<span class="hljs-number">20</span> [scrapy.middleware] INFO: Enabled extensions:
[<span class="hljs-string">'scrapy.extensions.logstats.LogStats'</span>,
 <span class="hljs-string">'scrapy.extensions.telnet.TelnetConsole'</span>,
 <span class="hljs-string">'scrapy.extensions.corestats.CoreStats'</span>]
<span class="hljs-number">2020</span>-<span class="hljs-number">02</span>-<span class="hljs-number">19</span> <span class="hljs-number">13</span>:<span class="hljs-number">37</span>:<span class="hljs-number">20</span> [scrapy.middleware] INFO: Enabled downloader middlewares:
[<span class="hljs-string">'scrapy.downloadermiddlewares.robotstxt.RobotsTxtMiddleware'</span>,
 <span class="hljs-string">'scrapy.downloadermiddlewares.httpauth.HttpAuthMiddleware'</span>,
 <span class="hljs-string">'scrapy.downloadermiddlewares.downloadtimeout.DownloadTimeoutMiddleware'</span>,
 <span class="hljs-string">'scrapy.downloadermiddlewares.defaultheaders.DefaultHeadersMiddleware'</span>,
 <span class="hljs-string">'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware'</span>,
 <span class="hljs-string">'scrapy.downloadermiddlewares.retry.RetryMiddleware'</span>,
 <span class="hljs-string">'scrapy.downloadermiddlewares.redirect.MetaRefreshMiddleware'</span>,
 <span class="hljs-string">'scrapy.downloadermiddlewares.httpcompression.HttpCompressionMiddleware'</span>,
 <span class="hljs-string">'scrapy.downloadermiddlewares.redirect.RedirectMiddleware'</span>,
 <span class="hljs-string">'scrapy.downloadermiddlewares.cookies.CookiesMiddleware'</span>,
 <span class="hljs-string">'scrapy.downloadermiddlewares.stats.DownloaderStats'</span>]
<span class="hljs-number">2020</span>-<span class="hljs-number">02</span>-<span class="hljs-number">19</span> <span class="hljs-number">13</span>:<span class="hljs-number">37</span>:<span class="hljs-number">20</span> [scrapy.middleware] INFO: Enabled spider middlewares:
[<span class="hljs-string">'scrapy.spidermiddlewares.httperror.HttpErrorMiddleware'</span>,
 <span class="hljs-string">'scrapy.spidermiddlewares.offsite.OffsiteMiddleware'</span>,
 <span class="hljs-string">'scrapy.spidermiddlewares.referer.RefererMiddleware'</span>,
 <span class="hljs-string">'scrapy.spidermiddlewares.urllength.UrlLengthMiddleware'</span>,
 <span class="hljs-string">'scrapy.spidermiddlewares.depth.DepthMiddleware'</span>]
<span class="hljs-number">2020</span>-<span class="hljs-number">02</span>-<span class="hljs-number">19</span> <span class="hljs-number">13</span>:<span class="hljs-number">37</span>:<span class="hljs-number">20</span> [scrapy.middleware] INFO: Enabled item pipelines:
[]
<span class="hljs-number">2020</span>-<span class="hljs-number">02</span>-<span class="hljs-number">19</span> <span class="hljs-number">13</span>:<span class="hljs-number">37</span>:<span class="hljs-number">20</span> [scrapy.core.engine] INFO: Spider opened
<span class="hljs-number">2020</span>-<span class="hljs-number">02</span>-<span class="hljs-number">19</span> <span class="hljs-number">13</span>:<span class="hljs-number">37</span>:<span class="hljs-number">20</span> [scrapy.extensions.logstats] INFO: Crawled <span class="hljs-number">0</span> pages (at <span class="hljs-number">0</span> pages/min), scraped <span class="hljs-number">0</span> items (at <span class="hljs-number">0</span> items/min)
<span class="hljs-number">2020</span>-<span class="hljs-number">02</span>-<span class="hljs-number">19</span> <span class="hljs-number">13</span>:<span class="hljs-number">37</span>:<span class="hljs-number">20</span> [scrapy.extensions.telnet] DEBUG: Telnet console listening on <span class="hljs-number">127.0</span>.<span class="hljs-number">0.1</span>:<span class="hljs-number">6023</span>
<span class="hljs-number">2020</span>-<span class="hljs-number">02</span>-<span class="hljs-number">19</span> <span class="hljs-number">13</span>:<span class="hljs-number">37</span>:<span class="hljs-number">21</span> [scrapy.core.engine] DEBUG: Crawled (<span class="hljs-number">404</span>) &lt;GET http:<span class="hljs-comment">//quotes.toscrape.com/robots.txt&gt; (referer: None)</span>
<span class="hljs-number">2020</span>-<span class="hljs-number">02</span>-<span class="hljs-number">19</span> <span class="hljs-number">13</span>:<span class="hljs-number">37</span>:<span class="hljs-number">21</span> [scrapy.core.engine] DEBUG: Crawled (<span class="hljs-number">200</span>) &lt;GET http:<span class="hljs-comment">//quotes.toscrape.com/&gt; (referer: None)</span>
<span class="hljs-number">2020</span>-<span class="hljs-number">02</span>-<span class="hljs-number">19</span> <span class="hljs-number">13</span>:<span class="hljs-number">37</span>:<span class="hljs-number">21</span> [scrapy.core.scraper] DEBUG: Scraped from &lt;<span class="hljs-number">200</span> http:<span class="hljs-comment">//quotes.toscrape.com/&gt;</span>
{<span class="hljs-string">'author'</span>: u<span class="hljs-string">'Albert Einstein'</span>,
 <span class="hljs-string">'tags'</span>: [u<span class="hljs-string">'change'</span>, u<span class="hljs-string">'deep-thoughts'</span>, u<span class="hljs-string">'thinking'</span>, u<span class="hljs-string">'world'</span>],
 <span class="hljs-string">'text'</span>: u<span class="hljs-string">'\u201cThe world as we have created it is a process of our thinking. It cannot be changed without changing our thinking.\u201d'</span>}
<span class="hljs-number">2020</span>-<span class="hljs-number">02</span>-<span class="hljs-number">19</span> <span class="hljs-number">13</span>:<span class="hljs-number">37</span>:<span class="hljs-number">21</span> [scrapy.core.scraper] DEBUG: Scraped from &lt;<span class="hljs-number">200</span> http:<span class="hljs-comment">//quotes.toscrape.com/&gt;</span>
{<span class="hljs-string">'author'</span>: u<span class="hljs-string">'J.K. Rowling'</span>,
 <span class="hljs-string">'tags'</span>: [u<span class="hljs-string">'abilities'</span>, u<span class="hljs-string">'choices'</span>],
 <span class="hljs-string">'text'</span>: u<span class="hljs-string">'\u201cIt is our choices, Harry, that show what we truly are, far more than our abilities.\u201d'</span>}
...
<span class="hljs-number">2020</span>-<span class="hljs-number">02</span>-<span class="hljs-number">19</span> <span class="hljs-number">13</span>:<span class="hljs-number">37</span>:<span class="hljs-number">27</span> [scrapy.core.engine] INFO: <span class="hljs-function">Closing <span class="hljs-title">spider</span> <span class="hljs-params">(finished)</span>
2020-02-19 13:37:27 [scrapy.statscollectors] INFO: Dumping Scrapy stats:
</span>{<span class="hljs-string">'downloader/request_bytes'</span>: <span class="hljs-number">2859</span>,
 <span class="hljs-string">'downloader/request_count'</span>: <span class="hljs-number">11</span>,
 <span class="hljs-string">'downloader/request_method_count/GET'</span>: <span class="hljs-number">11</span>,
 <span class="hljs-string">'downloader/response_bytes'</span>: <span class="hljs-number">24871</span>,
 <span class="hljs-string">'downloader/response_count'</span>: <span class="hljs-number">11</span>,
 <span class="hljs-string">'downloader/response_status_count/200'</span>: <span class="hljs-number">10</span>,
 <span class="hljs-string">'downloader/response_status_count/404'</span>: <span class="hljs-number">1</span>,
 <span class="hljs-string">'dupefilter/filtered'</span>: <span class="hljs-number">1</span>,
 <span class="hljs-string">'finish_reason'</span>: <span class="hljs-string">'finished'</span>,
 <span class="hljs-string">'finish_time'</span>: datetime.datetime(<span class="hljs-number">2017</span>, <span class="hljs-number">2</span>, <span class="hljs-number">19</span>, <span class="hljs-number">5</span>, <span class="hljs-number">37</span>, <span class="hljs-number">27</span>, <span class="hljs-number">227438</span>),
 <span class="hljs-string">'item_scraped_count'</span>: <span class="hljs-number">100</span>,
 <span class="hljs-string">'log_count/DEBUG'</span>: <span class="hljs-number">113</span>,
 <span class="hljs-string">'log_count/INFO'</span>: <span class="hljs-number">7</span>,
 <span class="hljs-string">'request_depth_max'</span>: <span class="hljs-number">10</span>,
 <span class="hljs-string">'response_received_count'</span>: <span class="hljs-number">11</span>,
 <span class="hljs-string">'scheduler/dequeued'</span>: <span class="hljs-number">10</span>,
 <span class="hljs-string">'scheduler/dequeued/memory'</span>: <span class="hljs-number">10</span>,
 <span class="hljs-string">'scheduler/enqueued'</span>: <span class="hljs-number">10</span>,
 <span class="hljs-string">'scheduler/enqueued/memory'</span>: <span class="hljs-number">10</span>,
 <span class="hljs-string">'start_time'</span>: datetime.datetime(<span class="hljs-number">2017</span>, <span class="hljs-number">2</span>, <span class="hljs-number">19</span>, <span class="hljs-number">5</span>, <span class="hljs-number">37</span>, <span class="hljs-number">20</span>, <span class="hljs-number">321557</span>)}
<span class="hljs-number">2020</span>-<span class="hljs-number">02</span>-<span class="hljs-number">19</span> <span class="hljs-number">13</span>:<span class="hljs-number">37</span>:<span class="hljs-number">27</span> [scrapy.core.engine] INFO: <span class="hljs-function">Spider <span class="hljs-title">closed</span> <span class="hljs-params">(finished)</span>
</span></code></pre>

<p data-nodeid="96">这里只是部分运行结果，中间一些抓取结果已省略。<br>
首先，Scrapy 输出了当前的版本号，以及正在启动的项目名称。接着输出了当前 settings.py 中一些重写后的配置。然后输出了当前所应用的 Middlewares 和 Pipelines。Middlewares 默认是启用的，可以在 settings.py 中修改。Pipelines 默认是空，同样也可以在 settings.py 中配置。后面会对它们进行讲解。</p>
<p data-nodeid="97">接下来就是输出各个页面的抓取结果了，可以看到爬虫一边解析，一边翻页，直至将所有内容抓取完毕，然后终止。</p>
<p data-nodeid="98">最后，Scrapy 输出了整个抓取过程的统计信息，如请求的字节数、请求次数、响应次数、完成原因等。</p>
<p data-nodeid="99">整个 Scrapy 程序成功运行。我们通过非常简单的代码就完成了一个网站内容的爬取，这样相比之前一点点写程序简洁很多。</p>
<h3 data-nodeid="80194" class="">保存到文件</h3>

<p data-nodeid="101">运行完 Scrapy 后，我们只在控制台看到了输出结果。如果想保存结果该怎么办呢？</p>
<p data-nodeid="102">要完成这个任务其实不需要任何额外的代码，Scrapy 提供的 Feed Exports 可以轻松将抓取结果输出。例如，我们想将上面的结果保存成 JSON 文件，可以执行如下命令：</p>
<pre class="lang-java" data-nodeid="80910"><code data-language="java">scrapy crawl quotes -o quotes.json
</code></pre>

<p data-nodeid="81625">命令运行后，项目内多了一个 quotes.json 文件，文件包含了刚才抓取的所有内容，内容是 JSON 格式。</p>
<p data-nodeid="81626">另外我们还可以每一个 Item 输出一行 JSON，输出后缀为 jl，为 jsonline 的缩写，命令如下所示：</p>

<pre class="lang-java" data-nodeid="82343"><code data-language="java">scrapy crawl quotes -o quotes.jl
</code></pre>

<p data-nodeid="106">或</p>
<pre class="lang-java" data-nodeid="83058"><code data-language="java">scrapy crawl quotes -o quotes.jsonlines
</code></pre>

<p data-nodeid="108">输出格式还支持很多种，例如 csv、xml、pickle、marshal 等，还支持 ftp、s3 等远程输出，另外还可以通过自定义 ItemExporter 来实现其他的输出。<br>
例如，下面命令对应的输出分别为 csv、xml、pickle、marshal 格式以及 ftp 远程输出：</p>
<pre class="lang-java" data-nodeid="83773"><code data-language="java">scrapy crawl quotes -o quotes.csv
scrapy crawl quotes -o quotes.xml
scrapy crawl quotes -o quotes.pickle
scrapy crawl quotes -o quotes.marshal
scrapy crawl quotes -o ftp:<span class="hljs-comment">//user:pass@ftp.example.com/path/to/quotes.csv</span>
</code></pre>

<p data-nodeid="84488">其中，ftp 输出需要正确配置用户名、密码、地址、输出路径，否则会报错。</p>
<p data-nodeid="84489">通过 Scrapy 提供的 Feed Exports，我们可以轻松地输出抓取结果到文件。对于一些小型项目来说，这应该足够了。不过如果想要更复杂的输出，如输出到数据库等，我们可以使用 Item Pileline 来完成。</p>

<h3 data-nodeid="85206" class="">使用 Item Pipeline</h3>

<p data-nodeid="112">如果想进行更复杂的操作，如将结果保存到 MongoDB 数据库，或者筛选某些有用的 Item，则我们可以定义 Item Pipeline 来实现。</p>
<p data-nodeid="113">Item Pipeline 为项目管道。当 Item 生成后，它会自动被送到 Item Pipeline 进行处理，我们常用 Item Pipeline 来做如下操作。</p>
<ul data-nodeid="114">
<li data-nodeid="115">
<p data-nodeid="116">清洗 HTML 数据；</p>
</li>
<li data-nodeid="117">
<p data-nodeid="118">验证爬取数据，检查爬取字段；</p>
</li>
<li data-nodeid="119">
<p data-nodeid="120">查重并丢弃重复内容；</p>
</li>
<li data-nodeid="121">
<p data-nodeid="122">将爬取结果储存到数据库。</p>
</li>
</ul>
<p data-nodeid="123">要实现 Item Pipeline 很简单，只需要定义一个类并实现 process_item 方法即可。启用 Item Pipeline 后，Item Pipeline 会自动调用这个方法。process_item 方法必须返回包含数据的字典或 Item 对象，或者抛出 DropItem 异常。</p>
<p data-nodeid="124">process_item 方法有两个参数。一个参数是 item，每次 Spider 生成的 Item 都会作为参数传递过来。另一个参数是 spider，就是 Spider 的实例。</p>
<p data-nodeid="125">接下来，我们实现一个 Item Pipeline，筛掉 text 长度大于 50 的 Item，并将结果保存到 MongoDB。</p>
<p data-nodeid="126">修改项目里的 pipelines.py 文件，之前用命令行自动生成的文件内容可以删掉，增加一个 TextPipeline 类，内容如下所示：</p>
<pre class="lang-dart" data-nodeid="88782"><code data-language="dart">from scrapy.exceptions <span class="hljs-keyword">import</span> DropItem
​
<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">TextPipeline</span>(<span class="hljs-title">object</span>):
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">__init__</span>(<span class="hljs-title">self</span>):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">self</span>.<span class="hljs-title">limit</span> = 50
 &nbsp; &nbsp;
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">process_item</span>(<span class="hljs-title">self</span>, <span class="hljs-title">item</span>, <span class="hljs-title">spider</span>):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">if</span> <span class="hljs-title">item</span>['<span class="hljs-title">text</span>']:
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">if</span> <span class="hljs-title">len</span>(<span class="hljs-title">item</span>['<span class="hljs-title">text</span>']) &gt; <span class="hljs-title">self</span>.<span class="hljs-title">limit</span>:
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">item</span>['<span class="hljs-title">text</span>'] = <span class="hljs-title">item</span>['<span class="hljs-title">text</span>'][0:<span class="hljs-title">self</span>.<span class="hljs-title">limit</span>].<span class="hljs-title">rstrip</span>() + '...'
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">return</span> <span class="hljs-title">item</span>
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">else</span>:
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">return</span> <span class="hljs-title">DropItem</span>('<span class="hljs-title">Missing</span> <span class="hljs-title">Text</span>')
</span></code></pre>





<p data-nodeid="89497">这段代码在构造方法里定义了限制长度为 50，实现了 process_item 方法，其参数是 item 和 spider。首先该方法判断 item 的 text 属性是否存在，如果不存在，则抛出 DropItem 异常；如果存在，再判断长度是否大于 50，如果大于，那就截断然后拼接省略号，再将 item 返回即可。</p>
<p data-nodeid="89498">接下来，我们将处理后的 item 存入 MongoDB，定义另外一个 Pipeline。同样在 pipelines.py 中，我们实现另一个类 MongoPipeline，内容如下所示：</p>

<pre class="lang-dart" data-nodeid="97367"><code data-language="dart"><span class="hljs-keyword">import</span> pymongo
​
<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">MongoPipeline</span>(<span class="hljs-title">object</span>):
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">__init__</span>(<span class="hljs-title">self</span>, <span class="hljs-title">mongo_uri</span>, <span class="hljs-title">mongo_db</span>):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">self</span>.<span class="hljs-title">mongo_uri</span> = <span class="hljs-title">mongo_uri</span>
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">self</span>.<span class="hljs-title">mongo_db</span> = <span class="hljs-title">mongo_db</span>
​
 &nbsp; &nbsp;@<span class="hljs-title">classmethod</span>
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">from_crawler</span>(<span class="hljs-title">cls</span>, <span class="hljs-title">crawler</span>):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">return</span> <span class="hljs-title">cls</span>(<span class="hljs-title">mongo_uri</span>=<span class="hljs-title">crawler</span>.<span class="hljs-title">settings</span>.<span class="hljs-title">get</span>('<span class="hljs-title">MONGO_URI</span>'),
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">mongo_db</span>=<span class="hljs-title">crawler</span>.<span class="hljs-title">settings</span>.<span class="hljs-title">get</span>('<span class="hljs-title">MONGO_DB</span>')
 &nbsp; &nbsp; &nbsp;  )
​
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">open_spider</span>(<span class="hljs-title">self</span>, <span class="hljs-title">spider</span>):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">self</span>.<span class="hljs-title">client</span> = <span class="hljs-title">pymongo</span>.<span class="hljs-title">MongoClient</span>(<span class="hljs-title">self</span>.<span class="hljs-title">mongo_uri</span>)
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">self</span>.<span class="hljs-title">db</span> = <span class="hljs-title">self</span>.<span class="hljs-title">client</span>[<span class="hljs-title">self</span>.<span class="hljs-title">mongo_db</span>]
​
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">process_item</span>(<span class="hljs-title">self</span>, <span class="hljs-title">item</span>, <span class="hljs-title">spider</span>):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">name</span> = <span class="hljs-title">item</span>.<span class="hljs-title">__class__</span>.<span class="hljs-title">__name__</span>
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">self</span>.<span class="hljs-title">db</span>[<span class="hljs-title">name</span>].<span class="hljs-title">insert</span>(<span class="hljs-title">dict</span>(<span class="hljs-title">item</span>))
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">return</span> <span class="hljs-title">item</span>
​
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">close_spider</span>(<span class="hljs-title">self</span>, <span class="hljs-title">spider</span>):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">self</span>.<span class="hljs-title">client</span>.<span class="hljs-title">close</span>()
</span></code></pre>











<p data-nodeid="130">MongoPipeline 类实现了 API 定义的另外几个方法。</p>
<ul data-nodeid="131">
<li data-nodeid="132">
<p data-nodeid="133">from_crawler：这是一个类方法，用 @classmethod 标识，是一种依赖注入的方式，方法的参数就是 crawler，通过 crawler 这个参数我们可以拿到全局配置的每个配置信息，在全局配置 settings.py 中我们可以定义 MONGO_URI 和 MONGO_DB 来指定 MongoDB 连接需要的地址和数据库名称，拿到配置信息之后返回类对象即可。所以这个方法的定义主要是用来获取 settings.py 中的配置的。</p>
</li>
<li data-nodeid="134">
<p data-nodeid="135">open_spider：当 Spider 被开启时，这个方法被调用。在这里主要进行了一些初始化操作。</p>
</li>
<li data-nodeid="136">
<p data-nodeid="137">close_spider：当 Spider 被关闭时，这个方法会调用，在这里将数据库连接关闭。</p>
</li>
</ul>
<p data-nodeid="138">最主要的 process_item 方法则执行了数据插入操作。</p>
<p data-nodeid="139">定义好 TextPipeline 和 MongoPipeline 这两个类后，我们需要在 settings.py 中使用它们。MongoDB 的连接信息还需要定义。</p>
<p data-nodeid="140">我们在 settings.py 中加入如下内容：</p>
<pre class="lang-java" data-nodeid="98082"><code data-language="java">ITEM_PIPELINES = {
 &nbsp; <span class="hljs-string">'tutorial.pipelines.TextPipeline'</span>: <span class="hljs-number">300</span>,
 &nbsp; <span class="hljs-string">'tutorial.pipelines.MongoPipeline'</span>: <span class="hljs-number">400</span>,
}
MONGO_URI=<span class="hljs-string">'localhost'</span>
MONGO_DB=<span class="hljs-string">'tutorial'</span>
</code></pre>

<p data-nodeid="98797">赋值 ITEM_PIPELINES 字典，键名是 Pipeline 的类名称，键值是调用优先级，是一个数字，数字越小则对应的 Pipeline 越先被调用。</p>
<p data-nodeid="98798">再重新执行爬取，命令如下所示：</p>

<pre class="lang-java" data-nodeid="99517"><code data-language="java">scrapy crawl quotes
</code></pre>

<p data-nodeid="100940">爬取结束后，MongoDB 中创建了一个 tutorial 的数据库、QuoteItem 的表，如图所示。</p>
<p data-nodeid="100941" class=""><img src="https://s0.lgstatic.com/i/image/M00/2B/8D/CgqCHl7-rYOAPdxFAAZ43djzJ7s319.png" alt="image (7).png" data-nodeid="100949"></p>


<p data-nodeid="145">长的 text 已经被处理并追加了省略号，短的 text 保持不变，author 和 tags 也都相应保存。</p>
<h3 data-nodeid="101672" class="">代码</h3>

<p data-nodeid="147">本节代码地址：<a href="https://github.com/Python3WebSpider/ScrapyTutorial" data-nodeid="350">https://github.com/Python3WebSpider/ScrapyTutorial</a>。</p>
<h3 data-nodeid="102396" class="te-preview-highlight">结语</h3>

<p data-nodeid="149">我们通过抓取 Quotes 网站完成了整个 Scrapy 的简单入门。但这只是冰山一角，还有很多内容等待我们去探索。</p>


# 灵活好用的Spider的用法
<p data-nodeid="41690">在上一节课我们通过实例了解了 Scrapy 的基本使用方法，在这个过程中，我们用到了 Spider 来编写爬虫逻辑，同时用到了一些选择器来对结果进行选择。</p>



<p data-nodeid="40244">在这一节课，我们就对 Spider 和 Selector 的基本用法作一个总结。</p>
<h3 data-nodeid="40245">Spider 的用法</h3>
<p data-nodeid="40246">在 Scrapy 中，要抓取网站的链接配置、抓取逻辑、解析逻辑等其实都是在 Spider 中配置的。在前一节课的实例中，我们发现抓取逻辑也是在 Spider 中完成的。本节课我们就来专门了解一下 Spider 的基本用法。</p>
<h4 data-nodeid="40247">Spider 运行流程</h4>
<p data-nodeid="40248">在实现 Scrapy 爬虫项目时，最核心的类便是 Spider 类了，它定义了如何爬取某个网站的流程和解析方式。简单来讲，Spider 要做的事就是如下两件：</p>
<ul data-nodeid="40249">
<li data-nodeid="40250">
<p data-nodeid="40251">定义爬取网站的动作；</p>
</li>
<li data-nodeid="40252">
<p data-nodeid="40253">分析爬取下来的网页。</p>
</li>
</ul>
<p data-nodeid="40254">对于 Spider 类来说，整个爬取循环如下所述。</p>
<ul data-nodeid="40255">
<li data-nodeid="40256">
<p data-nodeid="40257">以初始的 URL 初始化 Request，并设置回调函数。 当该 Request 成功请求并返回时，将生成 Response，并作为参数传给该回调函数。</p>
</li>
<li data-nodeid="40258">
<p data-nodeid="40259">在回调函数内分析返回的网页内容。返回结果可以有两种形式，一种是解析到的有效结果返回字典或 Item 对象。下一步可经过处理后（或直接）保存，另一种是解析到的下一个（如下一页）链接，可以利用此链接构造 Request 并设置新的回调函数，返回 Request。</p>
</li>
<li data-nodeid="40260">
<p data-nodeid="40261">如果返回的是字典或 Item 对象，可通过 Feed Exports 等形式存入文件，如果设置了 Pipeline 的话，可以经由 Pipeline 处理（如过滤、修正等）并保存。</p>
</li>
<li data-nodeid="40262">
<p data-nodeid="40263">如果返回的是 Reqeust，那么 Request 执行成功得到 Response 之后会再次传递给 Request 中定义的回调函数，可以再次使用选择器来分析新得到的网页内容，并根据分析的数据生成 Item。</p>
</li>
</ul>
<p data-nodeid="40264">通过以上几步循环往复进行，便完成了站点的爬取。</p>
<h4 data-nodeid="40265">Spider 类分析</h4>
<p data-nodeid="40266">在上一节课的例子中我们定义的 Spider 继承自 scrapy.spiders.Spider，这个类是最简单最基本的 Spider 类，每个其他的 Spider 必须继承自这个类，还有后面要说明的一些特殊 Spider 类也都是继承自它。</p>
<p data-nodeid="40267">这个类里提供了 start_requests 方法的默认实现，读取并请求 start_urls 属性，并根据返回的结果调用 parse 方法解析结果。另外它还有一些基础属性，下面对其进行讲解。</p>
<ul data-nodeid="40268">
<li data-nodeid="40269">
<p data-nodeid="40270">name：爬虫名称，是定义 Spider 名字的字符串。Spider 的名字定义了 Scrapy 如何定位并初始化 Spider，所以其必须是唯一的。 不过我们可以生成多个相同的 Spider 实例，这没有任何限制。 name 是 Spider 最重要的属性，而且是必需的。如果该 Spider 爬取单个网站，一个常见的做法是以该网站的域名名称来命名 Spider。例如，如果 Spider 爬取 mywebsite.com，该 Spider 通常会被命名为 mywebsite。</p>
</li>
<li data-nodeid="40271">
<p data-nodeid="40272">allowed_domains：允许爬取的域名，是可选配置，不在此范围的链接不会被跟进爬取。</p>
</li>
<li data-nodeid="40273">
<p data-nodeid="40274">start_urls：起始 URL 列表，当我们没有实现 start_requests 方法时，默认会从这个列表开始抓取。</p>
</li>
<li data-nodeid="40275">
<p data-nodeid="40276">custom_settings：这是一个字典，是专属于本 Spider 的配置，此设置会覆盖项目全局的设置，而且此设置必须在初始化前被更新，所以它必须定义成类变量。</p>
</li>
<li data-nodeid="40277">
<p data-nodeid="40278">crawler：此属性是由 from_crawler 方法设置的，代表的是本 Spider 类对应的 Crawler 对象，Crawler 对象中包含了很多项目组件，利用它我们可以获取项目的一些配置信息，如最常见的就是获取项目的设置信息，即 Settings。</p>
</li>
<li data-nodeid="40279">
<p data-nodeid="40280">settings：是一个 Settings 对象，利用它我们可以直接获取项目的全局设置变量。</p>
</li>
</ul>
<p data-nodeid="40281">除了一些基础属性，Spider 还有一些常用的方法，在此介绍如下。</p>
<ul data-nodeid="40282">
<li data-nodeid="40283">
<p data-nodeid="40284">start_requests：此方法用于生成初始请求，它必须返回一个可迭代对象，此方法会默认使用 start_urls 里面的 URL 来构造 Request，而且 Request 是 GET 请求方式。如果我们想在启动时以 POST 方式访问某个站点，可以直接重写这个方法，发送 POST 请求时我们使用 FormRequest 即可。</p>
</li>
<li data-nodeid="40285">
<p data-nodeid="40286">parse：当 Response 没有指定回调函数时，该方法会默认被调用，它负责处理 Response，处理返回结果，并从中提取出想要的数据和下一步的请求，然后返回。该方法需要返回一个包含 Request 或 Item 的可迭代对象。</p>
</li>
<li data-nodeid="40287">
<p data-nodeid="40288">closed：当 Spider 关闭时，该方法会被调用，在这里一般会定义释放资源的一些操作或其他收尾操作。</p>
</li>
</ul>
<h3 data-nodeid="40289">Selector 的用法</h3>
<p data-nodeid="40290">我们之前介绍了利用 Beautiful Soup、PyQuery，以及正则表达式来提取网页数据，这确实非常方便。而 Scrapy 还提供了自己的数据提取方法，即 Selector（选择器）。</p>
<p data-nodeid="40291">Selector 是基于 lxml 构建的，支持 XPath 选择器、CSS 选择器，以及正则表达式，功能全面，解析速度和准确度非常高。</p>
<p data-nodeid="40292">接下来我们将介绍 Selector 的用法。</p>
<h4 data-nodeid="40293">直接使用</h4>
<p data-nodeid="40294">Selector 是一个可以独立使用的模块。我们可以直接利用 Selector 这个类来构建一个选择器对象，然后调用它的相关方法如 xpath、css 等来提取数据。</p>
<p data-nodeid="40295">例如，针对一段 HTML 代码，我们可以用如下方式构建 Selector 对象来提取数据：</p>
<pre class="lang-java" data-nodeid="42542"><code data-language="java">from scrapy <span class="hljs-keyword">import</span> Selector
​
body = <span class="hljs-string">'&lt;html&gt;&lt;head&gt;&lt;title&gt;Hello World&lt;/title&gt;&lt;/head&gt;&lt;body&gt;&lt;/body&gt;&lt;/html&gt;'</span>
selector = Selector(text=body)
title = selector.xpath(<span class="hljs-string">'//title/text()'</span>).extract_first()
print(title)
</code></pre>


<p data-nodeid="40297">运行结果：</p>
<pre class="lang-java" data-nodeid="43109"><code data-language="java">Hello World
</code></pre>

<p data-nodeid="43676">这里我们没有在 Scrapy 框架中运行，而是把 Scrapy 中的 Selector 单独拿出来使用了，构建的时候传入 text 参数，就生成了一个 Selector 选择器对象，然后就可以像前面我们所用的 Scrapy 中的解析方式一样，调用 xpath、css 等方法来提取了。</p>
<p data-nodeid="43677">在这里我们查找的是源代码中的 title 中的文本，在 XPath 选择器最后加 text 方法就可以实现文本的提取了。</p>

<p data-nodeid="40300">以上内容就是 Selector 的直接使用方式。同 Beautiful Soup 等库类似，Selector 其实也是强大的网页解析库。如果方便的话，我们也可以在其他项目中直接使用 Selector 来提取数据。</p>
<p data-nodeid="40301">接下来，我们用实例来详细讲解 Selector 的用法。</p>
<h4 data-nodeid="40302">Scrapy Shell</h4>
<p data-nodeid="40303">由于 Selector 主要是与 Scrapy 结合使用，如 Scrapy 的回调函数中的参数 response 直接调用 xpath() 或者 css() 方法来提取数据，所以在这里我们借助 Scrapy Shell 来模拟 Scrapy 请求的过程，来讲解相关的提取方法。</p>
<p data-nodeid="40304">我们用官方文档的一个样例页面来做演示：<a href="http://doc.scrapy.org/en/latest/_static/selectors-sample1.html" data-nodeid="40429">http://doc.scrapy.org/en/latest/_static/selectors-sample1.html</a>。</p>
<p data-nodeid="40305">开启 Scrapy Shell，在命令行中输入如下命令：</p>
<pre class="lang-java" data-nodeid="44246"><code data-language="java">scrapy shell http:<span class="hljs-comment">//doc.scrapy.org/en/latest/_static/selectors-sample1.html</span>
</code></pre>

<p data-nodeid="45373">这样我们就进入了 Scrapy Shell 模式。这个过程其实是 Scrapy 发起了一次请求，请求的 URL 就是刚才命令行下输入的 URL，然后把一些可操作的变量传递给我们，如 request、response 等，如图所示。</p>
<p data-nodeid="45374" class=""><img src="https://s0.lgstatic.com/i/image/M00/2E/D0/Ciqc1F8FqLiAdcLhAAESdhrMDNE818.png" alt="image (6).png" data-nodeid="45382"></p>


<p data-nodeid="40308">我们可以在命令行模式下输入命令调用对象的一些操作方法，回车之后实时显示结果。这与 Python 的命令行交互模式是类似的。</p>
<p data-nodeid="40309">接下来，演示的实例都将页面的源码作为分析目标，页面源码如下所示：</p>
<pre class="lang-dart" data-nodeid="48832"><code data-language="dart">&lt;html&gt;
 &lt;head&gt;
 &nbsp;&lt;base href=<span class="hljs-string">'http://example.com/'</span> /&gt;
 &nbsp;&lt;title&gt;Example website&lt;/title&gt;
 &lt;/head&gt;
 &lt;body&gt;
 &nbsp;&lt;div id=<span class="hljs-string">'images'</span>&gt;
 &nbsp; &lt;a href=<span class="hljs-string">'image1.html'</span>&gt;Name: My image <span class="hljs-number">1</span> &lt;br /&gt;&lt;img src=<span class="hljs-string">'image1_thumb.jpg'</span> /&gt;&lt;/a&gt;
 &nbsp; &lt;a href=<span class="hljs-string">'image2.html'</span>&gt;Name: My image <span class="hljs-number">2</span> &lt;br /&gt;&lt;img src=<span class="hljs-string">'image2_thumb.jpg'</span> /&gt;&lt;/a&gt;
 &nbsp; &lt;a href=<span class="hljs-string">'image3.html'</span>&gt;Name: My image <span class="hljs-number">3</span> &lt;br /&gt;&lt;img src=<span class="hljs-string">'image3_thumb.jpg'</span> /&gt;&lt;/a&gt;
 &nbsp; &lt;a href=<span class="hljs-string">'image4.html'</span>&gt;Name: My image <span class="hljs-number">4</span> &lt;br /&gt;&lt;img src=<span class="hljs-string">'image4_thumb.jpg'</span> /&gt;&lt;/a&gt;
 &nbsp; &lt;a href=<span class="hljs-string">'image5.html'</span>&gt;Name: My image <span class="hljs-number">5</span> &lt;br /&gt;&lt;img src=<span class="hljs-string">'image5_thumb.jpg'</span> /&gt;&lt;/a&gt;
 &nbsp;&lt;/div&gt;
 &lt;/body&gt;
&lt;/html&gt;
</code></pre>






<h4 data-nodeid="40311">XPath 选择器</h4>
<p data-nodeid="40312">进入 Scrapy Shell 之后，我们将主要操作 response 变量来进行解析。因为我们解析的是 HTML 代码，Selector 将自动使用 HTML 语法来分析。</p>
<p data-nodeid="40313">response 有一个属性 selector，我们调用 response.selector 返回的内容就相当于用 response 的 text 构造了一个 Selector 对象。通过这个 Selector 对象我们可以调用解析方法如 xpath、css 等，通过向方法传入 XPath 或 CSS 选择器参数就可以实现信息的提取。</p>
<p data-nodeid="40314">我们用一个实例感受一下，如下所示：</p>
<pre class="lang-java" data-nodeid="49407"><code data-language="java">&gt;&gt;&gt; result = response.selector.xpath(<span class="hljs-string">'//a'</span>)
&gt;&gt;&gt; result
[&lt;Selector xpath=<span class="hljs-string">'//a'</span> data=<span class="hljs-string">'&lt;a href="image1.html"&gt;Name: My image 1 &lt;'</span>&gt;,
 &lt;Selector xpath=<span class="hljs-string">'//a'</span> data=<span class="hljs-string">'&lt;a href="image2.html"&gt;Name: My image 2 &lt;'</span>&gt;,
 &lt;Selector xpath=<span class="hljs-string">'//a'</span> data=<span class="hljs-string">'&lt;a href="image3.html"&gt;Name: My image 3 &lt;'</span>&gt;,
 &lt;Selector xpath=<span class="hljs-string">'//a'</span> data=<span class="hljs-string">'&lt;a href="image4.html"&gt;Name: My image 4 &lt;'</span>&gt;,
 &lt;Selector xpath=<span class="hljs-string">'//a'</span> data=<span class="hljs-string">'&lt;a href="image5.html"&gt;Name: My image 5 &lt;'</span>&gt;]
&gt;&gt;&gt; type(result)
scrapy.selector.unified.SelectorList
</code></pre>

<p data-nodeid="49982">打印结果的形式是 Selector 组成的列表，其实它是 SelectorList 类型，SelectorList 和 Selector 都可以继续调用 xpath 和 css 等方法来进一步提取数据。</p>
<p data-nodeid="49983">在上面的例子中，我们提取了 a 节点。接下来，我们尝试继续调用 xpath 方法来提取 a 节点内包含的 img 节点，如下所示：</p>

<pre class="lang-java" data-nodeid="50560"><code data-language="java">&gt;&gt;&gt; result.xpath(<span class="hljs-string">'./img'</span>)
[&lt;Selector xpath=<span class="hljs-string">'./img'</span> data=<span class="hljs-string">'&lt;img src="image1_thumb.jpg"&gt;'</span>&gt;,
 &lt;Selector xpath=<span class="hljs-string">'./img'</span> data=<span class="hljs-string">'&lt;img src="image2_thumb.jpg"&gt;'</span>&gt;,
 &lt;Selector xpath=<span class="hljs-string">'./img'</span> data=<span class="hljs-string">'&lt;img src="image3_thumb.jpg"&gt;'</span>&gt;,
 &lt;Selector xpath=<span class="hljs-string">'./img'</span> data=<span class="hljs-string">'&lt;img src="image4_thumb.jpg"&gt;'</span>&gt;,
 &lt;Selector xpath=<span class="hljs-string">'./img'</span> data=<span class="hljs-string">'&lt;img src="image5_thumb.jpg"&gt;'</span>&gt;]
</code></pre>

<p data-nodeid="51135">我们获得了 a 节点里面的所有 img 节点，结果为 5。</p>
<p data-nodeid="51136">值得注意的是，选择器的最前方加 <code data-backticks="1" data-nodeid="51139">.</code>（点），这代表提取元素内部的数据，如果没有加点，则代表从根节点开始提取。此处我们用了 ./img 的提取方式，则代表从 a 节点里进行提取。如果此处我们用 //img，则还是从 html 节点里进行提取。</p>

<p data-nodeid="40319">我们刚才使用了 response.selector.xpath 方法对数据进行了提取。Scrapy 提供了两个实用的快捷方法，response.xpath 和 response.css，它们二者的功能完全等同于 response.selector.xpath 和 response.selector.css。方便起见，后面我们统一直接调用 response 的 xpath 和 css 方法进行选择。</p>
<p data-nodeid="40320">现在我们得到的是 SelectorList 类型的变量，该变量是由 Selector 对象组成的列表。我们可以用索引单独取出其中某个 Selector 元素，如下所示：</p>
<pre class="lang-java" data-nodeid="51715"><code data-language="java">&gt;&gt;&gt; result[<span class="hljs-number">0</span>]
&lt;Selector xpath=<span class="hljs-string">'//a'</span> data=<span class="hljs-string">'&lt;a href="image1.html"&gt;Name: My image 1 &lt;'</span>&gt;
</code></pre>

<p data-nodeid="40322">我们可以像操作列表一样操作这个 SelectorList。但是现在获取的内容是 Selector 或者 SelectorList 类型，并不是真正的文本内容。那么具体的内容怎么提取呢？<br>
比如我们现在想提取出 a 节点元素，就可以利用 extract 方法，如下所示：</p>
<pre class="lang-java" data-nodeid="52290"><code data-language="java">&gt;&gt;&gt; result.extract()
[<span class="hljs-string">'&lt;a href="image1.html"&gt;Name: My image 1 &lt;br&gt;&lt;img src="image1_thumb.jpg"&gt;&lt;/a&gt;'</span>, <span class="hljs-string">'&lt;a href="image2.html"&gt;Name: My image 2 &lt;br&gt;&lt;img src="image2_thumb.jpg"&gt;&lt;/a&gt;'</span>, <span class="hljs-string">'&lt;a href="image3.html"&gt;Name: My image 3 &lt;br&gt;&lt;img src="image3_thumb.jpg"&gt;&lt;/a&gt;'</span>, <span class="hljs-string">'&lt;a href="image4.html"&gt;Name: My image 4 &lt;br&gt;&lt;img src="image4_thumb.jpg"&gt;&lt;/a&gt;'</span>, <span class="hljs-string">'&lt;a href="image5.html"&gt;Name: My image 5 &lt;br&gt;&lt;img src="image5_thumb.jpg"&gt;&lt;/a&gt;'</span>]
</code></pre>

<p data-nodeid="53440">这里使用了 extract 方法，我们就可以把真实需要的内容获取下来。</p>
<p data-nodeid="53441">我们还可以改写 XPath 表达式，来选取节点的内部文本和属性，如下所示：</p>

<pre class="lang-java" data-nodeid="52865"><code data-language="java">&gt;&gt;&gt; response.xpath(<span class="hljs-string">'//a/text()'</span>).extract()
[<span class="hljs-string">'Name: My image 1 '</span>, <span class="hljs-string">'Name: My image 2 '</span>, <span class="hljs-string">'Name: My image 3 '</span>, <span class="hljs-string">'Name: My image 4 '</span>, <span class="hljs-string">'Name: My image 5 '</span>]
&gt;&gt;&gt; response.xpath(<span class="hljs-string">'//a/@href'</span>).extract()
[<span class="hljs-string">'image1.html'</span>, <span class="hljs-string">'image2.html'</span>, <span class="hljs-string">'image3.html'</span>, <span class="hljs-string">'image4.html'</span>, <span class="hljs-string">'image5.html'</span>]
</code></pre>

<p data-nodeid="54018">我们只需要再加一层 /text() 就可以获取节点的内部文本，或者加一层 /@href 就可以获取节点的 href 属性。其中，@ 符号后面内容就是要获取的属性名称。</p>
<p data-nodeid="54019">现在我们可以用一个规则把所有符合要求的节点都获取下来，返回的类型是列表类型。</p>

<p data-nodeid="40327">但是这里有一个问题：如果符合要求的节点只有一个，那么返回的结果会是什么呢？我们再用一个实例来感受一下，如下所示：</p>
<pre class="lang-java" data-nodeid="54596"><code data-language="java">&gt;&gt;&gt; response.xpath(<span class="hljs-string">'//a[@href="image1.html"]/text()'</span>).extract()
[<span class="hljs-string">'Name: My image 1 '</span>]
</code></pre>

<p data-nodeid="40329">我们用属性限制了匹配的范围，使 XPath 只可以匹配到一个元素。然后用 extract 方法提取结果，其结果还是一个列表形式，其文本是列表的第一个元素。但很多情况下，我们其实想要的数据就是第一个元素内容，这里我们通过加一个索引来获取，如下所示：</p>
<pre class="lang-java" data-nodeid="55171"><code data-language="java"><span class="hljs-string">'Name: My image 1 '</span>
</code></pre>

<p data-nodeid="40331">但是，这个写法很明显是有风险的。一旦 XPath 有问题，那么 extract 后的结果可能是一个空列表。如果我们再用索引来获取，那不就可能会导致数组越界吗？<br>
所以，另外一个方法可以专门提取单个元素，它叫作 extract_first。我们可以改写上面的例子如下所示：</p>
<pre class="lang-java" data-nodeid="55746"><code data-language="java">&gt;&gt;&gt; response.xpath(<span class="hljs-string">'//a[@href="image1.html"]/text()'</span>).extract_first()
<span class="hljs-string">'Name: My image 1 '</span>
</code></pre>

<p data-nodeid="40333">这样，我们直接利用 extract_first 方法将匹配的第一个结果提取出来，同时我们也不用担心数组越界的问题。<br>
另外我们也可以为 extract_first 方法设置一个默认值参数，这样当 XPath 规则提取不到内容时会直接使用默认值。例如将 XPath 改成一个不存在的规则，重新执行代码，如下所示：</p>
<pre class="lang-java" data-nodeid="56321"><code data-language="java">&gt;&gt;&gt; response.xpath(<span class="hljs-string">'//a[@href="image1"]/text()'</span>).extract_first()&gt;&gt;&gt; response.xpath(<span class="hljs-string">'//a[@href="image1"]/text()'</span>).extract_first(<span class="hljs-string">'Default Image'</span>)
<span class="hljs-string">'Default Image'</span>
</code></pre>

<p data-nodeid="56896">这里，如果 XPath 匹配不到任何元素，调用 extract_first 会返回空，也不会报错。在第二行代码中，我们还传递了一个参数当作默认值，如 Default Image。这样如果 XPath 匹配不到结果的话，返回值会使用这个参数来代替，可以看到输出正是如此。</p>
<p data-nodeid="56897">到现在为止，我们了解了 Scrapy 中的 XPath 的相关用法，包括嵌套查询、提取内容、提取单个内容、获取文本和属性等。</p>

<h4 data-nodeid="40336">CSS 选择器</h4>
<p data-nodeid="40337">接下来，我们看看 CSS 选择器的用法。Scrapy 的选择器同时还对接了 CSS 选择器，使用 response.css() 方法可以使用 CSS 选择器来选择对应的元素。</p>
<p data-nodeid="40338">例如在上文我们选取了所有的 a 节点，那么 CSS 选择器同样可以做到，如下所示：</p>
<pre class="lang-java" data-nodeid="57476"><code data-language="java">&gt;&gt;&gt; response.css(<span class="hljs-string">'a'</span>)
[&lt;Selector xpath=<span class="hljs-string">'descendant-or-self::a'</span> data=<span class="hljs-string">'&lt;a href="image1.html"&gt;Name: My image 1 &lt;'</span>&gt;, 
&lt;Selector xpath=<span class="hljs-string">'descendant-or-self::a'</span> data=<span class="hljs-string">'&lt;a href="image2.html"&gt;Name: My image 2 &lt;'</span>&gt;, 
&lt;Selector xpath=<span class="hljs-string">'descendant-or-self::a'</span> data=<span class="hljs-string">'&lt;a href="image3.html"&gt;Name: My image 3 &lt;'</span>&gt;, 
&lt;Selector xpath=<span class="hljs-string">'descendant-or-self::a'</span> data=<span class="hljs-string">'&lt;a href="image4.html"&gt;Name: My image 4 &lt;'</span>&gt;, 
&lt;Selector xpath=<span class="hljs-string">'descendant-or-self::a'</span> data=<span class="hljs-string">'&lt;a href="image5.html"&gt;Name: My image 5 &lt;'</span>&gt;]
</code></pre>

<p data-nodeid="40340">同样，调用 extract 方法就可以提取出节点，如下所示：</p>
<pre class="lang-java" data-nodeid="58051"><code data-language="java">[<span class="hljs-string">'&lt;a href="image1.html"&gt;Name: My image 1 &lt;br&gt;&lt;img src="image1_thumb.jpg"&gt;&lt;/a&gt;'</span>, <span class="hljs-string">'&lt;a href="image2.html"&gt;Name: My image 2 &lt;br&gt;&lt;img src="image2_thumb.jpg"&gt;&lt;/a&gt;'</span>, <span class="hljs-string">'&lt;a href="image3.html"&gt;Name: My image 3 &lt;br&gt;&lt;img src="image3_thumb.jpg"&gt;&lt;/a&gt;'</span>, <span class="hljs-string">'&lt;a href="image4.html"&gt;Name: My image 4 &lt;br&gt;&lt;img src="image4_thumb.jpg"&gt;&lt;/a&gt;'</span>, <span class="hljs-string">'&lt;a href="image5.html"&gt;Name: My image 5 &lt;br&gt;&lt;img src="image5_thumb.jpg"&gt;&lt;/a&gt;'</span>]
</code></pre>

<p data-nodeid="40342">用法和 XPath 选择是完全一样的。另外，我们也可以进行属性选择和嵌套选择，如下所示：</p>
<pre class="lang-java" data-nodeid="58626"><code data-language="java">&gt;&gt;&gt; response.css(<span class="hljs-string">'a[href="image1.html"]'</span>).extract()
[<span class="hljs-string">'&lt;a href="image1.html"&gt;Name: My image 1 &lt;br&gt;&lt;img src="image1_thumb.jpg"&gt;&lt;/a&gt;'</span>]
&gt;&gt;&gt; response.css(<span class="hljs-string">'a[href="image1.html"] img'</span>).extract()
[<span class="hljs-string">'&lt;img src="image1_thumb.jpg"&gt;'</span>]
</code></pre>

<p data-nodeid="59201">这里用 [href="image.html"] 限定了 href 属性，可以看到匹配结果就只有一个了。另外如果想查找 a 节点内的 img 节点，只需要再加一个空格和 img 即可。选择器的写法和标准 CSS 选择器写法如出一辙。</p>
<p data-nodeid="59202">我们也可以使用 extract_first() 方法提取列表的第一个元素，如下所示：</p>

<pre class="lang-java" data-nodeid="59788"><code data-language="java">&gt;&gt;&gt; response.css(<span class="hljs-string">'a[href="image1.html"] img'</span>).extract_first()
<span class="hljs-string">'&lt;img src="image1_thumb.jpg"&gt;'</span>
</code></pre>

<p data-nodeid="40346">接下来的两个用法不太一样。节点的内部文本和属性的获取是这样实现的，如下所示：</p>
<pre class="lang-java" data-nodeid="60363"><code data-language="java">&gt;&gt;&gt; response.css(<span class="hljs-string">'a[href="image1.html"]::text'</span>).extract_first()
<span class="hljs-string">'Name: My image 1 '</span>
&gt;&gt;&gt; response.css(<span class="hljs-string">'a[href="image1.html"] img::attr(src)'</span>).extract_first()
<span class="hljs-string">'image1_thumb.jpg'</span>
</code></pre>

<p data-nodeid="60938">获取文本和属性需要用 ::text 和 ::attr() 的写法。而其他库如 Beautiful Soup 或 PyQuery 都有单独的方法。</p>
<p data-nodeid="60939">另外，CSS 选择器和 XPath 选择器一样可以嵌套选择。我们可以先用 XPath 选择器选中所有 a 节点，再利用 CSS 选择器选中 img 节点，再用 XPath 选择器获取属性。我们用一个实例来感受一下，如下所示：</p>

<pre class="lang-java" data-nodeid="61516"><code data-language="java">&gt;&gt;&gt; response.xpath(<span class="hljs-string">'//a'</span>).css(<span class="hljs-string">'img'</span>).xpath(<span class="hljs-string">'@src'</span>).extract()
[<span class="hljs-string">'image1_thumb.jpg'</span>, <span class="hljs-string">'image2_thumb.jpg'</span>, <span class="hljs-string">'image3_thumb.jpg'</span>, <span class="hljs-string">'image4_thumb.jpg'</span>, <span class="hljs-string">'image5_thumb.jpg'</span>]
</code></pre>

<p data-nodeid="40350">我们成功获取了所有 img 节点的 src 属性。<br>
因此，我们可以随意使用 xpath 和 css 方法二者自由组合实现嵌套查询，二者是完全兼容的。</p>
<h4 data-nodeid="40351">正则匹配</h4>
<p data-nodeid="40352">Scrapy 的选择器还支持正则匹配。比如，在示例的 a 节点中的文本类似于 Name: My image 1，现在我们只想把 Name: 后面的内容提取出来，这时就可以借助 re 方法，实现如下：</p>
<pre class="lang-java" data-nodeid="62091"><code data-language="java">&gt;&gt;&gt; response.xpath(<span class="hljs-string">'//a/text()'</span>).re(<span class="hljs-string">'Name:\s(.*)'</span>)
[<span class="hljs-string">'My image 1 '</span>, <span class="hljs-string">'My image 2 '</span>, <span class="hljs-string">'My image 3 '</span>, <span class="hljs-string">'My image 4 '</span>, <span class="hljs-string">'My image 5 '</span>]
</code></pre>

<p data-nodeid="62666">我们给 re 方法传入一个正则表达式，其中 <code data-backticks="1" data-nodeid="62669">(.*)</code> 就是要匹配的内容，输出的结果就是正则表达式匹配的分组，结果会依次输出。</p>
<p data-nodeid="62667">如果同时存在两个分组，那么结果依然会被按序输出，如下所示：</p>

<pre class="lang-java" data-nodeid="63246"><code data-language="java">&gt;&gt;&gt; response.xpath(<span class="hljs-string">'//a/text()'</span>).re(<span class="hljs-string">'(.*?):\s(.*)'</span>)
[<span class="hljs-string">'Name'</span>, <span class="hljs-string">'My image 1 '</span>, <span class="hljs-string">'Name'</span>, <span class="hljs-string">'My image 2 '</span>, <span class="hljs-string">'Name'</span>, <span class="hljs-string">'My image 3 '</span>, <span class="hljs-string">'Name'</span>, <span class="hljs-string">'My image 4 '</span>, <span class="hljs-string">'Name'</span>, <span class="hljs-string">'My image 5 '</span>]
</code></pre>

<p data-nodeid="40356">类似 extract_first 方法，re_first 方法可以选取列表的第一个元素，用法如下：</p>
<pre class="lang-java" data-nodeid="63821"><code data-language="java">&gt;&gt;&gt; response.xpath(<span class="hljs-string">'//a/text()'</span>).re_first(<span class="hljs-string">'(.*?):\s(.*)'</span>)
<span class="hljs-string">'Name'</span>
&gt;&gt;&gt; response.xpath(<span class="hljs-string">'//a/text()'</span>).re_first(<span class="hljs-string">'Name:\s(.*)'</span>)
<span class="hljs-string">'My image 1 '</span>
</code></pre>

<p data-nodeid="64396">不论正则匹配了几个分组，结果都会等于列表的第一个元素。</p>
<p data-nodeid="64397">值得注意的是，response 对象不能直接调用 re 和 re_first 方法。如果想要对全文进行正则匹配，可以先调用 xpath 方法然后再进行正则匹配，如下所示：</p>

<pre class="lang-java" data-nodeid="64976"><code data-language="java">&gt;&gt;&gt; response.re(<span class="hljs-string">'Name:\s(.*)'</span>)
Traceback (most recent call last):
 &nbsp;File <span class="hljs-string">"&lt;console&gt;"</span>, line <span class="hljs-number">1</span>, in &lt;<span class="hljs-keyword">module</span>&gt;
AttributeError: <span class="hljs-string">'HtmlResponse'</span> object has no attribute <span class="hljs-string">'re'</span>
&gt;&gt;&gt; response.xpath(<span class="hljs-string">'.'</span>).re(<span class="hljs-string">'Name:\s(.*)&lt;br&gt;'</span>)
[<span class="hljs-string">'My image 1 '</span>, <span class="hljs-string">'My image 2 '</span>, <span class="hljs-string">'My image 3 '</span>, <span class="hljs-string">'My image 4 '</span>, <span class="hljs-string">'My image 5 '</span>]
&gt;&gt;&gt; response.xpath(<span class="hljs-string">'.'</span>).re_first(<span class="hljs-string">'Name:\s(.*)&lt;br&gt;'</span>)
<span class="hljs-string">'My image 1 '</span>
</code></pre>

<p data-nodeid="65551">通过上面的例子，我们可以看到，直接调用 re 方法会提示没有 re 属性。但是这里首先调用了 <code data-backticks="1" data-nodeid="65554">xpath('.')</code>选中全文，然后调用 re 和 re_first 方法，就可以进行正则匹配了。</p>
<p data-nodeid="65552">以上内容便是 Scrapy 选择器的用法，它包括两个常用选择器和正则匹配功能。如果你熟练掌握 XPath 语法、CSS 选择器语法、正则表达式语法可以大大提高数据提取效率。</p>


# 功能强大的Middleware的用法
<p data-nodeid="328225" class="">我们在 Scrapy 架构中，可以看到有一个叫作 Middleware 的概念，中文翻译过来就叫作中间件，在 Scrapy 中有两种 Middleware，一种是 Spider Middleware，另一种是 Downloader Middleware，本节课我们分别来介绍下。</p>
<h3 data-nodeid="328226">Spider Middleware 的用法</h3>
<p data-nodeid="328227">Spider Middleware 是介入 Scrapy 的 Spider 处理机制的钩子框架。</p>
<p data-nodeid="328228">当 Downloader 生成 Response 之后，Response 会被发送给 Spider，在发送给 Spider 之前，Response 会首先经过 Spider Middleware 处理，当 Spider 处理生成 Item 和 Request 之后，Item 和 Request 还会经过 Spider Middleware 的处理。</p>
<p data-nodeid="328229">Spider Middleware 有如下三个作用。</p>
<ul data-nodeid="328230">
<li data-nodeid="328231">
<p data-nodeid="328232">我们可以在 Downloader 生成的 Response 发送给 Spider 之前，也就是在 Response 发送给 Spider 之前对 Response 进行处理。</p>
</li>
<li data-nodeid="328233">
<p data-nodeid="328234">我们可以在 Spider 生成的 Request 发送给 Scheduler 之前，也就是在 Request 发送给 Scheduler 之前对 Request 进行处理。</p>
</li>
<li data-nodeid="328235">
<p data-nodeid="328236">我们可以在 Spider 生成的 Item 发送给 Item Pipeline 之前，也就是在 Item 发送给 Item Pipeline 之前对 Item 进行处理。</p>
</li>
</ul>
<h4 data-nodeid="328237">使用说明</h4>
<p data-nodeid="328238">需要说明的是，Scrapy 其实已经提供了许多 Spider Middleware，它们被 SPIDER_MIDDLEWARES_BASE 这个变量所定义。</p>
<p data-nodeid="328239">SPIDER_MIDDLEWARES_BASE 变量的内容如下：</p>
<pre class="lang-java" data-nodeid="328240"><code data-language="java">{
 &nbsp; &nbsp;<span class="hljs-string">'scrapy.spidermiddlewares.httperror.HttpErrorMiddleware'</span>: <span class="hljs-number">50</span>,
 &nbsp; &nbsp;<span class="hljs-string">'scrapy.spidermiddlewares.offsite.OffsiteMiddleware'</span>: <span class="hljs-number">500</span>,
 &nbsp; &nbsp;<span class="hljs-string">'scrapy.spidermiddlewares.referer.RefererMiddleware'</span>: <span class="hljs-number">700</span>,
 &nbsp; &nbsp;<span class="hljs-string">'scrapy.spidermiddlewares.urllength.UrlLengthMiddleware'</span>: <span class="hljs-number">800</span>,
 &nbsp; &nbsp;<span class="hljs-string">'scrapy.spidermiddlewares.depth.DepthMiddleware'</span>: <span class="hljs-number">900</span>,
}
</code></pre>
<p data-nodeid="328241">和 Downloader Middleware 一样，Spider Middleware 首先加入 SPIDER_MIDDLEWARES 的设置中，该设置会和 Scrapy 中 SPIDER_MIDDLEWARES_BASE 定义的 Spider Middleware 合并。然后根据键值的数字优先级排序，得到一个有序列表。第一个 Middleware 是最靠近引擎的，最后一个 Middleware 是最靠近 Spider 的。</p>
<h4 data-nodeid="328242">核心方法</h4>
<p data-nodeid="328243">Scrapy 内置的 Spider Middleware 为 Scrapy 提供了基础的功能。如果我们想要扩展其功能，只需要实现某几个方法即可。</p>
<p data-nodeid="328244">每个 Spider Middleware 都定义了以下一个或多个方法的类，核心方法有如下 4 个。</p>
<ul data-nodeid="328245">
<li data-nodeid="328246">
<p data-nodeid="328247">process_spider_input(response, spider)</p>
</li>
<li data-nodeid="328248">
<p data-nodeid="328249">process_spider_output(response, result, spider)</p>
</li>
<li data-nodeid="328250">
<p data-nodeid="328251">process_spider_exception(response, exception, spider)</p>
</li>
<li data-nodeid="328252">
<p data-nodeid="328253">process_start_requests(start_requests, spider)</p>
</li>
</ul>
<p data-nodeid="328254">只需要实现其中一个方法就可以定义一个 Spider Middleware。下面我们来看看这 4 个方法的详细用法。</p>
<h5 data-nodeid="328255">process_spider_input(response, spider)</h5>
<p data-nodeid="328256">当 Response 通过 Spider Middleware 时，该方法被调用，处理该 Response。</p>
<p data-nodeid="328257">方法的参数有两个：</p>
<ul data-nodeid="328258">
<li data-nodeid="328259">
<p data-nodeid="328260">response，即 Response 对象，即被处理的 Response；</p>
</li>
<li data-nodeid="328261">
<p data-nodeid="328262">spider，即 Spider 对象，即该 response 对应的 Spider。</p>
</li>
</ul>
<p data-nodeid="328263">process_spider_input() 应该返回 None 或者抛出一个异常。</p>
<ul data-nodeid="328264">
<li data-nodeid="328265">
<p data-nodeid="328266">如果其返回 None，Scrapy 将会继续处理该 Response，调用所有其他的 Spider Middleware 直到 Spider 处理该 Response。</p>
</li>
<li data-nodeid="328267">
<p data-nodeid="328268">如果其抛出一个异常，Scrapy 将不会调用任何其他 Spider Middleware 的 process_spider_input() 方法，并调用 Request 的 errback() 方法。 errback 的输出将会以另一个方向被重新输入到中间件中，使用 process_spider_output() 方法来处理，当其抛出异常时则调用 process_spider_exception() 来处理。</p>
</li>
</ul>
<h5 data-nodeid="328269">process_spider_output(response, result, spider)</h5>
<p data-nodeid="328270">当 Spider 处理 Response 返回结果时，该方法被调用。</p>
<p data-nodeid="328271">方法的参数有三个：</p>
<ul data-nodeid="328272">
<li data-nodeid="328273">
<p data-nodeid="328274">response，即 Response 对象，即生成该输出的 Response；</p>
</li>
<li data-nodeid="328275">
<p data-nodeid="328276">result，包含 Request 或 Item 对象的可迭代对象，即 Spider 返回的结果；</p>
</li>
<li data-nodeid="328277">
<p data-nodeid="328278">spider，即 Spider 对象，即其结果对应的 Spider。</p>
</li>
</ul>
<p data-nodeid="328279">process_spider_output() 必须返回包含 Request 或 Item 对象的可迭代对象。</p>
<h5 data-nodeid="328280">process_spider_exception(response, exception, spider)</h5>
<p data-nodeid="328281">当 Spider 或 Spider Middleware 的 process_spider_input() 方法抛出异常时， 该方法被调用。</p>
<p data-nodeid="328282">方法的参数有三个：</p>
<ul data-nodeid="328283">
<li data-nodeid="328284">
<p data-nodeid="328285">response，即 Response 对象，即异常被抛出时被处理的 Response；</p>
</li>
<li data-nodeid="328286">
<p data-nodeid="328287">exception，即 Exception 对象，被抛出的异常；</p>
</li>
<li data-nodeid="328288">
<p data-nodeid="328289">spider，即 Spider 对象，即抛出该异常的 Spider。</p>
</li>
</ul>
<p data-nodeid="328290">process_spider_exception() 必须返回结果，要么返回 None ， 要么返回一个包含 Response 或 Item 对象的可迭代对象。</p>
<ul data-nodeid="328291">
<li data-nodeid="328292">
<p data-nodeid="328293">如果其返回 None ，Scrapy 将继续处理该异常，调用其他 Spider Middleware 中的 process_spider_exception() 方法，直到所有 Spider Middleware 都被调用。</p>
</li>
<li data-nodeid="328294">
<p data-nodeid="328295">如果其返回一个可迭代对象，则其他 Spider Middleware 的 process_spider_output() 方法被调用， 其他的 process_spider_exception() 将不会被调用。</p>
</li>
</ul>
<h5 data-nodeid="328296">process_start_requests(start_requests, spider)</h5>
<p data-nodeid="328297">该方法以 Spider 启动的 Request 为参数被调用，执行的过程类似于 process_spider_output() ，只不过其没有相关联的 Response 并且必须返回 Request。</p>
<p data-nodeid="328298">方法的参数有两个：</p>
<ul data-nodeid="328299">
<li data-nodeid="328300">
<p data-nodeid="328301">start_requests，即包含 Request 的可迭代对象，即 Start Requests；</p>
</li>
<li data-nodeid="328302">
<p data-nodeid="328303">spider，即 Spider 对象，即 Start Requests 所属的 Spider。</p>
</li>
</ul>
<p data-nodeid="328304">其必须返回另一个包含 Request 对象的可迭代对象。</p>
<h3 data-nodeid="328305">Downloader Middleware 的用法</h3>
<p data-nodeid="328306">Downloader Middleware 即下载中间件，它是处于 Scrapy 的 Request 和 Response 之间的处理模块。</p>
<p data-nodeid="328307">Scheduler 从队列中拿出一个 Request 发送给 Downloader 执行下载，这个过程会经过 Downloader Middleware 的处理。另外，当 Downloader 将 Request 下载完成得到 Response 返回给 Spider 时会再次经过 Downloader Middleware 处理。</p>
<p data-nodeid="328308">也就是说，Downloader Middleware 在整个架构中起作用的位置是以下两个。</p>
<ul data-nodeid="328309">
<li data-nodeid="328310">
<p data-nodeid="328311">在 Scheduler 调度出队列的 Request 发送给 Downloader 下载之前，也就是我们可以在 Request 执行下载之前对其进行修改。</p>
</li>
<li data-nodeid="328312">
<p data-nodeid="328313">在下载后生成的 Response 发送给 Spider 之前，也就是我们可以在生成 Resposne 被 Spider 解析之前对其进行修改。</p>
</li>
</ul>
<p data-nodeid="328314">Downloader Middleware 的功能十分强大，修改 User-Agent、处理重定向、设置代理、失败重试、设置 Cookies 等功能都需要借助它来实现。下面我们来了解一下 Downloader Middleware 的详细用法。</p>
<h4 data-nodeid="328315">使用说明</h4>
<p data-nodeid="328316">需要说明的是，Scrapy 其实已经提供了许多 Downloader Middleware，比如负责失败重试、自动重定向等功能的 Middleware，它们被 DOWNLOADER_MIDDLEWARES_BASE 变量所定义。</p>
<p data-nodeid="328317">DOWNLOADER_MIDDLEWARES_BASE 变量的内容如下所示：</p>
<pre class="lang-java" data-nodeid="328318"><code data-language="java">{
 &nbsp; &nbsp;<span class="hljs-string">'scrapy.downloadermiddlewares.robotstxt.RobotsTxtMiddleware'</span>: <span class="hljs-number">100</span>,
 &nbsp; &nbsp;<span class="hljs-string">'scrapy.downloadermiddlewares.httpauth.HttpAuthMiddleware'</span>: <span class="hljs-number">300</span>,
 &nbsp; &nbsp;<span class="hljs-string">'scrapy.downloadermiddlewares.downloadtimeout.DownloadTimeoutMiddleware'</span>: <span class="hljs-number">350</span>,
 &nbsp; &nbsp;<span class="hljs-string">'scrapy.downloadermiddlewares.defaultheaders.DefaultHeadersMiddleware'</span>: <span class="hljs-number">400</span>,
 &nbsp; &nbsp;<span class="hljs-string">'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware'</span>: <span class="hljs-number">500</span>,
 &nbsp; &nbsp;<span class="hljs-string">'scrapy.downloadermiddlewares.retry.RetryMiddleware'</span>: <span class="hljs-number">550</span>,
 &nbsp; &nbsp;<span class="hljs-string">'scrapy.downloadermiddlewares.ajaxcrawl.AjaxCrawlMiddleware'</span>: <span class="hljs-number">560</span>,
 &nbsp; &nbsp;<span class="hljs-string">'scrapy.downloadermiddlewares.redirect.MetaRefreshMiddleware'</span>: <span class="hljs-number">580</span>,
 &nbsp; &nbsp;<span class="hljs-string">'scrapy.downloadermiddlewares.httpcompression.HttpCompressionMiddleware'</span>: <span class="hljs-number">590</span>,
 &nbsp; &nbsp;<span class="hljs-string">'scrapy.downloadermiddlewares.redirect.RedirectMiddleware'</span>: <span class="hljs-number">600</span>,
 &nbsp; &nbsp;<span class="hljs-string">'scrapy.downloadermiddlewares.cookies.CookiesMiddleware'</span>: <span class="hljs-number">700</span>,
 &nbsp; &nbsp;<span class="hljs-string">'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware'</span>: <span class="hljs-number">750</span>,
 &nbsp; &nbsp;<span class="hljs-string">'scrapy.downloadermiddlewares.stats.DownloaderStats'</span>: <span class="hljs-number">850</span>,
 &nbsp; &nbsp;<span class="hljs-string">'scrapy.downloadermiddlewares.httpcache.HttpCacheMiddleware'</span>: <span class="hljs-number">900</span>,
}
</code></pre>
<p data-nodeid="328800">这是一个字典格式，字典的键名是 Scrapy 内置的 Downloader Middleware 的名称，键值代表了调用的优先级，优先级是一个数字，数字越小代表越靠近 Scrapy 引擎，数字越大代表越靠近 Downloader。每个 Downloader Middleware 都可以定义 process_request() 和 request_response() 方法来分别处理请求和响应，对于 process_request() 方法来说，优先级数字越小越先被调用，对于 process_response() 方法来说，优先级数字越大越先被调用。</p>
<p data-nodeid="328801">如果自己定义的 Downloader Middleware 要添加到项目里，DOWNLOADER_MIDDLEWARES_BASE 变量不能直接修改。Scrapy 提供了另外一个设置变量 DOWNLOADER_MIDDLEWARES，我们直接修改这个变量就可以添加自己定义的 Downloader Middleware，以及禁用 DOWNLOADER_MIDDLEWARES_BASE 里面定义的 Downloader Middleware。下面我们具体来看看 Downloader Middleware 的使用方法。</p>

<h4 data-nodeid="328320">核心方法</h4>
<p data-nodeid="328321">Scrapy 内置的 Downloader Middleware 为 Scrapy 提供了基础的功能，但在项目实战中我们往往需要单独定义 Downloader Middleware。不用担心，这个过程非常简单，我们只需要实现某几个方法即可。</p>
<p data-nodeid="328322">每个 Downloader Middleware 都定义了一个或多个方法的类，核心的方法有如下三个。</p>
<ul data-nodeid="328323">
<li data-nodeid="328324">
<p data-nodeid="328325">process_request(request, spider)</p>
</li>
<li data-nodeid="328326">
<p data-nodeid="328327">process_response(request, response, spider)</p>
</li>
<li data-nodeid="328328">
<p data-nodeid="328329">process_exception(request, exception, spider)</p>
</li>
</ul>
<p data-nodeid="328330">我们只需要实现至少一个方法，就可以定义一个 Downloader Middleware。下面我们来看看这三个方法的详细用法。</p>
<h5 data-nodeid="328331">process_request(request, spider)</h5>
<p data-nodeid="328332">Request 被 Scrapy 引擎调度给 Downloader 之前，process_request() 方法就会被调用，也就是在 Request 从队列里调度出来到 Downloader 下载执行之前，我们都可以用 process_request() 方法对 Request 进行处理。方法的返回值必须为 None、Response 对象、Request 对象之一，或者抛出 IgnoreRequest 异常。</p>
<p data-nodeid="328333">process_request() 方法的参数有如下两个。</p>
<ul data-nodeid="328334">
<li data-nodeid="328335">
<p data-nodeid="328336">request，即 Request 对象，即被处理的 Request；</p>
</li>
<li data-nodeid="328337">
<p data-nodeid="328338">spider，即 Spider 对象，即此 Request 对应的 Spider。</p>
</li>
</ul>
<p data-nodeid="328339">返回类型不同，产生的效果也不同。下面归纳一下不同的返回情况。</p>
<ul data-nodeid="328340">
<li data-nodeid="328341">
<p data-nodeid="328342">当返回为 None 时，Scrapy 将继续处理该 Request，接着执行其他 Downloader Middleware 的 process_request() 方法，直到 Downloader 把 Request 执行后得到 Response 才结束。这个过程其实就是修改 Request 的过程，不同的 Downloader Middleware 按照设置的优先级顺序依次对 Request 进行修改，最后推送至 Downloader 执行。</p>
</li>
<li data-nodeid="328343">
<p data-nodeid="328344">当返回为 Response 对象时，更低优先级的 Downloader Middleware 的 process_request() 和 process_exception() 方法就不会被继续调用，每个 Downloader Middleware 的 process_response() 方法转而被依次调用。调用完毕之后，直接将 Response 对象发送给 Spider 来处理。</p>
</li>
<li data-nodeid="328345">
<p data-nodeid="328346">当返回为 Request 对象时，更低优先级的 Downloader Middleware 的 process_request() 方法会停止执行。这个 Request 会重新放到调度队列里，其实它就是一个全新的 Request，等待被调度。如果被 Scheduler 调度了，那么所有的 Downloader Middleware 的 process_request() 方法会被重新按照顺序执行。</p>
</li>
<li data-nodeid="328347">
<p data-nodeid="328348">如果 IgnoreRequest 异常抛出，则所有的 Downloader Middleware 的 process_exception() 方法会依次执行。如果没有一个方法处理这个异常，那么 Request 的 errorback() 方法就会回调。如果该异常还没有被处理，那么它便会被忽略。</p>
</li>
</ul>
<h5 data-nodeid="328349">process_response(request, response, spider)</h5>
<p data-nodeid="328350">Downloader 执行 Request 下载之后，会得到对应的 Response。Scrapy 引擎便会将 Response 发送给 Spider 进行解析。在发送之前，我们都可以用 process_response() 方法来对 Response 进行处理。方法的返回值必须为 Request 对象、Response 对象之一，或者抛出 IgnoreRequest 异常。</p>
<p data-nodeid="328351">process_response() 方法的参数有如下三个。</p>
<ul data-nodeid="328352">
<li data-nodeid="328353">
<p data-nodeid="328354">request，是 Request 对象，即此 Response 对应的 Request。</p>
</li>
<li data-nodeid="328355">
<p data-nodeid="328356">response，是 Response 对象，即此被处理的 Response。</p>
</li>
<li data-nodeid="328357">
<p data-nodeid="328358">spider，是 Spider 对象，即此 Response 对应的 Spider。</p>
</li>
</ul>
<p data-nodeid="328359">下面对不同的返回情况做一下归纳：</p>
<ul data-nodeid="328360">
<li data-nodeid="328361">
<p data-nodeid="328362">当返回为 Request 对象时，更低优先级的 Downloader Middleware 的 process_response() 方法不会继续调用。该 Request 对象会重新放到调度队列里等待被调度，它相当于一个全新的 Request。然后，该 Request 会被 process_request() 方法顺次处理。</p>
</li>
<li data-nodeid="328363">
<p data-nodeid="328364">当返回为 Response 对象时，更低优先级的 Downloader Middleware 的 process_response() 方法会继续调用，继续对该 Response 对象进行处理。</p>
</li>
<li data-nodeid="328365">
<p data-nodeid="328366">如果 IgnoreRequest 异常抛出，则 Request 的 errorback() 方法会回调。如果该异常还没有被处理，那么它便会被忽略。</p>
</li>
</ul>
<h5 data-nodeid="328367">process_exception(request, exception, spider)</h5>
<p data-nodeid="328368">当 Downloader 或 process_request() 方法抛出异常时，例如抛出 IgnoreRequest 异常，process_exception() 方法就会被调用。方法的返回值必须为 None、Response 对象、Request 对象之一。</p>
<p data-nodeid="328369">process_exception() 方法的参数有如下三个。</p>
<ul data-nodeid="328370">
<li data-nodeid="328371">
<p data-nodeid="328372">request，即 Request 对象，即产生异常的 Request。</p>
</li>
<li data-nodeid="328373">
<p data-nodeid="328374">exception，即 Exception 对象，即抛出的异常。</p>
</li>
<li data-nodeid="328375">
<p data-nodeid="328376">spdier，即 Spider 对象，即 Request 对应的 Spider。</p>
</li>
</ul>
<p data-nodeid="328377">下面归纳一下不同的返回值。</p>
<ul data-nodeid="328378">
<li data-nodeid="328379">
<p data-nodeid="328380">当返回为 None 时，更低优先级的 Downloader Middleware 的 process_exception() 会被继续顺次调用，直到所有的方法都被调度完毕。</p>
</li>
<li data-nodeid="328381">
<p data-nodeid="328382">当返回为 Response 对象时，更低优先级的 Downloader Middleware 的 process_exception() 方法不再被继续调用，每个 Downloader Middleware 的 process_response() 方法转而被依次调用。</p>
</li>
<li data-nodeid="328383">
<p data-nodeid="328384">当返回为 Request 对象时，更低优先级的 Downloader Middleware 的 process_exception() 也不再被继续调用，该 Request 对象会重新放到调度队列里面等待被调度，它相当于一个全新的 Request。然后，该 Request 又会被 process_request() 方法顺次处理。</p>
</li>
</ul>
<p data-nodeid="328385">以上内容便是这三个方法的详细使用逻辑。在使用它们之前，请先对这三个方法的返回值的处理情况有一个清晰的认识。在自定义 Downloader Middleware 的时候，也一定要注意每个方法的返回类型。</p>
<p data-nodeid="328386">下面我们用一个案例实战来加深一下对 Downloader Middleware 用法的理解。</p>
<h4 data-nodeid="328387">项目实战</h4>
<p data-nodeid="328388">新建一个项目，命令如下所示：</p>
<pre class="lang-java" data-nodeid="328389"><code data-language="java">scrapy startproject scrapydownloadertest
</code></pre>
<p data-nodeid="328390">新建了一个 Scrapy 项目，名为 scrapydownloadertest。进入项目，新建一个 Spider，命令如下所示：</p>
<pre class="lang-java" data-nodeid="328391"><code data-language="java">scrapy genspider httpbin httpbin.org
</code></pre>
<p data-nodeid="328392">新建了一个 Spider，名为 httpbin，源代码如下所示：</p>
<pre class="lang-java" data-nodeid="328393"><code data-language="java"><span class="hljs-function"><span class="hljs-keyword">import</span> scrapy
class <span class="hljs-title">HttpbinSpider</span><span class="hljs-params">(scrapy.Spider)</span>:
 &nbsp; &nbsp;name </span>= <span class="hljs-string">'httpbin'</span>
 &nbsp; &nbsp;allowed_domains = [<span class="hljs-string">'httpbin.org'</span>]
 &nbsp; &nbsp;start_urls = [<span class="hljs-string">'http://httpbin.org/'</span>]

 &nbsp; &nbsp;<span class="hljs-function">def <span class="hljs-title">parse</span><span class="hljs-params">(self, response)</span>:
 &nbsp; &nbsp; &nbsp; &nbsp;pass
</span></code></pre>
<p data-nodeid="328394">接下来我们修改 start_urls 为：<code data-backticks="1" data-nodeid="328716">['http://httpbin.org/']</code>。随后将 parse() 方法添加一行日志输出，将 response 变量的 text 属性输出，这样我们便可以看到 Scrapy 发送的 Request 信息了。<br>
修改 Spider 内容如下所示：</p>
<pre class="lang-dart" data-nodeid="328395"><code data-language="dart"><span class="hljs-keyword">import</span> scrapy

<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">HttpbinSpider</span>(<span class="hljs-title">scrapy</span>.<span class="hljs-title">Spider</span>):
 &nbsp; &nbsp;<span class="hljs-title">name</span> = '<span class="hljs-title">httpbin</span>'
 &nbsp; &nbsp;<span class="hljs-title">allowed_domains</span> = ['<span class="hljs-title">httpbin</span>.<span class="hljs-title">org</span>']
 &nbsp; &nbsp;<span class="hljs-title">start_urls</span> = ['<span class="hljs-title">http</span>://<span class="hljs-title">httpbin</span>.<span class="hljs-title">org</span>/<span class="hljs-title">get</span>']

 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">parse</span>(<span class="hljs-title">self</span>, <span class="hljs-title">response</span>):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">self</span>.<span class="hljs-title">logger</span>.<span class="hljs-title">debug</span>(<span class="hljs-title">response</span>.<span class="hljs-title">text</span>)
</span></code></pre>
<p data-nodeid="328396">接下来运行此 Spider，执行如下命令：</p>
<pre class="lang-java" data-nodeid="328397"><code data-language="java">scrapy crawl httpbin
</code></pre>
<p data-nodeid="328398">Scrapy 运行结果包含 Scrapy 发送的 Request 信息，内容如下所示：</p>
<pre class="lang-java" data-nodeid="328399"><code data-language="java">{<span class="hljs-string">"args"</span>: {}, 
 &nbsp;<span class="hljs-string">"headers"</span>: {
 &nbsp; &nbsp;<span class="hljs-string">"Accept"</span>: <span class="hljs-string">"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"</span>, 
 &nbsp; &nbsp;<span class="hljs-string">"Accept-Encoding"</span>: <span class="hljs-string">"gzip,deflate,br"</span>, 
 &nbsp; &nbsp;<span class="hljs-string">"Accept-Language"</span>: <span class="hljs-string">"en"</span>, 
 &nbsp; &nbsp;<span class="hljs-string">"Connection"</span>: <span class="hljs-string">"close"</span>, 
 &nbsp; &nbsp;<span class="hljs-string">"Host"</span>: <span class="hljs-string">"httpbin.org"</span>, 
 &nbsp; &nbsp;<span class="hljs-string">"User-Agent"</span>: <span class="hljs-string">"Scrapy/1.4.0 (+http://scrapy.org)"</span>
  }, 
 &nbsp;<span class="hljs-string">"origin"</span>: <span class="hljs-string">"60.207.237.85"</span>, 
 &nbsp;<span class="hljs-string">"url"</span>: <span class="hljs-string">"http://httpbin.org/get"</span>
}
</code></pre>
<p data-nodeid="328400">我们观察一下 Headers，Scrapy 发送的 Request 使用的 User-Agent 是 Scrapy/1.4.0(+<a href="http://scrapy.org" data-nodeid="328725">http://scrapy.org</a>)，这其实是由 Scrapy 内置的 UserAgentMiddleware 设置的，UserAgentMiddleware 的源码如下所示：</p>
<pre class="lang-dart" data-nodeid="328401"><code data-language="dart">from scrapy <span class="hljs-keyword">import</span> signals

<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">UserAgentMiddleware</span>(<span class="hljs-title">object</span>):
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">__init__</span>(<span class="hljs-title">self</span>, <span class="hljs-title">user_agent</span>='<span class="hljs-title">Scrapy</span>'):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">self</span>.<span class="hljs-title">user_agent</span> = <span class="hljs-title">user_agent</span>

 &nbsp; &nbsp;@<span class="hljs-title">classmethod</span>
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">from_crawler</span>(<span class="hljs-title">cls</span>, <span class="hljs-title">crawler</span>):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">o</span> = <span class="hljs-title">cls</span>(<span class="hljs-title">crawler</span>.<span class="hljs-title">settings</span>['<span class="hljs-title">USER_AGENT</span>'])
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">crawler</span>.<span class="hljs-title">signals</span>.<span class="hljs-title">connect</span>(<span class="hljs-title">o</span>.<span class="hljs-title">spider_opened</span>, <span class="hljs-title">signal</span>=<span class="hljs-title">signals</span>.<span class="hljs-title">spider_opened</span>)
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">return</span> <span class="hljs-title">o</span>

 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">spider_opened</span>(<span class="hljs-title">self</span>, <span class="hljs-title">spider</span>):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">self</span>.<span class="hljs-title">user_agent</span> = <span class="hljs-title">getattr</span>(<span class="hljs-title">spider</span>, '<span class="hljs-title">user_agent</span>', <span class="hljs-title">self</span>.<span class="hljs-title">user_agent</span>)

 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">process_request</span>(<span class="hljs-title">self</span>, <span class="hljs-title">request</span>, <span class="hljs-title">spider</span>):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">if</span> <span class="hljs-title">self</span>.<span class="hljs-title">user_agent</span>:
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">request</span>.<span class="hljs-title">headers</span>.<span class="hljs-title">setdefault</span>(<span class="hljs-title">b</span>'<span class="hljs-title">User</span>-<span class="hljs-title">Agent</span>', <span class="hljs-title">self</span>.<span class="hljs-title">user_agent</span>)
</span></code></pre>
<p data-nodeid="328402">在 from_crawler() 方法中，首先尝试获取 settings 里面的 USER_AGENT，然后把 USER_AGENT 传递给 <strong data-nodeid="328756">init</strong>() 方法进行初始化，其参数就是 user_agent。如果没有传递 USER_AGENT 参数就默认设置为 Scrapy 字符串。我们新建的项目没有设置 USER_AGENT，所以这里的 user_agent 变量就是 Scrapy。接下来，在 process_request() 方法中，将 user-agent 变量设置为 headers 变量的一个属性，这样就成功设置了 User-Agent。因此，User-Agent 就是通过此 Downloader Middleware 的 process_request() 方法设置的。<br>
修改请求时的 User-Agent 可以有两种方式：一是修改 settings 里面的 USER_AGENT 变量；二是通过 Downloader Middleware 的 process_request() 方法来修改。</p>
<p data-nodeid="328403">第一种方法非常简单，我们只需要在 setting.py 里面加一行 USER_AGENT 的定义即可：</p>
<pre class="lang-java" data-nodeid="328404"><code data-language="java">USER_AGENT = <span class="hljs-string">'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'</span>
</code></pre>
<p data-nodeid="328405">一般推荐使用此方法来设置。但是如果想设置得更灵活，比如设置随机的 User-Agent，那就需要借助 Downloader Middleware 了。所以接下来我们用 Downloader Middleware 实现一个随机 User-Agent 的设置。</p>
<p data-nodeid="328406">在 middlewares.py 里面添加一个 RandomUserAgentMiddleware 的类，如下所示：</p>
<pre class="lang-dart" data-nodeid="328407"><code data-language="dart"><span class="hljs-keyword">import</span> random

<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">RandomUserAgentMiddleware</span>():
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">__init__</span>(<span class="hljs-title">self</span>):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">self</span>.<span class="hljs-title">user_agents</span> = ['<span class="hljs-title">Mozilla</span>/5.0 (<span class="hljs-title">Windows</span>; <span class="hljs-title">U</span>; <span class="hljs-title">MSIE</span> 9.0; <span class="hljs-title">Windows</span> <span class="hljs-title">NT</span> 9.0; <span class="hljs-title">en</span>-<span class="hljs-title">US</span>)',
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;'<span class="hljs-title">Mozilla</span>/5.0 (<span class="hljs-title">Windows</span> <span class="hljs-title">NT</span> 6.1) <span class="hljs-title">AppleWebKit</span>/537.2 (<span class="hljs-title">KHTML</span>, <span class="hljs-title">like</span> <span class="hljs-title">Gecko</span>) <span class="hljs-title">Chrome</span>/22.0.1216.0 <span class="hljs-title">Safari</span>/537.2',
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;'<span class="hljs-title">Mozilla</span>/5.0 (<span class="hljs-title">X11</span>; <span class="hljs-title">Ubuntu</span>; <span class="hljs-title">Linux</span> <span class="hljs-title">i686</span>; <span class="hljs-title">rv</span>:15.0) <span class="hljs-title">Gecko</span>/20100101 <span class="hljs-title">Firefox</span>/15.0.1'
 &nbsp; &nbsp; &nbsp;  ]

 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">process_request</span>(<span class="hljs-title">self</span>, <span class="hljs-title">request</span>, <span class="hljs-title">spider</span>):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">request</span>.<span class="hljs-title">headers</span>['<span class="hljs-title">User</span>-<span class="hljs-title">Agent</span>'] = <span class="hljs-title">random</span>.<span class="hljs-title">choice</span>(<span class="hljs-title">self</span>.<span class="hljs-title">user_agents</span>)
</span></code></pre>
<p data-nodeid="328408">我们首先在类的 __init__() 方法中定义了三个不同的 User-Agent，并用一个列表来表示。接下来实现了 process_request() 方法，它有一个参数 request，我们直接修改 request 的属性即可。在这里我们直接设置了 request 对象的 headers 属性的 User-Agent，设置内容是随机选择的 User-Agent，这样一个 Downloader Middleware 就写好了。</p>
<p data-nodeid="328409">不过，要使之生效我们还需要再去调用这个 Downloader Middleware。在 settings.py 中，将 DOWNLOADER_MIDDLEWARES 取消注释，并设置成如下内容：</p>
<pre class="lang-java" data-nodeid="328410"><code data-language="java">DOWNLOADER_MIDDLEWARES = {<span class="hljs-string">'scrapydownloadertest.middlewares.RandomUserAgentMiddleware'</span>: <span class="hljs-number">543</span>,}
</code></pre>
<p data-nodeid="328411">接下来我们重新运行 Spider，就可以看到 User-Agent 被成功修改为列表中所定义的随机的一个 User-Agent 了：</p>
<pre class="lang-java" data-nodeid="328412"><code data-language="java">{<span class="hljs-string">"args"</span>: {}, 
 &nbsp;<span class="hljs-string">"headers"</span>: {
 &nbsp; &nbsp;<span class="hljs-string">"Accept"</span>: <span class="hljs-string">"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"</span>, 
 &nbsp; &nbsp;<span class="hljs-string">"Accept-Encoding"</span>: <span class="hljs-string">"gzip,deflate,br"</span>, 
 &nbsp; &nbsp;<span class="hljs-string">"Accept-Language"</span>: <span class="hljs-string">"en"</span>, 
 &nbsp; &nbsp;<span class="hljs-string">"Connection"</span>: <span class="hljs-string">"close"</span>, 
 &nbsp; &nbsp;<span class="hljs-string">"Host"</span>: <span class="hljs-string">"httpbin.org"</span>, 
 &nbsp; &nbsp;<span class="hljs-string">"User-Agent"</span>: <span class="hljs-string">"Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)"</span>
  }, 
 &nbsp;<span class="hljs-string">"origin"</span>: <span class="hljs-string">"60.207.237.85"</span>, 
 &nbsp;<span class="hljs-string">"url"</span>: <span class="hljs-string">"http://httpbin.org/get"</span>
}
</code></pre>
<p data-nodeid="328413">我们就通过实现 Downloader Middleware 并利用 process_request() 方法成功设置了随机的 User-Agent。</p>
<p data-nodeid="328414">另外，Downloader Middleware 还有 process_response() 方法。Downloader 对 Request 执行下载之后会得到 Response，随后 Scrapy 引擎会将 Response 发送回 Spider 进行处理。但是在 Response 被发送给 Spider 之前，我们同样可以使用 process_response() 方法对 Response 进行处理。比如这里修改一下 Response 的状态码，在 RandomUserAgentMiddleware 添加如下代码：</p>
<pre class="lang-java" data-nodeid="328415"><code data-language="java"><span class="hljs-function">def <span class="hljs-title">process_response</span><span class="hljs-params">(self, request, response, spider)</span>:
 &nbsp; &nbsp;response.status </span>= <span class="hljs-number">201</span>
 &nbsp; &nbsp;<span class="hljs-keyword">return</span> response
</code></pre>
<p data-nodeid="328416">我们将 response 对象的 status 属性修改为 201，随后将 response 返回，这个被修改后的 Response 就会被发送到 Spider。</p>
<p data-nodeid="328417">我们再在 Spider 里面输出修改后的状态码，在 parse() 方法中添加如下的输出语句：</p>
<pre class="lang-java" data-nodeid="328418"><code data-language="java">self.logger.debug(<span class="hljs-string">'Status Code: '</span> + str(response.status))
</code></pre>
<p data-nodeid="328419">重新运行之后，控制台输出了如下内容：</p>
<pre class="lang-java" data-nodeid="328420"><code data-language="java">[httpbin] DEBUG: Status Code: <span class="hljs-number">201</span>
</code></pre>
<p data-nodeid="328421">可以发现，Response 的状态码成功修改了。因此要想对 Response 进行处理，就可以借助于 process_response() 方法。</p>
<p data-nodeid="328422">另外还有一个 process_exception() 方法，它是用来处理异常的方法。如果需要异常处理的话，我们可以调用此方法。不过这个方法的使用频率相对低一些，在此不用实例演示。</p>
<h3 data-nodeid="328423">本节代码</h3>
<p data-nodeid="329972" class="te-preview-highlight">本节源代码为：<br>
<a href="https://github.com/Python3WebSpider/ScrapyDownloaderTest" data-nodeid="329977">https://github.com/Python3WebSpider/ScrapyDownloaderTest</a>。</p>

<h3 data-nodeid="328425">结语</h3>
<p data-nodeid="328426" class="">本节讲解了 Spider Middleware 和 Downloader Middleware 的基本用法。利用它们我们可以方便地实现爬虫逻辑的灵活处理，需要好好掌握。</p>


# 哪都能存，Item Pipeline的用法
<p data-nodeid="1449" class="">在前面的示例中我们已经了解了 Item Pipeline 项目管道的基本概念，本节课我们就深入详细讲解它的用法。</p>
<p data-nodeid="1450">首先我们看看 Item Pipeline 在 Scrapy 中的架构，如图所示。</p>
<p data-nodeid="1451"><img src="https://s0.lgstatic.com/i/image/M00/32/C9/Ciqc1F8OyNqAAbnKAAJygBiwVD4320.png" alt="Drawing 0.png" data-nodeid="1597"></p>
<p data-nodeid="1452">图中的最左侧即为 Item Pipeline，它的调用发生在 Spider 产生 Item 之后。当 Spider 解析完 Response 之后，Item 就会传递到 Item Pipeline，被定义的 Item Pipeline 组件会顺次调用，完成一连串的处理过程，比如数据清洗、存储等。</p>
<p data-nodeid="1453">它的主要功能有：</p>
<ul data-nodeid="1454">
<li data-nodeid="1455">
<p data-nodeid="1456">清洗 HTML 数据；</p>
</li>
<li data-nodeid="1457">
<p data-nodeid="1458">验证爬取数据，检查爬取字段；</p>
</li>
<li data-nodeid="1459">
<p data-nodeid="1460">查重并丢弃重复内容；</p>
</li>
<li data-nodeid="1461">
<p data-nodeid="1462">将爬取结果储存到数据库。</p>
</li>
</ul>
<h3 data-nodeid="1463">1. 核心方法</h3>
<p data-nodeid="1464">我们可以自定义 Item Pipeline，只需要实现指定的方法就可以，其中必须要实现的一个方法是：</p>
<ul data-nodeid="1465">
<li data-nodeid="1466">
<p data-nodeid="1467">process_item(item, spider)</p>
</li>
</ul>
<p data-nodeid="1468">另外还有几个比较实用的方法，它们分别是：</p>
<ul data-nodeid="1469">
<li data-nodeid="1470">
<p data-nodeid="1471">open_spider(spider)</p>
</li>
<li data-nodeid="1472">
<p data-nodeid="1473">close_spider(spider)</p>
</li>
<li data-nodeid="1474">
<p data-nodeid="1475">from_crawler(cls, crawler)</p>
</li>
</ul>
<p data-nodeid="1476">下面我们对这几个方法的用法做下详细的介绍：</p>
<h4 data-nodeid="1477">process_item(item, spider)</h4>
<p data-nodeid="1478">process_item() 是必须要实现的方法，被定义的 Item Pipeline 会默认调用这个方法对 Item 进行处理。比如，我们可以进行数据处理或者将数据写入数据库等操作。它必须返回 Item 类型的值或者抛出一个 DropItem 异常。</p>
<p data-nodeid="1479">process_item() 方法的参数有如下两个：</p>
<ul data-nodeid="1480">
<li data-nodeid="1481">
<p data-nodeid="1482">item，是 Item 对象，即被处理的 Item；</p>
</li>
<li data-nodeid="1483">
<p data-nodeid="1484">spider，是 Spider 对象，即生成该 Item 的 Spider。</p>
</li>
</ul>
<p data-nodeid="1485">下面对该方法的返回类型归纳如下：</p>
<ul data-nodeid="1486">
<li data-nodeid="1487">
<p data-nodeid="1488">如果返回的是 Item 对象，那么此 Item 会被低优先级的 Item Pipeline 的 process_item() 方法进行处理，直到所有的方法被调用完毕。</p>
</li>
<li data-nodeid="1489">
<p data-nodeid="1490">如果抛出的是 DropItem 异常，那么此 Item 就会被丢弃，不再进行处理。</p>
</li>
</ul>
<h4 data-nodeid="1491">open_spider(self, spider)</h4>
<p data-nodeid="1492">open_spider() 方法是在 Spider 开启的时候被自动调用的，在这里我们可以做一些初始化操作，如开启数据库连接等。其中参数 spider 就是被开启的 Spider 对象。</p>
<h4 data-nodeid="1493">close_spider(spider)</h4>
<p data-nodeid="1494">close_spider() 方法是在 Spider 关闭的时候自动调用的，在这里我们可以做一些收尾工作，如关闭数据库连接等，其中参数 spider 就是被关闭的 Spider 对象。</p>
<h4 data-nodeid="1495">from_crawler(cls, crawler)</h4>
<p data-nodeid="1496">from_crawler() 方法是一个类方法，用 @classmethod 标识，是一种依赖注入的方式。它的参数是 crawler，通过 crawler 对象，我们可以拿到 Scrapy 的所有核心组件，如全局配置的每个信息，然后创建一个 Pipeline 实例。参数 cls 就是 Class，最后返回一个 Class 实例。</p>
<p data-nodeid="1497">下面我们用一个实例来加深对 Item Pipeline 用法的理解。</p>
<h3 data-nodeid="1498">2. 本节目标</h3>
<p data-nodeid="1499">我们以爬取 360 摄影美图为例，来分别实现 MongoDB 存储、MySQL 存储、Image 图片存储的三个 Pipeline。</p>
<h3 data-nodeid="1500">3. 准备工作</h3>
<p data-nodeid="1501">请确保已经安装好 MongoDB 和 MySQL 数据库，安装好 Python 的 PyMongo、PyMySQL、Scrapy 框架，另外需要安装 pillow 图像处理库，如果没有安装可以参考前文的安装说明。</p>
<h3 data-nodeid="1502">4. 抓取分析</h3>
<p data-nodeid="1503">我们这次爬取的目标网站为：<a href="https://image.so.com" data-nodeid="1671">https://image.so.com</a>。打开此页面，切换到摄影页面，网页中呈现了许许多多的摄影美图。我们打开浏览器开发者工具，过滤器切换到 XHR 选项，然后下拉页面，可以看到下面就会呈现许多 Ajax 请求，如图所示。</p>
<p data-nodeid="1504"><img src="https://s0.lgstatic.com/i/image/M00/32/D4/CgqCHl8OyPCATwEaAA4FKjXCH38464.png" alt="Drawing 1.png" data-nodeid="1675"></p>
<p data-nodeid="1505">我们查看一个请求的详情，观察返回的数据结构，如图所示。</p>
<p data-nodeid="1506"><img src="https://s0.lgstatic.com/i/image/M00/32/C9/Ciqc1F8OyPiAHQO_AAKWgKacbAY613.png" alt="Drawing 2.png" data-nodeid="1679"></p>
<p data-nodeid="1507">返回格式是 JSON。其中 list 字段就是一张张图片的详情信息，包含了 30 张图片的 ID、名称、链接、缩略图等信息。另外观察 Ajax 请求的参数信息，有一个参数 sn 一直在变化，这个参数很明显就是偏移量。当 sn 为 30 时，返回的是前 30 张图片，sn 为 60 时，返回的就是第 31~60 张图片。另外，ch 参数是摄影类别，listtype 是排序方式，temp 参数可以忽略。</p>
<p data-nodeid="1508">所以我们抓取时只需要改变 sn 的数值就好了。下面我们用 Scrapy 来实现图片的抓取，将图片的信息保存到 MongoDB、MySQL，同时将图片存储到本地。</p>
<h3 data-nodeid="1509">5. 新建项目</h3>
<p data-nodeid="1510">首先新建一个项目，命令如下：</p>
<pre class="lang-java" data-nodeid="1511"><code data-language="java">scrapy startproject images360
</code></pre>
<p data-nodeid="1512">接下来新建一个 Spider，命令如下：</p>
<pre class="lang-java" data-nodeid="1513"><code data-language="java">scrapy genspider images images.so.com
</code></pre>
<p data-nodeid="1514">这样我们就成功创建了一个 Spider。</p>
<h3 data-nodeid="1515">6. 构造请求</h3>
<p data-nodeid="1516">接下来定义爬取的页数。比如爬取 50 页、每页 30 张，也就是 1500 张图片，我们可以先在 settings.py 里面定义一个变量 MAX_PAGE，添加如下定义：</p>
<pre class="lang-java" data-nodeid="1517"><code data-language="java">MAX_PAGE = <span class="hljs-number">50</span>
</code></pre>
<p data-nodeid="1518">定义 start_requests() 方法，用来生成 50 次请求，如下所示：</p>
<pre class="lang-java" data-nodeid="1519"><code data-language="java"><span class="hljs-function">def <span class="hljs-title">start_requests</span><span class="hljs-params">(self)</span>:
 &nbsp; &nbsp;data </span>= {<span class="hljs-string">'ch'</span>: <span class="hljs-string">'photography'</span>, <span class="hljs-string">'listtype'</span>: <span class="hljs-string">'new'</span>}
 &nbsp; &nbsp;base_url = <span class="hljs-string">'https://image.so.com/zjl?'</span>
 &nbsp; &nbsp;<span class="hljs-function"><span class="hljs-keyword">for</span> page in <span class="hljs-title">range</span><span class="hljs-params">(<span class="hljs-number">1</span>, self.settings.get(<span class="hljs-string">'MAX_PAGE'</span>)</span> + 1):
 &nbsp; &nbsp; &nbsp; &nbsp;data['sn'] </span>= page * <span class="hljs-number">30</span>
 &nbsp; &nbsp; &nbsp; &nbsp;params = urlencode(data)
 &nbsp; &nbsp; &nbsp; &nbsp;url = base_url + <span class="hljs-function">params
 &nbsp; &nbsp; &nbsp; &nbsp;yield <span class="hljs-title">Request</span><span class="hljs-params">(url, self.parse)</span>
</span></code></pre>
<p data-nodeid="1520">在这里我们首先定义了初始的两个参数，sn 参数是遍历循环生成的。然后利用 urlencode 方法将字典转化为 URL 的 GET 参数，构造出完整的 URL，构造并生成 Request。</p>
<p data-nodeid="1521">还需要引入 scrapy.Request 和 urllib.parse 模块，如下所示：</p>
<pre class="lang-java" data-nodeid="1522"><code data-language="java">from scrapy <span class="hljs-keyword">import</span> Spider, Request
from urllib.parse <span class="hljs-keyword">import</span> urlencode
</code></pre>
<p data-nodeid="1523">再修改 settings.py 中的 ROBOTSTXT_OBEY 变量，将其设置为 False，否则无法抓取，如下所示：</p>
<pre class="lang-java" data-nodeid="1524"><code data-language="java">ROBOTSTXT_OBEY = False
</code></pre>
<p data-nodeid="1525">运行爬虫，即可以看到链接都请求成功，执行命令如下所示：</p>
<pre class="lang-java" data-nodeid="1526"><code data-language="java">scrapy crawl images
</code></pre>
<p data-nodeid="1527">运行示例结果如图所示。</p>
<p data-nodeid="1528"><img src="https://s0.lgstatic.com/i/image/M00/32/C9/Ciqc1F8OyReAWgKjAAGstwAqxoU157.png" alt="Drawing 3.png" data-nodeid="1708"></p>
<p data-nodeid="1529">所有请求的状态码都是 200，这就证明图片信息爬取成功了。</p>
<h3 data-nodeid="1530">7. 提取信息</h3>
<p data-nodeid="1531">首先定义一个叫作 ImageItem 的 Item，如下所示：</p>
<pre class="lang-java" data-nodeid="1532"><code data-language="java">from scrapy <span class="hljs-keyword">import</span> Item, <span class="hljs-function">Field
class <span class="hljs-title">ImageItem</span><span class="hljs-params">(Item)</span>:
 &nbsp; &nbsp;collection </span>= table = <span class="hljs-string">'images'</span>
 &nbsp; &nbsp;id = Field()
 &nbsp; &nbsp;url = Field()
 &nbsp; &nbsp;title = Field()
 &nbsp; &nbsp;thumb = Field()
</code></pre>
<p data-nodeid="1533">在这里我们定义了 4 个字段，包括图片的 ID、链接、标题、缩略图。另外还有两个属性 collection 和 table，都定义为 images 字符串，分别代表 MongoDB 存储的 Collection 名称和 MySQL 存储的表名称。<br>
接下来我们提取 Spider 里有关信息，将 parse 方法改写为如下所示：</p>
<pre class="lang-java" data-nodeid="1534"><code data-language="java"><span class="hljs-function">def <span class="hljs-title">parse</span><span class="hljs-params">(self, response)</span>:
    result </span>= json.loads(response.text)
    <span class="hljs-keyword">for</span> image in result.get(<span class="hljs-string">'list'</span>):
        item = ImageItem()
        item[<span class="hljs-string">'id'</span>] = image.get(<span class="hljs-string">'id'</span>)
        item[<span class="hljs-string">'url'</span>] = image.get(<span class="hljs-string">'qhimg_url'</span>)
        item[<span class="hljs-string">'title'</span>] = image.get(<span class="hljs-string">'title'</span>)
        item[<span class="hljs-string">'thumb'</span>] = image.get(<span class="hljs-string">'qhimg_thumb'</span>)
        yield item
</code></pre>
<p data-nodeid="1535">首先解析 JSON，遍历其 list 字段，取出一个个图片信息，然后再对 ImageItem 进行赋值，生成 Item 对象。<br>
这样我们就完成了信息的提取。</p>
<h3 data-nodeid="1536">8. 存储信息</h3>
<p data-nodeid="1537">接下来我们需要将图片的信息保存到 MongoDB、MySQL 中，同时将图片保存到本地。</p>
<h4 data-nodeid="1538">MongoDB</h4>
<p data-nodeid="1539">首先确保 MongoDB 已经正常安装并且能够正常运行。</p>
<p data-nodeid="1540">我们用一个 MongoPipeline 将信息保存到 MongoDB 中，在 pipelines.py 里添加如下类的实现：</p>
<pre class="lang-java" data-nodeid="1541"><code data-language="java"><span class="hljs-function"><span class="hljs-keyword">import</span> pymongo
class <span class="hljs-title">MongoPipeline</span><span class="hljs-params">(object)</span>:
    def <span class="hljs-title">__init__</span><span class="hljs-params">(self, mongo_uri, mongo_db)</span>:
        self.mongo_uri </span>= mongo_uri
        self.mongo_db = mongo_db

    <span class="hljs-meta">@classmethod</span>
    <span class="hljs-function">def <span class="hljs-title">from_crawler</span><span class="hljs-params">(cls, crawler)</span>:
        return <span class="hljs-title">cls</span><span class="hljs-params">(mongo_uri=crawler.settings.get(<span class="hljs-string">'MONGO_URI'</span>)</span>,
            mongo_db</span>=crawler.settings.get(<span class="hljs-string">'MONGO_DB'</span>)
        )
    <span class="hljs-function">def <span class="hljs-title">open_spider</span><span class="hljs-params">(self, spider)</span>:
        self.client </span>= pymongo.MongoClient(self.mongo_uri)
        self.db = self.client[self.mongo_db]
    <span class="hljs-function">def <span class="hljs-title">process_item</span><span class="hljs-params">(self, item, spider)</span>:
        self.db[item.collection].<span class="hljs-title">insert</span><span class="hljs-params">(dict(item)</span>)
        return item
    def <span class="hljs-title">close_spider</span><span class="hljs-params">(self, spider)</span>:
        self.client.<span class="hljs-title">close</span><span class="hljs-params">()</span>
</span></code></pre>
<p data-nodeid="1542">这里需要用到两个变量，MONGO_URI 和 MONGO_DB，即存储到 MongoDB 的链接地址和数据库名称。我们在 settings.py 里添加这两个变量，如下所示：</p>
<pre class="lang-java" data-nodeid="1543"><code data-language="java">MONGO_URI = <span class="hljs-string">'localhost'</span>
MONGO_DB = <span class="hljs-string">'images360'</span>
</code></pre>
<p data-nodeid="1544">这样一个保存到 MongoDB 的 Pipeline 的就创建好了。这里最主要的方法是 process_item()，直接调用 Collection 对象的 insert 方法即可完成数据的插入，最后返回 Item 对象。</p>
<h4 data-nodeid="1545">MySQL</h4>
<p data-nodeid="1546">首先需要确保 MySQL 已经正确安装并且正常运行。</p>
<p data-nodeid="1547">新建一个数据库，名字还是 images360，SQL 语句如下所示：</p>
<pre class="lang-java" data-nodeid="1548"><code data-language="java">CREATE DATABASE images360 DEFAULT CHARACTER SET utf8 COLLATE utf8_general_ci
</code></pre>
<p data-nodeid="1549">新建一个数据表，包含 id、url、title、thumb 四个字段，SQL 语句如下所示：</p>
<pre class="lang-java" data-nodeid="1550"><code data-language="java"><span class="hljs-function">CREATE TABLE <span class="hljs-title">images</span> <span class="hljs-params">(id VARCHAR(<span class="hljs-number">255</span>)</span> NULL PRIMARY KEY, url <span class="hljs-title">VARCHAR</span><span class="hljs-params">(<span class="hljs-number">255</span>)</span> NULL , title <span class="hljs-title">VARCHAR</span><span class="hljs-params">(<span class="hljs-number">255</span>)</span> NULL , thumb <span class="hljs-title">VARCHAR</span><span class="hljs-params">(<span class="hljs-number">255</span>)</span> NULL)
</span></code></pre>
<p data-nodeid="1551">执行完 SQL 语句之后，我们就成功创建好了数据表。接下来就可以往表里存储数据了。</p>
<p data-nodeid="1552">接下来我们实现一个 MySQLPipeline，代码如下所示：</p>
<pre class="lang-java" data-nodeid="1553"><code data-language="java"><span class="hljs-function"><span class="hljs-keyword">import</span> pymysql
class <span class="hljs-title">MysqlPipeline</span><span class="hljs-params">()</span>:
    def <span class="hljs-title">__init__</span><span class="hljs-params">(self, host, database, user, password, port)</span>:
        self.host </span>= host
        self.database = database
        self.user = user
        self.password = password
        self.port = port

    <span class="hljs-meta">@classmethod</span>
    <span class="hljs-function">def <span class="hljs-title">from_crawler</span><span class="hljs-params">(cls, crawler)</span>:
        return <span class="hljs-title">cls</span><span class="hljs-params">(host=crawler.settings.get(<span class="hljs-string">'MYSQL_HOST'</span>)</span>,
            database</span>=crawler.settings.get(<span class="hljs-string">'MYSQL_DATABASE'</span>),
            user=crawler.settings.get(<span class="hljs-string">'MYSQL_USER'</span>),
            password=crawler.settings.get(<span class="hljs-string">'MYSQL_PASSWORD'</span>),
            port=crawler.settings.get(<span class="hljs-string">'MYSQL_PORT'</span>),
        )

    <span class="hljs-function">def <span class="hljs-title">open_spider</span><span class="hljs-params">(self, spider)</span>:
        self.db </span>= pymysql.connect(self.host, self.user, self.password, self.database, charset=<span class="hljs-string">'utf8'</span>, port=self.port)
        self.cursor = self.db.cursor()

    <span class="hljs-function">def <span class="hljs-title">close_spider</span><span class="hljs-params">(self, spider)</span>:
        self.db.<span class="hljs-title">close</span><span class="hljs-params">()</span>

    def <span class="hljs-title">process_item</span><span class="hljs-params">(self, item, spider)</span>:
        data </span>= dict(item)
        keys = <span class="hljs-string">', '</span>.join(data.keys())
        values = <span class="hljs-string">', '</span>.join([<span class="hljs-string">'% s'</span>] * len(data))
        sql = <span class="hljs-string">'insert into % s (% s) values (% s)'</span> % (item.table, keys, values)
        self.cursor.execute(sql, tuple(data.values()))
        self.db.commit()
        <span class="hljs-keyword">return</span> item
</code></pre>
<p data-nodeid="1554">如前所述，这里用到的数据插入方法是一个动态构造 SQL 语句的方法。</p>
<p data-nodeid="1555">这里还需要几个 MySQL 的配置，我们在 settings.py 里添加几个变量，如下所示：</p>
<pre class="lang-java" data-nodeid="1556"><code data-language="java">MYSQL_HOST = <span class="hljs-string">'localhost'</span>
MYSQL_DATABASE = <span class="hljs-string">'images360'</span>
MYSQL_PORT = <span class="hljs-number">3306</span>
MYSQL_USER = <span class="hljs-string">'root'</span>
MYSQL_PASSWORD = <span class="hljs-string">'123456'</span>
</code></pre>
<p data-nodeid="1557">这里分别定义了 MySQL 的地址、数据库名称、端口、用户名、密码。这样，MySQL Pipeline 就完成了。</p>
<h4 data-nodeid="1558">Image Pipeline</h4>
<p data-nodeid="1559">Scrapy 提供了专门处理下载的 Pipeline，包括文件下载和图片下载。下载文件和图片的原理与抓取页面的原理一样，因此下载过程支持异步和多线程，十分高效。下面我们来看看具体的实现过程。</p>
<p data-nodeid="1560">官方文档地址为：<a href="https://doc.scrapy.org/en/latest/topics/media-pipeline.html" data-nodeid="1749">https://doc.scrapy.org/en/latest/topics/media-pipeline.html</a>。</p>
<p data-nodeid="1561">首先定义存储文件的路径，需要定义一个 IMAGES_STORE 变量，在 settings.py 中添加如下代码：</p>
<pre class="lang-java" data-nodeid="1562"><code data-language="java">IMAGES_STORE = <span class="hljs-string">'./images'</span>
</code></pre>
<p data-nodeid="1563">在这里我们将路径定义为当前路径下的 images 子文件夹，即下载的图片都会保存到本项目的 images 文件夹中。</p>
<p data-nodeid="1564">内置的 ImagesPipeline 会默认读取 Item 的 image_urls 字段，并认为该字段是一个列表形式，它会遍历 Item 的 image_urls 字段，然后取出每个 URL 进行图片下载。</p>
<p data-nodeid="1565">但是现在生成的 Item 的图片链接字段并不是 image_urls 字段表示的，也不是列表形式，而是单个的 URL。所以为了实现下载，我们需要重新定义下载的部分逻辑，即需要自定义 ImagePipeline，继承内置的 ImagesPipeline，重写方法。</p>
<p data-nodeid="1566">我们定义 ImagePipeline，如下所示：</p>
<pre class="lang-java" data-nodeid="1567"><code data-language="java">from scrapy <span class="hljs-keyword">import</span> Request
from scrapy.exceptions <span class="hljs-keyword">import</span> DropItem
from scrapy.pipelines.<span class="hljs-function">images <span class="hljs-keyword">import</span> ImagesPipeline
class <span class="hljs-title">ImagePipeline</span><span class="hljs-params">(ImagesPipeline)</span>:
    def <span class="hljs-title">file_path</span><span class="hljs-params">(self, request, response=None, info=None)</span>:
        url </span>= request.url
        file_name = url.split(<span class="hljs-string">'/'</span>)[-<span class="hljs-number">1</span>]
        <span class="hljs-keyword">return</span> <span class="hljs-function">file_name

    def <span class="hljs-title">item_completed</span><span class="hljs-params">(self, results, item, info)</span>:
        image_paths </span>= [x[<span class="hljs-string">'path'</span>] <span class="hljs-keyword">for</span> ok, x in results <span class="hljs-keyword">if</span> ok]
        <span class="hljs-keyword">if</span> not image_paths:
            <span class="hljs-function">raise <span class="hljs-title">DropItem</span><span class="hljs-params">(<span class="hljs-string">'Image Downloaded Failed'</span>)</span>
        return item

    def <span class="hljs-title">get_media_requests</span><span class="hljs-params">(self, item, info)</span>:
        yield <span class="hljs-title">Request</span><span class="hljs-params">(item[<span class="hljs-string">'url'</span>])</span>
</span></code></pre>
<p data-nodeid="1568">在这里我们实现了 ImagePipeline，继承 Scrapy 内置的 ImagesPipeline，重写了下面几个方法。</p>
<ul data-nodeid="1569">
<li data-nodeid="1570">
<p data-nodeid="1571">get_media_requests()。它的第一个参数 item 是爬取生成的 Item 对象。我们将它的 url 字段取出来，然后直接生成 Request 对象。此 Request 加入调度队列，等待被调度，执行下载。</p>
</li>
<li data-nodeid="1572">
<p data-nodeid="1573">file_path()。它的第一个参数 request 就是当前下载对应的 Request 对象。这个方法用来返回保存的文件名，直接将图片链接的最后一部分当作文件名即可。它利用 split() 函数分割链接并提取最后一部分，返回结果。这样此图片下载之后保存的名称就是该函数返回的文件名。</p>
</li>
<li data-nodeid="1574">
<p data-nodeid="1575">item_completed()，它是当单个 Item 完成下载时的处理方法。因为并不是每张图片都会下载成功，所以我们需要分析下载结果并剔除下载失败的图片。如果某张图片下载失败，那么我们就不需保存此 Item 到数据库。该方法的第一个参数 results 就是该 Item 对应的下载结果，它是一个列表形式，列表每一个元素是一个元组，其中包含了下载成功或失败的信息。这里我们遍历下载结果找出所有成功的下载列表。如果列表为空，那么该 Item 对应的图片下载失败，随即抛出异常 DropItem，该 Item 忽略。否则返回该 Item，说明此 Item 有效。</p>
</li>
</ul>
<p data-nodeid="1576">现在为止，三个 Item Pipeline 的定义就完成了。最后只需要启用就可以了，修改 settings.py，设置 ITEM_PIPELINES，如下所示：</p>
<pre class="lang-java" data-nodeid="1577"><code data-language="java">ITEM_PIPELINES = {
    <span class="hljs-string">'images360.pipelines.ImagePipeline'</span>: <span class="hljs-number">300</span>,
    <span class="hljs-string">'images360.pipelines.MongoPipeline'</span>: <span class="hljs-number">301</span>,
    <span class="hljs-string">'images360.pipelines.MysqlPipeline'</span>: <span class="hljs-number">302</span>,
}
</code></pre>
<p data-nodeid="1578">这里注意调用的顺序。我们需要优先调用 ImagePipeline 对 Item 做下载后的筛选，下载失败的 Item 就直接忽略，它们就不会保存到 MongoDB 和 MySQL 里。随后再调用其他两个存储的 Pipeline，这样就能确保存入数据库的图片都是下载成功的。<br>
接下来运行程序，执行爬取，如下所示：</p>
<pre class="lang-java" data-nodeid="1579"><code data-language="java">scrapy crawl images
</code></pre>
<p data-nodeid="1580">爬虫一边爬取一边下载，下载速度非常快，对应的输出日志如图所示。</p>
<p data-nodeid="1581"><img src="https://s0.lgstatic.com/i/image/M00/32/D5/CgqCHl8OyV2AWLyNAAGALb5Nqd8706.png" alt="Drawing 4.png" data-nodeid="1785"></p>
<p data-nodeid="1582">查看本地 images 文件夹，发现图片都已经成功下载，如图所示。</p>
<p data-nodeid="1583"><img src="https://s0.lgstatic.com/i/image/M00/32/D5/CgqCHl8OyWWAY76eAAKKY0ouxBs666.png" alt="Drawing 5.png" data-nodeid="1789"></p>
<p data-nodeid="1584">查看 MySQL，下载成功的图片信息也已成功保存，如图所示。</p>
<p data-nodeid="1585"><img src="https://s0.lgstatic.com/i/image/M00/32/CA/Ciqc1F8OyXaAdJc7ACY7Z4cSzuE317.png" alt="Drawing 6.png" data-nodeid="1793"></p>
<p data-nodeid="1586">查看 MongoDB，下载成功的图片信息同样已成功保存，如图所示。</p>
<p data-nodeid="1587"><img src="https://s0.lgstatic.com/i/image/M00/32/CA/Ciqc1F8OydSAD3T6ABH5qD8uKgM923.png" alt="Drawing 7.png" data-nodeid="1797"></p>
<p data-nodeid="1588">这样我们就可以成功实现图片的下载并把图片的信息存入数据库了。</p>
<h3 data-nodeid="1589">9. 本节代码</h3>
<p data-nodeid="1811" class="te-preview-highlight">本节代码地址为：<br>
<a href="https://github.com/Python3WebSpider/Images360" data-nodeid="1816">https://github.com/Python3WebSpider/Images360</a>。</p>

<h3 data-nodeid="1591">10. 结语</h3>
<p data-nodeid="1592" class="">Item Pipeline 是 Scrapy 非常重要的组件，数据存储几乎都是通过此组件实现的。请你务必认真掌握此内容。</p>
