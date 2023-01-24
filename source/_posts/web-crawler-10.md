---
title: 52讲轻松搞定网络爬虫笔记10
tags: [Web Crawler]
categories: data analysis
date: 2023-1-26
---

# 资料
[52讲轻松搞定网络爬虫](https://kaiwu.lagou.com/course/courseInfo.htm?courseId=46#/sale)


# 遇到动态页面怎么办，详解渲染页面提取
<p data-nodeid="165308">前面我们已经介绍了 Scrapy 的一些常见用法，包括服务端渲染页面的抓取和 API 的抓取，Scrapy 发起 Request 之后，返回的 Response 里面就包含了想要的结果。</p>



<p data-nodeid="163554">但是现在越来越多的网页都已经演变为 SPA 页面，其页面在浏览器中呈现的结果是经过 JavaScript 渲染得到的，如果我们使用 Scrapy 直接对其进行抓取的话，其结果和使用 requests 没有什么区别。</p>
<p data-nodeid="163555">那我们真的要使用 Scrapy 完成对 JavaScript 渲染页面的抓取应该怎么办呢？</p>
<p data-nodeid="163556">之前我们介绍了 Selenium 和 Pyppeteer 都可以实现 JavaScript 渲染页面的抓取，那用了 Scrapy 之后应该这么办呢？Scrapy 能和 Selenium 或 Pyppeteer 一起使用吗？答案是肯定的，我们可以将 Selenium 或 Pyppeteer 通过 Downloader Middleware 和 Scrapy 融合起来，实现 JavaScript 渲染页面的抓取，本节我们就来了解下它的实现吧。</p>
<h3 data-nodeid="165800" class="">回顾</h3>

<p data-nodeid="163558">在前面我们介绍了 Downloader Middleware 的用法，在 Downloader Middleware 中有三个我们可以实现的方法 process_request、process_response 以及 process_exception 方法。</p>
<p data-nodeid="163559">我们再看下 process_request 方法和其不同的返回值的效果：</p>
<ul data-nodeid="163560">
<li data-nodeid="163561">
<p data-nodeid="163562">当返回为 None 时，Scrapy 将继续处理该 Request，接着执行其他 Downloader Middleware 的 process_request 方法，一直到 Downloader 把 Request 执行完后得到 Response 才结束。这个过程其实就是修改 Request 的过程，不同的 Downloader Middleware 按照设置的优先级顺序依次对 Request 进行修改，最后送至 Downloader 执行。</p>
</li>
<li data-nodeid="163563">
<p data-nodeid="163564">当返回为 Response 对象时，更低优先级的 Downloader Middleware 的 process_request 和 process_exception 方法就不会被继续调用，每个 Downloader Middleware 的 process_response 方法转而被依次调用。调用完毕之后，直接将 Response 对象发送给 Spider 来处理。</p>
</li>
<li data-nodeid="163565">
<p data-nodeid="163566">当返回为 Request 对象时，更低优先级的 Downloader Middleware 的 process_request 方法会停止执行。这个 Request 会重新放到调度队列里，其实它就是一个全新的 Request，等待被调度。如果被 Scheduler 调度了，那么所有的 Downloader Middleware 的 process_request 方法都会被重新按照顺序执行。</p>
</li>
<li data-nodeid="163567">
<p data-nodeid="163568">如果 IgnoreRequest 异常抛出，则所有的 Downloader Middleware 的 process_exception 方法会依次执行。如果没有一个方法处理这个异常，那么 Request 的 errorback 方法就会回调。如果该异常还没有被处理，那么它便会被忽略。</p>
</li>
</ul>
<p data-nodeid="163569">这里我们注意到第二个选项，当返回结果为 Response 对象时，低优先级的 process_request 方法就不会被继续调用了，这个 Response 对象会直接经由 process_response 方法处理后转交给 Spider 来解析。</p>
<p data-nodeid="163570">然后再接着想一想，process_request 接收的参数是 request，即 Request 对象，怎么会返回 Response 对象呢？原因可想而知了，这个 Request 对象不再经由 Scrapy 的 Downloader 来处理了，而是在 process_request 方法里面直接就完成了 Request 的发送操作，然后在得到了对应的 Response 结果后再将其返回就好了。</p>
<p data-nodeid="163571">那么对于 JavaScript 渲染的页面来说，照这个方法来做，我们就可以把 Selenium 或 Pyppeteer 加载页面的过程在 process_request 方法里面实现，得到网页渲染完后的源代码后直接构造 Response 返回即可，这样我们就完成了借助 Downloader Middleware 实现 Scrapy 爬取动态渲染页面的过程。</p>
<h3 data-nodeid="166292" class="">案例</h3>

<p data-nodeid="163573">本节我们就用实例来讲解一下 Scrapy 和 Pyppeteer 实现 JavaScript 渲染页面抓取的流程。</p>
<p data-nodeid="167272">本节使用的实例网站为 <a href="https://dynamic5.scrape.center/" data-nodeid="167277">https://dynamic5.scrape.center/</a>，这是一个 JavaScript 渲染页面，其内容是一本本的图书信息。</p>
<p data-nodeid="167273" class=""><img src="https://s0.lgstatic.com/i/image/M00/34/22/Ciqc1F8RXwWARdOHABEEvYwMNOY314.png" alt="image.png" data-nodeid="167281"></p>



<p data-nodeid="163576">同时这个网站的页面带有分页功能，只需要在 URL 加上 <code data-backticks="1" data-nodeid="163693">/page/</code> 和页码就可以跳转到下一页，如 <a href="https://dynamic5.scrape.center/page/2" data-nodeid="163697">https://dynamic5.scrape.center/page/2</a> 就是第二页内容，<a href="https://dynamic5.scrape.center/page/3" data-nodeid="163701">https://dynamic5.scrape.center/page/3</a> 就是第三页内容。</p>
<p data-nodeid="163577">那我们这个案例就来试着爬取前十页的图书信息吧。</p>
<h3 data-nodeid="167772" class="">实现</h3>

<p data-nodeid="163579">首先我们来新建一个项目，叫作 scrapypyppeteer，命令如下：</p>
<pre class="lang-java" data-nodeid="169737"><code data-language="java">scrapy startproject scrapypyppeteer
</code></pre>




<p data-nodeid="163581">接着进入项目，然后新建一个 Spider，名称为 book，命令如下：</p>
<pre class="lang-java" data-nodeid="170228"><code data-language="java">cd scrapypyppeteer
scrapy genspider book dynamic5.scrape.center
</code></pre>

<p data-nodeid="163583">这时候可以发现在项目的 spiders 文件夹下就出现了一个名为 spider.py 的文件，内容如下：</p>
<pre class="lang-dart" data-nodeid="176857"><code data-language="dart"># -*- coding: utf<span class="hljs-number">-8</span> -*-
<span class="hljs-keyword">import</span> scrapy
​
​
<span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">BookSpider</span>(<span class="hljs-title">scrapy</span>.<span class="hljs-title">Spider</span>):
 &nbsp; &nbsp;<span class="hljs-title">name</span> = '<span class="hljs-title">book</span>'
 &nbsp; &nbsp;<span class="hljs-title">allowed_domains</span> = ['<span class="hljs-title">dynamic5</span>.<span class="hljs-title">scrape</span>.<span class="hljs-title">center</span>']
 &nbsp; &nbsp;<span class="hljs-title">start_urls</span> = ['<span class="hljs-title">http</span>://<span class="hljs-title">dynamic5</span>.<span class="hljs-title">scrape</span>.<span class="hljs-title">center</span>/']
​
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">parse</span>(<span class="hljs-title">self</span>, <span class="hljs-title">response</span>):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">pass</span>
</span></code></pre>














<p data-nodeid="163585">首先我们构造列表页的初始请求，实现一个 start_requests 方法，如下所示：</p>
<pre class="lang-go" data-nodeid="191096"><code data-language="go"># -*- coding: utf<span class="hljs-number">-8</span> -*-
from scrapy <span class="hljs-keyword">import</span> Request, Spider
​
​
class BookSpider(Spider):
 &nbsp; &nbsp;name = <span class="hljs-string">'book'</span>
 &nbsp; &nbsp;allowed_domains = [<span class="hljs-string">'dynamic5.scrape.center'</span>]
 &nbsp; &nbsp;
 &nbsp; &nbsp;base_url = <span class="hljs-string">'https://dynamic5.scrape.center/page/{page}'</span>
 &nbsp; &nbsp;max_page = <span class="hljs-number">10</span>
 &nbsp; &nbsp;
 &nbsp; &nbsp;def start_requests(self):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">for</span> page in <span class="hljs-keyword">range</span>(<span class="hljs-number">1</span>, self.max_page + <span class="hljs-number">1</span>):
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;url = self.base_url.format(page=page)
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;yield Request(url, callback=self.parse_index)
 &nbsp; &nbsp;
 &nbsp; &nbsp;def parse_index(self, response):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-built_in">print</span>(response.text)
</code></pre>





























<p data-nodeid="192069">这时如果我们直接运行这个 Spider，在 parse_index 方法里面打印输出 Response 的内容，结果如下：</p>
<p data-nodeid="192070" class=""><img src="https://s0.lgstatic.com/i/image/M00/34/2D/CgqCHl8RX0iADQnIAAJk6NIEDak825.png" alt="image (1).png" data-nodeid="192080"></p>


<p data-nodeid="163589">我们可以发现所得到的内容并不是页面渲染后的真正 HTML 代码。此时如果我们想要获取 HTML 渲染结果的话就得使用 Downloader Middleware 实现了。</p>
<p data-nodeid="163590">这里我们直接以一个我已经写好的组件来演示了，组件的名称叫作 GerapyPyppeteer，组件里已经写好了 Scrapy 和 Pyppeteer 结合的中间件，下面我们来详细介绍下。</p>
<p data-nodeid="163591">我们可以借助于 pip3 来安装组件，命令如下：</p>
<pre class="lang-java" data-nodeid="192579"><code data-language="java">pip3 install gerapy-pyppeteer
</code></pre>

<p data-nodeid="163593">GerapyPyppeteer 提供了两部分内容，一部分是 Downloader Middleware，一部分是 Request。<br>
首先我们需要开启中间件，在 settings 里面开启 PyppeteerMiddleware，配置如下：</p>
<pre class="lang-java" data-nodeid="193078"><code data-language="java">DOWNLOADER_MIDDLEWARES = {
 &nbsp; &nbsp;<span class="hljs-string">'gerapy_pyppeteer.downloadermiddlewares.PyppeteerMiddleware'</span>: <span class="hljs-number">543</span>,
}
</code></pre>

<p data-nodeid="163595">然后我们把上文定义的 Request 修改为 PyppeteerRequest 即可：</p>
<pre class="lang-go" data-nodeid="197569"><code data-language="go"># -*- coding: utf<span class="hljs-number">-8</span> -*-
from gerapy_pyppeteer <span class="hljs-keyword">import</span> PyppeteerRequest
from scrapy <span class="hljs-keyword">import</span> Request, Spider
​
​
class BookSpider(Spider):
 &nbsp; &nbsp;name = <span class="hljs-string">'book'</span>
 &nbsp; &nbsp;allowed_domains = [<span class="hljs-string">'dynamic5.scrape.center'</span>]
 &nbsp; &nbsp;
 &nbsp; &nbsp;base_url = <span class="hljs-string">'https://dynamic5.scrape.center/page/{page}'</span>
 &nbsp; &nbsp;max_page = <span class="hljs-number">10</span>
 &nbsp; &nbsp;
 &nbsp; &nbsp;def start_requests(self):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">for</span> page in <span class="hljs-keyword">range</span>(<span class="hljs-number">1</span>, self.max_page + <span class="hljs-number">1</span>):
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;url = self.base_url.format(page=page)
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;yield PyppeteerRequest(url, callback=self.parse_index, wait_for=<span class="hljs-string">'.item .name'</span>)
 &nbsp; &nbsp;
 &nbsp; &nbsp;def parse_index(self, response):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-built_in">print</span>(response.text)
</code></pre>









<p data-nodeid="163597">这样其实就完成了 Pyppeteer 的对接了，非常简单。<br>
这里 PyppeteerRequest 和原本的 Request 多提供了一个参数，就是 wait_for，通过这个参数我们可以指定 Pyppeteer 需要等待特定的内容加载出来才算结束，然后才返回对应的结果。</p>
<p data-nodeid="163598">为了方便观察效果，我们把并发限制修改得小一点，然后把 Pyppeteer 的 Headless 模式设置为 False：</p>
<pre class="lang-java" data-nodeid="198068"><code data-language="java">CONCURRENT_REQUESTS = <span class="hljs-number">3</span>
GERAPY_PYPPETEER_HEADLESS = False
</code></pre>

<p data-nodeid="199057">这时我们重新运行 Spider，就可以看到在爬取的过程中，Pyppeteer 对应的 Chromium 浏览器就弹出来了，并逐个加载对应的页面内容，加载完成之后浏览器关闭。<br>
另外观察下控制台，我们发现对应的结果也就被提取出来了，如图所示：</p>
<p data-nodeid="199058" class=""><img src="https://s0.lgstatic.com/i/image/M00/34/2D/CgqCHl8RX2SAAVD5AAK2ktEbgAQ066.png" alt="image (2).png" data-nodeid="199068"></p>


<p data-nodeid="163602">这时候我们再重新修改下 parse_index 方法，提取对应的每本书的名称和作者即可：</p>
<pre class="lang-java" data-nodeid="201096"><code data-language="java"><span class="hljs-function">def <span class="hljs-title">parse_index</span><span class="hljs-params">(self, response)</span>:
 &nbsp; &nbsp;<span class="hljs-keyword">for</span> item in response.<span class="hljs-title">css</span><span class="hljs-params">(<span class="hljs-string">'.item'</span>)</span>:
 &nbsp; &nbsp; &nbsp; &nbsp;name </span>= item.css(<span class="hljs-string">'.name::text'</span>).extract_first()
 &nbsp; &nbsp; &nbsp; &nbsp;authors = item.css(<span class="hljs-string">'.authors::text'</span>).extract_first()
 &nbsp; &nbsp; &nbsp; &nbsp;name = name.strip() <span class="hljs-keyword">if</span> name <span class="hljs-keyword">else</span> None
 &nbsp; &nbsp; &nbsp; &nbsp;authors = authors.strip() <span class="hljs-keyword">if</span> authors <span class="hljs-keyword">else</span> None
 &nbsp; &nbsp; &nbsp; &nbsp;yield {
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-string">'name'</span>: name,
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-string">'authors'</span>: authors
 &nbsp; &nbsp; &nbsp;  }
</code></pre>




<p data-nodeid="202354">重新运行，即可发现对应的名称和作者就被提取出来了，运行结果如下：</p>
<p data-nodeid="202355" class=""><img src="https://s0.lgstatic.com/i/image/M00/34/2E/CgqCHl8RX4OAK3y6AAOIS7PLohc610.png" alt="image (3).png" data-nodeid="202363"></p>

<p data-nodeid="202102">这样我们就借助 GerapyPyppeteer 完成了 JavaScript 渲染页面的爬取。</p>



<h4 data-nodeid="202878" class="">原理分析</h4>

<p data-nodeid="163608">但上面仅仅是我们借助 GerapyPyppeteer 实现了 Scrapy 和 Pyppeteer 的对接，但其背后的原理是怎样的呢？</p>
<p data-nodeid="163609">我们可以详细分析它的源码，其 GitHub 地址为 <a href="https://github.com/Gerapy/GerapyPyppeteer" data-nodeid="163749">https://github.com/Gerapy/GerapyPyppeteer</a>。</p>
<p data-nodeid="163610">首先通过分析可以发现其最核心的内容就是实现了一个 PyppeteerMiddleware，这是一个 Downloader Middleware，这里最主要的就是 process_request  的实现，核心代码如下所示：</p>
<pre class="lang-java" data-nodeid="203394"><code data-language="java"><span class="hljs-function">def <span class="hljs-title">process_request</span><span class="hljs-params">(self, request, spider)</span>:
 &nbsp; &nbsp;logger.<span class="hljs-title">debug</span><span class="hljs-params">(<span class="hljs-string">'processing request %s'</span>, request)</span> &nbsp;
 &nbsp; &nbsp;return <span class="hljs-title">as_deferred</span><span class="hljs-params">(self._process_request(request, spider)</span>)
</span></code></pre>

<p data-nodeid="204424">这里其实就是调用了一个 _process_request 方法，这个方法的返回结果被 as_deferred 方法调用了。</p>
<p data-nodeid="204425">这个 as_deferred 是怎么定义的呢？代码如下：</p>

<pre class="lang-java" data-nodeid="203909"><code data-language="java"><span class="hljs-keyword">import</span> asyncio
from twisted.internet.defer <span class="hljs-keyword">import</span> Deferred
​
<span class="hljs-function">def <span class="hljs-title">as_deferred</span><span class="hljs-params">(f)</span>:
 &nbsp; &nbsp;return Deferred.<span class="hljs-title">fromFuture</span><span class="hljs-params">(asyncio.ensure_future(f)</span>)
</span></code></pre>

<p data-nodeid="204950">这个方法接收的就是一个 asyncio 库的 Future 对象，然后通过 fromFuture 方法转化成了 twisted 里面的 Deferred 对象。这是因为 Scrapy 本身的异步是借助 twisted 实现的，一个个的异步任务对应的就是一个个 Deferred 对象，而 Pyppeteer 又是基于 asyncio 的，它的异步任务是 Future 对象，所以这里我们需要借助 Deferred 的 fromFuture 方法将 Future 转为 Deferred 对象。</p>
<p data-nodeid="204951">另外为了支持这个功能，我们还需要在 Scrapy 中修改 reactor 对象，修改为 AsyncioSelectorReactor，实现如下：</p>

<pre class="lang-dart" data-nodeid="210618"><code data-language="dart"><span class="hljs-keyword">import</span> sys
from twisted.internet.asyncioreactor <span class="hljs-keyword">import</span> AsyncioSelectorReactor
<span class="hljs-keyword">import</span> twisted.internet
​
reactor = AsyncioSelectorReactor(asyncio.get_event_loop())
​
# install AsyncioSelectorReactor
twisted.internet.reactor = reactor
sys.modules[<span class="hljs-string">'twisted.internet.reactor'</span>] = reactor
</code></pre>











<p data-nodeid="163616">这段代码已经在 PyppeteerMiddleware 里面定义好了，在 Scrapy 正式开始爬取之前这段代码就会被执行，将 Scrapy 中的 reactor 修改为 AsyncioSelectorReactor，从而实现 Future 的调度。<br>
接下来我们再来看下 _process_request 方法，实现如下：</p>
<pre class="lang-dart" data-nodeid="218343"><code data-language="dart"><span class="hljs-keyword">async</span> def _process_request(self, request: PyppeteerRequest, spider):
 &nbsp; &nbsp;<span class="hljs-string">"""
 &nbsp;  use pyppeteer to process spider
 &nbsp;  :param request:
 &nbsp;  :param spider:
 &nbsp;  :return:
 &nbsp;  """</span>
 &nbsp; &nbsp;options = {
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-string">'headless'</span>: self.headless,
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-string">'dumpio'</span>: self.dumpio,
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-string">'devtools'</span>: self.devtools,
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-string">'args'</span>: [
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;f<span class="hljs-string">'--window-size={self.window_width},{self.window_height}'</span>,
 &nbsp; &nbsp; &nbsp;  ]
 &nbsp;  }
 &nbsp; &nbsp;<span class="hljs-keyword">if</span> self.executable_path: options[<span class="hljs-string">'executable_path'</span>] = self.executable_path
 &nbsp; &nbsp;<span class="hljs-keyword">if</span> self.disable_extensions: options[<span class="hljs-string">'args'</span>].append(<span class="hljs-string">'--disable-extensions'</span>)
 &nbsp; &nbsp;<span class="hljs-keyword">if</span> self.hide_scrollbars: options[<span class="hljs-string">'args'</span>].append(<span class="hljs-string">'--hide-scrollbars'</span>)
 &nbsp; &nbsp;<span class="hljs-keyword">if</span> self.mute_audio: options[<span class="hljs-string">'args'</span>].append(<span class="hljs-string">'--mute-audio'</span>)
 &nbsp; &nbsp;<span class="hljs-keyword">if</span> self.no_sandbox: options[<span class="hljs-string">'args'</span>].append(<span class="hljs-string">'--no-sandbox'</span>)
 &nbsp; &nbsp;<span class="hljs-keyword">if</span> self.disable_setuid_sandbox: options[<span class="hljs-string">'args'</span>].append(<span class="hljs-string">'--disable-setuid-sandbox'</span>)
 &nbsp; &nbsp;<span class="hljs-keyword">if</span> self.disable_gpu: options[<span class="hljs-string">'args'</span>].append(<span class="hljs-string">'--disable-gpu'</span>)
 &nbsp; &nbsp;
 &nbsp; &nbsp;# <span class="hljs-keyword">set</span> proxy
 &nbsp; &nbsp;proxy = request.proxy
 &nbsp; &nbsp;<span class="hljs-keyword">if</span> not proxy:
 &nbsp; &nbsp; &nbsp; &nbsp;proxy = request.meta.<span class="hljs-keyword">get</span>(<span class="hljs-string">'proxy'</span>)
 &nbsp; &nbsp;<span class="hljs-keyword">if</span> proxy: options[<span class="hljs-string">'args'</span>].append(f<span class="hljs-string">'--proxy-server={proxy}'</span>)
 &nbsp; &nbsp;
 &nbsp; &nbsp;logger.debug(<span class="hljs-string">'set options %s'</span>, options)
 &nbsp; &nbsp;
 &nbsp; &nbsp;browser = <span class="hljs-keyword">await</span> launch(options)
 &nbsp; &nbsp;page = <span class="hljs-keyword">await</span> browser.newPage()
 &nbsp; &nbsp;<span class="hljs-keyword">await</span> page.setViewport({<span class="hljs-string">'width'</span>: self.window_width, <span class="hljs-string">'height'</span>: self.window_height})
 &nbsp; &nbsp;
 &nbsp; &nbsp;# <span class="hljs-keyword">set</span> cookies
 &nbsp; &nbsp;<span class="hljs-keyword">if</span> isinstance(request.cookies, dict):
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">await</span> page.setCookie(*[
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  {<span class="hljs-string">'name'</span>: k, <span class="hljs-string">'value'</span>: v}
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> request.cookies.items()
 &nbsp; &nbsp; &nbsp;  ])
 &nbsp; &nbsp;<span class="hljs-keyword">else</span>:
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">await</span> page.setCookie(request.cookies)
 &nbsp; &nbsp;
 &nbsp; &nbsp;# the headers must be <span class="hljs-keyword">set</span> using request interception
 &nbsp; &nbsp;<span class="hljs-keyword">await</span> page.setRequestInterception(True)
 &nbsp; &nbsp;
 &nbsp; &nbsp;<span class="hljs-meta">@page</span>.<span class="hljs-keyword">on</span>(<span class="hljs-string">'request'</span>)
 &nbsp; &nbsp;<span class="hljs-keyword">async</span> def _handle_interception(pu_request):
 &nbsp; &nbsp; &nbsp; &nbsp;# handle headers
 &nbsp; &nbsp; &nbsp; &nbsp;overrides = {
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-string">'headers'</span>: {
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;k.decode(): <span class="hljs-string">','</span>.join(map(lambda v: v.decode(), v))
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> request.headers.items()
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  }
 &nbsp; &nbsp; &nbsp;  }
 &nbsp; &nbsp; &nbsp; &nbsp;# handle resource types
 &nbsp; &nbsp; &nbsp; &nbsp;_ignore_resource_types = self.ignore_resource_types
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">if</span> request.ignore_resource_types <span class="hljs-keyword">is</span> not None:
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;_ignore_resource_types = request.ignore_resource_types
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">if</span> pu_request.resourceType <span class="hljs-keyword">in</span> _ignore_resource_types:
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">await</span> pu_request.abort()
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">else</span>:
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">await</span> pu_request.continue_(overrides)
 &nbsp; &nbsp;
 &nbsp; &nbsp;timeout = self.download_timeout
 &nbsp; &nbsp;<span class="hljs-keyword">if</span> request.timeout <span class="hljs-keyword">is</span> not None:
 &nbsp; &nbsp; &nbsp; &nbsp;timeout = request.timeout
 &nbsp; &nbsp;
 &nbsp; &nbsp;logger.debug(<span class="hljs-string">'crawling %s'</span>, request.url)
 &nbsp; &nbsp;
 &nbsp; &nbsp;response = None
 &nbsp; &nbsp;<span class="hljs-keyword">try</span>:
 &nbsp; &nbsp; &nbsp; &nbsp;options = {
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-string">'timeout'</span>: <span class="hljs-number">1000</span> * timeout,
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-string">'waitUntil'</span>: request.wait_until
 &nbsp; &nbsp; &nbsp;  }
 &nbsp; &nbsp; &nbsp; &nbsp;logger.debug(<span class="hljs-string">'request %s with options %s'</span>, request.url, options)
 &nbsp; &nbsp; &nbsp; &nbsp;response = <span class="hljs-keyword">await</span> page.goto(
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;request.url,
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;options=options
 &nbsp; &nbsp; &nbsp;  )
 &nbsp; &nbsp;except (PageError, TimeoutError):
 &nbsp; &nbsp; &nbsp; &nbsp;logger.error(<span class="hljs-string">'error rendering url %s using pyppeteer'</span>, request.url)
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">await</span> page.close()
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">await</span> browser.close()
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">return</span> self._retry(request, <span class="hljs-number">504</span>, spider)
 &nbsp; &nbsp;
 &nbsp; &nbsp;<span class="hljs-keyword">if</span> request.wait_for:
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">try</span>:
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;logger.debug(<span class="hljs-string">'waiting for %s finished'</span>, request.wait_for)
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">await</span> page.waitFor(request.wait_for)
 &nbsp; &nbsp; &nbsp; &nbsp;except TimeoutError:
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;logger.error(<span class="hljs-string">'error waiting for %s of %s'</span>, request.wait_for, request.url)
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">await</span> page.close()
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">await</span> browser.close()
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">return</span> self._retry(request, <span class="hljs-number">504</span>, spider)
 &nbsp; &nbsp;
 &nbsp; &nbsp;# evaluate script
 &nbsp; &nbsp;<span class="hljs-keyword">if</span> request.script:
 &nbsp; &nbsp; &nbsp; &nbsp;logger.debug(<span class="hljs-string">'evaluating %s'</span>, request.script)
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">await</span> page.evaluate(request.script)
 &nbsp; &nbsp;
 &nbsp; &nbsp;# sleep
 &nbsp; &nbsp;<span class="hljs-keyword">if</span> request.sleep <span class="hljs-keyword">is</span> not None:
 &nbsp; &nbsp; &nbsp; &nbsp;logger.debug(<span class="hljs-string">'sleep for %ss'</span>, request.sleep)
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">await</span> asyncio.sleep(request.sleep)
 &nbsp; &nbsp;
 &nbsp; &nbsp;content = <span class="hljs-keyword">await</span> page.content()
 &nbsp; &nbsp;body = str.encode(content)
 &nbsp; &nbsp;
 &nbsp; &nbsp;# close page and browser
 &nbsp; &nbsp;logger.debug(<span class="hljs-string">'close pyppeteer'</span>)
 &nbsp; &nbsp;<span class="hljs-keyword">await</span> page.close()
 &nbsp; &nbsp;<span class="hljs-keyword">await</span> browser.close()
 &nbsp; &nbsp;
 &nbsp; &nbsp;<span class="hljs-keyword">if</span> not response:
 &nbsp; &nbsp; &nbsp; &nbsp;logger.error(<span class="hljs-string">'get null response by pyppeteer of url %s'</span>, request.url)
 &nbsp; &nbsp;
 &nbsp; &nbsp;# Necessary to bypass the compression middleware (?)
 &nbsp; &nbsp;response.headers.pop(<span class="hljs-string">'content-encoding'</span>, None)
 &nbsp; &nbsp;response.headers.pop(<span class="hljs-string">'Content-Encoding'</span>, None)
 &nbsp; &nbsp;
 &nbsp; &nbsp;<span class="hljs-keyword">return</span> HtmlResponse(
 &nbsp; &nbsp; &nbsp; &nbsp;page.url,
 &nbsp; &nbsp; &nbsp; &nbsp;status=response.status,
 &nbsp; &nbsp; &nbsp; &nbsp;headers=response.headers,
 &nbsp; &nbsp; &nbsp; &nbsp;body=body,
 &nbsp; &nbsp; &nbsp; &nbsp;encoding=<span class="hljs-string">'utf-8'</span>,
 &nbsp; &nbsp; &nbsp; &nbsp;request=request
 &nbsp;  )
</code></pre>















<p data-nodeid="218858">代码内容比较多，我们慢慢来说。</p>
<p data-nodeid="218859">首先最开始的部分是定义 Pyppeteer 的一些启动参数：</p>

<pre class="lang-java" data-nodeid="219376"><code data-language="java">options = {
 &nbsp; &nbsp;<span class="hljs-string">'headless'</span>: self.headless,
 &nbsp; &nbsp;<span class="hljs-string">'dumpio'</span>: self.dumpio,
 &nbsp; &nbsp;<span class="hljs-string">'devtools'</span>: self.devtools,
 &nbsp; &nbsp;<span class="hljs-string">'args'</span>: [
 &nbsp; &nbsp; &nbsp; &nbsp;f<span class="hljs-string">'--window-size={self.window_width},{self.window_height}'</span>,
 &nbsp;  ]
}
<span class="hljs-keyword">if</span> self.executable_path: options[<span class="hljs-string">'executable_path'</span>] = self.executable_path
<span class="hljs-keyword">if</span> self.disable_extensions: options[<span class="hljs-string">'args'</span>].append(<span class="hljs-string">'--disable-extensions'</span>)
<span class="hljs-keyword">if</span> self.hide_scrollbars: options[<span class="hljs-string">'args'</span>].append(<span class="hljs-string">'--hide-scrollbars'</span>)
<span class="hljs-keyword">if</span> self.mute_audio: options[<span class="hljs-string">'args'</span>].append(<span class="hljs-string">'--mute-audio'</span>)
<span class="hljs-keyword">if</span> self.no_sandbox: options[<span class="hljs-string">'args'</span>].append(<span class="hljs-string">'--no-sandbox'</span>)
<span class="hljs-keyword">if</span> self.disable_setuid_sandbox: options[<span class="hljs-string">'args'</span>].append(<span class="hljs-string">'--disable-setuid-sandbox'</span>)
<span class="hljs-keyword">if</span> self.disable_gpu: options[<span class="hljs-string">'args'</span>].append(<span class="hljs-string">'--disable-gpu'</span>)
</code></pre>

<p data-nodeid="219891">这些参数来自 from_crawler 里面读取项目 settings 的内容，如配置 Pyppeteer 对应浏览器的无头模式、窗口大小、是否隐藏滚动条、是否弃用沙箱，等等。</p>
<p data-nodeid="219892">紧接着就是利用 options 来启动 Pyppeteer：</p>

<pre class="lang-java" data-nodeid="220411"><code data-language="java">browser = <span class="hljs-function">await <span class="hljs-title">launch</span><span class="hljs-params">(options)</span>
page </span>= await browser.newPage()
await page.setViewport({<span class="hljs-string">'width'</span>: self.window_width, <span class="hljs-string">'height'</span>: self.window_height})
</code></pre>

<p data-nodeid="220926">这里启动了 Pyppeteer 对应的浏览器，将其赋值为 browser，然后新建了一个选项卡，赋值为 page，然后通过 setViewport 方法设定了窗口的宽高。</p>
<p data-nodeid="220927">接下来就是对一些 Cookies 进行处理，如果 Request 带有 Cookies 的话会被赋值到 Pyppeteer 中：</p>

<pre class="lang-dart" data-nodeid="226594"><code data-language="dart"># <span class="hljs-keyword">set</span> cookies
<span class="hljs-keyword">if</span> isinstance(request.cookies, dict):
 &nbsp; &nbsp;<span class="hljs-keyword">await</span> page.setCookie(*[
 &nbsp; &nbsp; &nbsp;  {<span class="hljs-string">'name'</span>: k, <span class="hljs-string">'value'</span>: v}
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> request.cookies.items()
 &nbsp;  ])
<span class="hljs-keyword">else</span>:
 &nbsp; &nbsp;<span class="hljs-keyword">await</span> page.setCookie(request.cookies)
</code></pre>











<p data-nodeid="163624">再然后关键的步骤就是进行页面的加载了：</p>
<pre class="lang-java" data-nodeid="227109"><code data-language="java"><span class="hljs-keyword">try</span>:
 &nbsp; &nbsp;options = {
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-string">'timeout'</span>: <span class="hljs-number">1000</span> * timeout,
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-string">'waitUntil'</span>: request.wait_until
 &nbsp;  }
 &nbsp; &nbsp;logger.debug(<span class="hljs-string">'request %s with options %s'</span>, request.url, options)
 &nbsp; &nbsp;response = await page.goto(
 &nbsp; &nbsp; &nbsp; &nbsp;request.url,
 &nbsp; &nbsp; &nbsp; &nbsp;options=options
 &nbsp;  )
except (PageError, TimeoutError):
 &nbsp; &nbsp;logger.error(<span class="hljs-string">'error rendering url %s using pyppeteer'</span>, request.url)
 &nbsp; &nbsp;await page.close()
 &nbsp; &nbsp;await browser.close()
 &nbsp; &nbsp;<span class="hljs-keyword">return</span> self._retry(request, <span class="hljs-number">504</span>, spider)
</code></pre>

<p data-nodeid="227624">这里我们首先制定了加载超时时间 timeout 还有要等待完成的事件 waitUntil，接着调用 page 的 goto 方法访问对应的页面，同时进行了异常检测，如果发生错误就关闭浏览器并重新发起一次重试请求。</p>
<p data-nodeid="227625">在页面加载出来之后，我们还需要判定我们期望的结果是不是加载出来了，所以这里又增加了 waitFor 的调用：</p>

<pre class="lang-java" data-nodeid="228142"><code data-language="java"><span class="hljs-keyword">if</span> request.wait_for:
 &nbsp; &nbsp;<span class="hljs-keyword">try</span>:
 &nbsp; &nbsp; &nbsp; &nbsp;logger.debug(<span class="hljs-string">'waiting for %s finished'</span>, request.wait_for)
 &nbsp; &nbsp; &nbsp; &nbsp;await page.waitFor(request.wait_for)
 &nbsp; &nbsp;except TimeoutError:
 &nbsp; &nbsp; &nbsp; &nbsp;logger.error(<span class="hljs-string">'error waiting for %s of %s'</span>, request.wait_for, request.url)
 &nbsp; &nbsp; &nbsp; &nbsp;await page.close()
 &nbsp; &nbsp; &nbsp; &nbsp;await browser.close()
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">return</span> self._retry(request, <span class="hljs-number">504</span>, spider)
</code></pre>

<p data-nodeid="228657">这里 request 有个 wait_for 属性，就可以定义想要加载的节点的选择器，如 <code data-backticks="1" data-nodeid="228662">.item .name</code> 等，这样如果页面在规定时间内加载出来就会继续向下执行，否则就会触发 TimeoutError 并被捕获，关闭浏览器并重新发起一次重试请求。</p>
<p data-nodeid="228658">等想要的结果加载出来之后，我们还可以执行一些自定义的 JavaScript 代码完成我们想要自定义的功能：</p>

<pre class="lang-dart" data-nodeid="234329"><code data-language="dart"># evaluate script
<span class="hljs-keyword">if</span> request.script:
 &nbsp; &nbsp;logger.debug(<span class="hljs-string">'evaluating %s'</span>, request.script)
 &nbsp; &nbsp;<span class="hljs-keyword">await</span> page.evaluate(request.script)
</code></pre>











<p data-nodeid="163630">最后关键的一步就是将当前页面的源代码打印出来，然后构造一个 HtmlResponse 返回即可：</p>
<pre class="lang-dart te-preview-highlight" data-nodeid="239994"><code data-language="dart">content = <span class="hljs-keyword">await</span> page.content()
body = str.encode(content)
​
# close page and browser
logger.debug(<span class="hljs-string">'close pyppeteer'</span>)
<span class="hljs-keyword">await</span> page.close()
<span class="hljs-keyword">await</span> browser.close()
​
<span class="hljs-keyword">if</span> not response:
 &nbsp; &nbsp;logger.error(<span class="hljs-string">'get null response by pyppeteer of url %s'</span>, request.url)
​
# Necessary to bypass the compression middleware (?)
response.headers.pop(<span class="hljs-string">'content-encoding'</span>, None)
response.headers.pop(<span class="hljs-string">'Content-Encoding'</span>, None)
​
<span class="hljs-keyword">return</span> HtmlResponse(
 &nbsp; &nbsp;page.url,
 &nbsp; &nbsp;status=response.status,
 &nbsp; &nbsp;headers=response.headers,
 &nbsp; &nbsp;body=body,
 &nbsp; &nbsp;encoding=<span class="hljs-string">'utf-8'</span>,
 &nbsp; &nbsp;request=request
)
</code></pre>











<p data-nodeid="164806">所以，如果代码可以执行到最后，返回到就是一个 Response 对象，这个 Resposne 对象的 body 就是 Pyppeteer  渲染页面后的结果，因此这个 Response 对象再传给 Spider 解析，就是 JavaScript 渲染后的页面结果了。</p>
<p data-nodeid="164807">这样我们就通过 Downloader Middleware 通过对接 Pyppeteer 完成 JavaScript 动态渲染页面的抓取了。</p>


# 大幅提速，分布式爬虫理念
<p data-nodeid="77766">我们在前面几节课了解了 Scrapy 爬虫框架的用法。但这些框架都是在同一台主机上运行的，爬取效率比较低。如果能够实现多台主机协同爬取，那么爬取效率必然会成倍增长，这就是分布式爬虫的优势。</p>



<p data-nodeid="77424">接下来我们就来了解一下分布式爬虫的基本原理，以及 Scrapy 实现分布式爬虫的流程。</p>
<p data-nodeid="77425">我们在前面已经实现了 Scrapy 基本的爬虫功能，虽然爬虫是异步加多线程的，但是我们却只能在一台主机上运行，所以爬取效率还是有限的，而分布式爬虫则是将多台主机组合起来，共同完成一个爬取任务，这将大大提高爬取的效率。</p>
<h3 data-nodeid="77426">分布式爬虫架构</h3>
<p data-nodeid="78204">在了解分布式爬虫架构之前，首先回顾一下 Scrapy 的架构，如图所示。</p>
<p data-nodeid="78205" class=""><img src="https://s0.lgstatic.com/i/image/M00/36/9E/CgqCHl8X3v-ALLasAAJygBiwVD4562.png" alt="Drawing 0.png" data-nodeid="78209"></p>


<p data-nodeid="78634">Scrapy 单机爬虫中有一个本地爬取队列 Queue，这个队列是利用 deque 模块实现的。如果新的 Request 生成就会放到队列里面，随后 Request 被 Scheduler 调度。之后，Request 交给 Downloader 执行爬取，简单的调度架构如图所示。</p>
<p data-nodeid="78635" class=""><img src="https://s0.lgstatic.com/i/image/M00/36/93/Ciqc1F8X3wqASz4bAAAu4I6VU_g788.png" alt="Drawing 1.png" data-nodeid="78639"></p>




<p data-nodeid="77433">如果两个 Scheduler 同时从队列里面获取 Request，每个 Scheduler 都会有其对应的 Downloader，那么在带宽足够、正常爬取且不考虑队列存取压力的情况下，爬取效率会有什么变化呢？没错，爬取效率会翻倍。</p>
<p data-nodeid="79040">这样，Scheduler 可以扩展多个，Downloader 也可以扩展多个。而爬取队列 Queue 必须始终为一个，也就是所谓的共享爬取队列。这样才能保证 Scheduer 从队列里调度某个 Request 之后，其他 Scheduler 不会重复调度此 Request，就可以做到多个 Schduler 同步爬取。这就是分布式爬虫的基本雏形，简单调度架构如图所示。</p>
<p data-nodeid="79041" class=""><img src="https://s0.lgstatic.com/i/image/M00/36/9E/CgqCHl8X3xOAIuL2AABY71AHaqQ611.png" alt="Drawing 3.png" data-nodeid="79045"></p>




<p data-nodeid="77438">我们需要做的就是在多台主机上同时运行爬虫任务协同爬取，而协同爬取的前提就是共享爬取队列。这样各台主机就不需要维护各自的爬取队列了，而是从共享爬取队列存取 Request。但是各台主机还有各自的 Scheduler 和 Downloader，所以调度和下载功能是分别完成的。如果不考虑队列存取性能消耗，爬取效率还是可以成倍提高的。</p>
<h3 data-nodeid="77439">维护爬取队列</h3>
<p data-nodeid="77440">那么如何维护这个队列呢？我们首先需要考虑的就是性能问题，那什么数据库存取效率高呢？这时我们自然想到了基于内存存储的 Redis，而且 Redis 还支持多种数据结构，例如列表 List、集合 Set、有序集合 Sorted Set 等，存取的操作也非常简单，所以在这里我们采用 Redis 来维护爬取队列。</p>
<p data-nodeid="77441">这几种数据结构存储实际各有千秋，分析如下：</p>
<ul data-nodeid="77442">
<li data-nodeid="77443">
<p data-nodeid="77444">列表数据结构有 lpush、lpop、rpush、rpop 方法，所以我们可以用它实现一个先进先出的爬取队列，也可以实现一个先进后出的栈式爬取队列。</p>
</li>
<li data-nodeid="77445">
<p data-nodeid="77446">集合的元素是无序且不重复的，这样我们就可以非常方便地实现一个随机排序的不重复的爬取队列。</p>
</li>
<li data-nodeid="77447">
<p data-nodeid="77448">有序集合带有分数表示，而 Scrapy 的 Request 也有优先级的控制，所以我们用有序集合就可以实现一个带优先级调度的队列。</p>
</li>
</ul>
<p data-nodeid="77449">这些不同的队列我们需要根据具体爬虫的需求灵活选择。</p>
<h3 data-nodeid="77450">怎样去重</h3>
<p data-nodeid="77451">Scrapy 有自动去重功能，它的去重使用了 Python 中的集合。这个集合记录了 Scrapy 中每个 Request 的指纹，这个指纹实际上就是 Request 的散列值。我们可以看看 Scrapy 的源代码，如下所示：</p>
<pre class="lang-java" data-nodeid="79344"><code data-language="java"><span class="hljs-function"><span class="hljs-keyword">import</span> hashlib
def <span class="hljs-title">request_fingerprint</span><span class="hljs-params">(request, include_headers=None)</span>:
 &nbsp; &nbsp;<span class="hljs-keyword">if</span> include_headers:
 &nbsp; &nbsp; &nbsp; &nbsp;include_headers </span>= tuple(to_bytes(h.lower())
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <span class="hljs-function"><span class="hljs-keyword">for</span> h in <span class="hljs-title">sorted</span><span class="hljs-params">(include_headers)</span>)
 &nbsp; &nbsp;cache </span>= _fingerprint_cache.setdefault(request, {})
 &nbsp; &nbsp;<span class="hljs-keyword">if</span> include_headers not in cache:
 &nbsp; &nbsp; &nbsp; &nbsp;fp = hashlib.sha1()
 &nbsp; &nbsp; &nbsp; &nbsp;fp.update(to_bytes(request.method))
 &nbsp; &nbsp; &nbsp; &nbsp;fp.update(to_bytes(canonicalize_url(request.url)))
 &nbsp; &nbsp; &nbsp; &nbsp;fp.update(request.body or b<span class="hljs-string">''</span>)
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">if</span> include_headers:
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">for</span> hdr in include_headers:
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">if</span> hdr in request.headers:
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;fp.update(hdr)
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">for</span> v in request.headers.getlist(hdr):
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;fp.update(v)
 &nbsp; &nbsp; &nbsp; &nbsp;cache[include_headers] = fp.hexdigest()
 &nbsp; &nbsp;<span class="hljs-keyword">return</span> cache[include_headers]
</code></pre>


<p data-nodeid="79543">request_fingerprint 就是计算 Request 指纹的方法，其方法内部使用的是 hashlib 的 sha1 方法。计算的字段包括 Request 的 Method、URL、Body、Headers 这几部分内容，这里只要有一点不同，那么计算的结果就不同。计算得到的结果是加密后的字符串，也就是指纹。每个 Request 都有独有的指纹，指纹就是一个字符串，判定字符串是否重复比判定 Request 对象是否重复容易得多，所以指纹可以作为判定 Request 是否重复的依据。</p>
<p data-nodeid="79544">那么我们如何判定是否重复呢？Scrapy 是这样实现的，如下所示：</p>

<pre class="lang-java" data-nodeid="79747"><code data-language="java"><span class="hljs-function">def <span class="hljs-title">__init__</span><span class="hljs-params">(self)</span>:
 &nbsp; &nbsp;self.fingerprints </span>= set()
 &nbsp; &nbsp;
<span class="hljs-function">def <span class="hljs-title">request_seen</span><span class="hljs-params">(self, request)</span>:
 &nbsp; &nbsp;fp </span>= self.request_fingerprint(request)
 &nbsp; &nbsp;<span class="hljs-keyword">if</span> fp in self.fingerprints:
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">return</span> True
 &nbsp; &nbsp;self.fingerprints.add(fp)
</code></pre>

<p data-nodeid="79946">在去重的类 RFPDupeFilter 中，有一个 request_seen 方法，这个方法有一个参数 request，它的作用就是检测该 Request 对象是否重复。这个方法调用 request_fingerprint 获取该 Request 的指纹，检测这个指纹是否存在于 fingerprints 变量中，而 fingerprints 是一个集合，集合的元素都是不重复的。如果指纹存在，那么就返回 True，说明该 Request 是重复的，否则将这个指纹加入集合中。如果下次还有相同的 Request 传递过来，指纹也是相同的，那么这时指纹就已经存在于集合中，Request 对象就会直接判定为重复。这样去重的目的就实现了。</p>
<p data-nodeid="79947">Scrapy 的去重过程就是，利用集合元素的不重复特性来实现 Request 的去重。</p>

<p data-nodeid="77456">对于分布式爬虫来说，我们肯定不能再使用每个爬虫各自的集合来去重了。因为这样还是每台主机单独维护自己的集合，不能做到共享。多台主机如果生成了相同的 Request，只能各自去重，各个主机之间就无法做到去重了。</p>
<p data-nodeid="77457">那么要实现多台主机去重，这个指纹集合也需要是共享的，Redis 正好有集合的存储数据结构，我们可以利用 Redis 的集合作为指纹集合，那么这样去重集合也是共享的。每台主机新生成 Request 之后，会把该 Request 的指纹与集合比对，如果指纹已经存在，说明该 Request 是重复的，否则将 Request 的指纹加入这个集合中即可。利用同样的原理不同的存储结构我们也可以实现分布式 Reqeust 的去重。</p>
<h3 data-nodeid="77458">防止中断</h3>
<p data-nodeid="77459">在 Scrapy 中，爬虫运行时的 Request 队列放在内存中。爬虫运行中断后，这个队列的空间就被释放，此队列就被销毁了。所以一旦爬虫运行中断，爬虫再次运行就相当于全新的爬取过程。</p>
<p data-nodeid="77460">要做到中断后继续爬取，我们可以将队列中的 Request 保存起来，下次爬取直接读取保存数据即可获取上次爬取的队列。我们在 Scrapy 中指定一个爬取队列的存储路径即可，这个路径使用 JOB_DIR 变量来标识，我们可以用如下命令来实现：</p>
<pre class="lang-java" data-nodeid="80152"><code data-language="java">scrapy crawl spider -s JOBDIR=crawls/spider
</code></pre>

<p data-nodeid="77462">更加详细的使用方法可以参见官方文档，链接为：<a href="https://doc.scrapy.org/en/latest/topics/jobs.html" data-nodeid="77527">https://doc.scrapy.org/en/latest/topics/jobs.html</a>。<br>
在 Scrapy 中，我们实际是把爬取队列保存到本地，第二次爬取直接读取并恢复队列即可。那么在分布式架构中我们还用担心这个问题吗？不需要。因为爬取队列本身就是用数据库保存的，如果爬虫中断了，数据库中的 Request 依然是存在的，下次启动就会接着上次中断的地方继续爬取。</p>
<p data-nodeid="77463">所以，当 Redis 的队列为空时，爬虫会重新爬取；当 Redis 的队列不为空时，爬虫便会接着上次中断之处继续爬取。</p>
<h3 data-nodeid="77464">架构实现</h3>
<p data-nodeid="77465">我们接下来就需要在程序中实现这个架构了。首先需要实现一个共享的爬取队列，还要实现去重功能。另外，还需要重写一个 Scheduer 的实现，使之可以从共享的爬取队列存取 Request。</p>
<p data-nodeid="77466">幸运的是，已经有人实现了这些逻辑和架构，并发布成了叫作 Scrapy-Redis 的 Python 包。</p>
<p data-nodeid="77467" class="">在下一节，我们便看看 Scrapy-Redis 的源码实现，以及它的详细工作原理。</p>

# 分布式利器Scrapy-Redis原理
<p data-nodeid="120223">在上节课我们提到过，Scrapy-Redis 库已经为我们提供了 Scrapy 分布式的队列、调度器、去重等功能，其 GitHub 地址为： <a href="https://github.com/rmax/scrapy-redis" data-nodeid="120227">https://github.com/rmax/scrapy-redis</a>。</p>



<p data-nodeid="119764">本节课我们深入掌握利用 Redis 实现 Scrapy 分布式的方法，并深入了解 Scrapy-Redis 的原理。</p>
<h3 data-nodeid="119765">获取源码</h3>
<p data-nodeid="119766">可以把源码克隆下来，执行如下命令：</p>
<pre class="lang-java" data-nodeid="120680"><code data-language="java">git clone https:<span class="hljs-comment">//github.com/rmax/scrapy-redis.git </span>
</code></pre>


<p data-nodeid="119768">核心源码在 scrapy-redis/src/scrapy_redis 目录下。</p>
<h3 data-nodeid="119769">爬取队列</h3>
<p data-nodeid="119770">我们从爬取队列入手，来看看它的具体实现。源码文件为 queue.py，它包含了三个队列的实现，首先它实现了一个父类 Base，提供一些基本方法和属性，如下所示：</p>
<pre class="lang-dart" data-nodeid="123088"><code data-language="dart"><span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">Base</span>(<span class="hljs-title">object</span>): 
 &nbsp; &nbsp;"""<span class="hljs-title">Per</span>-<span class="hljs-title">spider</span> <span class="hljs-title">base</span> <span class="hljs-title">queue</span> <span class="hljs-title">class</span>""" 
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">__init__</span>(<span class="hljs-title">self</span>, <span class="hljs-title">server</span>, <span class="hljs-title">spider</span>, <span class="hljs-title">key</span>, <span class="hljs-title">serializer</span>=<span class="hljs-title">None</span>): 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">if</span> <span class="hljs-title">serializer</span> <span class="hljs-title">is</span> <span class="hljs-title">None</span>: 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">serializer</span> = <span class="hljs-title">picklecompat</span> 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">if</span> <span class="hljs-title">not</span> <span class="hljs-title">hasattr</span>(<span class="hljs-title">serializer</span>, '<span class="hljs-title">loads</span>'): 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">raise</span> <span class="hljs-title">TypeError</span>("<span class="hljs-title">serializer</span> <span class="hljs-title">does</span> <span class="hljs-title">not</span> <span class="hljs-title">implement</span> '<span class="hljs-title">loads</span>' <span class="hljs-title">function</span>: % <span class="hljs-title">r</span>" 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;% <span class="hljs-title">serializer</span>) 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">if</span> <span class="hljs-title">not</span> <span class="hljs-title">hasattr</span>(<span class="hljs-title">serializer</span>, '<span class="hljs-title">dumps</span>'): 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">raise</span> <span class="hljs-title">TypeError</span>("<span class="hljs-title">serializer</span> '% <span class="hljs-title">s</span>' <span class="hljs-title">does</span> <span class="hljs-title">not</span> <span class="hljs-title">implement</span> '<span class="hljs-title">dumps</span>' <span class="hljs-title">function</span>: % <span class="hljs-title">r</span>" 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;% <span class="hljs-title">serializer</span>) 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">self</span>.<span class="hljs-title">server</span> = <span class="hljs-title">server</span> 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">self</span>.<span class="hljs-title">spider</span> = <span class="hljs-title">spider</span> 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">self</span>.<span class="hljs-title">key</span> = <span class="hljs-title">key</span> % </span>{<span class="hljs-string">'spider'</span>: spider.name} 
 &nbsp; &nbsp; &nbsp; &nbsp;self.serializer = serializer 
​ 
 &nbsp; &nbsp;def _encode_request(self, request): 
 &nbsp; &nbsp; &nbsp; &nbsp;obj = request_to_dict(request, self.spider) 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">return</span> self.serializer.dumps(obj) 
​ 
 &nbsp; &nbsp;def _decode_request(self, encoded_request): 
 &nbsp; &nbsp; &nbsp; &nbsp;obj = self.serializer.loads(encoded_request) 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">return</span> request_from_dict(obj, self.spider) 
​ 
 &nbsp; &nbsp;def __len__(self): 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-string">"""Return the length of the queue"""</span> 
 &nbsp; &nbsp; &nbsp; &nbsp;raise NotImplementedError 
​ 
 &nbsp; &nbsp;def push(self, request): 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-string">"""Push a request"""</span> 
 &nbsp; &nbsp; &nbsp; &nbsp;raise NotImplementedError 
​ 
 &nbsp; &nbsp;def pop(self, timeout=<span class="hljs-number">0</span>): 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-string">"""Pop a request"""</span> 
 &nbsp; &nbsp; &nbsp; &nbsp;raise NotImplementedError 
​ 
 &nbsp; &nbsp;def clear(self): 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-string">"""Clear queue/stack"""</span> 
 &nbsp; &nbsp; &nbsp; &nbsp;self.server.delete(self.key) 
</code></pre>








<p data-nodeid="123389">首先看一下 _encode_request 和 _decode_request 方法，因为我们需要把一个  Request 对象存储到数据库中，但数据库无法直接存储对象，所以需要将 Request 序列转化成字符串再存储，而这两个方法分别是序列化和反序列化的操作，利用 pickle 库来实现，一般在调用 push 将 Request 存入数据库时会调用 _encode_request 方法进行序列化，在调用 pop 取出 Request 的时候会调用 _decode_request 进行反序列化。</p>
<p data-nodeid="126432" class="">在父类中 __len__、push 和 pop 方法都是未实现的，会直接抛出 NotImplementedError，因此是不能直接使用这个类的，必须实现一个子类来重写这三个方法，而不同的子类就会有不同的实现，也就有着不同的功能。</p>



<p data-nodeid="119773">接下来我们就需要定义一些子类来继承 Base 类，并重写这几个方法，那在源码中就有三个子类的实现，它们分别是 FifoQueue、PriorityQueue、LifoQueue，我们分别来看下它们的实现原理。</p>
<p data-nodeid="119774">首先是 FifoQueue：</p>
<pre class="lang-dart" data-nodeid="125821"><code data-language="dart"><span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">FifoQueue</span>(<span class="hljs-title">Base</span>): 
 &nbsp; &nbsp;"""<span class="hljs-title">Per</span>-<span class="hljs-title">spider</span> <span class="hljs-title">FIFO</span> <span class="hljs-title">queue</span>""" 
​ 
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">__len__</span>(<span class="hljs-title">self</span>): 
 &nbsp; &nbsp; &nbsp; &nbsp;"""<span class="hljs-title">Return</span> <span class="hljs-title">the</span> <span class="hljs-title">length</span> <span class="hljs-title">of</span> <span class="hljs-title">the</span> <span class="hljs-title">queue</span>""" 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">return</span> <span class="hljs-title">self</span>.<span class="hljs-title">server</span>.<span class="hljs-title">llen</span>(<span class="hljs-title">self</span>.<span class="hljs-title">key</span>) 
​ 
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">push</span>(<span class="hljs-title">self</span>, <span class="hljs-title">request</span>): 
 &nbsp; &nbsp; &nbsp; &nbsp;"""<span class="hljs-title">Push</span> <span class="hljs-title">a</span> <span class="hljs-title">request</span>""" 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">self</span>.<span class="hljs-title">server</span>.<span class="hljs-title">lpush</span>(<span class="hljs-title">self</span>.<span class="hljs-title">key</span>, <span class="hljs-title">self</span>.<span class="hljs-title">_encode_request</span>(<span class="hljs-title">request</span>)) 
​ 
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">pop</span>(<span class="hljs-title">self</span>, <span class="hljs-title">timeout</span>=0): 
 &nbsp; &nbsp; &nbsp; &nbsp;"""<span class="hljs-title">Pop</span> <span class="hljs-title">a</span> <span class="hljs-title">request</span>""" 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">if</span> <span class="hljs-title">timeout</span> &gt; 0: 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">data</span> = <span class="hljs-title">self</span>.<span class="hljs-title">server</span>.<span class="hljs-title">brpop</span>(<span class="hljs-title">self</span>.<span class="hljs-title">key</span>, <span class="hljs-title">timeout</span>) 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">if</span> <span class="hljs-title">isinstance</span>(<span class="hljs-title">data</span>, <span class="hljs-title">tuple</span>): 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">data</span> = <span class="hljs-title">data</span>[1] 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">else</span>: 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">data</span> = <span class="hljs-title">self</span>.<span class="hljs-title">server</span>.<span class="hljs-title">rpop</span>(<span class="hljs-title">self</span>.<span class="hljs-title">key</span>) 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">if</span> <span class="hljs-title">data</span>: 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">return</span> <span class="hljs-title">self</span>.<span class="hljs-title">_decode_request</span>(<span class="hljs-title">data</span>) 
</span></code></pre>








<p data-nodeid="127848">可以看到这个类继承了 Base 类，并重写了 __len__、push、pop 这三个方法，在这三个方法中都是对 server 对象的操作，而 server 对象就是一个 Redis 连接对象，我们可以直接调用其操作 Redis 的方法对数据库进行操作，可以看到这里的操作方法有 llen、lpush、rpop 等，这就代表此爬取队列是使用的 Redis 的列表，序列化后的 Request 会被存入列表中，就是列表的其中一个元素，__len__ 方法是获取列表的长度，push 方法中调用了 lpush 操作，这代表从列表左侧存入数据，pop 方法中调用了 rpop 操作，这代表从列表右侧取出数据。</p>
<p data-nodeid="127849">所以 Request 在列表中的存取顺序是左侧进、右侧出，所以这是有序的进出，即先进先出，英文叫作 First Input First Output，也被简称为 FIFO，而此类的名称就叫作 FifoQueue。</p>





<p data-nodeid="119777">另外还有一个与之相反的实现类，叫作 LifoQueue，实现如下：</p>
<pre class="lang-dart" data-nodeid="130269"><code data-language="dart"><span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">LifoQueue</span>(<span class="hljs-title">Base</span>): 
 &nbsp; &nbsp;"""<span class="hljs-title">Per</span>-<span class="hljs-title">spider</span> <span class="hljs-title">LIFO</span> <span class="hljs-title">queue</span>.""" 
​ 
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">__len__</span>(<span class="hljs-title">self</span>): 
 &nbsp; &nbsp; &nbsp; &nbsp;"""<span class="hljs-title">Return</span> <span class="hljs-title">the</span> <span class="hljs-title">length</span> <span class="hljs-title">of</span> <span class="hljs-title">the</span> <span class="hljs-title">stack</span>""" 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">return</span> <span class="hljs-title">self</span>.<span class="hljs-title">server</span>.<span class="hljs-title">llen</span>(<span class="hljs-title">self</span>.<span class="hljs-title">key</span>) 
​ 
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">push</span>(<span class="hljs-title">self</span>, <span class="hljs-title">request</span>): 
 &nbsp; &nbsp; &nbsp; &nbsp;"""<span class="hljs-title">Push</span> <span class="hljs-title">a</span> <span class="hljs-title">request</span>""" 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">self</span>.<span class="hljs-title">server</span>.<span class="hljs-title">lpush</span>(<span class="hljs-title">self</span>.<span class="hljs-title">key</span>, <span class="hljs-title">self</span>.<span class="hljs-title">_encode_request</span>(<span class="hljs-title">request</span>)) 
​ 
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">pop</span>(<span class="hljs-title">self</span>, <span class="hljs-title">timeout</span>=0): 
 &nbsp; &nbsp; &nbsp; &nbsp;"""<span class="hljs-title">Pop</span> <span class="hljs-title">a</span> <span class="hljs-title">request</span>""" 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">if</span> <span class="hljs-title">timeout</span> &gt; 0: 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">data</span> = <span class="hljs-title">self</span>.<span class="hljs-title">server</span>.<span class="hljs-title">blpop</span>(<span class="hljs-title">self</span>.<span class="hljs-title">key</span>, <span class="hljs-title">timeout</span>) 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">if</span> <span class="hljs-title">isinstance</span>(<span class="hljs-title">data</span>, <span class="hljs-title">tuple</span>): 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">data</span> = <span class="hljs-title">data</span>[1] 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">else</span>: 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">data</span> = <span class="hljs-title">self</span>.<span class="hljs-title">server</span>.<span class="hljs-title">lpop</span>(<span class="hljs-title">self</span>.<span class="hljs-title">key</span>) 
​ 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">if</span> <span class="hljs-title">data</span>: 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">return</span> <span class="hljs-title">self</span>.<span class="hljs-title">_decode_request</span>(<span class="hljs-title">data</span>) 
</span></code></pre>








<p data-nodeid="130570">与 FifoQueue 不同的就是它的 pop 方法，在这里使用的是 lpop 操作，也就是从左侧出，而 push 方法依然是使用的 lpush 操作，是从左侧入。那么这样达到的效果就是先进后出、后进先出，英文叫作 Last In First Out，简称为 LIFO，而此类名称就叫作 LifoQueue。同时这个存取方式类似栈的操作，所以其实也可以称作 StackQueue。</p>
<p data-nodeid="130571">另外在源码中还有一个子类实现，叫作 PriorityQueue，顾名思义，它叫作优先级队列，实现如下：</p>

<pre class="lang-dart" data-nodeid="131777"><code data-language="dart"><span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">PriorityQueue</span>(<span class="hljs-title">Base</span>): 
 &nbsp; &nbsp;"""<span class="hljs-title">Per</span>-<span class="hljs-title">spider</span> <span class="hljs-title">priority</span> <span class="hljs-title">queue</span> <span class="hljs-title">abstraction</span> <span class="hljs-title">using</span> <span class="hljs-title">redis</span>' <span class="hljs-title">sorted</span> <span class="hljs-title">set</span>""" 
​ 
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">__len__</span>(<span class="hljs-title">self</span>): 
 &nbsp; &nbsp; &nbsp; &nbsp;"""<span class="hljs-title">Return</span> <span class="hljs-title">the</span> <span class="hljs-title">length</span> <span class="hljs-title">of</span> <span class="hljs-title">the</span> <span class="hljs-title">queue</span>""" 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">return</span> <span class="hljs-title">self</span>.<span class="hljs-title">server</span>.<span class="hljs-title">zcard</span>(<span class="hljs-title">self</span>.<span class="hljs-title">key</span>) 
​ 
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">push</span>(<span class="hljs-title">self</span>, <span class="hljs-title">request</span>): 
 &nbsp; &nbsp; &nbsp; &nbsp;"""<span class="hljs-title">Push</span> <span class="hljs-title">a</span> <span class="hljs-title">request</span>""" 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">data</span> = <span class="hljs-title">self</span>.<span class="hljs-title">_encode_request</span>(<span class="hljs-title">request</span>) 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">score</span> = -<span class="hljs-title">request</span>.<span class="hljs-title">priority</span> 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">self</span>.<span class="hljs-title">server</span>.<span class="hljs-title">execute_command</span>('<span class="hljs-title">ZADD</span>', <span class="hljs-title">self</span>.<span class="hljs-title">key</span>, <span class="hljs-title">score</span>, <span class="hljs-title">data</span>) 
​ 
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">pop</span>(<span class="hljs-title">self</span>, <span class="hljs-title">timeout</span>=0): 
 &nbsp; &nbsp; &nbsp; &nbsp;""" 
 &nbsp; &nbsp; &nbsp;  <span class="hljs-title">Pop</span> <span class="hljs-title">a</span> <span class="hljs-title">request</span> 
 &nbsp; &nbsp; &nbsp;  <span class="hljs-title">timeout</span> <span class="hljs-title">not</span> <span class="hljs-title">support</span> <span class="hljs-title">in</span> <span class="hljs-title">this</span> <span class="hljs-title">queue</span> <span class="hljs-title">class</span> 
 &nbsp; &nbsp; &nbsp;  """ 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">pipe</span> = <span class="hljs-title">self</span>.<span class="hljs-title">server</span>.<span class="hljs-title">pipeline</span>() 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">pipe</span>.<span class="hljs-title">multi</span>() 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">pipe</span>.<span class="hljs-title">zrange</span>(<span class="hljs-title">self</span>.<span class="hljs-title">key</span>, 0, 0).<span class="hljs-title">zremrangebyrank</span>(<span class="hljs-title">self</span>.<span class="hljs-title">key</span>, 0, 0) 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">results</span>, <span class="hljs-title">count</span> = <span class="hljs-title">pipe</span>.<span class="hljs-title">execute</span>() 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">if</span> <span class="hljs-title">results</span>: 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">return</span> <span class="hljs-title">self</span>.<span class="hljs-title">_decode_request</span>(<span class="hljs-title">results</span>[0]) 
</span></code></pre>




<p data-nodeid="133340">在这里我们可以看到 __len__、push、pop 方法中使用了 server 对象的 zcard、zadd、zrange 操作，可以知道这里使用的存储结果是有序集合 Sorted Set，在这个集合中每个元素都可以设置一个分数，那么这个分数就代表优先级。</p>
<p data-nodeid="133341">在 __len__ 方法里调用了 zcard 操作，返回的就是有序集合的大小，也就是爬取队列的长度，在 push 方法中调用了 zadd 操作，就是向集合中添加元素，这里的分数指定成 Request 的优先级的相反数，因为分数低的会排在集合的前面，所以这里高优先级的 Request 就会存在集合的最前面。pop 方法是首先调用了 zrange 操作取出了集合的第一个元素，因为最高优先级的 Request 会存在集合最前面，所以第一个元素就是最高优先级的 Request，然后再调用 zremrangebyrank 操作将这个元素删除，这样就完成了取出并删除的操作。</p>





<p data-nodeid="119782">此队列是默认使用的队列，也就是爬取队列默认是使用有序集合来存储的。</p>
<h3 data-nodeid="119783">去重过滤</h3>
<p data-nodeid="119784">前面说过 Scrapy 的去重是利用集合来实现的，而在 Scrapy 分布式中的去重就需要利用共享的集合，那么这里使用的就是 Redis 中的集合数据结构。我们来看看去重类是怎样实现的，源码文件是 dupefilter.py，其内实现了一个 RFPDupeFilter 类，如下所示：</p>
<pre class="lang-dart" data-nodeid="133654"><code data-language="dart"><span class="hljs-class"><span class="hljs-keyword">class</span> <span class="hljs-title">RFPDupeFilter</span>(<span class="hljs-title">BaseDupeFilter</span>): 
 &nbsp; &nbsp;"""<span class="hljs-title">Redis</span>-<span class="hljs-title">based</span> <span class="hljs-title">request</span> <span class="hljs-title">duplicates</span> <span class="hljs-title">filter</span>. 
 &nbsp;  <span class="hljs-title">This</span> <span class="hljs-title">class</span> <span class="hljs-title">can</span> <span class="hljs-title">also</span> <span class="hljs-title">be</span> <span class="hljs-title">used</span> <span class="hljs-title">with</span> <span class="hljs-title">default</span> <span class="hljs-title">Scrapy</span>'<span class="hljs-title">s</span> <span class="hljs-title">scheduler</span>. 
 &nbsp;  """ 
 &nbsp; &nbsp;<span class="hljs-title">logger</span> = <span class="hljs-title">logger</span> 
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">__init__</span>(<span class="hljs-title">self</span>, <span class="hljs-title">server</span>, <span class="hljs-title">key</span>, <span class="hljs-title">debug</span>=<span class="hljs-title">False</span>): 
 &nbsp; &nbsp; &nbsp; &nbsp;"""<span class="hljs-title">Initialize</span> <span class="hljs-title">the</span> <span class="hljs-title">duplicates</span> <span class="hljs-title">filter</span>. 
 &nbsp; &nbsp; &nbsp;  <span class="hljs-title">Parameters</span> 
 &nbsp; &nbsp; &nbsp;  ---------- 
 &nbsp; &nbsp; &nbsp;  <span class="hljs-title">server</span> : <span class="hljs-title">redis</span>.<span class="hljs-title">StrictRedis</span> 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  <span class="hljs-title">The</span> <span class="hljs-title">redis</span> <span class="hljs-title">server</span> <span class="hljs-title">instance</span>. 
 &nbsp; &nbsp; &nbsp;  <span class="hljs-title">key</span> : <span class="hljs-title">str</span> 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  <span class="hljs-title">Redis</span> <span class="hljs-title">key</span> <span class="hljs-title">Where</span> <span class="hljs-title">to</span> <span class="hljs-title">store</span> <span class="hljs-title">fingerprints</span>. 
 &nbsp; &nbsp; &nbsp;  <span class="hljs-title">debug</span> : <span class="hljs-title">bool</span>, <span class="hljs-title">optional</span> 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  <span class="hljs-title">Whether</span> <span class="hljs-title">to</span> <span class="hljs-title">log</span> <span class="hljs-title">filtered</span> <span class="hljs-title">requests</span>. 
 &nbsp; &nbsp; &nbsp;  """ 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">self</span>.<span class="hljs-title">server</span> = <span class="hljs-title">server</span> 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">self</span>.<span class="hljs-title">key</span> = <span class="hljs-title">key</span> 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">self</span>.<span class="hljs-title">debug</span> = <span class="hljs-title">debug</span> 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">self</span>.<span class="hljs-title">logdupes</span> = <span class="hljs-title">True</span> 
​ 
 &nbsp; &nbsp;@<span class="hljs-title">classmethod</span> 
 &nbsp; &nbsp;<span class="hljs-title">def</span> <span class="hljs-title">from_settings</span>(<span class="hljs-title">cls</span>, <span class="hljs-title">settings</span>): 
 &nbsp; &nbsp; &nbsp; &nbsp;"""<span class="hljs-title">Returns</span> <span class="hljs-title">an</span> <span class="hljs-title">instance</span> <span class="hljs-title">from</span> <span class="hljs-title">given</span> <span class="hljs-title">settings</span>. 
 &nbsp; &nbsp; &nbsp;  <span class="hljs-title">This</span> <span class="hljs-title">uses</span> <span class="hljs-title">by</span> <span class="hljs-title">default</span> <span class="hljs-title">the</span> <span class="hljs-title">key</span> ``<span class="hljs-title">dupefilter</span>:&lt;<span class="hljs-title">timestamp</span>&gt;``. <span class="hljs-title">When</span> <span class="hljs-title">using</span> <span class="hljs-title">the</span> 
 &nbsp; &nbsp; &nbsp;  ``<span class="hljs-title">scrapy_redis</span>.<span class="hljs-title">scheduler</span>.<span class="hljs-title">Scheduler</span>`` <span class="hljs-title">class</span>, <span class="hljs-title">this</span> <span class="hljs-title">method</span> <span class="hljs-title">is</span> <span class="hljs-title">not</span> <span class="hljs-title">used</span> <span class="hljs-title">as</span> 
 &nbsp; &nbsp; &nbsp;  <span class="hljs-title">it</span> <span class="hljs-title">needs</span> <span class="hljs-title">to</span> <span class="hljs-title">pass</span> <span class="hljs-title">the</span> <span class="hljs-title">spider</span> <span class="hljs-title">name</span> <span class="hljs-title">in</span> <span class="hljs-title">the</span> <span class="hljs-title">key</span>. 
 &nbsp; &nbsp; &nbsp;  <span class="hljs-title">Parameters</span> 
 &nbsp; &nbsp; &nbsp;  ---------- 
 &nbsp; &nbsp; &nbsp;  <span class="hljs-title">settings</span> : <span class="hljs-title">scrapy</span>.<span class="hljs-title">settings</span>.<span class="hljs-title">Settings</span> 
 &nbsp; &nbsp; &nbsp;  <span class="hljs-title">Returns</span> 
 &nbsp; &nbsp; &nbsp;  ------- 
 &nbsp; &nbsp; &nbsp;  <span class="hljs-title">RFPDupeFilter</span> 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  <span class="hljs-title">A</span> <span class="hljs-title">RFPDupeFilter</span> <span class="hljs-title">instance</span>. 
 &nbsp; &nbsp; &nbsp;  """ 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">server</span> = <span class="hljs-title">get_redis_from_settings</span>(<span class="hljs-title">settings</span>) 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-title">key</span> = <span class="hljs-title">defaults</span>.<span class="hljs-title">DUPEFILTER_KEY</span> % </span>{<span class="hljs-string">'timestamp'</span>: <span class="hljs-built_in">int</span>(time.time())} 
 &nbsp; &nbsp; &nbsp; &nbsp;debug = settings.getbool(<span class="hljs-string">'DUPEFILTER_DEBUG'</span>) 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">return</span> cls(server, key=key, debug=debug) 
​ 
 &nbsp; &nbsp;<span class="hljs-meta">@classmethod</span> 
 &nbsp; &nbsp;def from_crawler(cls, crawler): 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-string">"""Returns instance from crawler. 
 &nbsp; &nbsp; &nbsp;  Parameters 
 &nbsp; &nbsp; &nbsp;  ---------- 
 &nbsp; &nbsp; &nbsp;  crawler : scrapy.crawler.Crawler 
 &nbsp; &nbsp; &nbsp;  Returns 
 &nbsp; &nbsp; &nbsp;  ------- 
 &nbsp; &nbsp; &nbsp;  RFPDupeFilter 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  Instance of RFPDupeFilter. 
 &nbsp; &nbsp; &nbsp;  """</span> 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">return</span> cls.from_settings(crawler.settings) 
​ 
 &nbsp; &nbsp;def request_seen(self, request): 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-string">"""Returns True if request was already seen. 
 &nbsp; &nbsp; &nbsp;  Parameters 
 &nbsp; &nbsp; &nbsp;  ---------- 
 &nbsp; &nbsp; &nbsp;  request : scrapy.http.Request 
 &nbsp; &nbsp; &nbsp;  Returns 
 &nbsp; &nbsp; &nbsp;  ------- 
 &nbsp; &nbsp; &nbsp;  bool 
 &nbsp; &nbsp; &nbsp;  """</span> 
 &nbsp; &nbsp; &nbsp; &nbsp;fp = self.request_fingerprint(request) 
 &nbsp; &nbsp; &nbsp; &nbsp;added = self.server.sadd(self.key, fp) 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">return</span> added == <span class="hljs-number">0</span> 
​ 
 &nbsp; &nbsp;def request_fingerprint(self, request): 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-string">"""Returns a fingerprint for a given request. 
 &nbsp; &nbsp; &nbsp;  Parameters 
 &nbsp; &nbsp; &nbsp;  ---------- 
 &nbsp; &nbsp; &nbsp;  request : scrapy.http.Request 
​ 
 &nbsp; &nbsp; &nbsp;  Returns 
 &nbsp; &nbsp; &nbsp;  ------- 
 &nbsp; &nbsp; &nbsp;  str 
​ 
 &nbsp; &nbsp; &nbsp;  """</span> 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">return</span> request_fingerprint(request) 
​ 
 &nbsp; &nbsp;def close(self, reason=<span class="hljs-string">''</span>): 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-string">"""Delete data on close. Called by Scrapy's scheduler. 
 &nbsp; &nbsp; &nbsp;  Parameters 
 &nbsp; &nbsp; &nbsp;  ---------- 
 &nbsp; &nbsp; &nbsp;  reason : str, optional 
 &nbsp; &nbsp; &nbsp;  """</span> 
 &nbsp; &nbsp; &nbsp; &nbsp;self.clear() 
​ 
 &nbsp; &nbsp;def clear(self): 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-string">"""Clears fingerprints data."""</span> 
 &nbsp; &nbsp; &nbsp; &nbsp;self.server.delete(self.key) 
​ 
 &nbsp; &nbsp;def log(self, request, spider): 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-string">"""Logs given request. 
 &nbsp; &nbsp; &nbsp;  Parameters 
 &nbsp; &nbsp; &nbsp;  ---------- 
 &nbsp; &nbsp; &nbsp;  request : scrapy.http.Request 
 &nbsp; &nbsp; &nbsp;  spider : scrapy.spiders.Spider 
 &nbsp; &nbsp; &nbsp;  """</span> 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">if</span> self.debug: 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;msg = <span class="hljs-string">"Filtered duplicate request: %(request) s"</span> 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;self.logger.debug(msg, {<span class="hljs-string">'request'</span>: request}, extra={<span class="hljs-string">'spider'</span>: spider}) 
 &nbsp; &nbsp; &nbsp; &nbsp;elif self.logdupes: 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;msg = (<span class="hljs-string">"Filtered duplicate request %(request) s"</span> 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <span class="hljs-string">"- no more duplicates will be shown"</span> 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <span class="hljs-string">"(see DUPEFILTER_DEBUG to show all duplicates)"</span>) 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;self.logger.debug(msg, {<span class="hljs-string">'request'</span>: request}, extra={<span class="hljs-string">'spider'</span>: spider}) 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;self.logdupes = False 
</code></pre>

<p data-nodeid="133955">这里同样实现了一个 request_seen 方法，和 Scrapy 中的 request_seen 方法实现极其类似。不过这里集合使用的是 server 对象的 sadd 操作，也就是集合不再是一个简单数据结构了，而是直接换成了数据库的存储方式。</p>
<p data-nodeid="133956">鉴别重复的方式还是使用指纹，指纹同样是依靠 request_fingerprint 方法来获取的。获取指纹之后就直接向集合添加指纹，如果添加成功，说明这个指纹原本不存在于集合中，返回值 1。代码中最后的返回结果是判定添加结果是否为 0，如果刚才的返回值为 1，那这个判定结果就是 False，也就是不重复，否则判定为重复。</p>

<p data-nodeid="119787">这样我们就成功利用 Redis 的集合完成了指纹的记录和重复的验证。</p>
<h3 data-nodeid="119788">调度器</h3>
<p data-nodeid="119789">Scrapy-Redis 还帮我们实现了配合 Queue、DupeFilter 使用的调度器 Scheduler，源文件名称是 scheduler.py。我们可以指定一些配置，如 SCHEDULER_FLUSH_ON_START 即是否在爬取开始的时候清空爬取队列，SCHEDULER_PERSIST 即是否在爬取结束后保持爬取队列不清除。我们可以在 settings.py 里自由配置，而此调度器很好地实现了对接。</p>
<p data-nodeid="119790">接下来我们看看两个核心的存取方法，实现如下所示：</p>
<pre class="lang-dart" data-nodeid="134265"><code data-language="dart">def enqueue_request(self, request): 
 &nbsp; &nbsp;<span class="hljs-keyword">if</span> not request.dont_filter and self.df.request_seen(request): 
 &nbsp; &nbsp; &nbsp; &nbsp;self.df.log(request, self.spider) 
 &nbsp; &nbsp; &nbsp; &nbsp;<span class="hljs-keyword">return</span> False 
 &nbsp; &nbsp;<span class="hljs-keyword">if</span> self.stats: 
 &nbsp; &nbsp; &nbsp; &nbsp;self.stats.inc_value(<span class="hljs-string">'scheduler/enqueued/redis'</span>, spider=self.spider) 
 &nbsp; &nbsp;self.queue.push(request) 
 &nbsp; &nbsp;<span class="hljs-keyword">return</span> True 
​ 
def next_request(self): 
 &nbsp; &nbsp;block_pop_timeout = self.idle_before_close 
 &nbsp; &nbsp;request = self.queue.pop(block_pop_timeout) 
 &nbsp; &nbsp;<span class="hljs-keyword">if</span> request and self.stats: 
 &nbsp; &nbsp; &nbsp; &nbsp;self.stats.inc_value(<span class="hljs-string">'scheduler/dequeued/redis'</span>, spider=self.spider) 
 &nbsp; &nbsp;<span class="hljs-keyword">return</span> request 
</code></pre>

<p data-nodeid="119792">enqueue_request 可以向队列中添加 Request，核心操作就是调用 Queue 的 push 操作，还有一些统计和日志操作。next_request 就是从队列中取 Request，核心操作就是调用 Queue 的 pop 操作，此时如果队列中还有 Request，则 Request 会直接取出来，爬取继续，否则如果队列为空，爬取则会重新开始。</p>
<h3 data-nodeid="119793">总结</h3>
<p data-nodeid="119794">那么到现在为止我们就把之前所说的三个分布式的问题解决了，总结如下：</p>
<ul data-nodeid="119795">
<li data-nodeid="119796">
<p data-nodeid="119797">爬取队列的实现，在这里提供了三种队列，使用了 Redis 的列表或有序集合来维护。</p>
</li>
<li data-nodeid="119798">
<p data-nodeid="119799">去重的实现，使用了 Redis 的集合来保存 Request 的指纹以提供重复过滤。</p>
</li>
<li data-nodeid="119800">
<p data-nodeid="119801">中断后重新爬取的实现，中断后 Redis 的队列没有清空，再次启动时调度器的 next_request 会从队列中取到下一个 Request，继续爬取。</p>
</li>
</ul>
<h3 data-nodeid="119802">结语</h3>
<p data-nodeid="119803">以上内容便是 Scrapy-Redis 的核心源码解析。Scrapy-Redis 中还提供了 Spider、Item Pipeline 的实现，不过它们并不是必须使用的。</p>
<p data-nodeid="119804">在下一节，我们会将 Scrapy-Redis 集成到之前所实现的 Scrapy 项目中，实现多台主机协同爬取。</p>

# 实战上手，Scrapy-Redis分布式实现
<p data-nodeid="13155">在前面一节课我们了解了 Scrapy-Redis 的基本原理，本节课我们就结合之前的案例实现基于 Scrapy-Redis 的分布式爬虫吧。</p>
<h3 data-nodeid="13156" class="">环境准备</h3>
<p data-nodeid="13157" class="">本节案例我们基于第 46 讲 —— Scrapy 和 Pyppeteer 的动态渲染页面的抓取案例来进行学习，我们需要把它改写成基于 Redis 的分布式爬虫。</p>



<p data-nodeid="13583" class="">首先我们需要把代码下载下来，其 GitHub 地址为 <a href="https://github.com/Python3WebSpider/ScrapyPyppeteer" data-nodeid="13587">https://github.com/Python3WebSpider/ScrapyPyppeteer</a>，进入项目，试着运行代码确保可以顺利执行，运行效果如图所示：<br>
<img src="https://s0.lgstatic.com/i/image/M00/3B/73/CgqCHl8kBHeAIw9uABBbpUkhs8c906.png" alt="1.png" data-nodeid="13592"><br>
其次，我们需要有一个 Redis 数据库，可以直接下载安装包并安装，也可以使用 Docker 启动，保证能正常连接和使用即可，比如我这里就在本地 localhost 启动了一个 Redis 数据库，运行在 6379 端口，密码为空。</p>



<p data-nodeid="13815">另外我们还需要安装 Scrapy-Redis 包，安装命令如下：</p>
<pre class="lang-java" data-nodeid="14763"><code data-language="java">pip3 install scrapy-redis
</code></pre>




<p data-nodeid="14982">安装完毕之后确保其可以正常导入使用即可。</p>
<h3 data-nodeid="15270" class="">实现</h3>
<p data-nodeid="15556">接下来我们只需要简单的几步操作就可以实现分布式爬虫的配置了。</p>
<h4 data-nodeid="15557" class="">修改 Scheduler</h4>
<p data-nodeid="15840" class="">在前面的课时中我们讲解了 Scheduler 的概念，它是用来处理 Request、Item 等对象的调度逻辑的，默认情况下，Request 的队列是在内存中的，为了实现分布式，我们需要将队列迁移到 Redis 中，这时候我们就需要修改 Scheduler，修改非常简单，只需要在 settings.py 里面添加如下代码即可：</p>
<pre data-nodeid="15841" class=""><code>SCHEDULER = "scrapy_redis.scheduler.Scheduler"
</code></pre>
<p data-nodeid="16102">这里我们将 Scheduler 的类修改为 Scrapy-Redis 提供的 Scheduler 类，这样在我们运行爬虫时，Request 队列就会出现在 Redis 中了。</p>
<h4 data-nodeid="16103" class="">修改 Redis 连接信息</h4>
<p data-nodeid="16361" class="">另外我们还需要修改下 Redis 的连接信息，这样 Scrapy 才能成功连接到 Redis 数据库，修改格式如下：</p>
<pre data-nodeid="16362" class=""><code>REDIS_URL = 'redis://[user:pass]@hostname:9001'
</code></pre>
<p data-nodeid="16586">在这里我们需要根据如上的格式来修改，由于我的 Redis 是在本地运行的，所以在这里就不需要填写用户名密码了，直接设置为如下内容即可：</p>
<pre data-nodeid="16587" class=""><code>REDIS_URL = 'redis://localhost:6379'
</code></pre>
<h4 data-nodeid="16790" class="">修改去重类</h4>
<p data-nodeid="16991">既然 Request 队列迁移到了 Redis，那么相应的去重操作我们也需要迁移到 Redis 里面，前一节课我们讲解了 Dupefilter 的原理，这里我们就修改下去重类来实现基于 Redis 的去重：</p>
<pre data-nodeid="16992" class=""><code>DUPEFILTER_CLASS = "scrapy_redis.dupefilter.RFPDupeFilter"
</code></pre>
<h4 data-nodeid="17166" class="">配置持久化</h4>
<p data-nodeid="17338">一般来说开启了 Redis 分布式队列之后，我们不希望爬虫在关闭时将整个队列和去重信息全部删除，因为很有可能在某个情况下我们会手动关闭爬虫或者爬虫遭遇意外终止，为了解决这个问题，我们可以配置 Redis 队列的持久化，修改如下：</p>
<pre data-nodeid="17339" class=""><code>SCHEDULER_PERSIST = True
</code></pre>
<p data-nodeid="17499">好了，到此为止我们就完成分布式爬虫的配置了。</p>
<h3 data-nodeid="17500" class="">运行</h3>
<p data-nodeid="17501" class="">上面我们完成的实际上并不是真正意义的分布式爬虫，因为 Redis 队列我们使用的是本地的 Redis，所以多个爬虫需要运行在本地才可以，如果想实现真正意义的分布式爬虫，可以使用远程 Redis，这样我们就能在多台主机运行爬虫连接此 Redis 从而实现真正意义上的分布式爬虫了。</p>














<p data-nodeid="17657">不过没关系，我们可以在本地启动多个爬虫验证下爬取效果。我们在多个命令行窗口运行如下命令：</p>
<pre data-nodeid="17658" class=""><code>scrapy crawl book
</code></pre>
<p data-nodeid="17982">第一个爬虫出现了如下运行效果：<br>
<img src="https://s0.lgstatic.com/i/image/M00/3B/73/CgqCHl8kBSuAHjAvABs5G4LewHk110.png" alt="2.png" data-nodeid="17989"><br>
这时候不要关闭此窗口，再打开另外一个窗口，运行同样的爬取命令：</p>
<pre data-nodeid="17983" class=""><code>scrapy crawl book
</code></pre>
<p data-nodeid="18149" class="">运行效果如下：<br>
<img src="https://s0.lgstatic.com/i/image/M00/3B/68/Ciqc1F8kBUOAXh_TABoUyMu0pFo413.png" alt="3.png" data-nodeid="18154"><br>
这时候我们可以观察到它从第 24 页开始爬取了，因为当前爬取队列存在第一个爬虫生成的爬取 Request，第二个爬虫启动时检测到有 Request 存在就直接读取已经存在的 Request，然后接着爬取了。</p>






<p data-nodeid="12641">同样，我们可以启动第三个、第四个爬虫实现同样的爬取功能。这样，我们就基于 Scrapy-Redis 成功实现了基本的分布式爬虫功能。</p>

# Scrapy部署不用愁，Scrapyd的原理
<p data-nodeid="15153">上节课我们的分布式爬虫部署完成并可以成功运行了，但是有个环节非常烦琐，那就是代码部署。</p>
<p data-nodeid="15154">我们设想下面的几个场景：</p>
<ul data-nodeid="15155">
<li data-nodeid="15156">
<p data-nodeid="15157">如果采用上传文件的方式部署代码，我们首先需要将代码压缩，然后采用 SFTP 或 FTP 的方式将文件上传到服务器，之后再连接服务器将文件解压，每个服务器都需要这样配置。</p>
</li>
<li data-nodeid="15158">
<p data-nodeid="15159">如果采用 Git 同步的方式部署代码，我们可以先把代码 Push 到某个 Git 仓库里，然后再远程连接各台主机执行 Pull 操作，同步代码，每个服务器同样需要做一次操作。</p>
</li>
</ul>
<p data-nodeid="15160">如果代码突然有更新，那我们必须更新每个服务器，而且万一哪台主机的版本没控制好，还可能会影响整体的分布式爬取状况。</p>
<p data-nodeid="15161">所以我们需要一个更方便的工具来部署 Scrapy 项目，如果可以省去一遍遍逐个登录服务器部署的操作，那将会方便很多。</p>
<p data-nodeid="15162">本节我们就来看看提供分布式部署的工具 Scrapyd。</p>
<h3 data-nodeid="16508" class="">了解 Scrapyd</h3>

<p data-nodeid="15164">接下来，我们就来深入地了解 Scrapyd，Scrapyd 是一个运行 Scrapy 爬虫的服务程序，它提供一系列 HTTP 接口来帮助我们部署、启动、停止、删除爬虫程序。Scrapyd 支持版本管理，同时还可以管理多个爬虫任务，利用它我们可以非常方便地完成 Scrapy 爬虫项目的部署任务调度。</p>
<h3 data-nodeid="17048" class="">准备工作</h3>

<p data-nodeid="15166">首先我们需要安装 scrapyd，一般我们部署的服务器是 Linux，所以这里以 Linux 为例来进行说明。</p>
<p data-nodeid="15167">这里推荐使用 pip 安装，命令如下：</p>
<pre class="lang-java" data-nodeid="19205"><code data-language="java">pip3 install scrapyd 
</code></pre>




<p data-nodeid="15169">另外为了我们编写的项目能够运行成功，还需要安装项目本身的环境依赖，如上一节的项目需要依赖 Scrapy、Scrapy-Redis、Gerapy-Pyppeteer 等库，也需要在服务器上安装，否则会出现部署失败的问题。<br>
安装完毕之后，需要新建一个配置文件 /etc/scrapyd/scrapyd.conf，Scrapyd 在运行的时候会读取此配置文件。</p>
<p data-nodeid="15170">在 Scrapyd 1.2 版本之后，不会自动创建该文件，需要我们自行添加。首先，执行如下命令新建文件：</p>
<pre class="lang-java" data-nodeid="19744"><code data-language="java">sudo mkdir /etc/scrapyd &nbsp; 
sudo vi /etc/scrapyd/scrapyd.conf 
</code></pre>

<p data-nodeid="15172">接着写入如下内容：</p>
<pre class="lang-java" data-nodeid="20283"><code data-language="java">[scrapyd] &nbsp; 
eggs_dir &nbsp;  = eggs &nbsp; 
logs_dir &nbsp;  = logs &nbsp; 
items_dir &nbsp; = &nbsp; 
jobs_to_keep = <span class="hljs-number">5</span> &nbsp; 
dbs_dir &nbsp; &nbsp; = dbs &nbsp; 
max_proc &nbsp;  = <span class="hljs-number">0</span> &nbsp; 
max_proc_per_cpu = <span class="hljs-number">10</span> &nbsp; 
finished_to_keep = <span class="hljs-number">100</span> &nbsp; 
poll_interval = <span class="hljs-number">5.0</span> &nbsp; 
bind_address = <span class="hljs-number">0.0</span>.<span class="hljs-number">0.0</span> &nbsp; 
http_port &nbsp; = <span class="hljs-number">6800</span> &nbsp; 
debug &nbsp; &nbsp; &nbsp; = off &nbsp; 
runner &nbsp; &nbsp;  = scrapyd.runner &nbsp; 
application = scrapyd.app.application &nbsp; 
launcher &nbsp;  = scrapyd.launcher.Launcher &nbsp; 
webroot &nbsp; &nbsp; = scrapyd.website.Root &nbsp; 
​ 
[services] &nbsp; 
schedule.json &nbsp; &nbsp; = scrapyd.webservice.Schedule &nbsp; 
cancel.json &nbsp; &nbsp; &nbsp; = scrapyd.webservice.Cancel &nbsp; 
addversion.json &nbsp; = scrapyd.webservice.AddVersion &nbsp; 
listprojects.json = scrapyd.webservice.ListProjects &nbsp; 
listversions.json = scrapyd.webservice.ListVersions &nbsp; 
listspiders.json  = scrapyd.webservice.ListSpiders &nbsp; 
delproject.json &nbsp; = scrapyd.webservice.DeleteProject &nbsp; 
delversion.json &nbsp; = scrapyd.webservice.DeleteVersion &nbsp; 
listjobs.json &nbsp; &nbsp; = scrapyd.webservice.ListJobs &nbsp; 
daemonstatus.json = scrapyd.webservice.DaemonStatus 
</code></pre>

<p data-nodeid="21361">配置文件的内容可以参见官方文档 <a href="https://scrapyd.readthedocs.io/en/stable/config.html#example-configuration-file" data-nodeid="21366">https://scrapyd.readthedocs.io/en/stable/config.html#example-configuration-file</a>。这里的配置文件有所修改，其中之一是 max_proc_per_cpu 官方默认为 4，即一台主机每个 CPU 最多运行 4 个 Scrapy 任务，在此提高为 10。另外一个是 bind_address，默认为本地 127.0.0.1，在此修改为 0.0.0.0，以使外网可以访问。</p>
<p data-nodeid="21362">Scrapyd 是一个纯 Python 项目，这里可以直接调用它来运行。为了使程序一直在后台运行，Linux 和 Mac 可以使用如下命令：</p>

<pre class="lang-java" data-nodeid="20822"><code data-language="java">(scrapyd &gt; /dev/<span class="hljs-keyword">null</span> &amp;) 
</code></pre>

<p data-nodeid="15176">这样 Scrapyd 就可以在后台持续运行了，控制台输出直接忽略。当然，如果想记录输出日志，可以修改输出目标，如下所示：</p>
<pre class="lang-java" data-nodeid="21915"><code data-language="java">(scrapyd&gt; ~/scrapyd.log &amp;) 
</code></pre>

<p data-nodeid="22454">此时会将 Scrapyd 的运行结果输出到～/scrapyd.log 文件中。当然也可以使用 screen、tmux、supervisor 等工具来实现进程守护。</p>
<p data-nodeid="23530">安装并运行了 Scrapyd 之后，我们就可以访问服务器的 6800 端口看到一个 WebUI 页面了，例如我的服务器地址为 120.27.34.25，在上面安装好了 Scrapyd 并成功运行，那么我就可以在本地的浏览器中打开： <a href="http://120.27.34.25:6800" data-nodeid="23535">http://120.27.34.25:6800</a>，就可以看到 Scrapyd 的首页，这里请自行替换成你的服务器地址查看即可，如图所示：</p>
<p data-nodeid="23531" class=""><img src="https://s0.lgstatic.com/i/image/M00/3D/CF/CgqCHl8qmKaANTr_AAFuWnOPCQ0587.png" alt="image (6).png" data-nodeid="23543"></p>



<p data-nodeid="15180">如果可以成功访问到此页面，那么证明 Scrapyd 配置就没有问题了。</p>
<h3 data-nodeid="24090" class="">Scrapyd 的功能</h3>

<p data-nodeid="15182">Scrapyd 提供了一系列 HTTP 接口来实现各种操作，在这里我们可以将接口的功能梳理一下，以 Scrapyd 所在的 IP 为 120.27.34.25 为例进行讲解。</p>
<h4 data-nodeid="15183">daemonstatus.json</h4>
<p data-nodeid="15184">这个接口负责查看 Scrapyd 当前服务和任务的状态，我们可以用 curl 命令来请求这个接口，命令如下：</p>
<pre class="lang-java" data-nodeid="24638"><code data-language="java">curl http:<span class="hljs-comment">//139.217.26.30:6800/daemonstatus.json </span>
</code></pre>

<p data-nodeid="15186">这样我们就会得到如下结果：</p>
<pre class="lang-java" data-nodeid="25185"><code data-language="java">{<span class="hljs-string">"status"</span>: <span class="hljs-string">"ok"</span>, <span class="hljs-string">"finished"</span>: <span class="hljs-number">90</span>, <span class="hljs-string">"running"</span>: <span class="hljs-number">9</span>, <span class="hljs-string">"node_name"</span>: <span class="hljs-string">"datacrawl-vm"</span>, <span class="hljs-string">"pending"</span>: <span class="hljs-number">0</span>} 
</code></pre>

<p data-nodeid="15188">返回结果是 Json 字符串，status 是当前运行状态， finished 代表当前已经完成的 Scrapy 任务，running 代表正在运行的 Scrapy 任务，pending 代表等待被调度的 Scrapyd 任务，node_name 就是主机的名称。</p>
<h4 data-nodeid="15189">addversion.json</h4>
<p data-nodeid="15190">这个接口主要是用来部署 Scrapy 项目，在部署的时候我们需要首先将项目打包成 Egg 文件，然后传入项目名称和部署版本。</p>
<p data-nodeid="15191">我们可以用如下的方式实现项目部署：</p>
<pre class="lang-java" data-nodeid="25732"><code data-language="java">curl http:<span class="hljs-comment">//120.27.34.25:6800/addversion.json -F project=wenbo -F version=first -F egg=@weibo.egg </span>
</code></pre>

<p data-nodeid="15193">在这里 -F 即代表添加一个参数，同时我们还需要将项目打包成 Egg 文件放到本地。<br>
这样发出请求之后我们可以得到如下结果：</p>
<pre class="lang-java" data-nodeid="26279"><code data-language="java">{<span class="hljs-string">"status"</span>: <span class="hljs-string">"ok"</span>, <span class="hljs-string">"spiders"</span>: <span class="hljs-number">3</span>} 
</code></pre>

<p data-nodeid="15195">这个结果表明部署成功，并且其中包含的 Spider 的数量为 3。此方法部署可能比较烦琐，在后面我会介绍更方便的工具来实现项目的部署。</p>
<h4 data-nodeid="15196">schedule.json</h4>
<p data-nodeid="15197">这个接口负责调度已部署好的 Scrapy 项目运行。我们可以通过如下接口实现任务调度：</p>
<pre class="lang-java" data-nodeid="26826"><code data-language="java">curl http:<span class="hljs-comment">//120.27.34.25:6800/schedule.json -d project=weibo -d spider=weibocn </span>
</code></pre>

<p data-nodeid="15199">在这里需要传入两个参数，project 即 Scrapy 项目名称，spider 即 Spider 名称。返回结果如下：</p>
<pre class="lang-java" data-nodeid="27373"><code data-language="java">{<span class="hljs-string">"status"</span>: <span class="hljs-string">"ok"</span>, <span class="hljs-string">"jobid"</span>: <span class="hljs-string">"6487ec79947edab326d6db28a2d86511e8247444"</span>} 
</code></pre>

<p data-nodeid="15201">status 代表 Scrapy 项目启动情况，jobid 代表当前正在运行的爬取任务代号。</p>
<h4 data-nodeid="15202">cancel.json</h4>
<p data-nodeid="15203">这个接口可以用来取消某个爬取任务，如果这个任务是 pending 状态，那么它将会被移除，如果这个任务是 running 状态，那么它将会被终止。</p>
<p data-nodeid="15204">我们可以用下面的命令来取消任务的运行：</p>
<pre class="lang-java" data-nodeid="27920"><code data-language="java">curl http:<span class="hljs-comment">//120.27.34.25:6800/cancel.json -d project=weibo -d job=6487ec79947edab326d6db28a2d86511e8247444 </span>
</code></pre>

<p data-nodeid="15206">在这里需要传入两个参数，project 即项目名称，job 即爬取任务代号。返回结果如下：</p>
<pre class="lang-java" data-nodeid="28467"><code data-language="java">{<span class="hljs-string">"status"</span>: <span class="hljs-string">"ok"</span>, <span class="hljs-string">"prevstate"</span>: <span class="hljs-string">"running"</span>} 
</code></pre>

<p data-nodeid="15208">status 代表请求执行情况，prevstate 代表之前的运行状态。</p>
<h4 data-nodeid="15209">listprojects.json</h4>
<p data-nodeid="15210">这个接口用来列出部署到 Scrapyd 服务上的所有项目描述。我们可以用下面的命令来获取 Scrapyd 服务器上的所有项目描述：</p>
<pre class="lang-java" data-nodeid="29014"><code data-language="java">curl http:<span class="hljs-comment">//120.27.34.25:6800/listprojects.json </span>
</code></pre>

<p data-nodeid="15212">这里不需要传入任何参数。返回结果如下：</p>
<pre class="lang-java" data-nodeid="29561"><code data-language="java">{<span class="hljs-string">"status"</span>: <span class="hljs-string">"ok"</span>, <span class="hljs-string">"projects"</span>: [<span class="hljs-string">"weibo"</span>, <span class="hljs-string">"zhihu"</span>]} 
</code></pre>

<p data-nodeid="15214">status 代表请求执行情况，projects 是项目名称列表。</p>
<h4 data-nodeid="15215">listversions.json</h4>
<p data-nodeid="15216">这个接口用来获取某个项目的所有版本号，版本号是按序排列的，最后一个条目是最新的版本号。</p>
<p data-nodeid="15217">我们可以用如下命令来获取项目的版本号：</p>
<pre class="lang-java" data-nodeid="30108"><code data-language="java">curl http:<span class="hljs-comment">//120.27.34.25:6800/listversions.json?project=weibo </span>
</code></pre>

<p data-nodeid="15219">在这里需要一个参数 project，就是项目的名称。返回结果如下：</p>
<pre class="lang-java" data-nodeid="30655"><code data-language="java">{<span class="hljs-string">"status"</span>: <span class="hljs-string">"ok"</span>, <span class="hljs-string">"versions"</span>: [<span class="hljs-string">"v1"</span>, <span class="hljs-string">"v2"</span>]} 
</code></pre>

<p data-nodeid="15221">status 代表请求执行情况，versions 是版本号列表。</p>
<h4 data-nodeid="15222">listspiders.json</h4>
<p data-nodeid="15223">这个接口用来获取某个项目最新的一个版本的所有 Spider 名称。我们可以用如下命令来获取项目的 Spider 名称：</p>
<pre class="lang-java" data-nodeid="31202"><code data-language="java">curl http:<span class="hljs-comment">//120.27.34.25:6800/listspiders.json?project=weibo </span>
</code></pre>

<p data-nodeid="15225">在这里需要一个参数 project，就是项目的名称。返回结果如下：</p>
<pre class="lang-java" data-nodeid="31749"><code data-language="java">{<span class="hljs-string">"status"</span>: <span class="hljs-string">"ok"</span>, <span class="hljs-string">"spiders"</span>: [<span class="hljs-string">"weibocn"</span>]} 
</code></pre>

<p data-nodeid="15227">status 代表请求执行情况，spiders 是 Spider 名称列表。</p>
<h4 data-nodeid="15228">listjobs.json</h4>
<p data-nodeid="15229">这个接口用来获取某个项目当前运行的所有任务详情。我们可以用如下命令来获取所有任务详情：</p>
<pre class="lang-java" data-nodeid="32296"><code data-language="java">curl http:<span class="hljs-comment">//120.27.34.25:6800/listjobs.json?project=weibo </span>
</code></pre>

<p data-nodeid="15231">在这里需要一个参数 project，就是项目的名称。返回结果如下：</p>
<pre class="lang-java" data-nodeid="32843"><code data-language="java">{<span class="hljs-string">"status"</span>: <span class="hljs-string">"ok"</span>, 
 <span class="hljs-string">"pending"</span>: [{<span class="hljs-string">"id"</span>: <span class="hljs-string">"78391cc0fcaf11e1b0090800272a6d06"</span>, <span class="hljs-string">"spider"</span>: <span class="hljs-string">"weibocn"</span>}], 
 <span class="hljs-string">"running"</span>: [{<span class="hljs-string">"id"</span>: <span class="hljs-string">"422e608f9f28cef127b3d5ef93fe9399"</span>, <span class="hljs-string">"spider"</span>: <span class="hljs-string">"weibocn"</span>, <span class="hljs-string">"start_time"</span>: <span class="hljs-string">"2017-07-12 10:14:03.594664"</span>}], 
 <span class="hljs-string">"finished"</span>: [{<span class="hljs-string">"id"</span>: <span class="hljs-string">"2f16646cfcaf11e1b0090800272a6d06"</span>, <span class="hljs-string">"spider"</span>: <span class="hljs-string">"weibocn"</span>, <span class="hljs-string">"start_time"</span>: <span class="hljs-string">"2017-07-12 10:14:03.594664"</span>, <span class="hljs-string">"end_time"</span>: <span class="hljs-string">"2017-07-12 10:24:03.594664"</span>}]} 
</code></pre>

<p data-nodeid="15233">status 代表请求执行情况，pendings 代表当前正在等待的任务，running 代表当前正在运行的任务，finished 代表已经完成的任务。</p>
<h4 data-nodeid="15234">delversion.json</h4>
<p data-nodeid="15235">这个接口用来删除项目的某个版本。我们可以用如下命令来删除项目版本：</p>
<pre class="lang-java" data-nodeid="33390"><code data-language="java">curl http:<span class="hljs-comment">//120.27.34.25:6800/delversion.json -d project=weibo -d version=v1 </span>
</code></pre>

<p data-nodeid="15237">在这里需要一个参数 project，就是项目的名称，还需要一个参数 version，就是项目的版本。返回结果如下：</p>
<pre class="lang-java" data-nodeid="33937"><code data-language="java">{<span class="hljs-string">"status"</span>: <span class="hljs-string">"ok"</span>} 
</code></pre>

<p data-nodeid="15239">status 代表请求执行情况，这样就代表删除成功了。</p>
<h4 data-nodeid="15240">delproject.json</h4>
<p data-nodeid="15241">这个接口用来删除某个项目。我们可以用如下命令来删除某个项目：</p>
<pre class="lang-java" data-nodeid="34484"><code data-language="java">curl http:<span class="hljs-comment">//120.27.34.25:6800/delproject.json -d project=weibo </span>
</code></pre>

<p data-nodeid="15243">在这里需要一个参数 project，就是项目的名称。返回结果如下：</p>
<pre class="lang-java" data-nodeid="35031"><code data-language="java">{<span class="hljs-string">"status"</span>: <span class="hljs-string">"ok"</span>} 
</code></pre>

<p data-nodeid="15245">status 代表请求执行情况，这样就代表删除成功了。<br>
以上就是 Scrapyd 所有的接口，我们可以直接请求 HTTP 接口即可控制项目的部署、启动、运行等操作。</p>
<h3 data-nodeid="35578" class="">ScrapydAPI 的使用</h3>

<p data-nodeid="15247">以上的这些接口可能使用起来还不是很方便，没关系，还有一个 ScrapydAPI 库对这些接口又做了一层封装，其安装方式如下：</p>
<pre class="lang-java" data-nodeid="36126"><code data-language="java">pip3 install python-scrapyd-api 
</code></pre>

<p data-nodeid="15249">下面我们来看下 ScrapydAPI 的使用方法，其实核心原理和 HTTP 接口请求方式并无二致，只不过用 Python 封装后使用更加便捷。<br>
我们可以用如下方式建立一个 ScrapydAPI 对象：</p>
<pre class="lang-java" data-nodeid="36673"><code data-language="java">from scrapyd_api <span class="hljs-keyword">import</span> ScrapydAPI 
scrapyd = ScrapydAPI(<span class="hljs-string">'http://120.27.34.25:6800'</span>) 
</code></pre>

<p data-nodeid="15251">然后就可以通过调用它的方法来实现对应接口的操作了，例如部署的操作可以使用如下方式：</p>
<pre class="lang-java" data-nodeid="37220"><code data-language="java">egg = open(<span class="hljs-string">'weibo.egg'</span>, <span class="hljs-string">'rb'</span>) 
scrapyd.add_version(<span class="hljs-string">'weibo'</span>, <span class="hljs-string">'v1'</span>, egg) 
</code></pre>

<p data-nodeid="15253">这样我们就可以将项目打包为 Egg 文件，然后把本地打包的 Egg 项目部署到远程 Scrapyd 了。<br>
另外 ScrapydAPI 还实现了所有 Scrapyd 提供的 API 接口，名称都是相同的，参数也是相同的。</p>
<p data-nodeid="15254">例如我们可以调用 list_projects 方法即可列出 Scrapyd 中所有已部署的项目：</p>
<pre class="lang-java" data-nodeid="37767"><code data-language="java">scrapyd.list_projects() 
[<span class="hljs-string">'weibo'</span>, <span class="hljs-string">'zhihu'</span>] 
</code></pre>

<p data-nodeid="15256">另外还有其他的方法在此不再一一列举了，名称和参数都是相同的，更加详细的操作可以参考其官方文档： <a href="http://python-scrapyd-api.readthedocs.io/" data-nodeid="15397">http://python-scrapyd-api.readthedocs.io/</a>。<br>
我们可以通过它来部署项目，并通过 HTTP 接口来控制任务的运行，不过这里有一个不方便的地方就是部署过程，首先它需要打包 Egg 文件然后再上传，还是比较烦琐的，这里再介绍另外一个工具 Scrapyd-Client。</p>
<h3 data-nodeid="38314" class="">Scrapyd-Client 部署</h3>

<p data-nodeid="15258">Scrapyd-Client 为了方便 Scrapy 项目的部署，提供两个功能：</p>
<ul data-nodeid="15259">
<li data-nodeid="15260">
<p data-nodeid="15261">将项目打包成 Egg 文件。</p>
</li>
<li data-nodeid="15262">
<p data-nodeid="15263">将打包生成的 Egg 文件通过 addversion.json 接口部署到 Scrapyd 上。</p>
</li>
</ul>
<p data-nodeid="15264">也就是说，Scrapyd-Client 帮我们把部署全部实现了，我们不需要再去关心 Egg 文件是怎样生成的，也不需要再去读 Egg 文件并请求接口上传了，这一切的操作只需要执行一个命令即可一键部署。</p>
<p data-nodeid="15265">要部署 Scrapy 项目，我们首先需要修改一下项目的配置文件，例如我们之前写的 Scrapy 项目，在项目的第一层会有一个 scrapy.cfg 文件，它的内容如下：</p>
<pre class="lang-java" data-nodeid="38862"><code data-language="java">[settings] 
default = scrapypyppeteer.settings 
​ 
[deploy] 
#url = http://localhost:6800/ 
project = scrapypyppeteer 
</code></pre>

<p data-nodeid="15267">在这里我们需要配置 deploy，例如我们要将项目部署到 120.27.34.25 的 Scrapyd 上，就需要修改为如下内容：</p>
<pre class="lang-java" data-nodeid="39409"><code data-language="java">[deploy] 
url = http:<span class="hljs-comment">//120.27.34.25:6800/ </span>
project = scrapypyppeteer 
</code></pre>

<p data-nodeid="15269">这样我们再在 scrapy.cfg 文件所在路径执行如下命令：</p>
<pre class="lang-java" data-nodeid="39956"><code data-language="java">scrapyd-deploy 
</code></pre>

<p data-nodeid="15271">运行结果如下：</p>
<pre class="lang-java" data-nodeid="40503"><code data-language="java">Packing version <span class="hljs-number">1501682277</span> 
Deploying to project <span class="hljs-string">"weibo"</span> in http:<span class="hljs-comment">//120.27.34.25:6800/addversion.json </span>
<span class="hljs-function">Server <span class="hljs-title">response</span> <span class="hljs-params">(<span class="hljs-number">200</span>)</span>: 
</span>{<span class="hljs-string">"status"</span>: <span class="hljs-string">"ok"</span>, <span class="hljs-string">"spiders"</span>: <span class="hljs-number">1</span>, <span class="hljs-string">"node_name"</span>: <span class="hljs-string">"datacrawl-vm"</span>, <span class="hljs-string">"project"</span>: <span class="hljs-string">"scrapypyppeteer"</span>, <span class="hljs-string">"version"</span>: <span class="hljs-string">"1501682277"</span>} 
</code></pre>

<p data-nodeid="41050">返回这样的结果就代表部署成功了。</p>
<p data-nodeid="41051">我们也可以指定项目版本，如果不指定的话默认为当前时间戳，指定的话通过 version 参数传递即可，例如：</p>

<pre class="lang-java" data-nodeid="41600"><code data-language="java">scrapyd-deploy --version <span class="hljs-number">201707131455</span> 
</code></pre>

<p data-nodeid="42971">值得注意的是在 Python3 的 Scrapyd 1.2.0 版本中我们不要指定版本号为带字母的字符串，需要为纯数字，否则可能会出现报错。</p>
<p data-nodeid="42972">另外如果我们有多台主机，我们可以配置各台主机的别名，例如可以修改配置文件为：</p>



<pre class="lang-java" data-nodeid="42147"><code data-language="java">[deploy:vm1] 
url = http:<span class="hljs-comment">//120.27.34.24:6800/ </span>
project = scrapypyppeteer 
​ 
[deploy:vm2] 
url = http:<span class="hljs-comment">//139.217.26.30:6800/ </span>
project = scrapypyppeteer 
</code></pre>

<p data-nodeid="15277">有多台主机的话就在此统一配置，一台主机对应一组配置，在 deploy 后面加上主机的别名即可，这样如果我们想将项目部署到 IP 为 139.217.26.30 的 vm2 主机，我们只需要执行如下命令：</p>
<pre class="lang-java" data-nodeid="43521"><code data-language="java">scrapyd-deploy vm2 
</code></pre>

<p data-nodeid="15279">这样我们就可以将项目部署到名称为 vm2 的主机上了。<br>
如此一来，如果我们有多台主机，我们只需要在 scrapy.cfg 文件中配置好各台主机的 Scrapyd 地址，然后调用 scrapyd-deploy 命令加主机名称即可实现部署，非常方便。</p>
<p data-nodeid="15280">如果 Scrapyd 设置了访问限制的话，我们可以在配置文件中加入用户名和密码的配置，同时修改端口，修改成 Nginx 代理端口，如在模块一我们使用的是 6801，那么这里就需要改成 6801，修改如下：</p>
<pre class="lang-java" data-nodeid="44068"><code data-language="java">[deploy:vm1] 
url = http:<span class="hljs-comment">//120.27.34.24:6801/ </span>
project = scrapypyppeteer 
username = admin 
password = admin 
​ 
[deploy:vm2] 
url = http:<span class="hljs-comment">//139.217.26.30:6801/ </span>
project = scrapypyppeteer 
username = germey 
password = germey 
</code></pre>

<p data-nodeid="15282">这样通过加入 username 和 password 字段，我们就可以在部署时自动进行 Auth 验证，然后成功实现部署。</p>
<h3 data-nodeid="44615" class="">总结</h3>

<p data-nodeid="15284">以上我们介绍了 Scrapyd、Scrapyd-API、Scrapyd-Client 的部署方式，希望你可以多多尝试。</p>

# 容器化技术也得会，Scrapy对接docker
<p data-nodeid="211175">上一节课我们学习了 Scrapy 和 Scrapyd 的用法，虽然它们可以解决项目部署的一些问题，但其实这种方案并没有真正彻底解决环境配置的问题。</p>
<p data-nodeid="211176">比如使用 Scrapyd 时我们依然需要安装对应的依赖库，即使这样仍免不了还是会出现环境冲突和不一致的问题。因此，本节课我会再介绍另一种部署方案 —— Docker。</p>
<p data-nodeid="211177">Docker 可以提供操作系统级别的虚拟环境，一个 Docker 镜像一般都会包含一个完整的操作系统，而这些系统内也有已经配置好的开发环境，如 Python 3.6 环境等。</p>
<p data-nodeid="211178">我们可以直接使用此 Docker 的 Python 3 镜像运行一个容器，将项目直接放到容器里运行，就不用再额外配置 Python 3 环境了，这样就解决了环境配置的问题。</p>
<p data-nodeid="211179">我们也可以进一步将 Scrapy 项目制作成一个新的 Docker 镜像，镜像里只包含适用于本项目的 Python 环境。如果要部署到其他平台，只需要下载该镜像并运行就好了，因为 Docker 运行时采用虚拟环境，和宿主机是完全隔离的，所以也不需要担心环境冲突问题。</p>
<p data-nodeid="211180">如果我们能够把 Scrapy 项目制作成一个 Docker 镜像，只要其他主机安装了 Docker，那么只要将镜像下载并运行即可，而不必再担心环境配置问题或版本冲突问题。</p>
<p data-nodeid="211181">因此，利用 Docker 我们就能很好地解决环境配置、环境冲突等问题。接下来，我们就尝试把一个 Scrapy 项目制作成一个 Docker 镜像。</p>
<h3 data-nodeid="212405" class="">本节目标</h3>


<p data-nodeid="211183">我们要实现把前文 Scrapy 的入门项目打包成一个 Docker 镜像的过程。项目爬取的网址为： <a href="http://quotes.toscrape.com/" data-nodeid="211263">http://quotes.toscrape.com/</a>，本模块 Scrapy 入门一节已经实现了 Scrapy 对此站点的爬取过程，项目代码为： <a href="https://github.com/Python3WebSpider/ScrapyTutorial" data-nodeid="211267">https://github.com/Python3WebSpider/ScrapyTutorial</a>，如果本地不存在的话可以 Clone 下来。</p>
<h3 data-nodeid="212755" class="">准备工作</h3>

<p data-nodeid="211185">请确保已经安装好 Docker 并可以正常运行，如果没有安装可以参考 <a href="https://cuiqingcai.com/5438.html" data-nodeid="211273">https://cuiqingcai.com/5438.html</a>。</p>
<h3 data-nodeid="213105" class="">创建 Dockerfile</h3>

<p data-nodeid="211187">首先在项目的根目录下新建一个 requirements.txt 文件，将整个项目依赖的 Python 环境包都列出来，如下所示：</p>
<pre class="lang-java" data-nodeid="214502"><code data-language="java">scrapy 
pymongo 
</code></pre>




<p data-nodeid="211189">如果库需要特定的版本，我们还可以指定版本号，如下所示：</p>
<pre class="lang-java" data-nodeid="214851"><code data-language="java">scrapy&gt;=<span class="hljs-number">1.4</span>.<span class="hljs-number">0</span> 
pymongo&gt;=<span class="hljs-number">3.4</span>.<span class="hljs-number">0</span> 
</code></pre>

<p data-nodeid="211191">在项目根目录下新建一个 Dockerfile 文件，文件不加任何后缀名，修改内容如下所示：</p>
<pre class="lang-java" data-nodeid="215200"><code data-language="java">FROM python:<span class="hljs-number">3.7</span> 
ENV PATH /usr/local/bin:$PATH 
ADD . /code 
WORKDIR /code 
RUN pip3 install -r requirements.txt 
CMD scrapy crawl quotes 
</code></pre>

<p data-nodeid="215549">第一行的 FROM 代表使用的 Docker 基础镜像，在这里我们直接使用 python:3.7 的镜像，在此基础上运行 Scrapy 项目。</p>
<p data-nodeid="215550">第二行 ENV 是环境变量设置，将 /usr/local/bin:$PATH 赋值给 PATH，即增加 /usr/local/bin 这个环境的变量路径。</p>

<p data-nodeid="211194">第三行 ADD 是将本地的代码放置到虚拟容器中。它有两个参数：第一个参数是“.”，代表本地当前路径；第二个参数是 /code，代表虚拟容器中的路径，也就是将本地项目所有内容放置到虚拟容器的 /code 目录下，以便于在虚拟容器中运行代码。</p>
<p data-nodeid="211195">第四行 WORKDIR 是指定工作目录，这里将刚才添加的代码路径设置成工作路径。在这个路径下的目录结构和当前本地目录结构是相同的，所以我们可以直接执行库安装命令、爬虫运行命令等。</p>
<p data-nodeid="211196">第五行 RUN 是执行某些命令来做一些环境准备工作。由于 Docker 虚拟容器内只有 Python 3 环境，而没有所需要的 Python 库，所以我们运行此命令在虚拟容器中安装相应的 Python 库如 Scrapy，这样就可以在虚拟容器中执行 Scrapy 命令了。</p>
<p data-nodeid="211197">第六行 CMD 是容器启动命令。在容器运行时，此命令会被执行。在这里我们直接用 scrapy crawl quotes 来启动爬虫。</p>
<h3 data-nodeid="215901" class="">修改 MongoDB 连接</h3>

<p data-nodeid="211199">接下来我们需要修改 MongoDB 的连接信息。如果我们继续用 localhost 是无法找到 MongoDB 的，因为在 Docker 虚拟容器里 localhost 实际指向容器本身的运行 IP，而容器内部并没有安装 MongoDB，所以爬虫无法连接 MongoDB。</p>
<p data-nodeid="211200">这里的 MongoDB 地址可以有如下两种选择。</p>
<ul data-nodeid="211201">
<li data-nodeid="211202">
<p data-nodeid="211203">如果只想在本机测试，我们可以将地址修改为宿主机的 IP，也就是容器外部的本机 IP，一般是一个局域网 IP，使用 ifconfig 命令即可查看。</p>
</li>
<li data-nodeid="211204">
<p data-nodeid="211205">如果要部署到远程主机运行，一般 MongoDB 都是可公网访问的地址，修改为此地址即可。</p>
</li>
</ul>
<p data-nodeid="211206">但为了保证灵活性，我们可以将这个连接字符串通过环境变量传递进来，比如修改为：</p>
<pre class="lang-java" data-nodeid="216251"><code data-language="java"><span class="hljs-keyword">import</span> os 
​ 
MONGO_URI = os.getenv(<span class="hljs-string">'MONGO_URI'</span>) 
MONGO_DB = os.getenv(<span class="hljs-string">'MONGO_DB'</span>, <span class="hljs-string">'tutorial'</span>) 
</code></pre>

<p data-nodeid="211208">这样项目的配置就完成了。</p>
<h3 data-nodeid="216950" class="">构建镜像</h3>


<p data-nodeid="211210">接下来，我们便可以构建镜像了，执行如下命令：</p>
<pre class="lang-java" data-nodeid="217300"><code data-language="java">docker build -t quotes:latest . 
</code></pre>

<p data-nodeid="211212">这样输出就说明镜像构建成功。这时我们查看一下构建的镜像，如下所示：</p>
<pre class="lang-java" data-nodeid="217649"><code data-language="java">Sending build context to Docker daemon <span class="hljs-number">191.5</span> kB 
Step <span class="hljs-number">1</span>/<span class="hljs-number">6</span> : FROM python:<span class="hljs-number">3.7</span> 
 ---&gt; <span class="hljs-number">968120d</span>8cbe8 
Step <span class="hljs-number">2</span>/<span class="hljs-number">6</span> : ENV PATH /usr/local/bin:$PATH 
 ---&gt; Using cache 
 ---&gt; <span class="hljs-number">387</span>abbba1189 
Step <span class="hljs-number">3</span>/<span class="hljs-number">6</span> : ADD . /code 
 ---&gt; a844ee0db9c6 
Removing intermediate container <span class="hljs-number">4d</span>c41779c573 
Step <span class="hljs-number">4</span>/<span class="hljs-number">6</span> : WORKDIR /code 
 ---&gt; <span class="hljs-number">619</span>b2c064ae9 
Removing intermediate container bcd7cd7f7337 
Step <span class="hljs-number">5</span>/<span class="hljs-number">6</span> : RUN pip3 install -r requirements.txt 
 ---&gt; Running in <span class="hljs-number">9452</span>c83a12c5 
... 
Removing intermediate container <span class="hljs-number">9452</span>c83a12c5 
Step <span class="hljs-number">6</span>/<span class="hljs-number">6</span> : CMD scrapy crawl quotes 
 ---&gt; Running in c092b5557ab8 
 ---&gt; c8101aca6e2a 
Removing intermediate container c092b5557ab8 
Successfully built c8101aca6e2a 
</code></pre>

<p data-nodeid="211214">出现类似输出就证明镜像构建成功了，这时执行，比如我们查看一下构建的镜像：</p>
<pre class="lang-java" data-nodeid="217998"><code data-language="java">docker images 
</code></pre>

<p data-nodeid="211216">返回结果中其中有一行就是：</p>
<pre class="lang-java" data-nodeid="218347"><code data-language="java">quotes  latest  <span class="hljs-number">41</span>c8499ce210 &nbsp;  <span class="hljs-number">2</span> minutes ago &nbsp; <span class="hljs-number">769</span> MB 
</code></pre>

<p data-nodeid="211218">这就是我们新构建的镜像。</p>
<h3 data-nodeid="219046" class="">运行</h3>


<p data-nodeid="211220">我们可以先在本地测试运行，这时候我们需要指定 MongoDB 的连接信息，比如我在宿主机上启动了一个 MongoDB，找到当前宿主机的 IP 为 192.168.3.47，那么这里我就可以指定 MONGO_URI 并启动 Docker 镜像：</p>
<pre class="lang-java" data-nodeid="219396"><code data-language="java">docker run -e MONGO_URI=<span class="hljs-number">192.168</span>.<span class="hljs-number">3.47</span> quotes 
</code></pre>

<p data-nodeid="220094">当然我们还可以指定 MONGO_URI 为远程 MongoDB 连接字符串。</p>
<p data-nodeid="220095">另外我们也可以利用 Docker-Compose 来启动，与此同时顺便也可以使用 Docker 来新建一个 MongoDB。可以在项目目录下新建 docker-compose.yaml 文件，如下所示：</p>

<pre class="lang-java" data-nodeid="219745"><code data-language="java">version: <span class="hljs-string">'3'</span> 
services: 
  crawler: 
 &nbsp;  build: . 
 &nbsp;  image: quotes 
 &nbsp;  depends_on: 
 &nbsp; &nbsp;  - mongo 
 &nbsp;  environment: 
 &nbsp; &nbsp;  MONGO_URI: mongo:<span class="hljs-number">7017</span> 
  mongo: 
 &nbsp;  image: mongo 
 &nbsp;  ports: 
 &nbsp; &nbsp;  - <span class="hljs-number">7017</span>:<span class="hljs-number">27017</span> 
</code></pre>

<p data-nodeid="220630">这里我们使用 Docker-Compose 配置了两个容器，二者需要配合启动。</p>
<p data-nodeid="220631">首先是 crawler 这个容器，其 build 路径是当前路径，image 代表 build 生成的镜像名称，这里取名为 quotes，depends_on 代表容器的启动依赖顺序，这里依赖于 mongo 这个容器，environment 这里就是指定容器运行时的环境变量，这里指定为 <code data-backticks="1" data-nodeid="220636">mongo:7017</code> 。</p>



<p data-nodeid="211225">另外一个容器就是刚才的 crawler 这个容器所依赖的 MongoDB 数据库了，在这里我们直接指定了镜像为 mongo，运行端口配置为 <code data-backticks="1" data-nodeid="211316">7017:27017</code> ，这代表容器内的 MongoDB 运行在 27017 端口上，这个端口会映射到宿主机的 7017 端口上，所以我们在宿主机通过 7017 端口就能连接到这个 MongoDB 数据库。</p>
<p data-nodeid="211226">好，这时候我们运行一下：</p>
<pre class="lang-java" data-nodeid="220986"><code data-language="java">docker-compose up 
</code></pre>

<p data-nodeid="211228">然后 Docker 会构建镜像并运行，运行结果如下：</p>
<pre class="lang-java" data-nodeid="221335"><code data-language="java">Starting scrapytutorial_mongo_1 ... done 
Recreating scrapytutorial_crawler_1 ... done 
Attaching to scrapytutorial_mongo_1, scrapytutorial_crawler_1 
mongo_1 &nbsp;  | {<span class="hljs-string">"t"</span>:{<span class="hljs-string">"$date"</span>:<span class="hljs-string">"2020-08-06T16:18:05.310+00:00"</span>},<span class="hljs-string">"s"</span>:<span class="hljs-string">"I"</span>,  <span class="hljs-string">"c"</span>:<span class="hljs-string">"CONTROL"</span>,  <span class="hljs-string">"id"</span>:<span class="hljs-number">23285</span>, &nbsp; <span class="hljs-string">"ctx"</span>:<span class="hljs-string">"main"</span>,<span class="hljs-string">"msg"</span>:<span class="hljs-string">"Automatically disabling TLS 1.0, to force-enable TLS 1.0 specify --sslDisabledProtocols 'none'"</span>} 
mongo_1 &nbsp;  | {<span class="hljs-string">"t"</span>:{<span class="hljs-string">"$date"</span>:<span class="hljs-string">"2020-08-06T16:18:05.312+00:00"</span>},<span class="hljs-string">"s"</span>:<span class="hljs-string">"W"</span>,  <span class="hljs-string">"c"</span>:<span class="hljs-string">"ASIO"</span>, &nbsp; &nbsp; <span class="hljs-string">"id"</span>:<span class="hljs-number">22601</span>, &nbsp; <span class="hljs-string">"ctx"</span>:<span class="hljs-string">"main"</span>,<span class="hljs-string">"msg"</span>:<span class="hljs-string">"No TransportLayer configured during NetworkInterface startup"</span>} 
mongo_1 &nbsp;  | {<span class="hljs-string">"t"</span>:{<span class="hljs-string">"$date"</span>:<span class="hljs-string">"2020-08-06T16:18:05.312+00:00"</span>},<span class="hljs-string">"s"</span>:<span class="hljs-string">"I"</span>,  <span class="hljs-string">"c"</span>:<span class="hljs-string">"NETWORK"</span>,  <span class="hljs-string">"id"</span>:<span class="hljs-number">4648601</span>, <span class="hljs-string">"ctx"</span>:<span class="hljs-string">"main"</span>,<span class="hljs-string">"msg"</span>:<span class="hljs-string">"Implicit TCP FastOpen unavailable. If TCP FastOpen is required, set tcpFastOpenServer, tcpFastOpenClient, and tcpFastOpenQueueSize."</span>} 
... 
crawler_1  | <span class="hljs-number">2020</span>-<span class="hljs-number">08</span>-<span class="hljs-number">06</span> <span class="hljs-number">16</span>:<span class="hljs-number">18</span>:<span class="hljs-number">06</span> [scrapy.utils.log] INFO: Scrapy <span class="hljs-number">2.3</span>.<span class="hljs-number">0</span> started (bot: tutorial) 
crawler_1  | <span class="hljs-number">2020</span>-<span class="hljs-number">08</span>-<span class="hljs-number">06</span> <span class="hljs-number">16</span>:<span class="hljs-number">18</span>:<span class="hljs-number">06</span> [scrapy.utils.log] INFO: Versions: lxml <span class="hljs-number">4.5</span>.<span class="hljs-number">2.0</span>, libxml2 <span class="hljs-number">2.9</span>.<span class="hljs-number">10</span>, cssselect <span class="hljs-number">1.1</span>.<span class="hljs-number">0</span>, parsel <span class="hljs-number">1.6</span>.<span class="hljs-number">0</span>, w3lib <span class="hljs-number">1.22</span>.<span class="hljs-number">0</span>, Twisted <span class="hljs-number">20.3</span>.<span class="hljs-number">0</span>, Python <span class="hljs-number">3.7</span>.<span class="hljs-number">8</span> (<span class="hljs-keyword">default</span>, Jun <span class="hljs-number">30</span> <span class="hljs-number">2020</span>, <span class="hljs-number">18</span>:<span class="hljs-number">27</span>:<span class="hljs-number">23</span>) - [GCC <span class="hljs-number">8.3</span>.<span class="hljs-number">0</span>], pyOpenSSL <span class="hljs-number">19.1</span>.<span class="hljs-number">0</span> (OpenSSL <span class="hljs-number">1.1</span>.<span class="hljs-number">1</span>g  <span class="hljs-number">21</span> Apr <span class="hljs-number">2020</span>), cryptography <span class="hljs-number">3.0</span>, Platform Linux-<span class="hljs-number">4.19</span>.<span class="hljs-number">76</span>-linuxkit-x86_64-with-debian-<span class="hljs-number">10.4</span> 
crawler_1  | <span class="hljs-number">2020</span>-<span class="hljs-number">08</span>-<span class="hljs-number">06</span> <span class="hljs-number">16</span>:<span class="hljs-number">18</span>:<span class="hljs-number">06</span> [scrapy.utils.log] DEBUG: Using reactor: twisted.internet.epollreactor.EPollReactor 
crawler_1  | <span class="hljs-number">2020</span>-<span class="hljs-number">08</span>-<span class="hljs-number">06</span> <span class="hljs-number">16</span>:<span class="hljs-number">18</span>:<span class="hljs-number">06</span> [scrapy.crawler] INFO: Overridden settings: 
crawler_1  | {<span class="hljs-string">'BOT_NAME'</span>: <span class="hljs-string">'tutorial'</span>, 
</code></pre>

<p data-nodeid="222026">这时候就发现爬虫已经正常运行了，同时我们在宿主机上连接 localhost:7017 这个 MongoDB 服务就能看到爬取的结果了：</p>
<p data-nodeid="222027" class=""><img src="https://s0.lgstatic.com/i/image/M00/3E/C2/CgqCHl8tIB6AJUkZAAMRIUmdoXQ641.png" alt="Drawing 0.png" data-nodeid="222031"></p>


<p data-nodeid="211231">这就是用 Docker-Compose 启动的方式，其启动更加便捷，参数可以配置到 Docker-Compose 文件中。</p>
<h3 data-nodeid="222380" class="">推送至 Docker Hub</h3>

<p data-nodeid="211233">构建完成之后，我们可以将镜像 Push 到 Docker 镜像托管平台，如 Docker Hub 或者私有的 Docker Registry 等，这样我们就可以从远程服务器下拉镜像并运行了。</p>
<p data-nodeid="211234">以 Docker Hub 为例，如果项目包含一些私有的连接信息（如数据库），我们最好将 Repository 设为私有或者直接放到私有的 Docker Registry 中。</p>
<p data-nodeid="211235">首先在 <a href="https://hub.docker.com" data-nodeid="211332">https://hub.docker.com</a>注册一个账号，新建一个 Repository，名为 quotes。比如，我的用户名为 germey，新建的 Repository 名为 quotes，那么此 Repository 的地址就可以用 germey/quotes 来表示，当然你也可以自行修改。</p>
<p data-nodeid="211236">为新建的镜像打一个标签，命令如下所示：</p>
<pre class="lang-java" data-nodeid="222730"><code data-language="java">docker tag quotes:latest germey/quotes:latest 
</code></pre>

<p data-nodeid="211238">推送镜像到 Docker Hub 即可，命令如下所示：</p>
<pre class="lang-java" data-nodeid="223079"><code data-language="java">docker push germey/quotes 
</code></pre>

<p data-nodeid="223770">Docker Hub 中便会出现新推送的 Docker 镜像了，如图所示。</p>
<p data-nodeid="223771" class=""><img src="https://s0.lgstatic.com/i/image/M00/3E/C2/CgqCHl8tIC6AZRf7AAHlmVEMa3U133.png" alt="Drawing 1.png" data-nodeid="223775"></p>


<p data-nodeid="211241">如果我们想在其他的主机上运行这个镜像，在主机上装好 Docker 后，可以直接执行如下命令：</p>
<pre class="lang-java" data-nodeid="224124"><code data-language="java">docker run germey/quotes 
</code></pre>

<p data-nodeid="224822">这样就会自动下载镜像，然后启动容器运行，不需要配置 Python 环境，不需要关心版本冲突问题。</p>
<p data-nodeid="224823">当然我们也可以使用 Docker-Compose 来构建镜像和推送镜像，这里我们只需要修改 docker-compose.yaml 文件即可：</p>

<pre class="lang-java" data-nodeid="224473"><code data-language="java">version: <span class="hljs-string">'3'</span> 
services: 
  crawler: 
 &nbsp;  build: . 
 &nbsp;  image: germey/quotes 
 &nbsp;  depends_on: 
 &nbsp; &nbsp;  - mongo 
 &nbsp;  environment: 
 &nbsp; &nbsp;  MONGO_URI: mongo:<span class="hljs-number">7017</span> 
  mongo: 
 &nbsp;  image: mongo 
 &nbsp;  ports: 
 &nbsp; &nbsp;  - <span class="hljs-number">7017</span>:<span class="hljs-number">27017</span> 
</code></pre>

<p data-nodeid="211245">可以看到，这里我们就将 crawler 的 image 内容修改为了 <code data-backticks="1" data-nodeid="211346">germey/quotes</code> ，接下来执行：</p>
<pre class="lang-java" data-nodeid="225174"><code data-language="java">docker-compose build 
docker-compose push 
</code></pre>

<p data-nodeid="211247">就可以把镜像推送到 Docker Hub 了。</p>
<h3 data-nodeid="225523" class="">结语</h3>

<p data-nodeid="211249">本课时，我们讲解了将 Scrapy 项目制作成 Docker 镜像并部署到远程服务器运行的过程。使用此种方式，我们在本节课开始时所列出的问题都可以迎刃而解了。</p>

# Scrapy对接Kubernetes并实现定时爬取
<p data-nodeid="719">在上一节我们了解了如何制作一个 Scrapy 的 Docker 镜像，本节课我们来介绍下如何将镜像部署到 Kubernetes 上。</p>



<h3 data-nodeid="917" class="">Kubernetes</h3>

<p data-nodeid="5">Kubernetes 是谷歌开发的，用于自动部署，扩展和管理容器化应用程序的开源系统，其稳定性高、扩展性好，功能十分强大。现在业界已经越来越多地采用 Kubernetes 来部署和管理各个项目，</p>
<p data-nodeid="6">如果你还不了解 Kubernetes，可以参考其官方文档来学习一下： <a href="https://kubernetes.io/" data-nodeid="49">https://kubernetes.io/</a>。</p>
<h3 data-nodeid="1115" class="">准备工作</h3>

<p data-nodeid="8">如果我们需要将上一节的镜像部署到 Kubernetes 上，则首先需要我们有一个 Kubernetes 集群，同时需要能使用 kubectl 命令。</p>
<p data-nodeid="9">Kubernetes 集群可以自行配置，也可以使用各种云服务提供的集群，如阿里云、腾讯云、Azure 等，另外还可以使用 Minikube 等来快速搭建，当然也可以使用 Docker 本身提供的 Kubernetes 服务。</p>
<p data-nodeid="1501">比如我这里就直接使用了 Docker Desktop 提供的 Kubernetes 服务，勾选 Enable 直接开启即可。</p>
<p data-nodeid="1502" class=""><img src="https://s0.lgstatic.com/i/image/M00/40/A7/Ciqc1F8zZ0eAS3RkAADg5KBv-5U920.png" alt="image (13).png" data-nodeid="1510"></p>


<p data-nodeid="1715" class="">kubectl 是用来操作 Kubernetes 的命令行工具，可以参考 <a href="https://kubernetes.io/zh/docs/tasks/tools/install-kubectl/" data-nodeid="1719">https://kubernetes.io/zh/docs/tasks/tools/install-kubectl/</a> 来安装。</p>

<p data-nodeid="13">如果以上都安装好了，可以运行下 kubectl 命令验证下能否正常获取节点信息：</p>
<pre class="lang-java" data-nodeid="2540"><code data-language="java">kubectl get nodes 
</code></pre>




<p data-nodeid="15">运行结果类似如下：</p>
<pre class="lang-java" data-nodeid="2745"><code data-language="java">NAME &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; STATUS &nbsp; ROLES &nbsp;  AGE &nbsp; VERSION 
docker-desktop &nbsp; Ready &nbsp;  master &nbsp; <span class="hljs-number">75d</span> &nbsp; v1.<span class="hljs-number">16.6</span>-beta.<span class="hljs-number">0</span> 
</code></pre>

<h3 data-nodeid="2950" class="">部署</h3>

<p data-nodeid="18">要部署的话我们需要先创建一个命名空间 Namespace，这里直接使用 kubectl 命令创建即可，Namespace 的名称这里我们取名为 crawler。</p>
<p data-nodeid="19">创建命令如下：</p>
<pre class="lang-java" data-nodeid="3156"><code data-language="java">kubectl create namespace crawler 
</code></pre>

<p data-nodeid="21">运行结果如下：</p>
<pre class="lang-java" data-nodeid="3361"><code data-language="java">namespace/crawler created 
</code></pre>

<p data-nodeid="23">如果出现上述结果就说明命名空间创建成功了。接下来我们就需要把 Docker 镜像部署到这个 Namespace 下面了。<br>
Kubernetes 里面的资源是用 yaml 文件来定义的，如果要部署一次性任务或者为我们提供服务可以使用 Deployment，更多详情可以参考 Kubernetes 对于 Deployment 的说明： <a href="https://kubernetes.io/docs/concepts/workloads/controllers/deployment/" data-nodeid="74">https://kubernetes.io/docs/concepts/workloads/controllers/deployment/</a>。</p>
<p data-nodeid="24">新建 deployment.yaml 文件如下：</p>
<pre class="lang-yaml" data-nodeid="6641"><code data-language="yaml"><span class="hljs-attr">apiVersion:</span> <span class="hljs-string">apps/v1</span> 
<span class="hljs-attr">kind:</span> <span class="hljs-string">Deployment</span> 
<span class="hljs-attr">metadata:</span> 
  <span class="hljs-attr">name:</span> <span class="hljs-string">crawler-quotes</span> 
  <span class="hljs-attr">namespace:</span> <span class="hljs-string">crawler</span> 
  <span class="hljs-attr">labels:</span> 
 &nbsp;  <span class="hljs-attr">app:</span> <span class="hljs-string">crawler-quotes</span> 
<span class="hljs-attr">spec:</span> 
  <span class="hljs-attr">replicas:</span> <span class="hljs-number">1</span> 
  <span class="hljs-attr">selector:</span> 
 &nbsp;  <span class="hljs-attr">matchLabels:</span> 
 &nbsp; &nbsp;  <span class="hljs-attr">app:</span> <span class="hljs-string">crawler-quotes</span> 
  <span class="hljs-attr">template:</span> 
 &nbsp;  <span class="hljs-attr">metadata:</span> 
 &nbsp; &nbsp;  <span class="hljs-attr">labels:</span> 
 &nbsp; &nbsp; &nbsp;  <span class="hljs-attr">app:</span> <span class="hljs-string">crawler-quotes</span> 
 &nbsp;  <span class="hljs-attr">spec:</span> 
 &nbsp; &nbsp;  <span class="hljs-attr">containers:</span> 
 &nbsp; &nbsp; &nbsp;  <span class="hljs-bullet">-</span> <span class="hljs-attr">name:</span> <span class="hljs-string">crawler-quotes</span> 
 &nbsp; &nbsp; &nbsp; &nbsp;  <span class="hljs-attr">image:</span> <span class="hljs-string">germey/quotes</span> 
 &nbsp; &nbsp; &nbsp; &nbsp;  <span class="hljs-attr">env:</span> 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  <span class="hljs-bullet">-</span> <span class="hljs-attr">name:</span> <span class="hljs-string">MONGO_URI</span> 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  <span class="hljs-attr">value:</span> <span class="hljs-string">&lt;mongo&gt;</span> 
</code></pre>
















<p data-nodeid="7159">这里我们就可以按照 Deployment 的规范声明一个 yaml 文件了，指定 namespace 为 crawler，并指定 container 的 image 为我们已经 Push 到 Docker Hub 的镜像 germey/quotes，另外通过 env 指定了环境变量，注意这里需要将 <code data-backticks="1" data-nodeid="7162">&lt;mongo&gt;</code> 替换成一个有效的 MongoDB 连接字符串，如一个远程 MongoDB 服务。</p>
<p data-nodeid="7160">接下来我们只需要使用 kubectl 命令即可应用该部署：</p>



<pre class="lang-java" data-nodeid="6846"><code data-language="java">kubectl apply -f deployment.yaml 
</code></pre>

<p data-nodeid="28">运行完毕之后会提示类似如下结果：</p>
<pre class="lang-java" data-nodeid="7369"><code data-language="java">deployment.apps/crawler-quotes created 
</code></pre>

<p data-nodeid="7574">这样就说明部署成功了。如果 MongoDB 服务能够正常连接的话，这个爬虫就会运行并将结果存储到 MongoDB 中。</p>
<p data-nodeid="7575">另外我们还可以通过命令行或者 Kubernetes 的 Dashboard 查看部署任务的运行状态。</p>

<p data-nodeid="31">如果我们想爬虫定时运行的话，可以借助于 Kubernetes 提供的 cronjob 来将爬虫配置为定时任务，其运行模式就类似于 crontab 命令一样，详细用法可以参考： <a href="https://kubernetes.io/zh/docs/tasks/job/automated-tasks-with-cron-jobs/" data-nodeid="89">https://kubernetes.io/zh/docs/tasks/job/automated-tasks-with-cron-jobs/</a>。</p>
<p data-nodeid="32">可以新建 cronjob.yaml，内容如下：</p>
<pre class="lang-java" data-nodeid="7782"><code data-language="java">apiVersion: batch/v1beta1 
kind: CronJob 
metadata: 
  name: crawler-quotes 
  namespace: crawler 
spec: 
  schedule: <span class="hljs-string">"0 */1 * * *"</span> 
  jobTemplate: 
 &nbsp;  spec: 
 &nbsp; &nbsp;  template: 
 &nbsp; &nbsp; &nbsp;  spec: 
 &nbsp; &nbsp; &nbsp; &nbsp;  restartPolicy: OnFailure 
 &nbsp; &nbsp; &nbsp; &nbsp;  containers: 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  - name: crawler-quotes 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  image: germey/quotes 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  env: 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  - name: MONGO_URI 
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  value: &lt;mongo&gt; 
</code></pre>

<p data-nodeid="34">注意到这里 kind 我们不再使用 Deployment，而是改成了 CronJob，代表定时任务。 <code data-backticks="1" data-nodeid="93">spec.schedule</code> 里面定义了 crontab 格式的定时任务配置，这里代表每小时运行一次。其他的配置基本一致，同样注意这里需要将 <code data-backticks="1" data-nodeid="95">&lt;mongo&gt;</code> 替换成一个有效的 MongoDB 连接字符串，如一个远程 MongoDB 服务。<br>
接下来我们只需要使用 kubectl 命令即可应用该部署：</p>
<pre class="lang-java" data-nodeid="7987"><code data-language="java">kubectl apply -f cronjob.yaml 
</code></pre>

<p data-nodeid="36">运行完毕之后会提示类似如下结果：</p>
<pre class="lang-java" data-nodeid="8192"><code data-language="java">cronjob.batch/crawler-quotes created 
</code></pre>

<p data-nodeid="38">出现这样的结果这就说明部署成功了，这样这个爬虫就会每小时运行一次，并将数据存储到指定的 MongoDB 数据库中。</p>
<h3 data-nodeid="513" class="">总结</h3>


<p data-nodeid="40">以上我们就简单介绍了下 Kubernetes 部署爬虫的基本操作，Kubernetes 非常复杂，需要学习的内容很多，我们这一节介绍的只是冰山一角，还有更多的内容等待你去探索。</p>

# 从爬虫小白到高手的必经之路
<p data-nodeid="121522">如果你看到了本课时，那么恭喜你已经学完了本专栏课程的所有内容，爬虫的知识点很复杂，一路学过来相信你也经历了不少坎坷。</p>



<p data-nodeid="120853">本节课我们对网络爬虫所要学习的内容做一次总结，这里面也是我个人认为爬虫工程师应该具备的一些技术栈，由于专栏篇幅有限，肯定不可能把所有的知识点都覆盖到，但基础知识都已经涵盖了，下面我会把网络爬虫的知识点进行总结和梳理，如果你想深入学习网络爬虫的话可以参考。</p>
<p data-nodeid="120854" class="">网络爬虫的学习关系到计算机网络、编程基础、前端开发、后端开发、App 开发与逆向、网络安全、数据库、运维、机器学习、数据分析等各个方向的内容，它像一张大网一样把现在一些主流的技术栈都连接在了一起。正因为涵盖的方向多，因此学习的东西也非常零散和杂乱。</p>
<h3 data-nodeid="121964" class="">初学爬虫</h3>

<p data-nodeid="120856">一些最基本的网站，往往不带任何反爬措施。比如某个博客站点，我们要爬全站的话就顺着列表页爬到文章页，再把文章的时间、作者、正文等信息爬下来就可以了。</p>
<p data-nodeid="120857" class="">那代码怎么写呢？用 Python 的 requests 等库就够了，写一个基本的逻辑，顺带把一篇篇文章的源码获取下来，解析的话用 XPath、BeautifulSoup、PyQuery 或者正则表达式，或者粗暴的字符串匹配把想要的内容抠出来，再加个文本写入存下来就可以了。</p>
<p data-nodeid="120858" class="">代码也很简单，就只是几个方法的调用。逻辑也很简单，几个循环加存储。最后就能看到一篇篇文章被我们存到了自己的电脑里。当然如果你不太会写代码或者都懒得写，那么利用基本的可视化爬取工具，如某爪鱼、某裔采集器也能通过可视化点选的方式把数据爬下来。</p>
<p data-nodeid="120859">如果存储方面稍微扩展一下的话，可以对接上 MySQL、MongoDB、Elasticsearch、Kafka 等来保存数据，实现持久化存储。以后查询或者操作会更方便。</p>
<p data-nodeid="120860">反正，不管效率如何，一个完全没有反爬的网站用最基本的方式就可以搞定。到这里，你可以说自己会爬虫了吗？不，还差得远呢。</p>
<h3 data-nodeid="122406" class="">Ajax、动态渲染</h3>

<p data-nodeid="120862">随着互联网的发展，前端技术也在不断变化，数据的加载方式也不再是单纯的服务端渲染了。现在你可以看到很多网站的数据可能都是通过接口的形式传输的，或者即使不是接口那也是一些 JSON 数据，然后经过 JavaScript 渲染得出来的。</p>
<p data-nodeid="120863">这时候，你要再用 requests 来爬取就不行了，因为 requests 爬下来的源码是服务端渲染得到的，浏览器看到页面的和 requests 获取的结果是不一样的。真正的数据是经过 JavaScript 执行得出来的，数据来源可能是 Ajax，也可能是页面里的某些 Data，也可能是一些 ifame 页面等，不过大多数情况下可能是 Ajax 接口获取的。</p>
<p data-nodeid="120864">所以很多情况下需要分析 Ajax，知道这些接口的调用方式之后再用程序来模拟。但是有些接口带着加密参数，比如 token、sign 等，又不好模拟，怎么办呢？</p>
<p data-nodeid="120865">一种方法就是去分析网站的 JavaScript 逻辑，死抠里面的代码，研究这些参数是怎么构造的，找出思路之后再用爬虫模拟或重写就行了。如果你解析出来了，那么直接模拟的方式效率会高很多，这里面就需要一些 JavaScript 基础了，当然有些网站加密逻辑做得太厉害了，你可能花一个星期也解析不出来，最后只能放弃了。</p>
<p data-nodeid="120866">那这样解不出来或者不想解了，该怎么办呢？这时候可以用一种简单粗暴的方法，也就是直接用模拟浏览器的方式来爬取，比如用 Puppeteer、Pyppeteer、Selenium、Splash 等，这样爬取到的源代码就是真正的网页代码，数据自然就好提取了，同时也就绕过分析 Ajax 和一些 JavaScript 逻辑的过程。这种方式就做到了可见即可爬，难度也不大，同时模拟了浏览器，也不太会有一些法律方面的问题。</p>
<p data-nodeid="120867">但其实后面的这种方法也会遇到各种反爬的情况，现在很多网站都会去识别 webdriver，看到你是用的 Selenium 等工具，直接拒接或不返回数据，所以你碰到这种网站还得专门来解决这个问题。</p>
<h3 data-nodeid="122848" class="">多进程、多线程、协程</h3>

<p data-nodeid="120869">上面的情况如果用单线程的爬虫来模拟是比较简单的，但是有个问题就是速度慢啊。</p>
<p data-nodeid="120870">爬虫是 I/O 密集型的任务，所以可能大多数情况下都在等待网络的响应，如果网络响应速度慢，那就得一直等着。但这个空余的时间其实可以让 CPU 去做更多事情。那怎么办呢？多开一些线程吧。</p>
<p data-nodeid="120871">所以这个时候我们就可以在某些场景下加上多进程、多线程，虽然说多线程有 GIL 锁，但对于爬虫来说其实影响没那么大，所以用上多进程、多线程都可以成倍地提高爬取速度，对应的库就有 threading、multiprocessing 等。</p>
<p data-nodeid="120872">异步协程就更厉害了，用 aiohttp、gevent、tornado 等工具，基本上想开多少并发就开多少并发，但是还是得谨慎一些，别把目标网站搞挂了。</p>
<p data-nodeid="120873">总之，用上这几个工具，爬取速度就提上来了。但速度提上来了不一定都是好事，反爬措施接着肯定就要来了，封你 IP、封你账号、弹验证码、返回假数据，所以有时候龟速爬似乎也是个解决办法？</p>
<h3 data-nodeid="123290" class="">分布式</h3>

<p data-nodeid="120875">多线程、多进程、协程都能加速，但终究还是单机的爬虫。要真正做到规模化，还得靠分布式爬虫来搞定。</p>
<p data-nodeid="120876">分布式的核心是什么？资源共享。比如爬取队列共享、去重指纹共享，等等。</p>
<p data-nodeid="120877">我们可以使用一些基础的队列或组件来实现分布式，比如 RabbitMQ、Celery、Kafka、Redis 等，但经过很多人的尝试，自己去实现一个分布式爬虫，性能和扩展性总会出现一些问题。不少企业内部其实也有自己开发的一套分布式爬虫，和业务更紧密，这种当然是最好了。</p>
<p data-nodeid="120878">现在主流的 Python 分布式爬虫还是基于 Scrapy 的，对接 Scrapy-Redis、Scrapy-Redis-BloomFilter 或者使用 Scrapy-Cluster 等，它们都是基于 Redis 来共享爬取队列的，多多少少总会遇到一些内存的问题。所以一些人也考虑对接到其他的消息队列上面，比如 RabbitMQ、Kafka 等，可以解决一些问题，效率也不差。</p>
<p data-nodeid="120879">总之，要提高爬取效率，分布式还是必须要掌握的。</p>
<h3 data-nodeid="123732" class="">验证码</h3>

<p data-nodeid="120881">爬虫时难免遇到反爬，验证码就是其中之一。要会反爬，那首先就要会解验证码。</p>
<p data-nodeid="120882">现在你可以看到很多网站都会有各种各样的验证码，比如最简单的图形验证码，要是验证码的文字规则的话，OCR 检测或基本的模型库都能识别，你可以直接对接一个打码平台来解决，准确率还是可以的。</p>
<p data-nodeid="120883">然而现在你可能都见不到什么图形验证码了，都是一些行为验证码，如某验、某盾等，国外也有很多，比如 reCaptcha 等。一些稍微简单一点的，比如滑动的，你可以想办法识别缺口，比如图像处理比对、深度学习识别都是可以的。</p>
<p data-nodeid="120884">对于轨迹行为你可以自己写一个模拟正常人行为的，加入抖动等。有了轨迹之后如何模拟呢，如果你非常厉害，那么可以直接去分析验证码的 JavaScript 逻辑，把轨迹数据录入，就能得到里面的一些加密参数，直接将这些参数放到表单或接口里面就能直接用了。当然也可以用模拟浏览器的方式来拖动，也能通过一定的方式拿到加密参数，或者直接用模拟浏览器的方式把登录一起做了，拿着 Cookies 来爬也行。</p>
<p data-nodeid="120885">当然拖动只是一种验证码，还有文字点选、逻辑推理等，要是真不想自己解决，可以找打码平台来解析出来再模拟，但毕竟是花钱的，一些高手就会选择自己训练深度学习相关的模型，收集数据、标注、训练，针对不同的业务训练不同的模型。这样有了核心技术，也不用再去花钱找打码平台了，再研究下验证码的逻辑模拟一下，加密参数就能解析出来了。不过有的验证码解析非常难，以至于我也搞不定。</p>
<p data-nodeid="120886">当然有些验证码可能是请求过于频繁而弹出来的，这种如果换 IP 也能解决。</p>
<h3 data-nodeid="124174" class="">封 IP</h3>

<p data-nodeid="120888">封 IP 也是一件令人头疼的事，行之有效的方法就是换代理了。代理有很多种，市面上免费的，收费的太多太多了。</p>
<p data-nodeid="120889">首先可以把市面上免费的代理用起来，自己搭建一个代理池，收集现在全网所有的免费代理，然后加一个测试器一直不断测试，测试的网址可以改成你要爬的网址。这样测试通过的一般都能直接拿来爬取目标网站。我自己也搭建过一个代理池，现在对接了一些免费代理，定时爬、定时测，还写了个 API 来取，放在了 GitHub 上： <a href="https://github.com/Python3WebSpider/ProxyPool" data-nodeid="121006">https://github.com/Python3WebSpider/ProxyPool</a>，打好了 Docker 镜像，提供了 Kubernetes 脚本，你可以直接拿来用。</p>
<p data-nodeid="120890">付费代理也是一样，很多商家提供了代理提取接口，请求一下就能获取几十几百个代理，我们可以同样把它们接入到代理池里面。但这个代理服务也分各种套餐，什么开放代理、独享代理等的质量和被封的概率也是不一样的。</p>
<p data-nodeid="120891">有的商家还利用隧道技术搭建了代理，这样代理的地址和端口我们是不知道的，代理池是由他们来维护的，比如某布云，这样用起来更省心一些，但是可控性就差一些。</p>
<p data-nodeid="120892">还有更稳定的代理，比如拨号代理、蜂窝代理等，接入成本会高一些，但是一定程度上也能解决一些封 IP 的问题。</p>
<h3 data-nodeid="124616" class="">封账号</h3>

<p data-nodeid="120894">有些信息需要模拟登录才能爬取，如果爬得过快，目标网站直接把你的账号封禁了，就没办法了。比如爬公众号的，人家把你 WX 号封了，那就全完了。</p>
<p data-nodeid="120895">一种解决方法就是放慢频率，控制节奏。还有一种方法就是看看别的终端，比如手机页、App 页、wap 页，看看有没有能绕过登录的方法。</p>
<p data-nodeid="120896">另外比较好的方法，就是分流。如果你的号足够多，建一个池子，比如 Cookies 池、Token 池、Sign 池等，多个账号跑出来的 Cookies、Token 都放到这个池子里，用的时候随机从里面获取一个。如果你想保证爬取效率不变，那么 100 个账号相比 20 个账号，对于每个账号对应的 Cookies、Token 的取用频率就变成原来的了 1/5，那么被封的概率也就随之降低了。</p>
<h3 data-nodeid="125058" class="">奇葩的反爬</h3>

<p data-nodeid="120898">上面说的是几种比较主流的反爬方式，当然还有非常多奇葩的反爬。比如返回假数据、返回图片化数据、返回乱序数据，等等，那都需要具体情况具体分析。</p>
<p data-nodeid="120899">这些反爬也得小心点，之前见过一个反爬直接返回 <code data-backticks="1" data-nodeid="121018">rm -rf /</code> 的也不是没有，你要是正好有个脚本模拟执行返回结果，后果自己想象。</p>
<h3 data-nodeid="125500" class="">JavaScript 逆向</h3>

<p data-nodeid="120901">说到重点了。随着前端技术的进步和网站反爬意识的增强，很多网站选择在前端上下功夫，那就是在前端对一些逻辑或代码进行加密或混淆。当然这不仅仅是为了保护前端的代码不被轻易盗取，更重要的是反爬。比如很多 Ajax 接口都会带着一些参数，比如 sign、token 等，这些前文也讲过了。这种数据我们可以用前文所说的 Selenium 等方式来爬取，但总归来说效率太低了，毕竟它模拟的是网页渲染的整个过程，而真实的数据可能仅仅就藏在一个小接口里。</p>
<p data-nodeid="120902">如果我们能够找出一些接口参数的真正逻辑，用代码来模拟执行，那效率就会有成倍的提升，而且还能在一定程度上规避上述的反爬现象。但问题是什么？比较难实现啊。</p>
<p data-nodeid="120903">Webpack 是一方面，前端代码都被压缩和转码成一些 bundle 文件，一些变量的含义已经丢失，不好还原。然后一些网站再加上一些 obfuscator 的机制，把前端代码变成你完全看不懂的东西，比如字符串拆散打乱、变量十六进制化、控制流扁平化、无限 debug、控制台禁用等，前端的代码和逻辑已经面目全非。有的用 WebAssembly 等技术把前端核心逻辑直接编译，那就只能慢慢抠了，虽然说有些有一定的技巧，但是总归来说还是会花费很多时间。但一旦解析出来了，那就万事大吉了。</p>
<p data-nodeid="120904">很多公司招聘爬虫工程师都会问有没有 JavaScript 逆向基础，破解过哪些网站，比如某宝、某多、某条等，解出来某个他们需要的可能就直接录用你。每家网站的逻辑都不一样，难度也不一样。</p>
<h3 data-nodeid="125942" class="">App</h3>

<p data-nodeid="120906">当然爬虫不仅仅是网页爬虫了，随着互联网时代的发展，现在越来越多的公司都选择将数据放到 App 上，甚至有些公司只有 App 没有网站。所以数据只能通过 App 来爬。</p>
<p data-nodeid="120907">怎么爬呢？基本的就是抓包工具了，Charles、Fiddler 等抓到接口之后，直接拿来模拟就行了。</p>
<p data-nodeid="120908">如果接口有加密参数怎么办呢？一种方法你可以边爬边处理，比如 mitmproxy 直接监听接口数据。另一方面你可以走 Hook，比如上 Xposed 也可以拿到。</p>
<p data-nodeid="120909">那爬的时候又怎么实现自动化呢？总不能拿手来戳吧。其实工具也多，安卓原生的 adb 工具也行，Appium 现在已经是比较主流的方案了，当然还有其他的某精灵都是可以实现的。</p>
<p data-nodeid="120910">最后，有的时候可能真的就不想走自动化的流程，我就想把里面的一些接口逻辑抠出来，那就需要搞逆向了，IDA Pro、jdax、FRIDA 等工具就派上用场了，当然这个过程和 JavaScript 逆向一样很痛苦，甚至可能得读汇编指令。</p>
<h3 data-nodeid="126384" class="">智能化</h3>

<p data-nodeid="120912">上面的这些知识，都掌握了以后，恭喜你已经超过了百分之八九十的爬虫玩家了，当然专门搞 JavaScript 逆向、App 逆向的都是站在食物链顶端的人，这种严格来说已经不算爬虫范畴了。</p>
<p data-nodeid="120913">除了上面的技能，在一些场合下，我们可能还需要结合一些机器学习的技术，让我们的爬虫变得更智能起来。</p>
<p data-nodeid="120914">比如现在很多博客、新闻文章，其页面结构相似度比较高，要提取的信息也比较类似。</p>
<p data-nodeid="120915">比如如何区分一个页面是索引页还是详情页？如何提取详情页的文章链接？如何解析文章页的页面内容？这些其实都是可以通过一些算法来计算出来的。</p>
<p data-nodeid="120916">所以，一些智能解析技术也应运而生，比如提取详情页，我的一位朋友写的 GeneralNewsExtractor 表现就非常好。</p>
<p data-nodeid="120917">假如说有一个需求，需要爬取一万个新闻网站数据，要一个个写 XPath 吗？如果有了智能化解析技术，在容忍一定错误的条件下，完成这个就是分分钟的事情。</p>
<p data-nodeid="120918">总之，如果我们能把这一块也学会了，我们的爬虫技术就会如虎添翼。</p>
<h3 data-nodeid="126826" class="">运维</h3>

<p data-nodeid="120920">这块内容也是一个重头戏。爬虫和运维也是息息相关的。比如：</p>
<ul data-nodeid="120921">
<li data-nodeid="120922">
<p data-nodeid="120923">写完一个爬虫，怎样去快速部署到 100 台主机上运行起来。</p>
</li>
<li data-nodeid="120924">
<p data-nodeid="120925">怎么灵活地监控每个爬虫的运行状态。</p>
</li>
<li data-nodeid="120926">
<p data-nodeid="120927">爬虫有处代码改动，如何去快速更新。</p>
</li>
<li data-nodeid="120928">
<p data-nodeid="120929">怎样监控一些爬虫的占用内存、消耗的 CPU 状况。</p>
</li>
<li data-nodeid="120930">
<p data-nodeid="120931">怎样科学地控制爬虫的定时运行。</p>
</li>
<li data-nodeid="120932">
<p data-nodeid="120933">爬虫出现了问题，怎样能及时收到通知，怎样设置科学的报警机制。</p>
</li>
</ul>
<p data-nodeid="120934">这里面，部署大家各有各的方法，比如可以用 Ansible。如果用 Scrapy 的话有 Scrapyd，然后配合上一些管理工具也能完成一些监控和定时任务。不过我现在用的更多的还是 Docker + Kubernetes，再加上 DevOps 一套解决方案，比如 GitHub Actions、Azure Pipelines、Jenkins 等，快速实现分发和部署。</p>
<p data-nodeid="120935">定时任务大家有的用 crontab，有的用 apscheduler，有的用管理工具，有的用 Kubernetes，我的话用 Kubernetes 会多一些了，定时任务也很好实现。</p>
<p data-nodeid="120936">至于监控的话，也有很多，专门的爬虫管理工具自带了一些监控和报警功能。一些云服务也带了一些监控的功能。我用的是 Kubernetes + Prometheus + Grafana，什么 CPU、内存、运行状态，一目了然，报警机制在 Grafana 里面配置也很方便，支持 Webhook、邮件甚至某钉。</p>
<p data-nodeid="120937">数据的存储和监控，用 Kafka、Elasticsearch 个人感觉也挺方便的，我主要用的是后者，然后再和 Grafana 配合起来，数据爬取量、爬取速度等等监控也都一目了然。</p>
<h3 data-nodeid="127268" class="">法律</h3>

<p data-nodeid="120939">另外希望你在做网络爬虫的过程中注意一些法律问题，基本上就是：</p>
<ul data-nodeid="120940">
<li data-nodeid="120941">
<p data-nodeid="120942">不要碰个人隐私信息。</p>
</li>
<li data-nodeid="120943">
<p data-nodeid="120944">规避商业竞争，看清目标站点的法律条款限制。</p>
</li>
<li data-nodeid="120945">
<p data-nodeid="120946">限制并发速度，不要影响目标站点的正常运行。</p>
</li>
<li data-nodeid="120947">
<p data-nodeid="120948">不要碰黑产、黄赌毒。</p>
</li>
<li data-nodeid="120949">
<p data-nodeid="120950">不随便宣传和传播目标站点或 App 的破解方案。</p>
</li>
<li data-nodeid="120951">
<p data-nodeid="120952">非公开数据一定要谨慎。</p>
</li>
</ul>
<p data-nodeid="120953">更多的内容可以参考一些文章：</p>
<ul data-nodeid="120954">
<li data-nodeid="120955">
<p data-nodeid="120956"><a href="https://mp.weixin.qq.com/s/aXr-ZE0ZifTm2h5w8BGh_Q" data-nodeid="121064">https://mp.weixin.qq.com/s/aXr-ZE0ZifTm2h5w8BGh_Q</a></p>
</li>
<li data-nodeid="120957">
<p data-nodeid="120958"><a href="https://mp.weixin.qq.com/s/zVTMQz78L16i7j8wXGjbLA" data-nodeid="121067">https://mp.weixin.qq.com/s/zVTMQz78L16i7j8wXGjbLA</a></p>
</li>
<li data-nodeid="120959">
<p data-nodeid="120960"><a href="https://mp.weixin.qq.com/s/MrJbodU0tcW6FRZ3JQa3xQ" data-nodeid="121070">https://mp.weixin.qq.com/s/MrJbodU0tcW6FRZ3JQa3xQ</a></p>
</li>
</ul>
<h3 data-nodeid="127710" class="">结语</h3>

<p data-nodeid="120962">至此，爬虫的一些涵盖的知识点也就差不多了，通过梳理发现计算机网络、编程基础、前端开发、后端开发、App 开发与逆向、网络安全、数据库、运维、机器学习都涵盖到了？上面总结的可以算是从爬虫小白到爬虫高手的路径了，里面每个方向其实可研究的点非常多，每个点做精了，都会非常了不起。</p>
