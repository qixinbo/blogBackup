---
title: 52讲轻松搞定网络爬虫笔记3
tags: [Web Crawler]
categories: data analysis
date: 2023-1-15
---

# 资料
[52讲轻松搞定网络爬虫](https://kaiwu.lagou.com/course/courseInfo.htm?courseId=46#/sale)

# Ajax的原理和解析
<p>当我们在用 requests 抓取页面的时候，得到的结果可能会和在浏览器中看到的不一样：在浏览器中正常显示的页面数据，使用 requests 却没有得到结果。这是因为 requests 获取的都是原始 HTML 文档，而浏览器中的页面则是经过 JavaScript 数据处理后生成的结果。这些数据的来源有多种，可能是通过 Ajax 加载的，可能是包含在 HTML 文档中的，也可能是经过 JavaScript 和特定算法计算后生成的。</p>
<p>对于第 1 种情况，数据加载是一种异步加载方式，原始页面不会包含某些数据，只有在加载完后，才会向服务器请求某个接口获取数据，然后数据才被处理从而呈现到网页上，这个过程实际上就是向服务器接口发送了一个 Ajax 请求。</p>
<p>按照 Web 的发展趋势来看，这种形式的页面将会越来越多。网页的原始 HTML 文档不会包含任何数据，数据都是通过 Ajax 统一加载后再呈现出来的，这样在 Web 开发上可以做到前后端分离，并且降低服务器直接渲染页面带来的压力。</p>
<p>所以如果你遇到这样的页面，直接利用 requests 等库来抓取原始页面，是无法获取有效数据的。这时我们需要分析网页后台向接口发送的 Ajax 请求，如果可以用 requests 来模拟 Ajax 请求，就可以成功抓取了。</p>
<p>所以，本课时我们就来了解什么是 Ajax 以及如何去分析和抓取 Ajax 请求。</p>
<h3>什么是 Ajax</h3>
<p>Ajax，全称为 Asynchronous JavaScript and XML，即异步的 JavaScript 和 XML。它不是一门编程语言，而是利用 JavaScript 在保证页面不被刷新、页面链接不改变的情况下与服务器交换数据并更新部分网页的技术。</p>
<p>传统的网页，如果你想更新其内容，那么必须要刷新整个页面。有了 Ajax，便可以在页面不被全部刷新的情况下更新其内容。在这个过程中，页面实际上在后台与服务器进行了数据交互，获取到数据之后，再利用 JavaScript 改变网页，这样网页内容就会更新了。</p>
<p>你可以到 W3School 上体验几个 Demo 来感受一下：<a href="http://www.w3school.com.cn/ajax/ajax_xmlhttprequest_send.asp">http://www.w3school.com.cn/ajax/ajax_xmlhttprequest_send.asp</a>。</p>
<h3>实例引入</h3>
<p>浏览网页的时候，我们会发现很多网页都有下滑查看更多的选项。以我微博的主页为例：<a href="https://m.weibo.cn/u/2830678474">https://m.weibo.cn/u/2830678474</a>。我们切换到微博页面，发现下滑几个微博后，后面的内容不会直接显示，而是会出现一个加载动画，加载完成后下方才会继续出现新的微博内容，这个过程其实就是 Ajax 加载的过程，如图所示：</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/79/DC/CgpOIF54UF2AMNEcAACYMMK9jgk319.png" alt=""></p>
<p>我们注意到页面其实并没有整个刷新，这意味着页面的链接没有变化，但是网页中却多了新内容，也就是后面刷出来的新微博。这就是通过 Ajax 获取新数据并呈现的过程。</p>
<h3>基本原理</h3>
<p>初步了解了 Ajax 之后，我们再来详细了解它的基本原理。发送 Ajax 请求到网页更新的过程可以简单分为以下 3 步：</p>
<ul>
<li>发送请求</li>
<li>解析内容</li>
<li>渲染网页</li>
</ul>
<p>下面我们分别详细介绍一下这几个过程。</p>
<h3>发送请求</h3>
<p>我们知道 JavaScript 可以实现页面的各种交互功能，Ajax 也不例外，它是由 JavaScript 实现的，实际上执行了如下代码：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> xmlhttp;
<span class="hljs-keyword">if</span> (<span class="hljs-built_in">window</span>.XMLHttpRequest) {
    <span class="hljs-comment">//code for IE7+, Firefox, Chrome, Opera, Safari</span>
    xmlhttp=<span class="hljs-keyword">new</span> XMLHttpRequest();} <span class="hljs-keyword">else</span> {<span class="hljs-comment">//code for IE6, IE5</span>
    xmlhttp=<span class="hljs-keyword">new</span> ActiveXObject(<span class="hljs-string">"Microsoft.XMLHTTP"</span>);
}
xmlhttp.onreadystatechange=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>) </span>{<span class="hljs-keyword">if</span> (xmlhttp.readyState==<span class="hljs-number">4</span> &amp;&amp; xmlhttp.status==<span class="hljs-number">200</span>) {<span class="hljs-built_in">document</span>.getElementById(<span class="hljs-string">"myDiv"</span>).innerHTML=xmlhttp.responseText;
    }
}
xmlhttp.open(<span class="hljs-string">"POST"</span>,<span class="hljs-string">"/ajax/"</span>,<span class="hljs-literal">true</span>);
xmlhttp.send();
</code></pre>
<p>这是 JavaScript 对 Ajax 最底层的实现，这个过程实际上是新建了 XMLHttpRequest 对象，然后调用 onreadystatechange 属性设置监听，最后调用 open() 和 send() 方法向某个链接（也就是服务器）发送请求。</p>
<p>前面我们用 Python 实现请求发送之后，可以得到响应结果，但这里请求的发送由 JavaScript 来完成。由于设置了监听，所以当服务器返回响应时，onreadystatechange 对应的方法便会被触发，我们在这个方法里面解析响应内容即可。</p>
<h4>解析内容</h4>
<p>得到响应之后，onreadystatechange 属性对应的方法会被触发，此时利用 xmlhttp 的 responseText 属性便可取到响应内容。这类似于 Python 中利用 requests 向服务器发起请求，然后得到响应的过程。</p>
<p>返回的内容可能是 HTML，也可能是 JSON，接下来我们只需要在方法中用 JavaScript 进一步处理即可。比如，如果返回的内容是 JSON 的话，我们便可以对它进行解析和转化。</p>
<h4>渲染网页</h4>
<p>JavaScript 有改变网页内容的能力，解析完响应内容之后，就可以调用 JavaScript 针对解析完的内容对网页进行下一步处理。比如，通过 document.getElementById().innerHTML 这样的操作，对某个元素内的源代码进行更改，这样网页显示的内容就改变了，这种对 Document 网页文档进行如更改、删除等操作也被称作 DOM 操作。</p>
<p>上例中，document.getElementById("myDiv").innerHTML=xmlhttp.responseText这个操作便将 ID 为 myDiv 的节点内部的 HTML 代码更改为服务器返回的内容，这样 myDiv 元素内部便会呈现出服务器返回的新数据，网页的部分内容看上去就更新了。</p>
<p>可以看到，发送请求、解析内容和渲染网页这 3 个步骤其实都是由 JavaScript 完成的。</p>
<p>我们再回想微博的下拉刷新，这其实是 JavaScript 向服务器发送了一个 Ajax 请求，然后获取新的微博数据，将其解析，并将其渲染在网页中的过程。</p>
<p>因此，真实的数据其实都是通过一次次 Ajax 请求得到的，如果想要抓取这些数据，我们需要知道这些请求到底是怎么发送的，发往哪里，发了哪些参数。如果我们知道了这些，不就可以用 Python 模拟这个发送操作，获取到其中的结果了吗？</p>
<h3>Ajax 分析</h3>
<p>这里还是以前面的微博为例，我们知道拖动刷新的内容由 Ajax 加载，而且页面的 URL 没有变化，这时我们应该到哪里去查看这些 Ajax 请求呢？</p>
<p>这里还需要借助浏览器的开发者工具，下面以 Chrome 浏览器为例来介绍。</p>
<p>首先，用 Chrome 浏览器打开微博链接&nbsp;<a href="https://m.weibo.cn/u/2830678474">https://m.weibo.cn/u/2830678474</a>，随后在页面中点击鼠标右键，从弹出的快捷菜单中选择“检查” 选项，此时便会弹出开发者工具，如图所示：</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/79/DC/Cgq2xl54UF6ASQetAAP8yHnhg-A801.png" alt=""></p>
<p>前面也提到过，这里就是页面加载过程中浏览器与服务器之间发送请求和接收响应的所有记录。</p>
<p>Ajax 有其特殊的请求类型，它叫作 xhr。在图中我们可以发现一个以 getIndex 开头的请求，其 Type 为 xhr，这就是一个 Ajax 请求。用鼠标点击这个请求，可以查看这个请求的详细信息。</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/79/DC/CgpOIF54UF6AN8guAAbrzO2HTj8622.png" alt=""></p>
<p>在右侧可以观察到 Request Headers、URL 和 Response Headers 等信息。Request Headers 中有一个信息为 X-Requested-With:XMLHttpRequest，这就标记了此请求是 Ajax 请求，如图所示：</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/00/C6/Ciqah154UF6AG4amAAKPalYmb5k222.png" alt=""></p>
<p>随后我们点击 Preview，即可看到响应的内容，它是 JSON 格式的。这里 Chrome 为我们自动做了解析，点击箭头即可展开和收起相应内容。</p>
<p>我们可以观察到，返回结果是我的个人信息，包括昵称、简介、头像等，这也是用来渲染个人主页所使用的数据。JavaScript 接收到这些数据之后，再执行相应的渲染方法，整个页面就渲染出来了。</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/79/DC/Cgq2xl54UF6AEs-HAAMmqovudJc203.png" alt=""></p>
<p>另外，我们也可以切换到 Response 选项卡，从中观察到真实的返回数据，如图所示：</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/79/DC/CgpOIF54UF6AbF8YAABZ6_r-H8Q421.png" alt=""></p>
<p>接下来，切回到第一个请求，观察一下它的 Response 是什么，如图所示：</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/00/C6/Ciqah154UF-ALTcXAAgUk5WaJmM244.png" alt=""></p>
<p>这就是最原始链接&nbsp;<a href="https://m.weibo.cn/u/2830678474">https://m.weibo.cn/u/2830678474</a>&nbsp;返回的结果，其代码只有不到 50 行，结构也非常简单，只是执行了一些 JavaScript。</p>
<p>所以说，我们看到的微博页面的真实数据并不是最原始的页面返回的，而是在执行 JavaScript 后再次向后台发送 Ajax 请求，浏览器拿到数据后进一步渲染出来的。</p>
<h3>过滤请求</h3>
<p>接下来，我们再利用 Chrome 开发者工具的筛选功能筛选出所有的 Ajax 请求。在请求的上方有一层筛选栏，直接点击 XHR，此时在下方显示的所有请求便都是 Ajax 请求了，如图所示：</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/79/DC/Cgq2xl54UF-AMKo-AAXlNO948BY860.png" alt=""></p>
<p>接下来，不断滑动页面，可以看到页面底部有一条条新的微博被刷出，而开发者工具下方也不断地出现 Ajax 请求，这样我们就可以捕获到所有的 Ajax 请求了。</p>
<p>随意点开一个条目，都可以清楚地看到其 Request URL、Request Headers、Response Headers、Response Body 等内容，此时想要模拟请求和提取就非常简单了。</p>
<p>下图所示的内容便是我某一页微博的列表信息：</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/79/DC/CgpOIF54UF-AcIlMAAoXZXGIcTE140.png" alt=""></p>
<p>到现在为止，我们已经可以分析出 Ajax 请求的一些详细信息了，接下来只需要用程序模拟这些 Ajax 请求，就可以轻松提取我们所需要的信息了。</p>

# Ajax爬取案例实战
<p data-nodeid="97695" class="">上一课时我们学习了 Ajax 的基本原理和分析方法，这一课时我们结合实际案例，学习 Ajax 分析和爬取页面的具体实现。</p>
<h3 data-nodeid="97696">准备工作</h3>
<p data-nodeid="97697">在开始学习之前，我们需要做好如下的准备工作：</p>
<ul data-nodeid="97698">
<li data-nodeid="97699">
<p data-nodeid="97700">安装好 Python 3（最低为 3.6 版本），并能成功运行 Python 3 程序。</p>
</li>
<li data-nodeid="97701">
<p data-nodeid="97702">了解 Python HTTP 请求库 requests 的基本用法。</p>
</li>
<li data-nodeid="97703">
<p data-nodeid="97704">了解 Ajax 的基础知识和分析 Ajax 的基本方法。</p>
</li>
</ul>
<p data-nodeid="97705">以上内容在前面的课时中均有讲解，如你尚未准备好建议先熟悉一下这些内容。</p>
<h3 data-nodeid="97706">爬取目标</h3>
<p data-nodeid="98502" class="">本课时我们以一个动态渲染网站为例来试验一下 Ajax 的爬取。其链接为：<a href="https://dynamic1.scrape.center/" data-nodeid="98506">https://dynamic1.scrape.center/</a>，页面如图所示。</p>


<p data-nodeid="97708"><img src="https://s0.lgstatic.com/i/image3/M01/02/6C/Ciqah156_puAD5VOAAR5H7Xnnxs004.png" alt="" data-nodeid="97815"></p>
<p data-nodeid="97709">这个页面看似和我们上一课时的案例一模一样，但其实不是，它的后台实现逻辑和数据加载方式与上一课时完全不同，只不过最后呈现的样式是一样的。</p>
<p data-nodeid="97710">这个网站同样支持翻页，可以点击最下方的页码来切换到下一页，如图所示。</p>
<p data-nodeid="97711"><img src="https://s0.lgstatic.com/i/image3/M01/7B/82/Cgq2xl56_pyAdL3MAARFiFasoCA870.png" alt="" data-nodeid="97819"></p>
<p data-nodeid="97712">点击每一个电影的链接进入详情页，页面结构也是完全一样的，如图所示。</p>
<p data-nodeid="97713"><img src="https://s0.lgstatic.com/i/image3/M01/02/6C/Ciqah156_pyAMuo5AAXvLQH0FFU950.png" alt="" data-nodeid="97822"></p>
<p data-nodeid="97714">我们需要爬取的数据也和原来是相同的，包括电影的名称、封面、类别、上映日期、评分、剧情简介等信息。</p>
<p data-nodeid="97715">本课时我们需要完成的目标有：</p>
<ul data-nodeid="97716">
<li data-nodeid="97717">
<p data-nodeid="97718">分析页面数据的加载逻辑。</p>
</li>
<li data-nodeid="97719">
<p data-nodeid="97720">用 requests 实现 Ajax 数据的爬取。</p>
</li>
<li data-nodeid="97721">
<p data-nodeid="97722">将每部电影的数据保存成一个 JSON 数据文件。</p>
</li>
</ul>
<p data-nodeid="97723">由于本课时主要讲解 Ajax，所以对于数据存储和加速部分就不再展开实现，主要是讲解 Ajax 的分析和爬取。</p>
<p data-nodeid="97724">那么我们现在就开始正式学习吧。</p>
<h3 data-nodeid="97725">初步探索</h3>
<p data-nodeid="97726">首先，我们尝试用之前的 requests 来直接提取页面，看看会得到怎样的结果。用最简单的代码实现一下 requests 获取首页源码的过程，代码如下：</p>
<pre class="lang-python" data-nodeid="99042"><code data-language="python"><span class="hljs-keyword">import</span>&nbsp;requests

url&nbsp;=&nbsp;<span class="hljs-string">'https://dynamic1.scrape.center/'</span>
html&nbsp;=&nbsp;requests.get(url).text
print(html)
</code></pre>

<p data-nodeid="97728">运行结果如下：</p>
<p data-nodeid="97729"><img src="https://s0.lgstatic.com/i/image3/M01/7B/82/Cgq2xl56_p2ACwbQAAWO53Be3FE827.png" alt="" data-nodeid="97834"></p>
<p data-nodeid="97730">可以看到我们只爬取到了这么一点 HTML 内容，而在浏览器中打开这个页面却能看到这样的结果，如图所示。</p>
<p data-nodeid="97731"><img src="https://s0.lgstatic.com/i/image3/M01/02/6C/Ciqah156_p2AK2ScAAR5mLKj84Y332.png" alt="" data-nodeid="97837"></p>
<p data-nodeid="97732">也就是说在 HTML 中我们只能在源码中看到引用了一些 JavaScript 和 CSS 文件，并没有观察任何有关电影数据的信息。</p>
<p data-nodeid="97733">如果遇到这样的情况，说明我们现在看到的整个页面是通过 JavaScript 渲染得到的，浏览器执行了 HTML 中所引用的 JavaScript 文件，JavaScript 通过调用一些数据加载和页面渲染的方法，才最终呈现了图中所示的页面。</p>
<p data-nodeid="97734">在一般情况下，这些数据都是通过 Ajax 来加载的， JavaScript 在后台调用这些 Ajax 数据接口，得到数据之后，再把数据进行解析并渲染呈现出来，得到最终的页面。所以说，要想爬取这个页面，我们可以通过直接爬取 Ajax 接口获取数据。</p>
<p data-nodeid="97735">在上一课时中，我们已经了解了用 Ajax 分析的基本方法。下面我们就来分析下 Ajax 接口的逻辑并实现数据爬取吧。</p>
<h3 data-nodeid="97736">爬取列表页</h3>
<p data-nodeid="97737">首先我们来分析下列表页的 Ajax 接口逻辑，打开浏览器开发者工具，切换到 Network 面板，勾选上 「Preserve Log」并切换到 「XHR」选项卡，如图所示。</p>
<p data-nodeid="97738"><img src="https://s0.lgstatic.com/i/image3/M01/7B/82/Cgq2xl56_p2AHMwfAALZgWyJlOw751.png" alt="" data-nodeid="97845"></p>
<p data-nodeid="97739">接着，我们重新刷新页面，然后点击第 2 页、第 3 页、第 4 页的按钮，这时候可以看到页面上的数据发生了变化，同时在开发者工具下方会监听到几个 Ajax 请求，如图所示。</p>
<p data-nodeid="97740"><img src="https://s0.lgstatic.com/i/image3/M01/02/6C/Ciqah156_p6AbNjgAALk3soWZX0212.png" alt="" data-nodeid="97848"></p>
<p data-nodeid="97741">由于我们切换了 4 页，所以这里正好也出现了 4 个 Ajax 请求，我们可以任选一个点击查看其请求详情，观察其请求的 URL、参数以及响应内容是怎样的，如图所示。</p>
<p data-nodeid="97742"><img src="https://s0.lgstatic.com/i/image3/M01/7B/83/Cgq2xl56_p6AMiexAAELqD7xtMQ979.png" alt="" data-nodeid="97851"></p>
<p data-nodeid="100119" class="">这里我们点开第 2 个结果，观察到其 Ajax 接口请求的 URL 地址为：<a href="https://dynamic1.scrape.center/api/movie/?limit=10&amp;offset=10" data-nodeid="100125">https://dynamic1.scrape.center/api/movie/?limit=10&amp;offset=10</a>，这里有两个参数，一个是 limit，其值为 10，一个是 offset，它的值也是 10。</p>


<p data-nodeid="97744">通过观察多个 Ajax 接口的参数，我们可以发现这么一个规律：limit 的值一直为 10，这就正好对应着每页 10 条数据；offset 的值在依次变大，页面每加 1 页，offset 就加 10，这就代表着页面的数据偏移量，比如第 2 页的 offset 值为 10 代表跳过 10 条数据，返回从第 11 条数据开始的结果，再加上 limit 的限制，就代表返回第 11~20 条数据的结果。</p>
<p data-nodeid="97745">接着我们再观察下响应的数据，切换到 Preview 选项卡，结果如图所示。</p>
<p data-nodeid="97746" class="te-preview-highlight"><img src="https://s0.lgstatic.com/i/image3/M01/02/6C/Ciqah156_p6AbqhsAAGbdIN1Xj4797.png" alt="" data-nodeid="97864"></p>
<p data-nodeid="97747">可以看到结果是一些 JSON 数据，它有一个 results 字段，这是一个列表，列表的每一个元素都是一个字典。观察一下字典的内容，发现我们可以看到对应的电影数据的字段了，如 name、alias、cover、categories，对比下浏览器中的真实数据，各个内容是完全一致的，而且这个数据已经非常结构化了，完全就是我们想要爬取的数据，真是得来全不费工夫。</p>
<p data-nodeid="97748">这样的话，我们只需要把所有页面的 Ajax 接口构造出来，那么所有的列表页数据我们都可以轻松获取到了。</p>
<p data-nodeid="97749">我们先定义一些准备工作，导入一些所需的库并定义一些配置，代码如下：</p>
<pre class="lang-python" data-nodeid="100661"><code data-language="python"><span class="hljs-keyword">import</span>&nbsp;requests
<span class="hljs-keyword">import</span>&nbsp;logging

logging.basicConfig(level=logging.INFO,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;format=<span class="hljs-string">'%(asctime)s&nbsp;-&nbsp;%(levelname)s:&nbsp;%(message)s'</span>)

INDEX_URL&nbsp;=&nbsp;<span class="hljs-string">'https://dynamic1.scrape.center/api/movie/?limit={limit}&amp;offset={offset}'</span>
</code></pre>

<p data-nodeid="97751">这里我们引入了 requests 和 logging 库，并定义了 logging 的基本配置，接着我们定义 INDEX_URL，这里把 limit 和 offset 预留出来变成占位符，可以动态传入参数构造成一个完整的列表页 URL。</p>
<p data-nodeid="97752">下面我们来实现一下列表页的爬取，还是和原来一样，我们先定义一个通用的爬取方法，代码如下：</p>
<pre class="lang-python" data-nodeid="97753"><code data-language="python"><span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">scrape_api</span>(<span class="hljs-params">url</span>):</span>
&nbsp;&nbsp;&nbsp;&nbsp;logging.info(<span class="hljs-string">'scraping&nbsp;%s...'</span>,&nbsp;url)
&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">try</span>:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;response&nbsp;=&nbsp;requests.get(url)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">if</span>&nbsp;response.status_code&nbsp;==&nbsp;<span class="hljs-number">200</span>:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;response.json()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logging.error(<span class="hljs-string">'get&nbsp;invalid&nbsp;status&nbsp;code&nbsp;%s&nbsp;while&nbsp;scraping&nbsp;%s'</span>,&nbsp;response.status_code,&nbsp;url)
&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">except</span>&nbsp;requests.RequestException:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logging.error(<span class="hljs-string">'error&nbsp;occurred&nbsp;while&nbsp;scraping&nbsp;%s'</span>,&nbsp;url,&nbsp;exc_info=<span class="hljs-literal">True</span>)
</code></pre>
<p data-nodeid="97754">这里我们定义一个 scrape_api 方法，和之前不同的是，这个方法专门用来处理 JSON 接口，最后的 response 调用的是 json 方法，它可以解析响应的内容并将其转化成 JSON 字符串。</p>
<p data-nodeid="97755">在这个基础之上，我们定义一个爬取列表页的方法，代码如下：</p>
<pre class="lang-python" data-nodeid="97756"><code data-language="python">LIMIT&nbsp;=&nbsp;<span class="hljs-number">10</span>

<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">scrape_index</span>(<span class="hljs-params">page</span>):</span>
&nbsp;&nbsp;&nbsp;&nbsp;url&nbsp;=&nbsp;INDEX_URL.format(limit=LIMIT,&nbsp;offset=LIMIT&nbsp;*&nbsp;(page&nbsp;-&nbsp;<span class="hljs-number">1</span>))
&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;scrape_api(url)
</code></pre>
<p data-nodeid="97757">这里我们定义了一个 scrape_index 方法，用来接收参数 page，page 代表列表页的页码。</p>
<p data-nodeid="97758">这里我们先构造了一个 URL，通过字符串的 format 方法，传入 limit 和 offset 的值。这里的 limit 直接使用了全局变量 LIMIT 的值，offset 则是动态计算的，计算方法是页码数减 1 再乘以 limit，比如第 1 页的 offset 值就是 0，第 2 页的 offset 值就是 10，以此类推。构造好 URL 之后，直接调用 scrape_api 方法并返回结果即可。</p>
<p data-nodeid="97759">这样我们就完成了列表页的爬取，每次请求都会得到一页 10 部的电影数据。</p>
<p data-nodeid="97760">由于这时爬取到的数据已经是 JSON 类型了，所以我们不用像之前一样去解析 HTML 代码来提取数据，爬到的数据就是我们想要的结构化数据，因此解析这一步这里我们就可以直接省略啦。</p>
<p data-nodeid="97761">到此为止，我们就能成功爬取列表页并提取出电影列表信息了。</p>
<h3 data-nodeid="97762">爬取详情页</h3>
<p data-nodeid="97763">这时候我们已经可以拿到每一页的电影数据了，但是实际上这些数据还缺少一些我们想要的信息，如剧情简介等，所以我们需要进一步进入到详情页来获取这些内容。</p>
<p data-nodeid="101736" class="">这时候我们点击任意一部电影，如《教父》，进入到其详情页面，这时候可以发现页面的 URL 已经变成了&nbsp;<a href="https://dynamic1.scrape.center/detail/40" data-nodeid="101740">https://dynamic1.scrape.center/detail/40</a>，页面也成功展示了详情页的信息，如图所示。</p>


<p data-nodeid="97765"><img src="https://s0.lgstatic.com/i/image3/M01/7B/83/Cgq2xl56_p6AAN4hAASly3u9w8A543.png" alt="" data-nodeid="97893"></p>
<p data-nodeid="102816" class="">另外我们也可以观察到在开发者工具中又出现了一个 Ajax 请求，其 URL 为&nbsp;<a href="https://dynamic1.scrape.center/api/movie/40/" data-nodeid="102820">https://dynamic1.scrape.center/api/movie/40/</a>，通过 Preview 选项卡也能看到 Ajax 请求对应响应的信息，如图所示。</p>


<p data-nodeid="97767"><img src="https://s0.lgstatic.com/i/image3/M01/02/6C/Ciqah156_p-AMmb8AAUz2kfyQ9k795.png" alt="" data-nodeid="97900"></p>
<p data-nodeid="97768">稍加观察我们就可以发现，Ajax 请求的 URL 后面有一个参数是可变的，这个参数就是电影的 id，这里是 40，对应《教父》这部电影。</p>
<p data-nodeid="103896" class="">如果我们想要获取 id 为 50 的电影，只需要把 URL 最后的参数改成 50 即可，即&nbsp;<a href="https://dynamic1.scrape.center/api/movie/50/" data-nodeid="103900">https://dynamic1.scrape.center/api/movie/50/</a>，请求这个新的 URL 我们就能获取 id 为 50 的电影所对应的数据了。</p>


<p data-nodeid="97770">同样的，它响应的结果也是结构化的 JSON 数据，字段也非常规整，我们直接爬取即可。</p>
<p data-nodeid="97771">分析了详情页的数据提取逻辑，那么怎么把它和列表页关联起来呢？这个 id 又是从哪里来呢？我们回过头来再看看列表页的接口返回数据，如图所示。</p>
<p data-nodeid="97772"><img src="https://s0.lgstatic.com/i/image3/M01/7B/83/Cgq2xl56_p-AZElfAAGXCVHyIdg043.png" alt="" data-nodeid="97910"></p>
<p data-nodeid="97773">可以看到列表页原本的返回数据就带了 id 这个字段，所以我们只需要拿列表页结果中的 id 来构造详情页中 Ajax 请求的 URL 就好了。</p>
<p data-nodeid="97774">那么接下来，我们就先定义一个详情页的爬取逻辑吧，代码如下：</p>
<pre class="lang-python" data-nodeid="104436"><code data-language="python">DETAIL_URL&nbsp;=&nbsp;<span class="hljs-string">'https://dynamic1.scrape.center/api/movie/{id}'</span>

<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">scrape_detail</span>(<span class="hljs-params">id</span>):</span>
&nbsp;&nbsp;&nbsp;&nbsp;url&nbsp;=&nbsp;DETAIL_URL.format(id=id)
&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;scrape_api(url)
</code></pre>

<p data-nodeid="97776">这里我们定义了一个 scrape_detail 方法，它接收参数 id。这里的实现也非常简单，先根据定义好的 DETAIL_URL 加上 id，构造一个真实的详情页 Ajax 请求的 URL，然后直接调用 scrape_api 方法传入这个 URL 即可。</p>
<p data-nodeid="97777">接着，我们定义一个总的调用方法，将以上的方法串联调用起来，代码如下：</p>
<pre class="lang-python" data-nodeid="97778"><code data-language="python">TOTAL_PAGE&nbsp;=&nbsp;<span class="hljs-number">10</span>

<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">main</span>():</span>
&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">for</span>&nbsp;page&nbsp;<span class="hljs-keyword">in</span>&nbsp;range(<span class="hljs-number">1</span>,&nbsp;TOTAL_PAGE&nbsp;+&nbsp;<span class="hljs-number">1</span>):
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;index_data&nbsp;=&nbsp;scrape_index(page)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">for</span>&nbsp;item&nbsp;<span class="hljs-keyword">in</span>&nbsp;index_data.get(<span class="hljs-string">'results'</span>):
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;id&nbsp;=&nbsp;item.get(<span class="hljs-string">'id'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail_data&nbsp;=&nbsp;scrape_detail(id)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logging.info(<span class="hljs-string">'detail&nbsp;data&nbsp;%s'</span>,&nbsp;detail_data)
</code></pre>
<p data-nodeid="97779">这里我们定义了一个 main 方法，首先遍历获取页码 page，然后把 page 当成参数传递给 scrape_index 方法，得到列表页的数据。接着我们遍历所有列表页的结果，获取每部电影的 id，然后把 id 当作参数传递给 scrape_detail 方法，来爬取每部电影的详情数据，赋值为 detail_data，输出即可。</p>
<p data-nodeid="97780">运行结果如下：</p>
<pre class="lang-html" data-nodeid="106041"><code data-language="html">2020-03-19&nbsp;02:51:55,981&nbsp;-&nbsp;INFO:&nbsp;scraping&nbsp;https://dynamic1.scrape.center/api/movie/?limit=10&amp;offset=0...
2020-03-19&nbsp;02:51:56,446&nbsp;-&nbsp;INFO:&nbsp;scraping&nbsp;https://dynamic1.scrape.center/api/movie/1...
2020-03-19&nbsp;02:51:56,638&nbsp;-&nbsp;INFO:&nbsp;detail&nbsp;data&nbsp;{'id':&nbsp;1,&nbsp;'name':&nbsp;'霸王别姬',&nbsp;'alias':&nbsp;'Farewell&nbsp;My&nbsp;Concubine',&nbsp;'cover':&nbsp;'https://p0.meituan.net/movie/ce4da3e03e655b5b88ed31b5cd7896cf62472.jpg@464w_644h_1e_1c',&nbsp;'categories':&nbsp;['剧情',&nbsp;'爱情'],&nbsp;'regions':&nbsp;['中国大陆',&nbsp;'中国香港'],&nbsp;'actors':&nbsp;[{'name':&nbsp;'张国荣',&nbsp;'role':&nbsp;'程蝶衣',&nbsp;...},&nbsp;...],&nbsp;'directors':&nbsp;[{'name':&nbsp;'陈凯歌',&nbsp;'image':&nbsp;'https://p0.meituan.net/movie/8f9372252050095067e0e8d58ef3d939156407.jpg@128w_170h_1e_1c'}],&nbsp;'score':&nbsp;9.5,&nbsp;'rank':&nbsp;1,&nbsp;'minute':&nbsp;171,&nbsp;'drama':&nbsp;'影片借一出《霸王别姬》的京戏，牵扯出三个人之间一段随时代风云变幻的爱恨情仇。段小楼（张丰毅&nbsp;饰）与程蝶衣（张国荣&nbsp;饰）是一对打小一起长大的师兄弟，...',&nbsp;'photos':&nbsp;[...],&nbsp;'published_at':&nbsp;'1993-07-26',&nbsp;'updated_at':&nbsp;'2020-03-07T16:31:36.967843Z'}
2020-03-19&nbsp;02:51:56,640&nbsp;-&nbsp;INFO:&nbsp;scraping&nbsp;https://dynamic1.scrape.center/api/movie/2...
2020-03-19&nbsp;02:51:56,813&nbsp;-&nbsp;INFO:&nbsp;detail&nbsp;data&nbsp;{'id':&nbsp;2,&nbsp;'name':&nbsp;'这个杀手不太冷',&nbsp;'alias':&nbsp;'Léon',&nbsp;'cover':&nbsp;'https://p1.meituan.net/movie/6bea9af4524dfbd0b668eaa7e187c3df767253.jpg@464w_644h_1e_1c',&nbsp;'categories':&nbsp;['剧情',&nbsp;'动作',&nbsp;'犯罪'],&nbsp;'regions':&nbsp;['法国'],&nbsp;'actors':&nbsp;[{'name':&nbsp;'让·雷诺',&nbsp;'role':&nbsp;'莱昂&nbsp;Leon',&nbsp;...},&nbsp;...],&nbsp;'directors':&nbsp;[{'name':&nbsp;'吕克·贝松',&nbsp;'image':&nbsp;'https://p0.meituan.net/movie/0e7d67e343bd3372a714093e8340028d40496.jpg@128w_170h_1e_1c'}],&nbsp;'score':&nbsp;9.5,&nbsp;'rank':&nbsp;3,&nbsp;'minute':&nbsp;110,&nbsp;'drama':&nbsp;'里昂（让·雷诺&nbsp;饰）是名孤独的职业杀手，受人雇佣。一天，邻居家小姑娘马蒂尔德（纳塔丽·波特曼&nbsp;饰）敲开他的房门，要求在他那里暂避杀身之祸。...',&nbsp;'photos':&nbsp;[...],&nbsp;'published_at':&nbsp;'1994-09-14',&nbsp;'updated_at':&nbsp;'2020-03-07T16:31:43.826235Z'}
...
</code></pre>



<p data-nodeid="97782">由于内容较多，这里省略了部分内容。</p>
<p data-nodeid="97783">可以看到，其实整个爬取工作到这里就已经完成了，这里会先顺次爬取每一页列表页的 Ajax 接口，然后再顺次爬取每部电影详情页的 Ajax 接口，最后打印出每部电影的 Ajax 接口响应数据，而且都是 JSON 格式。这样，所有电影的详情数据都会被我们爬取到啦。</p>
<h3 data-nodeid="97784">保存数据</h3>
<p data-nodeid="97785">最后，让我们把爬取到的数据保存下来吧。之前我们是用 MongoDB 来存储数据，由于本课时重点讲解 Ajax 爬取，所以这里就一切从简，将数据保存为 JSON 文本。</p>
<p data-nodeid="97786">定义一个数据保存的方法，代码如下：</p>
<pre class="lang-python" data-nodeid="97787"><code data-language="python"><span class="hljs-keyword">import</span>&nbsp;json
<span class="hljs-keyword">from</span>&nbsp;os&nbsp;<span class="hljs-keyword">import</span>&nbsp;makedirs
<span class="hljs-keyword">from</span>&nbsp;os.path&nbsp;<span class="hljs-keyword">import</span>&nbsp;exists

RESULTS_DIR&nbsp;=&nbsp;<span class="hljs-string">'results'</span>
exists(RESULTS_DIR)&nbsp;<span class="hljs-keyword">or</span>&nbsp;makedirs(RESULTS_DIR)

<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">save_data</span>(<span class="hljs-params">data</span>):</span>
&nbsp;&nbsp;&nbsp;&nbsp;name&nbsp;=&nbsp;data.get(<span class="hljs-string">'name'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;data_path&nbsp;=&nbsp;<span class="hljs-string">f'<span class="hljs-subst">{RESULTS_DIR}</span>/<span class="hljs-subst">{name}</span>.json'</span>
&nbsp;&nbsp;&nbsp;&nbsp;json.dump(data,&nbsp;open(data_path,&nbsp;<span class="hljs-string">'w'</span>,&nbsp;encoding=<span class="hljs-string">'utf-8'</span>),&nbsp;ensure_ascii=<span class="hljs-literal">False</span>,&nbsp;indent=<span class="hljs-number">2</span>
</code></pre>
<p data-nodeid="97788">在这里我们首先定义了数据保存的文件夹 RESULTS_DIR，注意，我们先要判断这个文件夹是否存在，如果不存在则需要创建。</p>
<p data-nodeid="97789">接着，我们定义了保存数据的方法 save_data，首先我们获取数据的 name 字段，即电影的名称，把电影名称作为 JSON 文件的名称，接着构造 JSON 文件的路径，然后用 json 的 dump 方法将数据保存成文本格式。dump 的方法设置了两个参数，一个是 ensure_ascii，我们将其设置为 False，它可以保证中文字符在文件中能以正常的中文文本呈现，而不是 unicode 字符；另一个是 indent，它的数值为 2，这代表生成的 JSON 数据结果有两个空格缩进，让它的格式显得更加美观。</p>
<p data-nodeid="97790">最后，main 方法再调用下 save_data 方法即可，实现如下：</p>
<pre class="lang-python" data-nodeid="97791"><code data-language="python"><span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">main</span>():</span>
&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">for</span>&nbsp;page&nbsp;<span class="hljs-keyword">in</span>&nbsp;range(<span class="hljs-number">1</span>,&nbsp;TOTAL_PAGE&nbsp;+&nbsp;<span class="hljs-number">1</span>):
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;index_data&nbsp;=&nbsp;scrape_index(page)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">for</span>&nbsp;item&nbsp;<span class="hljs-keyword">in</span>&nbsp;index_data.get(<span class="hljs-string">'results'</span>):
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;id&nbsp;=&nbsp;item.get(<span class="hljs-string">'id'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail_data&nbsp;=&nbsp;scrape_detail(id)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logging.info(<span class="hljs-string">'detail&nbsp;data&nbsp;%s'</span>,&nbsp;detail_data)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;save_data(detail_data)
</code></pre>
<p data-nodeid="97792">重新运行一下，我们发现本地 results 文件夹下出现了各个电影的 JSON 文件，如图所示。<br>
<img src="https://s0.lgstatic.com/i/image3/M01/02/6C/Ciqah156_p-Ac2i7AAFVfgr3dcw686.png" alt="" data-nodeid="97948"></p>
<p data-nodeid="97793">这样我们就已经把所有的电影数据保存下来了，打开其中一个 JSON 文件，看看保存格式，如图所示。</p>
<p data-nodeid="97794"><img src="https://s0.lgstatic.com/i/image3/M01/7B/83/Cgq2xl56_qCAWVzhAAIR0qrWPCY515.png" alt="" data-nodeid="97951"></p>
<p data-nodeid="97795">可以看到 JSON 文件里面的数据都是经过格式化的中文文本数据，结构清晰，一目了然。</p>
<p data-nodeid="97796">至此，我们就完成了全站电影数据的爬取并把每部电影的数据保存成了 JSON 文件。</p>
<h3 data-nodeid="97797">总结</h3>
<p data-nodeid="97798">本课时我们通过一个案例来体会了 Ajax 分析和爬取的基本流程，希望你能够对 Ajax 的分析和爬取的实现更加熟悉。</p>
<p data-nodeid="97799">另外我们也可以观察到，由于 Ajax 接口大部分返回的是 JSON 数据，所以在一定程度上可以避免一些数据提取的工作，减轻我们的工作量。</p>

# Selenium的基本使用
<p data-nodeid="108788" class="">上个课时我们讲解了 Ajax 的分析方法，利用 Ajax 接口我们可以非常方便地完成数据的爬取。只要我们能找到 Ajax 接口的规律，就可以通过某些参数构造出对应的的请求，数据自然就能被轻松爬取到。</p>
<p data-nodeid="110453" class="">但是，在很多情况下，Ajax 请求的接口通常会包含加密的参数，如 token、sign 等，如：<a href="https://dynamic2.scrape.center/" data-nodeid="110457">https://dynamic2.scrape.center/</a>，它的 Ajax 接口是包含一个 token 参数的，如图所示。</p>


<p data-nodeid="108790"><img src="https://s0.lgstatic.com/i/image3/M01/7D/12/Cgq2xl59oBaAXsW1AADPHO1rr_g541.png" alt="" data-nodeid="108983"></p>
<p data-nodeid="108791">由于接口的请求加上了 token 参数，如果不深入分析并找到 token 的构造逻辑，我们是难以直接模拟这些 Ajax 请求的。</p>
<p data-nodeid="108792">此时解决方法通常有两种，一种是深挖其中的逻辑，把其中 token 的构造逻辑完全找出来，再用 Python 复现，构造 Ajax 请求；另外一种方法就是直接通过模拟浏览器的方式，绕过这个过程。因为在浏览器里面我们是可以看到这个数据的，如果能直接把看到的数据爬取下来，当然也就能获取对应的信息了。</p>
<p data-nodeid="108793">由于第 1 种方法难度较高，在这里我们就先介绍第 2 种方法，模拟浏览器爬取。</p>
<p data-nodeid="108794">这里使用的工具为 Selenium，我们先来了解一下 Selenium 的基本使用方法吧。</p>
<p data-nodeid="108795">Selenium 是一个自动化测试工具，利用它可以驱动浏览器执行特定的动作，如点击、下拉等操作，同时还可以获取浏览器当前呈现的页面源代码，做到可见即可爬。对于一些使用 JavaScript 动态渲染的页面来说，此种抓取方式非常有效。本课时就让我们来感受一下它的强大之处吧。</p>
<h3 data-nodeid="108796">准备工作</h3>
<p data-nodeid="108797">本课时以 Chrome 为例来讲解 Selenium 的用法。在开始之前，请确保已经正确安装好了 Chrome 浏览器并配置好了 ChromeDriver。另外，还需要正确安装好 Python 的 Selenium 库。</p>
<p data-nodeid="108798">安装过程可以参考：<a href="https://cuiqingcai.com/5135.html" data-nodeid="108994">https://cuiqingcai.com/5135.html</a>&nbsp;和&nbsp;<a href="https://cuiqingcai.com/5141.html" data-nodeid="108998">https://cuiqingcai.com/5141.html</a>。</p>
<h3 data-nodeid="108799">基本使用</h3>
<p data-nodeid="108800">准备工作做好之后，首先来看一下 Selenium 有一些怎样的功能。示例如下：</p>
<pre class="lang-python" data-nodeid="108801"><code data-language="python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver&nbsp;
<span class="hljs-keyword">from</span>&nbsp;selenium.webdriver.common.by&nbsp;<span class="hljs-keyword">import</span>&nbsp;By&nbsp;
<span class="hljs-keyword">from</span>&nbsp;selenium.webdriver.common.keys&nbsp;<span class="hljs-keyword">import</span>&nbsp;Keys&nbsp;
<span class="hljs-keyword">from</span>&nbsp;selenium.webdriver.support&nbsp;<span class="hljs-keyword">import</span>&nbsp;expected_conditions&nbsp;<span class="hljs-keyword">as</span>&nbsp;EC&nbsp;
<span class="hljs-keyword">from</span>&nbsp;selenium.webdriver.support.wait&nbsp;<span class="hljs-keyword">import</span>&nbsp;WebDriverWait
browser&nbsp;=&nbsp;webdriver.Chrome()&nbsp;
<span class="hljs-keyword">try</span>:
browser.get(<span class="hljs-string">'https://www.baidu.com'</span>)
input&nbsp;=&nbsp;browser.find_element_by_id(<span class="hljs-string">'kw'</span>)
input.send_keys(<span class="hljs-string">'Python'</span>)
input.send_keys(Keys.ENTER)
wait&nbsp;=&nbsp;WebDriverWait(browser,&nbsp;<span class="hljs-number">10</span>)
wait.until(EC.presence_of_element_located((By.ID,&nbsp;<span class="hljs-string">'content_left'</span>)))
print(browser.current_url)
print(browser.get_cookies())
print(browser.page_source)&nbsp;
<span class="hljs-keyword">finally</span>:
browser.close()
</code></pre>
<p data-nodeid="108802">运行代码后会自动弹出一个 Chrome 浏览器，浏览器会跳转到百度，然后在搜索框中输入 Python，接着跳转到搜索结果页，如图所示。</p>
<p data-nodeid="108803"><img src="https://s0.lgstatic.com/i/image3/M01/03/FC/Ciqah159oBaACo6HAAyBbpN_drw824.png" alt="" data-nodeid="109004"></p>
<p data-nodeid="108804">此时在控制台的输出结果如下：</p>
<pre class="lang-html" data-nodeid="108805"><code data-language="html">https://www.baidu.com/s?ie=utf-8&amp;f=8&amp;rsv_bp=0&amp;rsv_idx=1&amp;tn=baidu&amp;wd=Python&amp;rsv_pq=&nbsp;
c94d0df9000a72d0&amp;rsv_t=07099xvun1ZmC0bf6eQvygJ43IUTTUOl5FCJVPgwG2YREs70GplJjH2F%2BC
Q&amp;rqlang=cn&amp;rsv_enter=1&amp;rsv_sug3=6&amp;rsv_sug2=0&amp;inputT=87&amp;rsv_sug4=87&nbsp;
[{'secure':&nbsp;False,
&nbsp;'value':&nbsp;'B490B5EBF6F3CD402E515D22BCDA1598',&nbsp;
&nbsp;'domain':&nbsp;'.baidu.com',&nbsp;
&nbsp;'path':&nbsp;'/',
&nbsp;'httpOnly':&nbsp;False,&nbsp;
&nbsp;'name':&nbsp;'BDORZ',&nbsp;
&nbsp;'expiry':&nbsp;1491688071.707553},&nbsp;
&nbsp;
&nbsp;{'secure':&nbsp;False,&nbsp;
&nbsp;'value':&nbsp;'22473_1441_21084_17001',&nbsp;
&nbsp;'domain':&nbsp;'.baidu.com',&nbsp;
&nbsp;'path':&nbsp;'/',
&nbsp;'httpOnly':&nbsp;False,&nbsp;
&nbsp;'name':&nbsp;'H_PS_PSSID'},&nbsp;

&nbsp;{'secure':&nbsp;False,&nbsp;
&nbsp;'value':&nbsp;'12883875381399993259_00_0_I_R_2_0303_C02F_N_I_I_0',&nbsp;
&nbsp;'domain':&nbsp;'.www.baidu.com',
&nbsp;'path':&nbsp;'/',&nbsp;
&nbsp;'httpOnly':&nbsp;False,&nbsp;
&nbsp;'name':&nbsp;'__bsi',&nbsp;
&nbsp;'expiry':&nbsp;1491601676.69722}]
<span class="hljs-meta">&lt;!DOCTYPE&nbsp;<span class="hljs-meta-keyword">html</span>&gt;</span>
<span class="hljs-comment">&lt;!--STATUS&nbsp;OK--&gt;</span>...
<span class="hljs-tag">&lt;/<span class="hljs-name">html</span>&gt;</span>
</code></pre>
<p data-nodeid="108806">源代码过长，在此省略。可以看到，当前我们得到的 URL、Cookies 和源代码都是浏览器中的真实内容。</p>
<p data-nodeid="108807">所以说，如果用 Selenium 来驱动浏览器加载网页的话，就可以直接拿到 JavaScript 渲染的结果了，不用担心使用的是什么加密系统。</p>
<p data-nodeid="108808">下面来详细了解一下 Selenium 的用法。</p>
<h3 data-nodeid="108809">声明浏览器对象</h3>
<p data-nodeid="108810">Selenium 支持非常多的浏览器，如 Chrome、Firefox、Edge 等，还有 Android、BlackBerry 等手机端的浏览器。</p>
<p data-nodeid="108811">此外，我们可以用如下方式进行初始化：</p>
<pre class="lang-python" data-nodeid="108812"><code data-language="python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver
browser&nbsp;=&nbsp;webdriver.Chrome()&nbsp;
browser&nbsp;=&nbsp;webdriver.Firefox()&nbsp;
browser&nbsp;=&nbsp;webdriver.Edge()&nbsp;
browser&nbsp;=&nbsp;webdriver.Safari()
</code></pre>
<p data-nodeid="108813">这样就完成了浏览器对象的初始化并将其赋值为 browser 对象。接下来，我们要做的就是调用 browser 对象，让其执行各个动作以模拟浏览器操作。</p>
<h3 data-nodeid="108814">访问页面</h3>
<p data-nodeid="108815">我们可以用 get 方法来请求网页，只需要把参数传入链接 URL 即可。比如，这里用 get 方法访问淘宝，然后打印出源代码，代码如下：</p>
<pre class="lang-python" data-nodeid="108816"><code data-language="python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver
browser&nbsp;=&nbsp;webdriver.Chrome()&nbsp;
browser.get(<span class="hljs-string">'https://www.taobao.com'</span>)&nbsp;
print(browser.page_source)&nbsp;
browser.close()
</code></pre>
<p data-nodeid="108817">运行后会弹出 Chrome 浏览器并且自动访问淘宝，然后控制台会输出淘宝页面的源代码，随后浏览器关闭。</p>
<p data-nodeid="108818">通过这几行简单的代码，我们就可以驱动浏览器并获取网页源码，非常便捷。</p>
<h3 data-nodeid="108819">查找节点</h3>
<p data-nodeid="108820">Selenium 可以驱动浏览器完成各种操作，比如填充表单、模拟点击等。举个例子，当我们想要完成向某个输入框输入文字的操作时，首先需要知道这个输入框在哪，而 Selenium 提供了一系列查找节点的方法，我们可以用这些方法来获取想要的节点，以便执行下一步动作或者提取信息。</p>
<p data-nodeid="108821"><strong data-nodeid="109022">单个节点</strong></p>
<p data-nodeid="108822">当我们想要从淘宝页面中提取搜索框这个节点，首先要观察它的源代码，如图所示。</p>
<p data-nodeid="108823"><img src="https://s0.lgstatic.com/i/image3/M01/03/FC/Ciqah159oBaAN2vyAAKaGSF3EU4066.png" alt="" data-nodeid="109025"></p>
<p data-nodeid="108824">可以发现，它的 id 是 q，name 也是 q，此外还有许多其他属性。此时我们就可以用多种方式获取它了。比如，find_element_by_name 代表根据 name 值获取，find_element_by_id 则是根据 id 获取，另外，还有根据 XPath、CSS 选择器等获取的方式。</p>
<p data-nodeid="108825">我们用代码实现一下：</p>
<pre class="lang-python" data-nodeid="108826"><code data-language="python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver
browser&nbsp;=&nbsp;webdriver.Chrome()&nbsp;
browser.get(<span class="hljs-string">'https://www.taobao.com'</span>)&nbsp;
input_first&nbsp;=&nbsp;browser.find_element_by_id(<span class="hljs-string">'q'</span>)&nbsp;
input_second&nbsp;=&nbsp;browser.find_element_by_css_selector(<span class="hljs-string">'#q'</span>)&nbsp;
input_third&nbsp;=&nbsp;browser.find_element_by_xpath(<span class="hljs-string">'//*[@id="q"]'</span>)&nbsp;
print(input_first,&nbsp;input_second,&nbsp;input_third)&nbsp;
browser.close()
</code></pre>
<p data-nodeid="108827">这里我们使用 3 种方式获取输入框，分别是根据 id、CSS 选择器和 XPath 获取，它们返回的结果完全一致。运行结果如下：</p>
<pre class="lang-python" data-nodeid="108828"><code data-language="python">&lt;selenium.webdriver.remote.webelement.WebElement&nbsp;(session=<span class="hljs-string">"5e53d9e1c8646e44c14c1c2880d424af"</span>,
&nbsp;element=<span class="hljs-string">"0.5649563096161541-1"</span>)&gt;
&nbsp;
&nbsp;&lt;selenium.webdriver.remote.webelement.WebElement&nbsp;(session
&nbsp;=<span class="hljs-string">"5e53d9e1c8646e44c14c1c2880d424af"</span>,&nbsp;
&nbsp;element=<span class="hljs-string">"0.5649563096161541-1"</span>)&gt;
&nbsp;
&nbsp;&lt;selenium.webdriver.
&nbsp;remote.webelement.WebElement&nbsp;(session=<span class="hljs-string">"5e53d9e1c8646e44c14c1c2880d424af"</span>,&nbsp;
&nbsp;element=<span class="hljs-string">"0.5649563096161541-1"</span>)&gt;
</code></pre>
<p data-nodeid="108829">可以看到，这 3 个节点的类型是一致的，都是 WebElement。</p>
<p data-nodeid="108830">这里列出所有获取单个节点的方法：</p>
<pre class="lang-python" data-nodeid="108831"><code data-language="python">find_element_by_id&nbsp;
find_element_by_name&nbsp;
find_element_by_xpath&nbsp;
find_element_by_link_text&nbsp;
find_element_by_partial_link_text&nbsp;
find_element_by_tag_name&nbsp;
find_element_by_class_name&nbsp;
find_element_by_css_selector
</code></pre>
<p data-nodeid="108832">另外，Selenium 还提供了 find_element 这个通用方法，它需要传入两个参数：查找方式 By 和值。实际上，find_element 就是 find_element_by_id 这种方法的通用函数版本，比如 find_element_by_id(id) 就等价于 find_element(By.ID, id)，二者得到的结果完全一致。我们用代码实现一下：</p>
<pre class="lang-python" data-nodeid="108833"><code data-language="python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver&nbsp;
<span class="hljs-keyword">from</span>&nbsp;selenium.webdriver.common.by&nbsp;<span class="hljs-keyword">import</span>&nbsp;By
browser&nbsp;=&nbsp;webdriver.Chrome()&nbsp;
browser.get(<span class="hljs-string">'https://www.taobao.com'</span>)&nbsp;
input_first&nbsp;=&nbsp;browser.find_element(By.ID,&nbsp;<span class="hljs-string">'q'</span>)&nbsp;
print(input_first)&nbsp;
browser.close()
</code></pre>
<p data-nodeid="108834">这种查找方式的功能和上面列举的查找函数完全一致，不过参数更加灵活。</p>
<p data-nodeid="108835"><strong data-nodeid="109066">多个节点</strong></p>
<p data-nodeid="108836">如果在网页中只查找一个目标，那么完全可以用 find_element 方法。但如果有多个节点需要查找，再用 find_element 方法，就只能得到第 1 个节点了。如果要查找所有满足条件的节点，需要用 find_elements 这样的方法。<strong data-nodeid="109077">注意，在这个方法的名称中，element 多了一个 s，注意区分。</strong></p>
<p data-nodeid="108837">举个例子，假如你要查找淘宝左侧导航条的所有条目，就可以这样来实现：</p>
<pre class="lang-python" data-nodeid="108838"><code data-language="python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver&nbsp;
browser&nbsp;=&nbsp;webdriver.Chrome()&nbsp;
browser.get(<span class="hljs-string">'https://www.taobao.com'</span>)&nbsp;
lis&nbsp;=&nbsp;browser.find_elements_by_css_selector(<span class="hljs-string">'.service-bd&nbsp;li'</span>)&nbsp;
print(lis)&nbsp;
browser.close()
</code></pre>
<p data-nodeid="108839">运行结果如下：</p>
<pre class="lang-python" data-nodeid="108840"><code data-language="python">[&lt;selenium.webdriver.remote.webelement.WebElement&nbsp;
(session=<span class="hljs-string">"c26290835d4457ebf7d96bfab3740d19"</span>,&nbsp;element=<span class="hljs-string">"0.09221044033125603-1"</span>)&gt;,
&nbsp;
&lt;selenium.webdriver.remote.webelement.WebElement&nbsp;
(session=<span class="hljs-string">"c26290835d4457ebf7d96bfab3740d19"</span>,&nbsp;element=<span class="hljs-string">"0.09221044033125603-2"</span>)&gt;,

&lt;selenium.webdriver.remote.webelement.WebElement&nbsp;
(session=<span class="hljs-string">"c26290835d4457ebf7d96bfab3740d19"</span>,&nbsp;element=<span class="hljs-string">"0.09221044033125603-3"</span>)&gt;...

&lt;selenium.webdriver.remote.webelement.WebElement&nbsp;
(session=<span class="hljs-string">"c26290835d4457ebf7d96bfab3740d19"</span>,&nbsp;element=<span class="hljs-string">"0.09221044033125603-16"</span>)&gt;]
</code></pre>
<p data-nodeid="108841">这里简化了输出结果，中间部分省略。</p>
<p data-nodeid="108842">可以看到，得到的内容变成了列表类型，列表中的每个节点都是 WebElement 类型。</p>
<p data-nodeid="108843">也就是说，如果我们用 find_element 方法，只能获取匹配的第一个节点，结果是 WebElement 类型。如果用 find_elements 方法，则结果是列表类型，列表中的每个节点是 WebElement 类型。</p>
<p data-nodeid="108844">这里列出所有获取多个节点的方法：</p>
<pre class="lang-python" data-nodeid="108845"><code data-language="python">find_elements_by_id&nbsp;
find_elements_by_name&nbsp;
find_elements_by_xpath&nbsp;
find_elements_by_link_text&nbsp;
find_elements_by_partial_link_text&nbsp;
find_elements_by_tag_name&nbsp;
find_elements_by_class_name&nbsp;
find_elements_by_css_selector
</code></pre>
<p data-nodeid="108846">当然，我们也可以直接用 find_elements 方法来选择，这时可以这样写：</p>
<pre class="lang-python" data-nodeid="108847"><code data-language="python">lis&nbsp;=&nbsp;browser.find_elements(By.CSS_SELECTOR,&nbsp;<span class="hljs-string">'.service-bd&nbsp;li'</span>)
</code></pre>
<p data-nodeid="108848">结果是完全一致的。</p>
<h3 data-nodeid="108849">节点交互</h3>
<p data-nodeid="108850">Selenium 可以驱动浏览器来执行一些操作，或者说可以让浏览器模拟执行一些动作。比较常见的用法有：输入文字时用 send_keys 方法，清空文字时用 clear 方法，点击按钮时用 click 方法。示例如下：</p>
<pre class="lang-python" data-nodeid="108851"><code data-language="python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver&nbsp;
<span class="hljs-keyword">import</span>&nbsp;time&nbsp;
browser&nbsp;=&nbsp;webdriver.Chrome()&nbsp;
browser.get(<span class="hljs-string">'https://www.taobao.com'</span>)&nbsp;
input&nbsp;=&nbsp;browser.find_element_by_id(<span class="hljs-string">'q'</span>)&nbsp;
input.send_keys(<span class="hljs-string">'iPhone'</span>)&nbsp;
time.sleep(<span class="hljs-number">1</span>)&nbsp;
input.clear()&nbsp;
input.send_keys(<span class="hljs-string">'iPad'</span>)&nbsp;
button&nbsp;=&nbsp;browser.find_element_by_class_name(<span class="hljs-string">'btn-search'</span>)&nbsp;
button.click()
</code></pre>
<p data-nodeid="108852">这里首先驱动浏览器打开淘宝，用 find_element_by_id 方法获取输入框，然后用 send_keys 方法输入 iPhone 文字，等待一秒后用 clear 方法清空输入框，接着再次调用 send_keys 方法输入 iPad 文字，之后再用 find_element_by_class_name 方法获取搜索按钮，最后调用 click 方法完成搜索动作。</p>
<p data-nodeid="108853">通过上面的方法，我们就完成了一些常见节点的动作操作，更多的操作可以参见官方文档的交互动作介绍 ：<a href="http://selenium-python.readthedocs.io/api.html#module-selenium.webdriver.remote.webelement" data-nodeid="109118">http://selenium-python.readthedocs.io/api.html#module-selenium.webdriver.remote.webelement</a>。</p>
<h3 data-nodeid="108854">动作链</h3>
<p data-nodeid="108855">在上面的实例中，一些交互动作都是针对某个节点执行的。比如，对于输入框，我们调用它的输入文字和清空文字方法；对于按钮，我们调用它的点击方法。其实，还有另外一些操作，它们没有特定的执行对象，比如鼠标拖拽、键盘按键等，这些动作用另一种方式来执行，那就是动作链。</p>
<p data-nodeid="108856">比如，现在我要实现一个节点的拖拽操作，将某个节点从一处拖拽到另外一处，可以这样实现：</p>
<pre class="lang-python" data-nodeid="108857"><code data-language="python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver&nbsp;
<span class="hljs-keyword">from</span>&nbsp;selenium.webdriver&nbsp;<span class="hljs-keyword">import</span>&nbsp;ActionChains&nbsp;
browser&nbsp;=&nbsp;webdriver.Chrome()&nbsp;
url&nbsp;=&nbsp;<span class="hljs-string">'http://www.runoob.com/try/try.php?filename=jqueryui-api-droppable'</span>&nbsp;
browser.get(url)&nbsp;
browser.switch_to.frame(<span class="hljs-string">'iframeResult'</span>)&nbsp;
source&nbsp;=&nbsp;browser.find_element_by_css_selector(<span class="hljs-string">'#draggable'</span>)&nbsp;
target&nbsp;=&nbsp;browser.find_element_by_css_selector(<span class="hljs-string">'#droppable'</span>)&nbsp;
actions&nbsp;=&nbsp;ActionChains(browser)&nbsp;
actions.drag_and_drop(source,&nbsp;target)&nbsp;
actions.perform()
</code></pre>
<p data-nodeid="108858">首先，打开网页中的一个拖拽实例，依次选中要拖拽的节点和拖拽到的目标节点，接着声明 ActionChains 对象并将其赋值为 actions 变量，然后通过调用 actions 变量的 drag_and_drop 方法，再调用 perform 方法执行动作，此时就完成了拖拽操作，如图所示：</p>
<p data-nodeid="108859"><img src="https://s0.lgstatic.com/i/image3/M01/7D/12/Cgq2xl59oBaAebZXAACbaBgWl4k530.png" alt="" data-nodeid="109129"></p>
<p data-nodeid="108860">拖拽前页面<br>
<img src="https://s0.lgstatic.com/i/image3/M01/03/FC/Ciqah159oBeAZICwAACKn0bkfog611.png" alt="" data-nodeid="109133"></p>
<p data-nodeid="108861">拖拽后页面</p>
<p data-nodeid="108862">以上两图分别为在拖拽前和拖拽后的结果。</p>
<p data-nodeid="108863">更多的动作链操作可以参考官方文档的动作链介绍：<a href="http://selenium-python.readthedocs.io/api.html#module-selenium.webdriver.common.action_chains" data-nodeid="109141">http://selenium-python.readthedocs.io/api.html#module-selenium.webdriver.common.action_chains</a>。</p>
<h3 data-nodeid="108864">执行 JavaScript</h3>
<p data-nodeid="108865">Selenium API 并没有提供实现某些操作的方法，比如，下拉进度条。但它可以直接模拟运行 JavaScript，此时使用 execute_script 方法即可实现，代码如下：</p>
<pre class="lang-js" data-nodeid="108866"><code data-language="js"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver&nbsp;
browser&nbsp;=&nbsp;webdriver.Chrome()&nbsp;
browser.get(<span class="hljs-string">'https://www.zhihu.com/explore'</span>)&nbsp;
browser.execute_script(<span class="hljs-string">'window.scrollTo(0,&nbsp;document.body.scrollHeight)'</span>)&nbsp;
browser.execute_script(<span class="hljs-string">'alert("To&nbsp;Bottom")'</span>)
</code></pre>
<p data-nodeid="108867">这里利用 execute_script 方法将进度条下拉到最底部，然后弹出 alert 提示框。</p>
<p data-nodeid="108868">有了这个方法，基本上 API 没有提供的所有功能都可以用执行 JavaScript 的方式来实现了。</p>
<h3 data-nodeid="108869">获取节点信息</h3>
<p data-nodeid="108870">前面说过，通过 page_source 属性可以获取网页的源代码，接着就可以使用解析库（如正则表达式、Beautiful Soup、pyquery 等）来提取信息了。</p>
<p data-nodeid="108871">不过，既然 Selenium 已经提供了选择节点的方法，并且返回的是 WebElement 类型，那么它也有相关的方法和属性来直接提取节点信息，如属性、文本等。这样的话，我们就可以不用通过解析源代码来提取信息了，非常方便。</p>
<p data-nodeid="108872">接下来，我们就来看看可以通过怎样的方式来获取节点信息吧。</p>
<p data-nodeid="108873"><strong data-nodeid="109160">获取属性</strong></p>
<p data-nodeid="108874">我们可以使用 get_attribute 方法来获取节点的属性，但是前提是得先选中这个节点，示例如下：</p>
<pre class="lang-python" data-nodeid="111565"><code data-language="python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver&nbsp;
browser&nbsp;=&nbsp;webdriver.Chrome()&nbsp;
url&nbsp;=&nbsp;<span class="hljs-string">'https://dynamic2.scrape.center/'</span>&nbsp;
browser.get(url)&nbsp;
logo&nbsp;=&nbsp;browser.find_element_by_class_name(<span class="hljs-string">'logo-image'</span>)
print(logo)&nbsp;
print(logo.get_attribute(<span class="hljs-string">'src'</span>))
</code></pre>

<p data-nodeid="108876">运行之后，程序便会驱动浏览器打开该页面，然后获取 class 为 logo-image 的节点，最后打印出它的 src 属性。</p>
<p data-nodeid="108877">控制台的输出结果如下：</p>
<pre class="lang-python" data-nodeid="113779"><code data-language="python">&lt;selenium.webdriver.remote.webelement.WebElement&nbsp;
(session=<span class="hljs-string">"7f4745d35a104759239b53f68a6f27d0"</span>,&nbsp;
element=<span class="hljs-string">"cd7c72b4-4920-47ed-91c5-ea06601dc509"</span>)&gt;&nbsp;
https://dynamic2.scrape.center/img/logo.a508a8f0.png
</code></pre>


<p data-nodeid="108879">通过 get_attribute 方法，我们只需要传入想要获取的属性名，就可以得到它的值了。</p>
<p data-nodeid="108880"><strong data-nodeid="109172">获取文本值</strong></p>
<p data-nodeid="108881">每个 WebElement 节点都有 text 属性，直接调用这个属性就可以得到节点内部的文本信息，这相当于 pyquery 的 text 方法，示例如下：</p>
<pre class="lang-python" data-nodeid="114886"><code data-language="python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver&nbsp;
browser&nbsp;=&nbsp;webdriver.Chrome()
url&nbsp;=&nbsp;<span class="hljs-string">'https://dynamic2.scrape.center/'</span>&nbsp;
browser.get(url)
input&nbsp;=&nbsp;browser.find_element_by_class_name(<span class="hljs-string">'logo-title'</span>)&nbsp;
print(input.text)
</code></pre>

<p data-nodeid="108883">这里依然先打开页面，然后获取 class 为 logo-title 这个节点，再将其文本值打印出来。</p>
<p data-nodeid="108884">控制台的输出结果如下：</p>
<pre data-nodeid="108885"><code>Scrape
</code></pre>
<h3 data-nodeid="108886">获取 ID、位置、标签名、大小</h3>
<p data-nodeid="108887">另外，WebElement 节点还有一些其他属性，比如 id 属性可以获取节点 id，location 属性可以获取该节点在页面中的相对位置，tag_name 属性可以获取标签名称，size 属性可以获取节点的大小，也就是宽高，这些属性有时候还是很有用的。示例如下：</p>
<pre class="lang-python" data-nodeid="115993"><code data-language="python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver&nbsp;
browser&nbsp;=&nbsp;webdriver.Chrome()&nbsp;
url&nbsp;=&nbsp;<span class="hljs-string">'https://dynamic2.scrape.center/'</span>&nbsp;
browser.get(url)&nbsp;
input&nbsp;=&nbsp;browser.find_element_by_class_name(<span class="hljs-string">'logo-title'</span>)&nbsp;
print(input.id)&nbsp;
print(input.location)&nbsp;
print(input.tag_name)&nbsp;
print(input.size)
</code></pre>

<p data-nodeid="108889">这里首先获得 class 为 logo-title 这个节点，然后调用其 id、location、tag_name、size 属性来获取对应的属性值。</p>
<h3 data-nodeid="108890">切换 Frame</h3>
<p data-nodeid="108891">我们知道网页中有一种节点叫作 iframe，也就是子 Frame，相当于页面的子页面，它的结构和外部网页的结构完全一致。Selenium 打开页面后，默认是在父级 Frame 里面操作，而此时如果页面中还有子 Frame，Selenium 是不能获取到子 Frame 里面的节点的。这时就需要使用 switch_to.frame 方法来切换 Frame。示例如下：</p>
<pre class="lang-python" data-nodeid="108892"><code data-language="python"><span class="hljs-keyword">import</span>&nbsp;time&nbsp;
<span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver&nbsp;
<span class="hljs-keyword">from</span>&nbsp;selenium.common.exceptions&nbsp;<span class="hljs-keyword">import</span>&nbsp;NoSuchElementException&nbsp;
browser&nbsp;=&nbsp;webdriver.Chrome()
url&nbsp;=&nbsp;<span class="hljs-string">'http://www.runoob.com/try/try.php?filename=jqueryui-api-droppable'</span>&nbsp;
browser.get(url)&nbsp;
browser.switch_to.frame(<span class="hljs-string">'iframeResult'</span>)
<span class="hljs-keyword">try</span>:
&nbsp;&nbsp;&nbsp;&nbsp;logo&nbsp;=&nbsp;browser.find_element_by_class_name(<span class="hljs-string">'logo'</span>)&nbsp;
<span class="hljs-keyword">except</span>&nbsp;NoSuchElementException:
&nbsp;&nbsp;&nbsp;&nbsp;print(<span class="hljs-string">'NO&nbsp;LOGO'</span>)&nbsp;
browser.switch_to.parent_frame()&nbsp;
logo&nbsp;=&nbsp;browser.find_element_by_class_name(<span class="hljs-string">'logo'</span>)
print(logo)&nbsp;
print(logo.text)
</code></pre>
<p data-nodeid="108893">控制台输出：</p>
<pre class="lang-pytohn" data-nodeid="108894"><code data-language="pytohn">NO&nbsp;LOGO&nbsp;
&lt;selenium.webdriver.remote.webelement.WebElement
(session="4bb8ac03ced4ecbdefef03ffdc0e4ccd",&nbsp;
element="0.13792611320464965-2")&gt;&nbsp;
RUNOOB.COM
</code></pre>
<p data-nodeid="108895">这里还是以前面演示动作链操作的网页为例，首先通过 switch_to.frame 方法切换到子 Frame 里面，然后尝试获取子 Frame 里的 logo 节点（这是不能找到的），如果找不到的话，就会抛出 NoSuchElementException 异常，异常被捕捉之后，就会输出 NO LOGO。接下来，我们需要重新切换回父级 Frame，然后再次重新获取节点，发现此时可以成功获取了。</p>
<p data-nodeid="108896">所以，当页面中包含子 Frame 时，如果想获取子 Frame 中的节点，需要先调用 switch_to.frame 方法切换到对应的 Frame，然后再进行操作。</p>
<h3 data-nodeid="108897">延时等待</h3>
<p data-nodeid="108898">在 Selenium 中，get 方法会在网页框架加载结束后结束执行，此时如果获取 page_source，可能并不是浏览器完全加载完成的页面，如果某些页面有额外的 Ajax 请求，我们在网页源代码中也不一定能成功获取到。所以，这里需要延时等待一定时间，确保节点已经加载出来。</p>
<p data-nodeid="108899">这里等待的方式有两种：一种是隐式等待，一种是显式等待。</p>
<p data-nodeid="108900"><strong data-nodeid="109202">隐式等待</strong></p>
<p data-nodeid="108901">当使用隐式等待执行测试的时候，如果 Selenium 没有在 DOM 中找到节点，将继续等待，超出设定时间后，则抛出找不到节点的异常。换句话说，隐式等待可以在我们查找节点而节点并没有立即出现的时候，等待一段时间再查找 DOM，默认的时间是 0。示例如下：</p>
<pre class="lang-python" data-nodeid="118207"><code data-language="python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver&nbsp;
browser&nbsp;=&nbsp;webdriver.Chrome()&nbsp;
browser.implicitly_wait(<span class="hljs-number">10</span>)&nbsp;
browser.get(<span class="hljs-string">'https://dynamic2.scrape.center/'</span>)&nbsp;
input&nbsp;=&nbsp;browser.find_element_by_class_name(<span class="hljs-string">'logo-image'</span>)&nbsp;
print(input)
</code></pre>


<p data-nodeid="108903">在这里我们用 implicitly_wait 方法实现了隐式等待。</p>
<p data-nodeid="108904"><strong data-nodeid="109210">显式等待</strong></p>
<p data-nodeid="108905">隐式等待的效果其实并没有那么好，因为我们只规定了一个固定时间，而页面的加载时间会受到网络条件的影响。</p>
<p data-nodeid="108906">这里还有一种更合适的显式等待方法，它指定要查找的节点，然后指定一个最长等待时间。如果在规定时间内加载出来了这个节点，就返回查找的节点；如果到了规定时间依然没有加载出该节点，则抛出超时异常。示例如下：</p>
<pre class="lang-python" data-nodeid="108907"><code data-language="python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver&nbsp;
<span class="hljs-keyword">from</span>&nbsp;selenium.webdriver.common.by&nbsp;<span class="hljs-keyword">import</span>&nbsp;By&nbsp;
<span class="hljs-keyword">from</span>&nbsp;selenium.webdriver.support.ui&nbsp;<span class="hljs-keyword">import</span>&nbsp;WebDriverWait&nbsp;
<span class="hljs-keyword">from</span>&nbsp;selenium.webdriver.support&nbsp;<span class="hljs-keyword">import</span>&nbsp;expected_conditions&nbsp;<span class="hljs-keyword">as</span>&nbsp;EC&nbsp;
browser&nbsp;=&nbsp;webdriver.Chrome()&nbsp;
browser.get(<span class="hljs-string">'https://www.taobao.com/'</span>)&nbsp;
wait&nbsp;=&nbsp;WebDriverWait(browser,&nbsp;<span class="hljs-number">10</span>)&nbsp;
input&nbsp;=&nbsp;wait.until(EC.presence_of_element_located((By.ID,&nbsp;<span class="hljs-string">'q'</span>)))&nbsp;
button&nbsp;=&nbsp;&nbsp;wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR,&nbsp;<span class="hljs-string">'.btn-search'</span>)))&nbsp;
print(input,&nbsp;button)
</code></pre>
<p data-nodeid="108908">这里首先引入 WebDriverWait 这个对象，指定最长等待时间，然后调用它的 until() 方法，传入要等待条件 expected_conditions。比如，这里传入了 presence_of_element_located 这个条件，代表节点出现，其参数是节点的定位元组，也就是 ID 为 q 的节点搜索框。</p>
<p data-nodeid="108909">这样做的效果就是，在 10 秒内如果 ID 为 q 的节点（即搜索框）成功加载出来，就返回该节点；如果超过 10 秒还没有加载出来，就抛出异常。</p>
<p data-nodeid="108910">对于按钮，我们可以更改一下等待条件，比如改为 element_to_be_clickable，也就是可点击，所以查找按钮时先查找 CSS 选择器为.btn-search 的按钮，如果 10 秒内它是可点击的，也就代表它成功加载出来了，就会返回这个按钮节点；如果超过 10 秒还不可点击，也就是没有加载出来，就抛出异常。</p>
<p data-nodeid="108911">现在我们运行代码，它在网速较佳的情况下是可以成功加载出来的。</p>
<p data-nodeid="108912">控制台的输出如下：</p>
<pre class="lang-python" data-nodeid="108913"><code data-language="python">&lt;selenium.webdriver.remote.webelement.WebElement&nbsp;
(session=<span class="hljs-string">"07dd2fbc2d5b1ce40e82b9754aba8fa8"</span>,&nbsp;
element=<span class="hljs-string">"0.5642646294074107-1"</span>)&gt;
&lt;selenium.webdriver.remote.webelement.WebElement&nbsp;
(session=<span class="hljs-string">"07dd2fbc2d5b1ce40e82b9754aba8fa8"</span>,&nbsp;
element=<span class="hljs-string">"0.5642646294074107-2"</span>)&gt;
</code></pre>
<p data-nodeid="108914">可以看到，控制台成功输出了两个节点，它们都是 WebElement 类型。</p>
<p data-nodeid="108915">如果网络有问题，10 秒内没有成功加载，那就抛出 TimeoutException 异常，此时控制台的输出如下：</p>
<pre class="lang-python" data-nodeid="108916"><code data-language="python">TimeoutException&nbsp;Traceback&nbsp;(most&nbsp;recent&nbsp;call&nbsp;last)&nbsp;
&lt;ipython-input-4-f3d73973b223&gt;&nbsp;in&nbsp;&lt;module&gt;()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7&nbsp;browser.get('https://www.taobao.com/')
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;8&nbsp;wait&nbsp;=&nbsp;WebDriverWait(browser,&nbsp;10)&nbsp;
----&gt;&nbsp;9&nbsp;input&nbsp;=&nbsp;wait.until(EC.presence_of_element_located((By.ID,&nbsp;'q')))
</code></pre>
<p data-nodeid="108917">关于等待条件，其实还有很多，比如判断标题内容，判断某个节点内是否出现了某文字等。下表我列出了所有的等待条件。</p>
<p data-nodeid="108918"><img src="https://s0.lgstatic.com/i/image3/M01/04/3A/Ciqah1596FyAIAjtAAECe0Jujuw745.png" alt="" data-nodeid="109236"></p>
<p data-nodeid="108919"><img src="https://s0.lgstatic.com/i/image3/M01/04/3A/Ciqah1596R2Af973AAEiFfxC3E4161.png" alt="" data-nodeid="109238"></p>
<p data-nodeid="108920">更多详细的等待条件的参数及用法介绍可以参考官方文档：<a href="http://selenium-python.readthedocs.io/api.html#module-selenium.webdriver.support.expected_conditions" data-nodeid="109244">http://selenium-python.readthedocs.io/api.html#module-selenium.webdriver.support.expected_conditions</a>。</p>
<h3 data-nodeid="108921">前进后退</h3>
<p data-nodeid="108922">平常我们使用浏览器时都有前进和后退功能，Selenium 也可以完成这个操作，它使用 back 方法后退，使用 forward 方法前进。示例如下：</p>
<pre class="lang-python" data-nodeid="108923"><code data-language="python"><span class="hljs-keyword">import</span>&nbsp;time&nbsp;
<span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver&nbsp;
browser&nbsp;=&nbsp;webdriver.Chrome()&nbsp;
browser.get(<span class="hljs-string">'https://www.baidu.com/'</span>)&nbsp;
browser.get(<span class="hljs-string">'https://www.taobao.com/'</span>)&nbsp;
browser.get(<span class="hljs-string">'https://www.python.org/'</span>)&nbsp;
browser.back()&nbsp;
time.sleep(<span class="hljs-number">1</span>)&nbsp;
browser.forward()&nbsp;
browser.close()
</code></pre>
<p data-nodeid="108924">这里我们连续访问 3 个页面，然后调用 back &nbsp;方法回到第 2 个页面，接下来再调用 forward 方法又可以前进到第 3 个页面。</p>
<h3 data-nodeid="108925">Cookies</h3>
<p data-nodeid="108926">使用 Selenium，还可以方便地对 Cookies 进行操作，例如获取、添加、删除 Cookies 等。示例如下：</p>
<pre class="lang-python" data-nodeid="108927"><code data-language="python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver&nbsp;
browser&nbsp;=&nbsp;webdriver.Chrome()&nbsp;
browser.get(<span class="hljs-string">'https://www.zhihu.com/explore'</span>)&nbsp;
print(browser.get_cookies())&nbsp;
browser.add_cookie({<span class="hljs-string">'name'</span>:&nbsp;<span class="hljs-string">'name'</span>,&nbsp;<span class="hljs-string">'domain'</span>:&nbsp;<span class="hljs-string">'www.zhihu.com'</span>,&nbsp;<span class="hljs-string">'value'</span>:&nbsp;<span class="hljs-string">'germey'</span>})&nbsp;
print(browser.get_cookies())&nbsp;
browser.delete_all_cookies()&nbsp;
print(browser.get_cookies())
</code></pre>
<p data-nodeid="108928">首先，我们访问知乎，加载完成后，浏览器实际上已经生成 Cookies 了。接着，调用 get_cookies 方法获取所有的 Cookies。然后，我们再添加一个 Cookie，这里传入一个字典，有 name、domain 和 value 等内容。接下来，再次获取所有的 Cookies，可以发现，结果会多出这一项新加的 Cookie。最后，调用 delete_all_cookies 方法删除所有的 Cookies。再重新获取，发现结果就为空了。</p>
<p data-nodeid="108929">控制台的输出如下：</p>
<pre class="lang-python" data-nodeid="108930"><code data-language="python">[{<span class="hljs-string">'secure'</span>:&nbsp;<span class="hljs-literal">False</span>,&nbsp;
<span class="hljs-string">'value'</span>:&nbsp;<span class="hljs-string">'"NGM0ZTM5NDAwMWEyNDQwNDk5ODlkZWY3OTkxY2I0NDY=|1491604091|236e34290a6f407bfbb517888849ea509ac366d0"'</span>,&nbsp;
<span class="hljs-string">'domain'</span>:&nbsp;<span class="hljs-string">'.zhihu.com'</span>,
<span class="hljs-string">'path'</span>:&nbsp;<span class="hljs-string">'/'</span>,&nbsp;
<span class="hljs-string">'httpOnly'</span>:&nbsp;<span class="hljs-literal">False</span>,&nbsp;
<span class="hljs-string">'name'</span>:&nbsp;<span class="hljs-string">'l_cap_id'</span>,&nbsp;
<span class="hljs-string">'expiry'</span>:&nbsp;<span class="hljs-number">1494196091.403418</span>},...]&nbsp;
[{<span class="hljs-string">'secure'</span>:&nbsp;<span class="hljs-literal">False</span>,&nbsp;
<span class="hljs-string">'value'</span>:&nbsp;<span class="hljs-string">'germey'</span>,&nbsp;
<span class="hljs-string">'domain'</span>:&nbsp;<span class="hljs-string">'.www.zhihu.com'</span>,&nbsp;
<span class="hljs-string">'path'</span>:&nbsp;<span class="hljs-string">'/'</span>,&nbsp;
<span class="hljs-string">'httpOnly'</span>:&nbsp;<span class="hljs-literal">False</span>,&nbsp;
<span class="hljs-string">'name'</span>:&nbsp;<span class="hljs-string">'name'</span>},&nbsp;
{<span class="hljs-string">'secure'</span>:&nbsp;<span class="hljs-literal">False</span>,&nbsp;
<span class="hljs-string">'value'</span>:&nbsp;<span class="hljs-string">'"NGM0ZTM5NDAwMWEyNDQwNDk5ODlkZWY3OTkxY2I0NDY=|1491604091|236e34290a6f407bfbb517888849ea509ac366d0"'</span>,&nbsp;
<span class="hljs-string">'domain'</span>:&nbsp;<span class="hljs-string">'.zhihu.com'</span>,&nbsp;
<span class="hljs-string">'path'</span>:<span class="hljs-string">'/'</span>,&nbsp;
<span class="hljs-string">'httpOnly'</span>:&nbsp;<span class="hljs-literal">False</span>,&nbsp;
<span class="hljs-string">'name'</span>:&nbsp;<span class="hljs-string">'l_cap_id'</span>,&nbsp;
<span class="hljs-string">'expiry'</span>:&nbsp;<span class="hljs-number">1494196091.403418</span>},&nbsp;...]&nbsp;
[]
</code></pre>
<p data-nodeid="108931">通过以上方法来操作 Cookies 还是非常方便的。</p>
<h3 data-nodeid="108932">选项卡管理</h3>
<p data-nodeid="108933">在访问网页的时候，我们通常会开启多个选项卡。在 Selenium 中，我们也可以对选项卡进行操作。示例如下：</p>
<pre class="lang-python" data-nodeid="108934"><code data-language="python"><span class="hljs-keyword">import</span>&nbsp;time&nbsp;
<span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver&nbsp;
browser&nbsp;=&nbsp;webdriver.Chrome()&nbsp;
browser.get(<span class="hljs-string">'https://www.baidu.com'</span>)&nbsp;
browser.execute_script(<span class="hljs-string">'window.open()'</span>)&nbsp;
print(browser.window_handles)&nbsp;
browser.switch_to.window(browser.window_handles[<span class="hljs-number">1</span>])
browser.get(<span class="hljs-string">'https://www.taobao.com'</span>)&nbsp;
time.sleep(<span class="hljs-number">1</span>)&nbsp;
browser.switch_to.window(browser.window_handles[<span class="hljs-number">0</span>])&nbsp;
browser.get(<span class="hljs-string">'https://python.org'</span>
</code></pre>
<p data-nodeid="108935">控制台输出如下：</p>
<pre class="lang-python" data-nodeid="108936"><code data-language="python">[<span class="hljs-string">'CDwindow-4f58e3a7-7167-4587-bedf-9cd8c867f435'</span>,&nbsp;<span class="hljs-string">'CDwindow-6e05f076-6d77-453a-a36c-32baacc447df'</span>]
</code></pre>
<p data-nodeid="108937">首先访问百度，然后调用 execute_script 方法，这里我们传入 window.open 这个 JavaScript 语句新开启一个选项卡，然后切换到该选项卡，调用 window_handles 属性获取当前开启的所有选项卡，后面的参数代表返回选项卡的代号列表。要想切换选项卡，只需要调用 switch_to.window 方法即可，其中的参数是选项卡的代号。这里我们将第 2 个选项卡代号传入，即跳转到第 2 个选项卡，接下来在第 2 个选项卡下打开一个新页面，如果你想要切换回第 2 个选项卡，只需要重新调用 switch_to.window 方法，再执行其他操作即可。</p>
<h3 data-nodeid="108938">异常处理</h3>
<p data-nodeid="108939">在使用 Selenium 的过程中，难免会遇到一些异常，例如超时、节点未找到等错误，一旦出现此类错误，程序便不会继续运行了。这里我们可以使用 try except 语句来捕获各种异常。</p>
<p data-nodeid="108940">首先，演示一下节点未找到的异常，示例如下：</p>
<pre class="lang-python" data-nodeid="108941"><code data-language="python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver&nbsp;
browser&nbsp;=&nbsp;webdriver.Chrome()&nbsp;
browser.get(<span class="hljs-string">'https://www.baidu.com'</span>)&nbsp;
browser.find_element_by_id(<span class="hljs-string">'hello'</span>)
</code></pre>
<p data-nodeid="108942">这里我们首先打开百度页面，然后尝试选择一个并不存在的节点，此时就会遇到异常。</p>
<p data-nodeid="108943">运行之后控制台的输出如下：</p>
<pre class="lang-python" data-nodeid="108944"><code data-language="python">NoSuchElementException&nbsp;Traceback&nbsp;(most&nbsp;recent&nbsp;call&nbsp;last)&nbsp;
&lt;ipython-input-23-978945848a1b&gt;&nbsp;in&nbsp;&lt;module&gt;()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3&nbsp;browser&nbsp;=&nbsp;webdriver.Chrome()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4&nbsp;browser.get&nbsp;('https://www.baidu.com')
----&gt;&nbsp;5&nbsp;browser.find_element_by_id('hello')
</code></pre>
<p data-nodeid="108945">可以看到，这里抛出了 NoSuchElementException 异常，通常代表节点未找到。为了防止程序遇到异常而中断，我们需要捕获这些异常，示例如下：</p>
<pre class="lang-python" data-nodeid="108946"><code data-language="python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver&nbsp;
<span class="hljs-keyword">from</span>&nbsp;selenium.common.exceptions&nbsp;<span class="hljs-keyword">import</span>&nbsp;TimeoutException,&nbsp;
NoSuchElementException&nbsp;
browser&nbsp;=&nbsp;webdriver.Chrome()
<span class="hljs-keyword">try</span>:
&nbsp;&nbsp;&nbsp;&nbsp;browser.get(<span class="hljs-string">'https://www.baidu.com'</span>)&nbsp;
<span class="hljs-keyword">except</span>&nbsp;TimeoutException:
&nbsp;&nbsp;&nbsp;&nbsp;print(<span class="hljs-string">'Time&nbsp;Out'</span>)&nbsp;
<span class="hljs-keyword">try</span>:
&nbsp;&nbsp;&nbsp;&nbsp;browser.find_element_by_id(<span class="hljs-string">'hello'</span>)&nbsp;
<span class="hljs-keyword">except</span>&nbsp;NoSuchElementException:
&nbsp;&nbsp;&nbsp;&nbsp;print(<span class="hljs-string">'No&nbsp;Element'</span>)&nbsp;
<span class="hljs-keyword">finally</span>:
&nbsp;&nbsp;&nbsp;&nbsp;browser.close()
</code></pre>
<p data-nodeid="108947">这里我们使用 try except 来捕获各类异常。比如，我们用 find_element_by_id 查找节点的方法捕获 NoSuchElementException 异常，这样一旦出现这样的错误，就进行异常处理，程序也不会中断了。</p>
<p data-nodeid="108948">控制台的输出如下：</p>
<pre data-nodeid="108949"><code>No&nbsp;Element
</code></pre>
<p data-nodeid="108950">关于更多的异常类，可以参考官方文档：：<a href="http://selenium-python.readthedocs.io/api.html#module-selenium.common.exceptions" data-nodeid="109289">http://selenium-python.readthedocs.io/api.html#module-selenium.common.exceptions</a>。</p>
<h3 data-nodeid="108951">反屏蔽</h3>
<p data-nodeid="108952">现在很多网站都加上了对 Selenium 的检测，来防止一些爬虫的恶意爬取。即如果检测到有人在使用 Selenium 打开浏览器，那就直接屏蔽。</p>
<p data-nodeid="108953">其大多数情况下，检测基本原理是检测当前浏览器窗口下的 window.navigator 对象是否包含 webdriver 这个属性。因为在正常使用浏览器的情况下，这个属性是 undefined，然而一旦我们使用了 Selenium，Selenium 会给 window.navigator 设置 webdriver 属性。很多网站就通过 JavaScript 判断如果 webdriver 属性存在，那就直接屏蔽。</p>
<p data-nodeid="120426" class="">这边有一个典型的案例网站：<a href="https://antispider1.scrape.center/" data-nodeid="120430">https://antispider1.scrape.center/</a>，这个网站就是使用了上述原理实现了 WebDriver 的检测，如果使用 Selenium 直接爬取的话，那就会返回如下页面：</p>


<p data-nodeid="108955"><img src="https://s0.lgstatic.com/i/image3/M01/7D/12/Cgq2xl59oBeATES6AABGMW_83Oc577.png" alt="" data-nodeid="109300"></p>
<p data-nodeid="108956">这时候我们可能想到直接使用 JavaScript 直接把这个 webdriver 属性置空，比如通过调用 execute_script 方法来执行如下代码：</p>
<pre class="lang-python" data-nodeid="108957"><code data-language="python">Object.defineProperty(navigator,&nbsp;"webdriver",&nbsp;{get:&nbsp;()&nbsp;=&gt;&nbsp;undefined})
</code></pre>
<p data-nodeid="108958">这行 JavaScript 的确是可以把 webdriver 属性置空，但是 execute_script 调用这行 JavaScript 语句实际上是在页面加载完毕之后才执行的，执行太晚了，网站早在最初页面渲染之前就已经对 webdriver 属性进行了检测，所以用上述方法并不能达到效果。</p>
<p data-nodeid="108959">在 Selenium 中，我们可以使用 CDP（即 Chrome Devtools-Protocol，Chrome 开发工具协议）来解决这个问题，通过 CDP 我们可以实现在每个页面刚加载的时候执行 JavaScript 代码，执行的 CDP 方法叫作 Page.addScriptToEvaluateOnNewDocument，然后传入上文的 JavaScript 代码即可，这样我们就可以在每次页面加载之前将 webdriver 属性置空了。另外我们还可以加入几个选项来隐藏 WebDriver 提示条和自动化扩展信息，代码实现如下：</p>
<pre class="lang-python te-preview-highlight" data-nodeid="121538"><code data-language="python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver
<span class="hljs-keyword">from</span>&nbsp;selenium.webdriver&nbsp;<span class="hljs-keyword">import</span>&nbsp;ChromeOptions

option&nbsp;=&nbsp;ChromeOptions()
option.add_experimental_option(<span class="hljs-string">'excludeSwitches'</span>,&nbsp;[<span class="hljs-string">'enable-automation'</span>])
option.add_experimental_option(<span class="hljs-string">'useAutomationExtension'</span>,&nbsp;<span class="hljs-literal">False</span>)
browser&nbsp;=&nbsp;webdriver.Chrome(options=option)
browser.execute_cdp_cmd(<span class="hljs-string">'Page.addScriptToEvaluateOnNewDocument'</span>,&nbsp;{
&nbsp;&nbsp;&nbsp;<span class="hljs-string">'source'</span>:&nbsp;<span class="hljs-string">'Object.defineProperty(navigator,&nbsp;"webdriver",&nbsp;{get:&nbsp;()&nbsp;=&gt;&nbsp;undefined})'</span>
})
browser.get(<span class="hljs-string">'https://antispider1.scrape.center/'</span>)
</code></pre>

<p data-nodeid="108961">这样整个页面就能被加载出来了：</p>
<p data-nodeid="108962"><img src="https://s0.lgstatic.com/i/image3/M01/03/FC/Ciqah159oBeAHd49AAMjMlBnsHE279.png" alt="" data-nodeid="109310"></p>
<p data-nodeid="108963">对于大多数的情况，以上的方法均可以实现 Selenium 反屏蔽。但对于一些特殊的网站，如果其有更多的 WebDriver 特征检测，可能需要具体排查。</p>
<h3 data-nodeid="108964">无头模式</h3>
<p data-nodeid="108965">上面的案例在运行的时候，我们可以观察到其总会弹出一个浏览器窗口，虽然有助于观察页面爬取状况，但在有些时候窗口弹来弹去也会形成一些干扰。</p>
<p data-nodeid="108966">Chrome 浏览器从 60 版本已经支持了无头模式，即 Headless。无头模式在运行的时候不会再弹出浏览器窗口，减少了干扰，而且它减少了一些资源的加载，如图片等资源，所以也在一定程度上节省了资源加载时间和网络带宽。</p>
<p data-nodeid="108967">我们可以借助于 ChromeOptions 来开启 Chrome Headless 模式，代码实现如下：</p>
<pre class="lang-python" data-nodeid="108968"><code data-language="python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver
<span class="hljs-keyword">from</span>&nbsp;selenium.webdriver&nbsp;<span class="hljs-keyword">import</span>&nbsp;ChromeOptions

option&nbsp;=&nbsp;ChromeOptions()
option.add_argument(<span class="hljs-string">'--headless'</span>)
browser&nbsp;=&nbsp;webdriver.Chrome(options=option)
browser.set_window_size(<span class="hljs-number">1366</span>,&nbsp;<span class="hljs-number">768</span>)
browser.get(<span class="hljs-string">'https://www.baidu.com'</span>)
browser.get_screenshot_as_file(<span class="hljs-string">'preview.png'</span>)
</code></pre>
<p data-nodeid="108969">这里我们通过 ChromeOptions 的 add_argument 方法添加了一个参数 --headless，开启了无头模式。在无头模式下，我们最好需要设置下窗口的大小，接着打开页面，最后我们调用 get_screenshot_as_file 方法输出了页面的截图。</p>
<p data-nodeid="108970">运行代码之后，我们发现 Chrome 窗口就不会再弹出来了，代码依然正常运行，最后输出了页面截图如图所示。</p>
<p data-nodeid="108971"><img src="https://s0.lgstatic.com/i/image3/M01/7D/12/Cgq2xl59oBeAdYtPAACc0m2Jx3Y415.png" alt="" data-nodeid="109327"></p>
<p data-nodeid="108972">这样我们就在无头模式下完成了页面的抓取和截图操作。</p>
<p data-nodeid="108973">现在，我们基本对 Selenium 的常规用法有了大体的了解。使用 Selenium，处理 JavaScript 渲染的页面不再是难事。</p>

# Selenium爬取实战
<p data-nodeid="123809" class="">在上一课时我们学习了 Selenium 的基本用法，本课时我们就来结合一个实际的案例来体会一下 Selenium 的适用场景以及使用方法。</p>
<h3 data-nodeid="123810">准备工作</h3>
<p data-nodeid="123811">在本课时开始之前，请确保已经做好了如下准备工作：</p>
<ul data-nodeid="123812">
<li data-nodeid="123813">
<p data-nodeid="123814">安装好 Chrome 浏览器并正确配置了 ChromeDriver。</p>
</li>
<li data-nodeid="123815">
<p data-nodeid="123816">安装好 Python （至少为 3.6 版本）并能成功运行 Python 程序。</p>
</li>
<li data-nodeid="123817">
<p data-nodeid="123818">安装好了 Selenium 相关的包并能成功用 Selenium 打开 Chrome 浏览器。</p>
</li>
</ul>
<h3 data-nodeid="123819">适用场景</h3>
<p data-nodeid="123820">在前面的实战案例中，有的网页我们可以直接用 requests 来爬取，有的可以直接通过分析 Ajax 来爬取，不同的网站类型有其适用的爬取方法。</p>
<p data-nodeid="123821">Selenium 同样也有其适用场景。对于那些带有 JavaScript 渲染的网页，我们多数情况下是无法直接用 requests 爬取网页源码的，不过在有些情况下我们可以直接用 requests 来模拟 Ajax 请求来直接得到数据。</p>
<p data-nodeid="123822">然而在有些情况下 Ajax 的一些请求接口可能带有一些加密参数，如 token、sign 等等，如果不分析清楚这些参数是怎么生成的话，我们就难以模拟和构造这些参数。怎么办呢？这时候我们可以直接选择使用 Selenium 驱动浏览器渲染的方式来另辟蹊径，实现所见即所得的爬取，这样我们就无需关心在这个网页背后发生了什么请求、得到什么数据以及怎么渲染页面这些过程，我们看到的页面就是最终浏览器帮我们模拟了 Ajax 请求和 JavaScript 渲染得到的最终结果，而 Selenium 正好也能拿到这个最终结果，相当于绕过了 Ajax 请求分析和模拟的阶段，直达目标。</p>
<p data-nodeid="123823">然而 Selenium 当然也有其局限性，它的爬取效率较低，有些爬取需要模拟浏览器的操作，实现相对烦琐。不过在某些场景下也不失为一种有效的爬取手段。</p>
<h3 data-nodeid="123824">爬取目标</h3>
<p data-nodeid="126447" class="">本课时我们就拿一个适用 Selenium 的站点来做案例，其链接为：<a href="https://dynamic2.scrape.center/" data-nodeid="126451">https://dynamic2.scrape.center/</a>，还是和之前一样的电影网站，页面如图所示。</p>


<p data-nodeid="123826"><img src="https://s0.lgstatic.com/i/image3/M01/7E/EC/Cgq2xl6Bvw-AStH9AASuzBi8U3Y686.png" alt="" data-nodeid="123927"></p>
<p data-nodeid="123827">初看之下页面和之前也没有什么区别，但仔细观察可以发现其 Ajax 请求接口和每部电影的 URL 都包含了加密参数。</p>
<p data-nodeid="123828">比如我们点击任意一部电影，观察一下 URL 的变化，如图所示。</p>
<p data-nodeid="123829"><img src="https://s0.lgstatic.com/i/image3/M01/05/D6/Ciqah16Bvw-AZFzUAAWYkjYkT8Y467.png" alt="" data-nodeid="123931"></p>
<p data-nodeid="123830">这里我们可以看到详情页的 URL 和之前就不一样了，在之前的案例中，URL 的 detail 后面本来直接跟的是 id，如 1、2、3 等数字，但是这里直接变成了一个长字符串，看似是一个 Base64 编码的内容，所以这里我们无法直接根据规律构造详情页的 URL 了。</p>
<p data-nodeid="123831">好，那么接下来我们直接看看 Ajax 的请求，我们从列表页的第 1 页到第 10 页依次点一下，观察一下 Ajax 请求是怎样的，如图所示。</p>
<p data-nodeid="123832"><img src="https://s0.lgstatic.com/i/image3/M01/7E/ED/Cgq2xl6Bvw-AI4IOAAO62yR--O0446.png" alt="" data-nodeid="123935"></p>
<p data-nodeid="123833">可以看到这里接口的参数比之前多了一个 token，而且每次请求的 token 都是不同的，这个 token 同样看似是一个 Base64 编码的字符串。更困难的是，这个接口还是有时效性的，如果我们把 Ajax 接口 URL 直接复制下来，短期内是可以访问的，但是过段时间之后就无法访问了，会直接返回 401 状态码。</p>
<p data-nodeid="123834">那现在怎么办呢？之前我们可以直接用 requests 来构造 Ajax 请求，但现在 Ajax 请求接口带了这个 token，而且还是可变的，现在我们也不知道 token 的生成逻辑，那就没法直接通过构造 Ajax 请求的方式来爬取了。这时候我们可以把 token 的生成逻辑分析出来再模拟 Ajax 请求，但这种方式相对较难。所以这里我们可以直接用 Selenium 来绕过这个阶段，直接获取最终 JavaScript 渲染完成的页面源码，再提取数据就好了。</p>
<p data-nodeid="123835">所以本课时我们要完成的目标有：</p>
<ul data-nodeid="123836">
<li data-nodeid="123837">
<p data-nodeid="123838">通过 Selenium 遍历列表页，获取每部电影的详情页 URL。</p>
</li>
<li data-nodeid="123839">
<p data-nodeid="123840">通过 Selenium 根据上一步获取的详情页 URL 爬取每部电影的详情页。</p>
</li>
<li data-nodeid="123841">
<p data-nodeid="123842">提取每部电影的名称、类别、分数、简介、封面等内容。</p>
</li>
</ul>
<h3 data-nodeid="123843">爬取列表页</h3>
<p data-nodeid="123844">首先要我们要做如下初始化的工作，代码如下：</p>
<pre class="lang-python" data-nodeid="124100"><code data-language="python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver
<span class="hljs-keyword">from</span>&nbsp;selenium.common.exceptions&nbsp;<span class="hljs-keyword">import</span>&nbsp;TimeoutException
<span class="hljs-keyword">from</span>&nbsp;selenium.webdriver.common.by&nbsp;<span class="hljs-keyword">import</span>&nbsp;By
<span class="hljs-keyword">from</span>&nbsp;selenium.webdriver.support&nbsp;<span class="hljs-keyword">import</span>&nbsp;expected_conditions&nbsp;<span class="hljs-keyword">as</span>&nbsp;EC
<span class="hljs-keyword">from</span>&nbsp;selenium.webdriver.support.wait&nbsp;<span class="hljs-keyword">import</span>&nbsp;WebDriverWait
<span class="hljs-keyword">import</span>&nbsp;logging
logging.basicConfig(level=logging.INFO,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;format=<span class="hljs-string">'%(asctime)s&nbsp;-&nbsp;%(levelname)s:&nbsp;%(message)s'</span>)
INDEX_URL&nbsp;=&nbsp;<span class="hljs-string">'https://dynamic2.scrape.center/page/{page}'</span>
TIME_OUT&nbsp;=&nbsp;<span class="hljs-number">10</span>
TOTAL_PAGE&nbsp;=&nbsp;<span class="hljs-number">10</span>
browser&nbsp;=&nbsp;webdriver.Chrome()
wait&nbsp;=&nbsp;WebDriverWait(browser,&nbsp;TIME_OUT)
</code></pre>

<p data-nodeid="123846">首先我们导入了一些必要的 Selenium 模块，包括 webdriver、WebDriverWait 等等，后面我们会用到它们来实现页面的爬取和延迟等待等设置。然后接着定义了一些变量和日志配置，和之前几课时的内容是类似的。接着我们使用 Chrome 类生成了一个 webdriver 对象，赋值为 browser，这里我们可以通过 browser 调用 Selenium 的一些 API 来完成一些浏览器的操作，如截图、点击、下拉等等。最后我们又声明了一个 WebDriverWait 对象，利用它我们可以配置页面加载的最长等待时间。</p>
<p data-nodeid="125271" class="">好，接下来我们就观察下列表页，实现列表页的爬取吧。这里可以观察到列表页的 URL 还是有一定规律的，比如第一页为&nbsp;<a href="https://dynamic2.scrape.center/page/1" data-nodeid="125275">https://dynamic2.scrape.center/page/1</a>，页码就是 URL 最后的数字，所以这里我们可以直接来构造每一页的 URL。</p>


<p data-nodeid="123848">那么每个列表页要怎么判断是否加载成功了呢？很简单，当页面出现了我们想要的内容就代表加载成功了。在这里我们就可以用 Selenium 的隐式判断条件来判定，比如每部电影的信息区块的 CSS 选择器为 #index .item，如图所示。</p>
<p data-nodeid="123849"><img src="https://s0.lgstatic.com/i/image3/M01/05/D6/Ciqah16Bvw-AdZyFAAQm5r5A8I0442.png" alt="" data-nodeid="123952"></p>
<p data-nodeid="123850">所以这里我们直接使用 visibility_of_all_elements_located 判断条件加上 CSS 选择器的内容即可判定页面有没有加载出来，配合 WebDriverWait 的超时配置，我们就可以实现 10 秒的页面的加载监听。如果 10 秒之内，我们所配置的条件符合，则代表页面加载成功，否则则会抛出 TimeoutException 异常。</p>
<p data-nodeid="123851">代码实现如下：</p>
<pre class="lang-python" data-nodeid="123852"><code data-language="python"><span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">scrape_page</span>(<span class="hljs-params">url,&nbsp;condition,&nbsp;locator</span>):</span>
&nbsp;&nbsp;&nbsp;logging.info(<span class="hljs-string">'scraping&nbsp;%s'</span>,&nbsp;url)
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">try</span>:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;browser.get(url)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;wait.until(condition(locator))
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">except</span>&nbsp;TimeoutException:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logging.error(<span class="hljs-string">'error&nbsp;occurred&nbsp;while&nbsp;scraping&nbsp;%s'</span>,&nbsp;url,&nbsp;exc_info=<span class="hljs-literal">True</span>)
<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">scrape_index</span>(<span class="hljs-params">page</span>):</span>
&nbsp;&nbsp;&nbsp;url&nbsp;=&nbsp;INDEX_URL.format(page=page)
&nbsp;&nbsp;&nbsp;scrape_page(url,&nbsp;condition=EC.visibility_of_all_elements_located,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;locator=(By.CSS_SELECTOR,&nbsp;<span class="hljs-string">'#index&nbsp;.item'</span>))
</code></pre>
<p data-nodeid="123853">这里我们定义了两个方法。</p>
<p data-nodeid="123854">第一个方法 scrape_page 依然是一个通用的爬取方法，它可以实现任意 URL 的爬取和状态监听以及异常处理，它接收 url、condition、locator 三个参数，其中 url 参数就是要爬取的页面 URL；condition 就是页面加载的判定条件，它可以是 expected_conditions 的其中某一项判定条件，如 visibility_of_all_elements_located、visibility_of_element_located 等等；locator 代表定位器，是一个元组，它可以通过配置查询条件和参数来获取一个或多个节点，如 (By.CSS_SELECTOR, '#index .item') 则代表通过 CSS 选择器查找 #index .item 来获取列表页所有电影信息节点。另外爬取的过程添加了 TimeoutException 检测，如果在规定时间（这里为 10 秒）没有加载出来对应的节点，那就抛出 TimeoutException 异常并输出错误日志。</p>
<p data-nodeid="123855">第二个方法 scrape_index 则是爬取列表页的方法，它接收一个参数 page，通过调用 scrape_page 方法并传入 condition 和 locator 对象，完成页面的爬取。这里 condition 我们用的是 visibility_of_all_elements_located，代表所有的节点都加载出来才算成功。</p>
<p data-nodeid="123856">注意，这里爬取页面我们不需要返回任何结果，因为执行完 scrape_index 后，页面正好处在对应的页面加载完成的状态，我们利用 browser 对象可以进一步进行信息的提取。</p>
<p data-nodeid="123857">好，现在我们已经可以加载出来列表页了，下一步当然就是进行列表页的解析，提取出详情页 URL ，我们定义一个如下的解析列表页的方法：</p>
<pre class="lang-python" data-nodeid="123858"><code data-language="python"><span class="hljs-keyword">from</span>&nbsp;urllib.parse&nbsp;<span class="hljs-keyword">import</span>&nbsp;urljoin
<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">parse_index</span>():</span>
&nbsp;&nbsp;&nbsp;elements&nbsp;=&nbsp;browser.find_elements_by_css_selector(<span class="hljs-string">'#index&nbsp;.item&nbsp;.name'</span>)
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">for</span>&nbsp;element&nbsp;<span class="hljs-keyword">in</span>&nbsp;elements:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;href&nbsp;=&nbsp;element.get_attribute(<span class="hljs-string">'href'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">yield</span>&nbsp;urljoin(INDEX_URL,&nbsp;href)
</code></pre>
<p data-nodeid="123859">这里我们通过 find_elements_by_css_selector 方法直接提取了所有电影的名称，接着遍历结果，通过 get_attribute 方法提取了详情页的 href，再用 urljoin 方法合并成一个完整的 URL。</p>
<p data-nodeid="123860">最后，我们再用一个 main 方法把上面的方法串联起来，实现如下：</p>
<pre class="lang-python" data-nodeid="123861"><code data-language="python"><span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">main</span>():</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">try</span>:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">for</span>&nbsp;page&nbsp;<span class="hljs-keyword">in</span>&nbsp;range(<span class="hljs-number">1</span>,&nbsp;TOTAL_PAGE&nbsp;+&nbsp;<span class="hljs-number">1</span>):
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;scrape_index(page)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail_urls&nbsp;=&nbsp;parse_index()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logging.info(<span class="hljs-string">'details&nbsp;urls&nbsp;%s'</span>,&nbsp;list(detail_urls))
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">finally</span>:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;browser.close()
</code></pre>
<p data-nodeid="123862">这里我们就是遍历了所有页码，依次爬取了每一页的列表页并提取出来了详情页的 URL。</p>
<p data-nodeid="123863">运行结果如下：</p>
<pre class="lang-python" data-nodeid="129367"><code data-language="python"><span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-29</span>&nbsp;<span class="hljs-number">12</span>:<span class="hljs-number">03</span>:<span class="hljs-number">09</span>,<span class="hljs-number">896</span>&nbsp;-&nbsp;INFO:&nbsp;scraping&nbsp;https://dynamic2.scrape.center/page/<span class="hljs-number">1</span>
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-29</span>&nbsp;<span class="hljs-number">12</span>:<span class="hljs-number">03</span>:<span class="hljs-number">13</span>,<span class="hljs-number">724</span>&nbsp;-&nbsp;INFO:&nbsp;details&nbsp;urls&nbsp;[<span class="hljs-string">'https://dynamic2.scrape.center/detail/ZWYzNCN0ZXVxMGJ0dWEjKC01N3cxcTVvNS0takA5OHh5Z2ltbHlmeHMqLSFpLTAtbWIx'</span>,
...
<span class="hljs-string">'https://dynamic2.scrape.center/detail/ZWYzNCN0ZXVxMGJ0dWEjKC01N3cxcTVvNS0takA5OHh5Z2ltbHlmeHMqLSFpLTAtbWI5'</span>,&nbsp;<span class="hljs-string">'https://dynamic2.scrape.center/detail/ZWYzNCN0ZXVxMGJ0dWEjKC01N3cxcTVvNS0takA5OHh5Z2ltbHlmeHMqLSFpLTAtbWIxMA=='</span>]
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-29</span>&nbsp;<span class="hljs-number">12</span>:<span class="hljs-number">03</span>:<span class="hljs-number">13</span>,<span class="hljs-number">724</span>&nbsp;-&nbsp;INFO:&nbsp;scraping&nbsp;https://dynamic2.scrape.center/page/<span class="hljs-number">2</span>
...
</code></pre>





<p data-nodeid="123865">由于输出内容较多，这里省略了部分内容。</p>
<p data-nodeid="123866">观察结果我们可以发现，详情页那一个个不规则的 URL 就成功被我们提取到了！</p>
<h3 data-nodeid="123867">爬取详情页</h3>
<p data-nodeid="123868">好了，既然现在我们已经可以成功拿到详情页的 URL 了，接下来我们就进一步完成详情页的爬取并提取对应的信息吧。</p>
<p data-nodeid="123869">同样的逻辑，详情页我们也可以加一个判定条件，如判断电影名称加载出来了就代表详情页加载成功，同样调用 scrape_page 方法即可，代码实现如下：</p>
<pre class="lang-python" data-nodeid="123870"><code data-language="python"><span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">scrape_detail</span>(<span class="hljs-params">url</span>):</span>
&nbsp;&nbsp;&nbsp;scrape_page(url,&nbsp;condition=EC.visibility_of_element_located,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;locator=(By.TAG_NAME,&nbsp;<span class="hljs-string">'h2'</span>))
</code></pre>
<p data-nodeid="123871">这里的判定条件 condition 我们使用的是 visibility_of_element_located，即判断单个元素出现即可，locator 我们传入的是 (By.TAG_NAME, 'h2')，即 h2 这个节点，也就是电影的名称对应的节点，如图所示。</p>
<p data-nodeid="123872"><img src="https://s0.lgstatic.com/i/image3/M01/7E/ED/Cgq2xl6Bvw-AdrCfAAV8yEmeyb4309.png" alt="" data-nodeid="124041"></p>
<p data-nodeid="123873">如果执行了 scrape_detail 方法，没有出现 TimeoutException 的话，页面就加载成功了，接着我们再定义一个解析详情页的方法，来提取出我们想要的信息就可以了，实现如下：</p>
<pre class="lang-python" data-nodeid="123874"><code data-language="python"><span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">parse_detail</span>():</span>
&nbsp;&nbsp;&nbsp;url&nbsp;=&nbsp;browser.current_url
&nbsp;&nbsp;&nbsp;name&nbsp;=&nbsp;browser.find_element_by_tag_name(<span class="hljs-string">'h2'</span>).text
&nbsp;&nbsp;&nbsp;categories&nbsp;=&nbsp;[element.text&nbsp;<span class="hljs-keyword">for</span>&nbsp;element&nbsp;<span class="hljs-keyword">in</span>&nbsp;browser.find_elements_by_css_selector(<span class="hljs-string">'.categories&nbsp;button&nbsp;span'</span>)]
&nbsp;&nbsp;&nbsp;cover&nbsp;=&nbsp;browser.find_element_by_css_selector(<span class="hljs-string">'.cover'</span>).get_attribute(<span class="hljs-string">'src'</span>)
&nbsp;&nbsp;&nbsp;score&nbsp;=&nbsp;browser.find_element_by_class_name(<span class="hljs-string">'score'</span>).text
&nbsp;&nbsp;&nbsp;drama&nbsp;=&nbsp;browser.find_element_by_css_selector(<span class="hljs-string">'.drama&nbsp;p'</span>).text
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;{
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">'url'</span>:&nbsp;url,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">'name'</span>:&nbsp;name,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">'categories'</span>:&nbsp;categories,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">'cover'</span>:&nbsp;cover,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">'score'</span>:&nbsp;score,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">'drama'</span>:&nbsp;drama
&nbsp;&nbsp;&nbsp;}
</code></pre>
<p data-nodeid="123875">这里我们定义了一个 parse_detail 方法，提取了 URL、名称、类别、封面、分数、简介等内容，提取方式如下：</p>
<ul data-nodeid="123876">
<li data-nodeid="123877">
<p data-nodeid="123878">URL：直接调用 browser 对象的 current_url 属性即可获取当前页面的 URL。</p>
</li>
<li data-nodeid="123879">
<p data-nodeid="123880">名称：通过提取 h2 节点内部的文本即可获取，这里使用了 find_element_by_tag_name 方法并传入 h2，提取到了名称的节点，然后调用 text 属性即提取了节点内部的文本，即电影名称。</p>
</li>
<li data-nodeid="123881">
<p data-nodeid="123882">类别：为了方便，类别我们可以通过 CSS 选择器来提取，其对应的 CSS 选择器为 .categories button span，可以选中多个类别节点，这里我们通过 find_elements_by_css_selector 即可提取 CSS 选择器对应的多个类别节点，然后依次遍历这个结果，调用它的 text 属性获取节点内部文本即可。</p>
</li>
<li data-nodeid="123883">
<p data-nodeid="123884">封面：同样可以使用 CSS 选择器 .cover 直接获取封面对应的节点，但是由于其封面的 URL 对应的是 src 这个属性，所以这里用 get_attribute 方法并传入 src 来提取。</p>
</li>
<li data-nodeid="123885">
<p data-nodeid="123886">分数：分数对应的 CSS 选择器为 .score ，我们可以用上面同样的方式来提取，但是这里我们换了一个方法，叫作 find_element_by_class_name，它可以使用 class 的名称来提取节点，能达到同样的效果，不过这里传入的参数就是 class 的名称 score 而不是 .score 了。提取节点之后，我们再调用 text 属性提取节点文本即可。</p>
</li>
<li data-nodeid="123887">
<p data-nodeid="123888">简介：同样可以使用 CSS 选择器 .drama p 直接获取简介对应的节点，然后调用 text 属性提取文本即可。</p>
</li>
</ul>
<p data-nodeid="123889">最后，我们把结果构造成一个字典返回即可。</p>
<p data-nodeid="123890">接下来，我们在 main 方法中再添加这两个方法的调用，实现如下：</p>
<pre class="lang-python" data-nodeid="123891"><code data-language="python"><span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">main</span>():</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">try</span>:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">for</span>&nbsp;page&nbsp;<span class="hljs-keyword">in</span>&nbsp;range(<span class="hljs-number">1</span>,&nbsp;TOTAL_PAGE&nbsp;+&nbsp;<span class="hljs-number">1</span>):
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;scrape_index(page)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail_urls&nbsp;=&nbsp;parse_index()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">for</span>&nbsp;detail_url&nbsp;<span class="hljs-keyword">in</span>&nbsp;list(detail_urls):
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logging.info(<span class="hljs-string">'get&nbsp;detail&nbsp;url&nbsp;%s'</span>,&nbsp;detail_url)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;scrape_detail(detail_url)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;detail_data&nbsp;=&nbsp;parse_detail()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logging.info(<span class="hljs-string">'detail&nbsp;data&nbsp;%s'</span>,&nbsp;detail_data)
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">finally</span>:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;browser.close()
</code></pre>
<p data-nodeid="123892">这样，爬取完列表页之后，我们就可以依次爬取详情页，来提取每部电影的具体信息了。</p>
<pre class="lang-python te-preview-highlight" data-nodeid="133448"><code data-language="python"><span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-29</span>&nbsp;<span class="hljs-number">12</span>:<span class="hljs-number">24</span>:<span class="hljs-number">10</span>,<span class="hljs-number">723</span>&nbsp;-&nbsp;INFO:&nbsp;scraping&nbsp;https://dynamic2.scrape.center/page/<span class="hljs-number">1</span>
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-29</span>&nbsp;<span class="hljs-number">12</span>:<span class="hljs-number">24</span>:<span class="hljs-number">16</span>,<span class="hljs-number">997</span>&nbsp;-&nbsp;INFO:&nbsp;get&nbsp;detail&nbsp;url&nbsp;https://dynamic2.scrape.center/detail/ZWYzNCN0ZXVxMGJ0dWEjKC01N3cxcTVvNS0takA5OHh5Z2ltbHlmeHMqLSFpLTAtbWIx
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-29</span>&nbsp;<span class="hljs-number">12</span>:<span class="hljs-number">24</span>:<span class="hljs-number">16</span>,<span class="hljs-number">997</span>&nbsp;-&nbsp;INFO:&nbsp;scraping&nbsp;https://dynamic2.scrape.center/detail/ZWYzNCN0ZXVxMGJ0dWEjKC01N3cxcTVvNS0takA5OHh5Z2ltbHlmeHMqLSFpLTAtbWIx
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-29</span>&nbsp;<span class="hljs-number">12</span>:<span class="hljs-number">24</span>:<span class="hljs-number">19</span>,<span class="hljs-number">289</span>&nbsp;-&nbsp;INFO:&nbsp;detail&nbsp;data&nbsp;{<span class="hljs-string">'url'</span>:&nbsp;<span class="hljs-string">'https://dynamic2.scrape.center/detail/ZWYzNCN0ZXVxMGJ0dWEjKC01N3cxcTVvNS0takA5OHh5Z2ltbHlmeHMqLSFpLTAtbWIx'</span>,&nbsp;<span class="hljs-string">'name'</span>:&nbsp;<span class="hljs-string">'霸王别姬&nbsp;-&nbsp;Farewell&nbsp;My&nbsp;Concubine'</span>,&nbsp;<span class="hljs-string">'categories'</span>:&nbsp;[<span class="hljs-string">'剧情'</span>,&nbsp;<span class="hljs-string">'爱情'</span>],&nbsp;<span class="hljs-string">'cover'</span>:&nbsp;<span class="hljs-string">'https://p0.meituan.net/movie/ce4da3e03e655b5b88ed31b5cd7896cf62472.jpg@464w_644h_1e_1c'</span>,&nbsp;<span class="hljs-string">'score'</span>:&nbsp;<span class="hljs-string">'9.5'</span>,&nbsp;<span class="hljs-string">'drama'</span>:&nbsp;<span class="hljs-string">'影片借一出《霸王别姬》的京戏，牵扯出三个人之间一段随时代风云变幻的爱恨情仇。段小楼（张丰毅&nbsp;饰）与程蝶衣（张国荣&nbsp;饰）是一对打小一起长大的师兄弟，两人一个演生，一个饰旦，一向配合天衣无缝，尤其一出《霸王别姬》，更是誉满京城，为此，两人约定合演一辈子《霸王别姬》。但两人对戏剧与人生关系的理解有本质不同，段小楼深知戏非人生，程蝶衣则是人戏不分。段小楼在认为该成家立业之时迎娶了名妓菊仙（巩俐&nbsp;饰），致使程蝶衣认定菊仙是可耻的第三者，使段小楼做了叛徒，自此，三人围绕一出《霸王别姬》生出的爱恨情仇战开始随着时代风云的变迁不断升级，终酿成悲剧。'</span>}
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-29</span>&nbsp;<span class="hljs-number">12</span>:<span class="hljs-number">24</span>:<span class="hljs-number">19</span>,<span class="hljs-number">291</span>&nbsp;-&nbsp;INFO:&nbsp;get&nbsp;detail&nbsp;url&nbsp;https://dynamic2.scrape.center/detail/ZWYzNCN0ZXVxMGJ0dWEjKC01N3cxcTVvNS0takA5OHh5Z2ltbHlmeHMqLSFpLTAtbWIy
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-29</span>&nbsp;<span class="hljs-number">12</span>:<span class="hljs-number">24</span>:<span class="hljs-number">19</span>,<span class="hljs-number">291</span>&nbsp;-&nbsp;INFO:&nbsp;scraping&nbsp;https://dynamic2.scrape.center/detail/ZWYzNCN0ZXVxMGJ0dWEjKC01N3cxcTVvNS0takA5OHh5Z2ltbHlmeHMqLSFpLTAtbWIy
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-29</span>&nbsp;<span class="hljs-number">12</span>:<span class="hljs-number">24</span>:<span class="hljs-number">21</span>,<span class="hljs-number">524</span>&nbsp;-&nbsp;INFO:&nbsp;detail&nbsp;data&nbsp;{<span class="hljs-string">'url'</span>:&nbsp;<span class="hljs-string">'https://dynamic2.scrape.center/detail/ZWYzNCN0ZXVxMGJ0dWEjKC01N3cxcTVvNS0takA5OHh5Z2ltbHlmeHMqLSFpLTAtbWIy'</span>,&nbsp;<span class="hljs-string">'name'</span>:&nbsp;<span class="hljs-string">'这个杀手不太冷&nbsp;-&nbsp;Léon'</span>,&nbsp;<span class="hljs-string">'categories'</span>:&nbsp;[<span class="hljs-string">'剧情'</span>,&nbsp;<span class="hljs-string">'动作'</span>,&nbsp;<span class="hljs-string">'犯罪'</span>],&nbsp;<span class="hljs-string">'cover'</span>:&nbsp;<span class="hljs-string">'https://p1.meituan.net/movie/6bea9af4524dfbd0b668eaa7e187c3df767253.jpg@464w_644h_1e_1c'</span>,&nbsp;<span class="hljs-string">'score'</span>:&nbsp;<span class="hljs-string">'9.5'</span>,&nbsp;<span class="hljs-string">'drama'</span>:&nbsp;<span class="hljs-string">'里昂（让·雷诺&nbsp;饰）是名孤独的职业杀手，受人雇佣。一天，邻居家小姑娘马蒂尔德（纳塔丽·波特曼&nbsp;饰）敲开他的房门，要求在他那里暂避杀身之祸。原来邻居家的主人是警方缉毒组的眼线，只因贪污了一小包毒品而遭恶警（加里·奥德曼&nbsp;饰）杀害全家的惩罚。马蒂尔德&nbsp;得到里昂的留救，幸免于难，并留在里昂那里。里昂教小女孩使枪，她教里昂法文，两人关系日趋亲密，相处融洽。&nbsp;女孩想着去报仇，反倒被抓，里昂及时赶到，将女孩救回。混杂着哀怨情仇的正邪之战渐次升级，更大的冲突在所难免……'</span>}
...
</code></pre>







<p data-nodeid="123894">这样详情页数据我们也可以提取到了。</p>
<h3 data-nodeid="123895">数据存储</h3>
<p data-nodeid="123896">最后，我们再像之前一样添加一个数据存储的方法，为了方便，这里还是保存为 JSON 文本文件，实现如下：</p>
<pre class="lang-python" data-nodeid="123897"><code data-language="python"><span class="hljs-keyword">from</span>&nbsp;os&nbsp;<span class="hljs-keyword">import</span>&nbsp;makedirs
<span class="hljs-keyword">from</span>&nbsp;os.path&nbsp;<span class="hljs-keyword">import</span>&nbsp;exists
RESULTS_DIR&nbsp;=&nbsp;<span class="hljs-string">'results'</span>
exists(RESULTS_DIR)&nbsp;<span class="hljs-keyword">or</span>&nbsp;makedirs(RESULTS_DIR)
<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">save_data</span>(<span class="hljs-params">data</span>):</span>
&nbsp;&nbsp;&nbsp;name&nbsp;=&nbsp;data.get(<span class="hljs-string">'name'</span>)
&nbsp;&nbsp;&nbsp;data_path&nbsp;=&nbsp;<span class="hljs-string">f'<span class="hljs-subst">{RESULTS_DIR}</span>/<span class="hljs-subst">{name}</span>.json'</span>
&nbsp;&nbsp;&nbsp;json.dump(data,&nbsp;open(data_path,&nbsp;<span class="hljs-string">'w'</span>,&nbsp;encoding=<span class="hljs-string">'utf-8'</span>),&nbsp;ensure_ascii=<span class="hljs-literal">False</span>,&nbsp;indent=<span class="hljs-number">2</span>)
</code></pre>
<p data-nodeid="123898">这里原理和实现方式与 Ajax 爬取实战课时是完全相同的，不再赘述。</p>
<p data-nodeid="123899">最后添加上 save_data 的调用，完整看下运行效果。</p>
<h3 data-nodeid="123900">Headless</h3>
<p data-nodeid="123901">如果觉得爬取过程中弹出浏览器有所干扰，我们可以开启 Chrome 的 Headless 模式，这样爬取过程中便不会再弹出浏览器了，同时爬取速度还有进一步的提升。</p>
<p data-nodeid="123902">只需要做如下修改即可：</p>
<pre class="lang-python" data-nodeid="123903"><code data-language="python">options&nbsp;=&nbsp;webdriver.ChromeOptions()
options.add_argument(<span class="hljs-string">'--headless'</span>)
browser&nbsp;=&nbsp;webdriver.Chrome(options=options)
</code></pre>
<p data-nodeid="123904">这里通过 ChromeOptions 添加了 --headless 参数，然后用 ChromeOptions 来进行 Chrome 的初始化即可。</p>
<p data-nodeid="123905">修改后再重新运行代码，Chrome 浏览器就不会弹出来了，爬取结果是完全一样的。</p>
<h3 data-nodeid="123906">总结</h3>
<p data-nodeid="123907">本课时我们通过一个案例了解了 Selenium 的适用场景，并结合案例使用 Selenium 实现了页面的爬取，从而对 Selenium 的使用有进一步的掌握。</p>
