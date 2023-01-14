---
title: 52讲轻松搞定网络爬虫笔记2
tags: [Web Crawler]
categories: data analysis
date: 2023-1-14
---

# 资料
[52讲轻松搞定网络爬虫](https://kaiwu.lagou.com/course/courseInfo.htm?courseId=46#/sale)


# Requests库的基本使用
<p data-nodeid="18640">学习爬虫，最基础的便是模拟浏览器向服务器发出请求，那么我们需要从什么地方做起呢？请求需要我们自己来构造吗？需要关心请求这个数据结构的实现吗？需要了解 HTTP、TCP、IP 层的网络传输通信吗？需要知道服务器的响应和应答原理吗？</p>
<p data-nodeid="18641">可能你无从下手，不过不用担心，Python 的强大之处就是提供了功能齐全的类库来帮助我们完成这些请求。利用 Python 现有的库我们可以非常方便地实现网络请求的模拟，常见的库有 urllib、requests 等。</p>
<p data-nodeid="18642">拿 requests 这个库来说，有了它，我们只需要关心请求的链接是什么，需要传的参数是什么，以及如何设置可选的参数就好了，不用深入到底层去了解它到底是怎样传输和通信的。有了它，两行代码就可以完成一个请求和响应的处理过程，非常方便地得到网页内容。</p>
<p data-nodeid="18643">接下来，就让我们用 Python 的 requests 库开始我们的爬虫之旅吧。</p>
<h3 data-nodeid="18644">安装</h3>
<p data-nodeid="18645">首先，requests 库是 Python 的一个第三方库，不是自带的。所以我们需要额外安装。</p>
<p data-nodeid="18646">在这之前需要你先安装好 Python3 环境，如 Python 3.6 版本，如若没有安装可以参考：<a href="https://cuiqingcai.com/5059.html" data-nodeid="18882">https://cuiqingcai.com/5059.html</a>。</p>
<p data-nodeid="18647">安装好 Python3 之后，我们使用 pip3 即可轻松地安装好 requests 库：</p>
<pre data-nodeid="18648"><code>pip3 install requests
</code></pre>
<p data-nodeid="18649">更详细的安装方式可以参考：<a href="https://cuiqingcai.com/5132.html" data-nodeid="18888">https://cuiqingcai.com/5132.html</a>。</p>
<p data-nodeid="18650">安装完成之后，我们就可以开始我们的网络爬虫之旅了。</p>
<h3 data-nodeid="18651">实例引入</h3>
<p data-nodeid="18652">用 Python 写爬虫的第一步就是模拟发起一个请求，把网页的源代码获取下来。</p>
<p data-nodeid="18653">当我们在浏览器中输入一个 URL 并回车，实际上就是让浏览器帮我们发起一个 GET 类型的 HTTP 请求，浏览器得到源代码后，把它渲染出来就可以看到网页内容了。</p>
<p data-nodeid="18654">那如果我们想用 requests 来获取源代码，应该怎么办呢？很简单，requests 这个库提供了一个 get 方法，我们调用这个方法，并传入对应的 URL 就能得到网页的源代码。</p>
<p data-nodeid="21541" class="">比如这里有一个示例网站：<a href="https://static1.scrape.center/" data-nodeid="21545">https://static1.scrape.center/</a>，其内容如下：</p>


<p data-nodeid="18656"><img src="https://s0.lgstatic.com/i/image3/M01/72/AB/Cgq2xl5on76ANrhqAAOUr7Y20R4873.png" alt="" data-nodeid="18901"></p>
<p data-nodeid="18657">这个网站展示了一些电影数据，如果我们想要把这个网页里面的数据爬下来，比如获取各个电影的名称、上映时间等信息，然后把它存下来的话，该怎么做呢？</p>
<p data-nodeid="18658">第一步当然就是获取它的网页源代码了。</p>
<p data-nodeid="18659">我们可以用 requests 这个库轻松地完成这个过程，代码的写法是这样的：</p>
<pre class="lang-python" data-nodeid="26361"><code data-language="python"><span class="hljs-keyword">import</span> requests

r = requests.get(<span class="hljs-string">'https://static1.scrape.center/'</span>)
print(r.text)
</code></pre>

<p data-nodeid="18661">运行结果如下：</p>
<pre class="lang-html" data-nodeid="18662"><code data-language="html"><span class="hljs-tag">&lt;<span class="hljs-name">html</span> <span class="hljs-attr">lang</span>=<span class="hljs-string">"en"</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">head</span>&gt;</span>
  <span class="hljs-tag">&lt;<span class="hljs-name">meta</span> <span class="hljs-attr">charset</span>=<span class="hljs-string">"utf-8"</span>&gt;</span>
  <span class="hljs-tag">&lt;<span class="hljs-name">meta</span> <span class="hljs-attr">http-equiv</span>=<span class="hljs-string">"X-UA-Compatible"</span> <span class="hljs-attr">content</span>=<span class="hljs-string">"IE=edge"</span>&gt;</span>
  <span class="hljs-tag">&lt;<span class="hljs-name">meta</span> <span class="hljs-attr">name</span>=<span class="hljs-string">"viewport"</span> <span class="hljs-attr">content</span>=<span class="hljs-string">"width=device-width,initial-scale=1"</span>&gt;</span>
  <span class="hljs-tag">&lt;<span class="hljs-name">link</span> <span class="hljs-attr">rel</span>=<span class="hljs-string">"icon"</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"/static/img/favicon.ico"</span>&gt;</span>
  <span class="hljs-tag">&lt;<span class="hljs-name">title</span>&gt;</span>Scrape | Movie<span class="hljs-tag">&lt;/<span class="hljs-name">title</span>&gt;</span>
  <span class="hljs-tag">&lt;<span class="hljs-name">link</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"/static/css/app.css"</span> <span class="hljs-attr">type</span>=<span class="hljs-string">"text/css"</span> <span class="hljs-attr">rel</span>=<span class="hljs-string">"stylesheet"</span>&gt;</span>
  <span class="hljs-tag">&lt;<span class="hljs-name">link</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"/static/css/index.css"</span> <span class="hljs-attr">type</span>=<span class="hljs-string">"text/css"</span> <span class="hljs-attr">rel</span>=<span class="hljs-string">"stylesheet"</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">head</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">body</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">id</span>=<span class="hljs-string">"app"</span>&gt;</span>
...
<span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">data-v-7f856186</span>=<span class="hljs-string">""</span> <span class="hljs-attr">id</span>=<span class="hljs-string">"index"</span>&gt;</span>
  <span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">data-v-7f856186</span>=<span class="hljs-string">""</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"el-row"</span>&gt;</span>
    <span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">data-v-7f856186</span>=<span class="hljs-string">""</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"el-col el-col-18 el-col-offset-3"</span>&gt;</span>
      <span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">data-v-7f856186</span>=<span class="hljs-string">""</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"el-card item m-t is-hover-shadow"</span>&gt;</span>
        <span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"el-card__body"</span>&gt;</span>
          <span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">data-v-7f856186</span>=<span class="hljs-string">""</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"el-row"</span>&gt;</span>
            <span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">data-v-7f856186</span>=<span class="hljs-string">""</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"el-col el-col-24 el-col-xs-8 el-col-sm-6 el-col-md-4"</span>&gt;</span>
              <span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">data-v-7f856186</span>=<span class="hljs-string">""</span>
                 <span class="hljs-attr">href</span>=<span class="hljs-string">"/detail/1"</span>
                 <span class="hljs-attr">class</span>=<span class="hljs-string">""</span>&gt;</span>
                <span class="hljs-tag">&lt;<span class="hljs-name">img</span>
                    <span class="hljs-attr">data-v-7f856186</span>=<span class="hljs-string">""</span> <span class="hljs-attr">src</span>=<span class="hljs-string">"https://p0.meituan.net/movie/283292171619cdfd5b240c8fd093f1eb255670.jpg@464w_644h_1e_1c"</span>
                    <span class="hljs-attr">class</span>=<span class="hljs-string">"cover"</span>&gt;</span>
              <span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span>
            <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
            <span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">data-v-7f856186</span>=<span class="hljs-string">""</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"p-h el-col el-col-24 el-col-xs-9 el-col-sm-13 el-col-md-16"</span>&gt;</span>
              <span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">data-v-7f856186</span>=<span class="hljs-string">""</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"/detail/1"</span> <span class="hljs-attr">class</span>=<span class="hljs-string">""</span>&gt;</span>
                <span class="hljs-tag">&lt;<span class="hljs-name">h2</span> <span class="hljs-attr">data-v-7f856186</span>=<span class="hljs-string">""</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"m-b-sm"</span>&gt;</span>肖申克的救赎 - The Shawshank Redemption<span class="hljs-tag">&lt;/<span class="hljs-name">h2</span>&gt;</span>
              <span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span>
              <span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">data-v-7f856186</span>=<span class="hljs-string">""</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"categories"</span>&gt;</span>
                <span class="hljs-tag">&lt;<span class="hljs-name">button</span> <span class="hljs-attr">data-v-7f856186</span>=<span class="hljs-string">""</span> <span class="hljs-attr">type</span>=<span class="hljs-string">"button"</span>
                        <span class="hljs-attr">class</span>=<span class="hljs-string">"el-button category el-button--primary el-button--mini"</span>&gt;</span>
                  <span class="hljs-tag">&lt;<span class="hljs-name">span</span>&gt;</span>剧情<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span>
                <span class="hljs-tag">&lt;/<span class="hljs-name">button</span>&gt;</span>
                <span class="hljs-tag">&lt;<span class="hljs-name">button</span> <span class="hljs-attr">data-v-7f856186</span>=<span class="hljs-string">""</span> <span class="hljs-attr">type</span>=<span class="hljs-string">"button"</span>
                        <span class="hljs-attr">class</span>=<span class="hljs-string">"el-button category el-button--primary el-button--mini"</span>&gt;</span>
                  <span class="hljs-tag">&lt;<span class="hljs-name">span</span>&gt;</span>犯罪<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span>
                <span class="hljs-tag">&lt;/<span class="hljs-name">button</span>&gt;</span>
              <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
              <span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">data-v-7f856186</span>=<span class="hljs-string">""</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"m-v-sm info"</span>&gt;</span>
                <span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">data-v-7f856186</span>=<span class="hljs-string">""</span>&gt;</span>美国<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span>
                <span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">data-v-7f856186</span>=<span class="hljs-string">""</span>&gt;</span> / <span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span>
                <span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">data-v-7f856186</span>=<span class="hljs-string">""</span>&gt;</span>142 分钟<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span>
              <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
              <span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">data-v-7f856186</span>=<span class="hljs-string">""</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"m-v-sm info"</span>&gt;</span>
                <span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">data-v-7f856186</span>=<span class="hljs-string">""</span>&gt;</span>1994-09-10 上映<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span>
              <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
            <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
          <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
        <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
      <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
    <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
  <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
  ...
<span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">body</span>&gt;</span>
</code></pre>
<p data-nodeid="18663">由于网页内容比较多，这里省略了大部分内容。</p>
<p data-nodeid="18664">不过看运行结果，我们已经成功获取网页的 HTML 源代码，里面包含了电影的标题、类型、上映时间，等等。把网页源代码获取下来之后，下一步我们把想要的数据提取出来，数据的爬取就完成了。</p>
<p data-nodeid="18665">这个实例的目的是让你体会一下 requests 这个库能帮我们实现什么功能。我们仅仅用 requests 的 get 方法就成功发起了一个 GET 请求，把网页源代码获取下来了，是不是很方便呢？</p>
<h3 data-nodeid="18666">请求</h3>
<p data-nodeid="18667">HTTP 中最常见的请求之一就是 GET 请求，下面我们来详细了解利用 requests 库构建 GET 请求的方法。</p>
<h4 data-nodeid="18668">GET 请求</h4>
<p data-nodeid="18669">我们换一个示例网站，其 URL 为&nbsp;<a href="http://httpbin.org/get" data-nodeid="18915">http://httpbin.org/get</a>，如果客户端发起的是 GET 请求的话，该网站会判断并返回相应的请求信息，包括 Headers、IP 等。</p>
<p data-nodeid="18670">我们还是用相同的方法来发起一个 GET 请求，代码如下：</p>
<pre class="lang-python" data-nodeid="18671"><code data-language="python"><span class="hljs-keyword">import</span> requests

r = requests.get(<span class="hljs-string">'http://httpbin.org/get'</span>)
print(r.text)
</code></pre>
<p data-nodeid="18672">运行结果如下：</p>
<pre class="lang-json" data-nodeid="18673"><code data-language="json">{<span class="hljs-attr">"args"</span>: {},
  <span class="hljs-attr">"headers"</span>: {
    <span class="hljs-attr">"Accept"</span>: <span class="hljs-string">"*/*"</span>,
    <span class="hljs-attr">"Accept-Encoding"</span>: <span class="hljs-string">"gzip, deflate"</span>,
    <span class="hljs-attr">"Host"</span>: <span class="hljs-string">"httpbin.org"</span>,
    <span class="hljs-attr">"User-Agent"</span>: <span class="hljs-string">"python-requests/2.10.0"</span>
  },
  <span class="hljs-attr">"origin"</span>: <span class="hljs-string">"122.4.215.33"</span>,
  <span class="hljs-attr">"url"</span>: <span class="hljs-string">"http://httpbin.org/get"</span>
}
</code></pre>
<p data-nodeid="18674">可以发现，我们成功发起了 GET 请求，也通过这个网站的返回结果得到了请求所携带的信息，包括 Headers、URL、IP，等等。</p>
<p data-nodeid="18675">对于 GET 请求，我们知道 URL 后面是可以跟上一些参数的，如果我们现在想添加两个参数，其中 name 是 germey，age 是 25，URL 就可以写成如下内容：</p>
<pre class="lang-python" data-nodeid="18676"><code data-language="python">http://httpbin.org/get?name=germey&amp;age=25
</code></pre>
<p data-nodeid="18677">要构造这个请求链接，是不是要直接写成这样呢？</p>
<pre class="lang-python" data-nodeid="18678"><code data-language="python">r = requests.get(<span class="hljs-string">'http://httpbin.org/get?name=germey&amp;age=25'</span>)
</code></pre>
<p data-nodeid="18679">这样也可以，但如果这些参数还需要我们手动拼接，未免有点不人性化。</p>
<p data-nodeid="18680">一般情况下，这种信息我们利用 params 这个参数就可以直接传递了，示例如下：</p>
<pre class="lang-python" data-nodeid="18681"><code data-language="python"><span class="hljs-keyword">import</span> requests

data = {
    <span class="hljs-string">'name'</span>: <span class="hljs-string">'germey'</span>,
    <span class="hljs-string">'age'</span>: <span class="hljs-number">25</span>
}
r = requests.get(<span class="hljs-string">'http://httpbin.org/get'</span>, params=data)
print(r.text)
</code></pre>
<p data-nodeid="18682">运行结果如下：</p>
<pre class="lang-python" data-nodeid="18683"><code data-language="python">{
  <span class="hljs-string">"args"</span>: {
    <span class="hljs-string">"age"</span>: <span class="hljs-string">"25"</span>,
    <span class="hljs-string">"name"</span>: <span class="hljs-string">"germey"</span>
  },
  <span class="hljs-string">"headers"</span>: {
    <span class="hljs-string">"Accept"</span>: <span class="hljs-string">"*/*"</span>,
    <span class="hljs-string">"Accept-Encoding"</span>: <span class="hljs-string">"gzip, deflate"</span>,
    <span class="hljs-string">"Host"</span>: <span class="hljs-string">"httpbin.org"</span>,
    <span class="hljs-string">"User-Agent"</span>: <span class="hljs-string">"python-requests/2.10.0"</span>
  },
  <span class="hljs-string">"origin"</span>: <span class="hljs-string">"122.4.215.33"</span>,
  <span class="hljs-string">"url"</span>: <span class="hljs-string">"http://httpbin.org/get?age=22&amp;name=germey"</span>
}
</code></pre>
<p data-nodeid="18684">在这里我们把 URL 参数通过字典的形式传给 get 方法的 params 参数，通过返回信息我们可以判断，请求的链接自动被构造成了：<a href="http://httpbin.org/get?age=22&amp;name=germey" data-nodeid="18930">http://httpbin.org/get?age=22&amp;name=germey</a>，这样我们就不用再去自己构造 URL 了，非常方便。</p>
<p data-nodeid="18685">另外，网页的返回类型实际上是 str 类型，但是它很特殊，是 JSON 格式的。所以，如果想直接解析返回结果，得到一个 JSON 格式的数据的话，可以直接调用 json 方法。</p>
<p data-nodeid="18686">示例如下：</p>
<pre class="lang-python" data-nodeid="18687"><code data-language="python"><span class="hljs-keyword">import</span> requests

r = requests.get(<span class="hljs-string">'http://httpbin.org/get'</span>)
print(type(r.text))
print(r.json())
print(type(r.json()))
</code></pre>
<p data-nodeid="18688">运行结果如下：</p>
<pre class="lang-html" data-nodeid="18689"><code data-language="html"><span class="hljs-tag">&lt;<span class="hljs-name">class'str'</span>&gt;</span>
{'headers': {'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Host': 'httpbin.org', 'User-Agent': 'python-requests/2.10.0'}, 'url': 'http://httpbin.org/get', 'args': {}, 'origin': '182.33.248.131'}
<span class="hljs-tag">&lt;<span class="hljs-name">class</span> '<span class="hljs-attr">dict</span>'&gt;</span>
</code></pre>
<p data-nodeid="18690">可以发现，调用 json 方法，就可以将返回结果是 JSON 格式的字符串转化为字典。</p>
<p data-nodeid="18691">但需要注意的是，如果返回结果不是 JSON 格式，便会出现解析错误，抛出 json.decoder.JSONDecodeError 异常。</p>
<h4 data-nodeid="18692">抓取网页</h4>
<p data-nodeid="18693">上面的请求链接返回的是 JSON 形式的字符串，那么如果请求普通的网页，则肯定能获得相应的内容了。下面以本课时最初的实例页面为例，我们再加上一点提取信息的逻辑，将代码完善成如下的样子：</p>
<pre class="lang-python" data-nodeid="27324"><code data-language="python"><span class="hljs-keyword">import</span> requests
<span class="hljs-keyword">import</span> re

r = requests.get(<span class="hljs-string">'https://static1.scrape.center/'</span>)
pattern = re.compile(<span class="hljs-string">'&lt;h2.*?&gt;(.*?)&lt;/h2&gt;'</span>, re.S)
titles = re.findall(pattern, r.text)
print(titles)
</code></pre>

<p data-nodeid="18695">在这个例子中我们用到了最基础的正则表达式来匹配出所有的标题。关于正则表达式的相关内容，我们会在下一课时详细介绍，这里作为实例来配合讲解。</p>
<p data-nodeid="18696">运行结果如下：</p>
<pre class="lang-html" data-nodeid="18697"><code data-language="html">['肖申克的救赎 - The Shawshank Redemption', '霸王别姬 - Farewell My Concubine', '泰坦尼克号 - Titanic', '罗马假日 - Roman Holiday', '这个杀手不太冷 - Léon', '魂断蓝桥 - Waterloo Bridge', '唐伯虎点秋香 - Flirting Scholar', '喜剧之王 - The King of Comedy', '楚门的世界 - The Truman Show', '活着 - To Live']
</code></pre>
<p data-nodeid="18698">我们发现，这里成功提取出了所有的电影标题。一个最基本的抓取和提取流程就完成了。</p>
<h4 data-nodeid="18699">抓取二进制数据</h4>
<p data-nodeid="18700">在上面的例子中，我们抓取的是网站的一个页面，实际上它返回的是一个 HTML 文档。如果想抓取图片、音频、视频等文件，应该怎么办呢？</p>
<p data-nodeid="18701">图片、音频、视频这些文件本质上都是由二进制码组成的，由于有特定的保存格式和对应的解析方式，我们才可以看到这些形形色色的多媒体。所以，想要抓取它们，就要拿到它们的二进制数据。</p>
<p data-nodeid="18702">下面以 GitHub 的站点图标为例来看一下：</p>
<pre class="lang-python" data-nodeid="18703"><code data-language="python"><span class="hljs-keyword">import</span> requests

r = requests.get(<span class="hljs-string">'https://github.com/favicon.ico'</span>)
print(r.text)
print(r.content)
</code></pre>
<p data-nodeid="18704">这里抓取的内容是站点图标，也就是在浏览器每一个标签上显示的小图标，如图所示：</p>
<p data-nodeid="18705"><img src="https://s0.lgstatic.com/i/image3/M01/72/AB/CgpOIF5on8CAb3vnAAANA7l3qLE757.png" alt="" data-nodeid="18948"></p>
<p data-nodeid="18706">这里打印了 Response 对象的两个属性，一个是 text，另一个是 content。</p>
<p data-nodeid="18707">运行结果如图所示，其中前两行是 r.text 的结果，最后一行是 r.content 的结果。</p>
<p data-nodeid="18708"><img src="https://s0.lgstatic.com/i/image3/M01/72/AB/Cgq2xl5on8GARhzvAAF-fz8cGvg942.png" alt="" data-nodeid="18952"></p>
<p data-nodeid="18709">可以注意到，前者出现了乱码，后者结果前带有一个 b，这代表是 bytes 类型的数据。</p>
<p data-nodeid="18710">由于图片是二进制数据，所以前者在打印时转化为 str 类型，也就是图片直接转化为字符串，这当然会出现乱码。</p>
<p data-nodeid="18711">上面返回的结果我们并不能看懂，它实际上是图片的二进制数据，没关系，我们将刚才提取到的信息保存下来就好了，代码如下：</p>
<pre class="lang-python" data-nodeid="18712"><code data-language="python"><span class="hljs-keyword">import</span> requests

r = requests.get(<span class="hljs-string">'https://github.com/favicon.ico'</span>)
<span class="hljs-keyword">with</span> open(<span class="hljs-string">'favicon.ico'</span>, <span class="hljs-string">'wb'</span>) <span class="hljs-keyword">as</span> f:
    f.write(r.content)
</code></pre>
<p data-nodeid="18713">这里用了 open 方法，它的第一个参数是文件名称，第二个参数代表以二进制的形式打开，可以向文件里写入二进制数据。</p>
<p data-nodeid="18714">运行结束之后，可以发现在文件夹中出现了名为 favicon.ico 的图标，如图所示。</p>
<p data-nodeid="18715"><img src="https://s0.lgstatic.com/i/image3/M01/72/AB/Cgq2xl5on8GAFP1xAAADVlDCrKg987.png" alt="" data-nodeid="18959"></p>
<p data-nodeid="18716">这样，我们就把二进制数据成功保存成一张图片了，这个小图标就被我们成功爬取下来了。</p>
<p data-nodeid="18717">同样地，音频和视频文件我们也可以用这种方法获取。</p>
<h4 data-nodeid="18718">添加 headers</h4>
<p data-nodeid="18719">我们知道，在发起一个 HTTP 请求的时候，会有一个请求头 Request Headers，那么这个怎么来设置呢？</p>
<p data-nodeid="18720">很简单，我们使用 headers 参数就可以完成了。</p>
<p data-nodeid="18721">在刚才的实例中，实际上我们是没有设置 Request Headers 信息的，如果不设置，某些网站会发现这不是一个正常的浏览器发起的请求，网站可能会返回异常的结果，导致网页抓取失败。</p>
<p data-nodeid="18722">要添加 Headers 信息，比如我们这里想添加一个 User-Agent 字段，我们可以这么来写：</p>
<pre class="lang-python" data-nodeid="28287"><code data-language="python"><span class="hljs-keyword">import</span> requests


headers = {
    <span class="hljs-string">'User-Agent'</span>: <span class="hljs-string">'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'</span>
}
r = requests.get(<span class="hljs-string">'https://static1.scrape.center/'</span>, headers=headers)
print(r.text)
</code></pre>

<p data-nodeid="18724">当然，我们可以在 headers 这个参数中任意添加其他的字段信息。</p>
<h4 data-nodeid="18725">POST 请求</h4>
<p data-nodeid="18726">前面我们了解了最基本的 GET 请求，另外一种比较常见的请求方式是 POST。使用 requests 实现 POST 请求同样非常简单，示例如下：</p>
<pre class="lang-python" data-nodeid="18727"><code data-language="python"><span class="hljs-keyword">import</span> requests

data = {<span class="hljs-string">'name'</span>: <span class="hljs-string">'germey'</span>, <span class="hljs-string">'age'</span>: <span class="hljs-string">'25'</span>}
r = requests.post(<span class="hljs-string">"http://httpbin.org/post"</span>, data=data)
print(r.text)
</code></pre>
<p data-nodeid="18728">这里还是请求&nbsp;<a href="http://httpbin.org/post" data-nodeid="18973">http://httpbin.org/post</a>，该网站可以判断如果请求是 POST 方式，就把相关请求信息返回。</p>
<p data-nodeid="18729">运行结果如下：</p>
<pre class="lang-python" data-nodeid="18730"><code data-language="python">{
  <span class="hljs-string">"args"</span>: {}, 
  <span class="hljs-string">"data"</span>: <span class="hljs-string">""</span>, 
  <span class="hljs-string">"files"</span>: {}, 
  <span class="hljs-string">"form"</span>: {
    <span class="hljs-string">"age"</span>: <span class="hljs-string">"25"</span>, 
    <span class="hljs-string">"name"</span>: <span class="hljs-string">"germey"</span>
  }, 
  <span class="hljs-string">"headers"</span>: {
    <span class="hljs-string">"Accept"</span>: <span class="hljs-string">"*/*"</span>, 
    <span class="hljs-string">"Accept-Encoding"</span>: <span class="hljs-string">"gzip, deflate"</span>, 
    <span class="hljs-string">"Content-Length"</span>: <span class="hljs-string">"18"</span>, 
    <span class="hljs-string">"Content-Type"</span>: <span class="hljs-string">"application/x-www-form-urlencoded"</span>, 
    <span class="hljs-string">"Host"</span>: <span class="hljs-string">"httpbin.org"</span>, 
    <span class="hljs-string">"User-Agent"</span>: <span class="hljs-string">"python-requests/2.22.0"</span>, 
    <span class="hljs-string">"X-Amzn-Trace-Id"</span>: <span class="hljs-string">"Root=1-5e5bdc26-b40d7e9862e3715f689cb5e6"</span>
  }, 
  <span class="hljs-string">"json"</span>: null, 
  <span class="hljs-string">"origin"</span>: <span class="hljs-string">"167.220.232.237"</span>, 
  <span class="hljs-string">"url"</span>: <span class="hljs-string">"http://httpbin.org/post"</span>
}
</code></pre>
<p data-nodeid="18731">可以发现，我们成功获得了返回结果，其中 form 部分就是提交的数据，这就证明 POST 请求成功发送了。</p>
<h3 data-nodeid="18732">响应</h3>
<p data-nodeid="18733">发送请求后，得到的自然就是响应，即 Response。</p>
<p data-nodeid="18734">在上面的实例中，我们使用 text 和 content 获取了响应的内容。此外，还有很多属性和方法可以用来获取其他信息，比如状态码、响应头、Cookies 等。示例如下：</p>
<pre class="lang-python" data-nodeid="29250"><code data-language="python"><span class="hljs-keyword">import</span> requests

r = requests.get(<span class="hljs-string">'https://static1.scrape.center/'</span>)
print(type(r.status_code), r.status_code)
print(type(r.headers), r.headers)
print(type(r.cookies), r.cookies)
print(type(r.url), r.url)
print(type(r.history), r.history)
</code></pre>

<p data-nodeid="18736">这里分别打印输出 status_code 属性得到状态码，输出 headers 属性得到响应头，输出 cookies 属性得到 Cookies，输出 url 属性得到 URL，输出 history 属性得到请求历史。</p>
<p data-nodeid="18737">运行结果如下：</p>
<pre class="lang-html" data-nodeid="24435"><code data-language="html"><span class="hljs-tag">&lt;<span class="hljs-name">class</span> '<span class="hljs-attr">int</span>'&gt;</span> 200
<span class="hljs-tag">&lt;<span class="hljs-name">class</span> '<span class="hljs-attr">requests.structures.CaseInsensitiveDict</span>'&gt;</span> {'Server': 'nginx/1.17.8', 'Date': 'Sun, 01 Mar 2020 13:31:54 GMT', 'Content-Type': 'text/html; charset=utf-8', 'Transfer-Encoding': 'chunked', 'Connection': 'keep-alive', 'Vary': 'Accept-Encoding', 'X-Frame-Options': 'SAMEORIGIN', 'Strict-Transport-Security': 'max-age=15724800; includeSubDomains', 'Content-Encoding': 'gzip'}
<span class="hljs-tag">&lt;<span class="hljs-name">class</span> '<span class="hljs-attr">requests.cookies.RequestsCookieJar</span>'&gt;</span> <span class="hljs-tag">&lt;<span class="hljs-name">RequestsCookieJar[]</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">class</span> '<span class="hljs-attr">str</span>'&gt;</span> https://static1.scrape.center/
<span class="hljs-tag">&lt;<span class="hljs-name">class</span> '<span class="hljs-attr">list</span>'&gt;</span> []
</code></pre>



<p data-nodeid="18739">可以看到，headers 和 cookies 这两个属性得到的结果分别是 CaseInsensitiveDict 和 RequestsCookieJar 类型。</p>
<p data-nodeid="18740">在第一课时我们知道，状态码是用来表示响应状态的，比如返回 200 代表我们得到的响应是没问题的，上面的例子正好输出的结果也是 200，所以我们可以通过判断 Response 的状态码来确认是否爬取成功。</p>
<p data-nodeid="18741">requests 还提供了一个内置的状态码查询对象 requests.codes，用法示例如下：</p>
<pre class="lang-python" data-nodeid="25398"><code data-language="python"><span class="hljs-keyword">import</span> requests

r = requests.get(<span class="hljs-string">'https://static1.scrape.center/'</span>)
exit() <span class="hljs-keyword">if</span> <span class="hljs-keyword">not</span> r.status_code == requests.codes.ok <span class="hljs-keyword">else</span> print(<span class="hljs-string">'Request Successfully'</span>)
</code></pre>

<p data-nodeid="18743">这里通过比较返回码和内置的成功的返回码，来保证请求得到了正常响应，输出成功请求的消息，否则程序终止，这里我们用 requests.codes.ok 得到的是成功的状态码 200。</p>
<p data-nodeid="18744">这样的话，我们就不用再在程序里面写状态码对应的数字了，用字符串表示状态码会显得更加直观。</p>
<p data-nodeid="18745">当然，肯定不能只有 ok 这个条件码。</p>
<p data-nodeid="18746">下面列出了返回码和相应的查询条件：</p>
<pre class="lang-python" data-nodeid="18747"><code data-language="python"><span class="hljs-comment"># 信息性状态码</span>
<span class="hljs-number">100</span>: (<span class="hljs-string">'continue'</span>,),
<span class="hljs-number">101</span>: (<span class="hljs-string">'switching_protocols'</span>,),
<span class="hljs-number">102</span>: (<span class="hljs-string">'processing'</span>,),
<span class="hljs-number">103</span>: (<span class="hljs-string">'checkpoint'</span>,),
<span class="hljs-number">122</span>: (<span class="hljs-string">'uri_too_long'</span>, <span class="hljs-string">'request_uri_too_long'</span>),

<span class="hljs-comment"># 成功状态码</span>
<span class="hljs-number">200</span>: (<span class="hljs-string">'ok'</span>, <span class="hljs-string">'okay'</span>, <span class="hljs-string">'all_ok'</span>, <span class="hljs-string">'all_okay'</span>, <span class="hljs-string">'all_good'</span>, <span class="hljs-string">'\\o/'</span>, <span class="hljs-string">'✓'</span>),
<span class="hljs-number">201</span>: (<span class="hljs-string">'created'</span>,),
<span class="hljs-number">202</span>: (<span class="hljs-string">'accepted'</span>,),
<span class="hljs-number">203</span>: (<span class="hljs-string">'non_authoritative_info'</span>, <span class="hljs-string">'non_authoritative_information'</span>),
<span class="hljs-number">204</span>: (<span class="hljs-string">'no_content'</span>,),
<span class="hljs-number">205</span>: (<span class="hljs-string">'reset_content'</span>, <span class="hljs-string">'reset'</span>),
<span class="hljs-number">206</span>: (<span class="hljs-string">'partial_content'</span>, <span class="hljs-string">'partial'</span>),
<span class="hljs-number">207</span>: (<span class="hljs-string">'multi_status'</span>, <span class="hljs-string">'multiple_status'</span>, <span class="hljs-string">'multi_stati'</span>, <span class="hljs-string">'multiple_stati'</span>),
<span class="hljs-number">208</span>: (<span class="hljs-string">'already_reported'</span>,),
<span class="hljs-number">226</span>: (<span class="hljs-string">'im_used'</span>,),

<span class="hljs-comment"># 重定向状态码</span>
<span class="hljs-number">300</span>: (<span class="hljs-string">'multiple_choices'</span>,),
<span class="hljs-number">301</span>: (<span class="hljs-string">'moved_permanently'</span>, <span class="hljs-string">'moved'</span>, <span class="hljs-string">'\\o-'</span>),
<span class="hljs-number">302</span>: (<span class="hljs-string">'found'</span>,),
<span class="hljs-number">303</span>: (<span class="hljs-string">'see_other'</span>, <span class="hljs-string">'other'</span>),
<span class="hljs-number">304</span>: (<span class="hljs-string">'not_modified'</span>,),
<span class="hljs-number">305</span>: (<span class="hljs-string">'use_proxy'</span>,),
<span class="hljs-number">306</span>: (<span class="hljs-string">'switch_proxy'</span>,),
<span class="hljs-number">307</span>: (<span class="hljs-string">'temporary_redirect'</span>, <span class="hljs-string">'temporary_moved'</span>, <span class="hljs-string">'temporary'</span>),
<span class="hljs-number">308</span>: (<span class="hljs-string">'permanent_redirect'</span>,
      <span class="hljs-string">'resume_incomplete'</span>, <span class="hljs-string">'resume'</span>,), <span class="hljs-comment"># These 2 to be removed in 3.0</span>

<span class="hljs-comment"># 客户端错误状态码</span>
<span class="hljs-number">400</span>: (<span class="hljs-string">'bad_request'</span>, <span class="hljs-string">'bad'</span>),
<span class="hljs-number">401</span>: (<span class="hljs-string">'unauthorized'</span>,),
<span class="hljs-number">402</span>: (<span class="hljs-string">'payment_required'</span>, <span class="hljs-string">'payment'</span>),
<span class="hljs-number">403</span>: (<span class="hljs-string">'forbidden'</span>,),
<span class="hljs-number">404</span>: (<span class="hljs-string">'not_found'</span>, <span class="hljs-string">'-o-'</span>),
<span class="hljs-number">405</span>: (<span class="hljs-string">'method_not_allowed'</span>, <span class="hljs-string">'not_allowed'</span>),
<span class="hljs-number">406</span>: (<span class="hljs-string">'not_acceptable'</span>,),
<span class="hljs-number">407</span>: (<span class="hljs-string">'proxy_authentication_required'</span>, <span class="hljs-string">'proxy_auth'</span>, <span class="hljs-string">'proxy_authentication'</span>),
<span class="hljs-number">408</span>: (<span class="hljs-string">'request_timeout'</span>, <span class="hljs-string">'timeout'</span>),
<span class="hljs-number">409</span>: (<span class="hljs-string">'conflict'</span>,),
<span class="hljs-number">410</span>: (<span class="hljs-string">'gone'</span>,),
<span class="hljs-number">411</span>: (<span class="hljs-string">'length_required'</span>,),
<span class="hljs-number">412</span>: (<span class="hljs-string">'precondition_failed'</span>, <span class="hljs-string">'precondition'</span>),
<span class="hljs-number">413</span>: (<span class="hljs-string">'request_entity_too_large'</span>,),
<span class="hljs-number">414</span>: (<span class="hljs-string">'request_uri_too_large'</span>,),
<span class="hljs-number">415</span>: (<span class="hljs-string">'unsupported_media_type'</span>, <span class="hljs-string">'unsupported_media'</span>, <span class="hljs-string">'media_type'</span>),
<span class="hljs-number">416</span>: (<span class="hljs-string">'requested_range_not_satisfiable'</span>, <span class="hljs-string">'requested_range'</span>, <span class="hljs-string">'range_not_satisfiable'</span>),
<span class="hljs-number">417</span>: (<span class="hljs-string">'expectation_failed'</span>,),
<span class="hljs-number">418</span>: (<span class="hljs-string">'im_a_teapot'</span>, <span class="hljs-string">'teapot'</span>, <span class="hljs-string">'i_am_a_teapot'</span>),
<span class="hljs-number">421</span>: (<span class="hljs-string">'misdirected_request'</span>,),
<span class="hljs-number">422</span>: (<span class="hljs-string">'unprocessable_entity'</span>, <span class="hljs-string">'unprocessable'</span>),
<span class="hljs-number">423</span>: (<span class="hljs-string">'locked'</span>,),
<span class="hljs-number">424</span>: (<span class="hljs-string">'failed_dependency'</span>, <span class="hljs-string">'dependency'</span>),
<span class="hljs-number">425</span>: (<span class="hljs-string">'unordered_collection'</span>, <span class="hljs-string">'unordered'</span>),
<span class="hljs-number">426</span>: (<span class="hljs-string">'upgrade_required'</span>, <span class="hljs-string">'upgrade'</span>),
<span class="hljs-number">428</span>: (<span class="hljs-string">'precondition_required'</span>, <span class="hljs-string">'precondition'</span>),
<span class="hljs-number">429</span>: (<span class="hljs-string">'too_many_requests'</span>, <span class="hljs-string">'too_many'</span>),
<span class="hljs-number">431</span>: (<span class="hljs-string">'header_fields_too_large'</span>, <span class="hljs-string">'fields_too_large'</span>),
<span class="hljs-number">444</span>: (<span class="hljs-string">'no_response'</span>, <span class="hljs-string">'none'</span>),
<span class="hljs-number">449</span>: (<span class="hljs-string">'retry_with'</span>, <span class="hljs-string">'retry'</span>),
<span class="hljs-number">450</span>: (<span class="hljs-string">'blocked_by_windows_parental_controls'</span>, <span class="hljs-string">'parental_controls'</span>),
<span class="hljs-number">451</span>: (<span class="hljs-string">'unavailable_for_legal_reasons'</span>, <span class="hljs-string">'legal_reasons'</span>),
<span class="hljs-number">499</span>: (<span class="hljs-string">'client_closed_request'</span>,),

<span class="hljs-comment"># 服务端错误状态码</span>
<span class="hljs-number">500</span>: (<span class="hljs-string">'internal_server_error'</span>, <span class="hljs-string">'server_error'</span>, <span class="hljs-string">'/o\\'</span>, <span class="hljs-string">'✗'</span>),
<span class="hljs-number">501</span>: (<span class="hljs-string">'not_implemented'</span>,),
<span class="hljs-number">502</span>: (<span class="hljs-string">'bad_gateway'</span>,),
<span class="hljs-number">503</span>: (<span class="hljs-string">'service_unavailable'</span>, <span class="hljs-string">'unavailable'</span>),
<span class="hljs-number">504</span>: (<span class="hljs-string">'gateway_timeout'</span>,),
<span class="hljs-number">505</span>: (<span class="hljs-string">'http_version_not_supported'</span>, <span class="hljs-string">'http_version'</span>),
<span class="hljs-number">506</span>: (<span class="hljs-string">'variant_also_negotiates'</span>,),
<span class="hljs-number">507</span>: (<span class="hljs-string">'insufficient_storage'</span>,),
<span class="hljs-number">509</span>: (<span class="hljs-string">'bandwidth_limit_exceeded'</span>, <span class="hljs-string">'bandwidth'</span>),
<span class="hljs-number">510</span>: (<span class="hljs-string">'not_extended'</span>,),
<span class="hljs-number">511</span>: (<span class="hljs-string">'network_authentication_required'</span>, <span class="hljs-string">'network_auth'</span>, <span class="hljs-string">'network_authentication'</span>)
</code></pre>
<p data-nodeid="18748">比如，如果想判断结果是不是 404 状态，可以用 requests.codes.not_found 来比对。</p>
<h3 data-nodeid="18749">高级用法</h3>
<p data-nodeid="18750">刚才，我们了解了 requests 的基本用法，如基本的 GET、POST 请求以及 Response 对象。当然 requests 能做到的不仅这些，它几乎可以帮我们完成 HTTP 的所有操作。</p>
<p data-nodeid="18751">下面我们再来了解下 requests 的一些高级用法，如文件上传、Cookies 设置、代理设置等。</p>
<h4 data-nodeid="18752">文件上传</h4>
<p data-nodeid="18753">我们知道 requests 可以模拟提交一些数据。假如有的网站需要上传文件，我们也可以用它来实现，示例如下：</p>
<pre class="lang-python" data-nodeid="18754"><code data-language="python"><span class="hljs-keyword">import</span> requests

files = {<span class="hljs-string">'file'</span>: open(<span class="hljs-string">'favicon.ico'</span>, <span class="hljs-string">'rb'</span>)}
r = requests.post(<span class="hljs-string">'http://httpbin.org/post'</span>, files=files)
print(r.text)
</code></pre>
<p data-nodeid="18755">在上一课时中我们保存了一个文件 favicon.ico，这次用它来模拟文件上传的过程。需要注意的是，favicon.ico 需要和当前脚本在同一目录下。如果有其他文件，当然也可以使用其他文件来上传，更改下代码即可。</p>
<p data-nodeid="18756">运行结果如下：</p>
<pre class="lang-python" data-nodeid="18757"><code data-language="python">{<span class="hljs-string">"args"</span>: {}, 
  <span class="hljs-string">"data"</span>: <span class="hljs-string">""</span>,<span class="hljs-string">"files"</span>: {<span class="hljs-string">"file"</span>:<span class="hljs-string">"data:application/octet-stream;base64,AAAAAA...="</span>},<span class="hljs-string">"form"</span>: {},<span class="hljs-string">"headers"</span>: {<span class="hljs-string">"Accept"</span>:<span class="hljs-string">"*/*"</span>,<span class="hljs-string">"Accept-Encoding"</span>:<span class="hljs-string">"gzip, deflate"</span>,<span class="hljs-string">"Content-Length"</span>:<span class="hljs-string">"6665"</span>,<span class="hljs-string">"Content-Type"</span>:<span class="hljs-string">"multipart/form-data; boundary=809f80b1a2974132b133ade1a8e8e058"</span>,<span class="hljs-string">"Host"</span>:<span class="hljs-string">"httpbin.org"</span>,<span class="hljs-string">"User-Agent"</span>:<span class="hljs-string">"python-requests/2.10.0"</span>},<span class="hljs-string">"json"</span>: null,<span class="hljs-string">"origin"</span>:<span class="hljs-string">"60.207.237.16"</span>,<span class="hljs-string">"url"</span>:<span class="hljs-string">"http://httpbin.org/post"</span>}
</code></pre>
<p data-nodeid="18758">以上省略部分内容，这个网站会返回响应，里面包含 files 这个字段，而 form 字段是空的，这证明文件上传部分会单独有一个 files 字段来标识。</p>
<h4 data-nodeid="18759">Cookies</h4>
<p data-nodeid="18760">我们如果想用 requests 获取和设置 Cookies 也非常方便，只需一步即可完成。</p>
<p data-nodeid="18761">我们先用一个实例看一下获取 Cookies 的过程：</p>
<pre class="lang-python" data-nodeid="18762"><code data-language="python"><span class="hljs-keyword">import</span> requests

r = requests.get(<span class="hljs-string">'http://www.baidu.com'</span>)
print(r.cookies)
<span class="hljs-keyword">for</span> key, value <span class="hljs-keyword">in</span> r.cookies.items():
    print(key + <span class="hljs-string">'='</span> + value)
</code></pre>
<p data-nodeid="18763">运行结果如下：</p>
<pre class="lang-python" data-nodeid="18764"><code data-language="python">&lt;RequestsCookieJar[&lt;Cookie BDORZ=<span class="hljs-number">27315</span> <span class="hljs-keyword">for</span> .baidu.com/&gt;]&gt;
BDORZ=<span class="hljs-number">27315</span>
</code></pre>
<p data-nodeid="18765">这里我们首先调用 cookies 属性即可成功得到 Cookies，可以发现它是 RequestCookieJar 类型。然后用 items 方法将其转化为元组组成的列表，遍历输出每一个 Cookie 的名称和值，实现 Cookie 的遍历解析。</p>
<p data-nodeid="18766">当然，我们也可以直接用 Cookie 来维持登录状态，下面我们以 GitHub 为例来说明一下，首先我们登录 GitHub，然后将 Headers 中的 Cookie 内容复制下来，如图所示：</p>
<p data-nodeid="18767"><img src="https://s0.lgstatic.com/i/image3/M01/72/AB/CgpOIF5on8SAX03uAAOB7v9rrD8925.png" alt="" data-nodeid="19009"><br>
这里可以替换成你自己的 Cookie，将其设置到 Headers 里面，然后发送请求，示例如下：</p>
<pre class="lang-python" data-nodeid="18768"><code data-language="python"><span class="hljs-keyword">import</span> requests

headers = {
    <span class="hljs-string">'Cookie'</span>: <span class="hljs-string">'_octo=GH1.1.1849343058.1576602081; _ga=GA1.2.90460451.1576602111; __Host-user_session_same_site=nbDv62kHNjp4N5KyQNYZ208waeqsmNgxFnFC88rnV7gTYQw_; _device_id=a7ca73be0e8f1a81d1e2ebb5349f9075; user_session=nbDv62kHNjp4N5KyQNYZ208waeqsmNgxFnFC88rnV7gTYQw_; logged_in=yes; dotcom_user=Germey; tz=Asia%2FShanghai; has_recent_activity=1; _gat=1; _gh_sess=your_session_info'</span>,
    <span class="hljs-string">'User-Agent'</span>: <span class="hljs-string">'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36'</span>,
}
r = requests.get(<span class="hljs-string">'https://github.com/'</span>, headers=headers)
print(r.text)
</code></pre>
<p data-nodeid="18769">我们发现，结果中包含了登录后才能显示的结果，如图所示：</p>
<p data-nodeid="18770"><img src="https://s0.lgstatic.com/i/image3/M01/72/AB/CgpOIF5on8SAKWukAAEoUzW61mM808.png" alt="" data-nodeid="19014"><br>
可以看到这里包含了我的 GitHub 用户名信息，你如果尝试同样可以得到你的用户信息。</p>
<p data-nodeid="18771">得到这样类似的结果，说明我们用 Cookies 成功模拟了登录状态，这样我们就能爬取登录之后才能看到的页面了。</p>
<p data-nodeid="18772">当然，我们也可以通过 cookies 参数来设置 Cookies 的信息，这里我们可以构造一个 RequestsCookieJar 对象，然后把刚才复制的 Cookie 处理下并赋值，示例如下：</p>
<pre class="lang-python" data-nodeid="18773"><code data-language="python"><span class="hljs-keyword">import</span> requests

cookies = <span class="hljs-string">'_octo=GH1.1.1849343058.1576602081; _ga=GA1.2.90460451.1576602111; __Host-user_session_same_site=nbDv62kHNjp4N5KyQNYZ208waeqsmNgxFnFC88rnV7gTYQw_; _device_id=a7ca73be0e8f1a81d1e2ebb5349f9075; user_session=nbDv62kHNjp4N5KyQNYZ208waeqsmNgxFnFC88rnV7gTYQw_; logged_in=yes; dotcom_user=Germey; tz=Asia%2FShanghai; has_recent_activity=1; _gat=1; _gh_sess=your_session_info'</span>
jar = requests.cookies.RequestsCookieJar()
headers = {
    <span class="hljs-string">'User-Agent'</span>: <span class="hljs-string">'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36'</span>
}
<span class="hljs-keyword">for</span> cookie <span class="hljs-keyword">in</span> cookies.split(<span class="hljs-string">';'</span>):
    key, value = cookie.split(<span class="hljs-string">'='</span>, <span class="hljs-number">1</span>)
    jar.set(key, value)
r = requests.get(<span class="hljs-string">'https://github.com/'</span>, cookies=jar, headers=headers)
print(r.text)
</code></pre>
<p data-nodeid="18774">这里我们首先新建一个 RequestCookieJar 对象，然后将复制下来的 cookies 利用 split 方法分割，接着利用 set 方法设置好每个 Cookie 的 key 和 value，最后通过调用 requests 的 get 方法并传递给 cookies 参数即可。</p>
<p data-nodeid="18775">测试后，发现同样可以正常登录。</p>
<h4 data-nodeid="18776">Session 维持</h4>
<p data-nodeid="18777">在 requests 中，如果直接利用 get 或 post 等方法的确可以做到模拟网页的请求，但是这实际上是相当于不同的 Session，相当于你用两个浏览器打开了不同的页面。</p>
<p data-nodeid="18778">设想这样一个场景，第一个请求利用 post 方法登录了某个网站，第二次想获取成功登录后的自己的个人信息，你又用了一次 get 方法去请求个人信息页面。实际上，这相当于打开了两个浏览器，是两个完全不相关的 Session，能成功获取个人信息吗？当然不能。</p>
<p data-nodeid="18779">有人会问，我在两次请求时设置一样的 Cookies 不就行了？可以，但这样做起来很烦琐，我们有更简单的解决方法。</p>
<p data-nodeid="18780">解决这个问题的主要方法就是维持同一个 Session，相当于打开一个新的浏览器选项卡而不是新开一个浏览器。但我又不想每次设置 Cookies，那该怎么办呢？这时候就有了新的利器 ——Session 对象。</p>
<p data-nodeid="18781">利用它，我们可以方便地维护一个 Session，而且不用担心 Cookies 的问题，它会帮我们自动处理好。示例如下：</p>
<pre class="lang-python" data-nodeid="18782"><code data-language="python"><span class="hljs-keyword">import</span> requests

requests.get(<span class="hljs-string">'http://httpbin.org/cookies/set/number/123456789'</span>)
r = requests.get(<span class="hljs-string">'http://httpbin.org/cookies'</span>)
print(r.text)
</code></pre>
<p data-nodeid="18783">这里我们请求了一个测试网址&nbsp;<a href="http://httpbin.org/cookies/set/number/123456789" data-nodeid="19030">http://httpbin.org/cookies/set/number/123456789</a>。请求这个网址时，可以设置一个 cookie，名称叫作 number，内容是 123456789，随后又请求了&nbsp;<a href="http://httpbin.org/cookies" data-nodeid="19034">http://httpbin.org/cookies</a>，此网址可以获取当前的 Cookies。</p>
<p data-nodeid="18784">这样能成功获取到设置的 Cookies 吗？试试看。</p>
<p data-nodeid="18785">运行结果如下：</p>
<pre class="lang-python" data-nodeid="18786"><code data-language="python">{
  <span class="hljs-string">"cookies"</span>: {}
}
</code></pre>
<p data-nodeid="18787">这并不行。我们再用 Session 试试看：</p>
<pre class="lang-python" data-nodeid="18788"><code data-language="python"><span class="hljs-keyword">import</span> requests

s = requests.Session()
s.get(<span class="hljs-string">'http://httpbin.org/cookies/set/number/123456789'</span>)
r = s.get(<span class="hljs-string">'http://httpbin.org/cookies'</span>)
print(r.text)
</code></pre>
<p data-nodeid="18789">再看下运行结果：</p>
<pre class="lang-python" data-nodeid="18790"><code data-language="python">{
  <span class="hljs-string">"cookies"</span>: {<span class="hljs-string">"number"</span>: <span class="hljs-string">"123456789"</span>}
}
</code></pre>
<p data-nodeid="18791">成功获取！这下能体会到同一个Session和不同Session的区别了吧！</p>
<p data-nodeid="18792">所以，利用 Session，可以做到模拟同一个 Session 而不用担心 Cookies 的问题。它通常用于模拟登录成功之后再进行下一步的操作。</p>
<h4 data-nodeid="18793">SSL 证书验证</h4>
<p data-nodeid="18794">现在很多网站都要求使用 HTTPS 协议，但是有些网站可能并没有设置好 HTTPS 证书，或者网站的 HTTPS 证书不被 CA 机构认可，这时候，这些网站可能就会出现 SSL 证书错误的提示。</p>
<p data-nodeid="31181" class="">比如这个示例网站：<a href="https://static2.scrape.center/" data-nodeid="31185">https://static2.scrape.center/</a>。</p>


<p data-nodeid="18796">如果我们用 Chrome 浏览器打开这个 URL，则会提示「您的连接不是私密连接」这样的错误，如图所示：</p>
<p data-nodeid="18797"><img src="https://s0.lgstatic.com/i/image3/M01/72/AB/Cgq2xl5on8WARK6YAABlizks2bg479.png" alt="" data-nodeid="19051"></p>
<p data-nodeid="18798">我们可以在浏览器中通过一些设置来忽略证书的验证。</p>
<p data-nodeid="18799">但是如果我们想用 requests 来请求这类网站，会遇到什么问题呢？我们用代码来试一下：</p>
<pre class="lang-python" data-nodeid="32149"><code data-language="python"><span class="hljs-keyword">import</span> requests

response = requests.get(<span class="hljs-string">'https://static2.scrape.center/'</span>)
print(response.status_code)
</code></pre>

<p data-nodeid="18801">运行结果如下：</p>
<pre class="lang-python" data-nodeid="33112"><code data-language="python">requests.exceptions.SSLError: HTTPSConnectionPool(host=<span class="hljs-string">'static2.scrape.center'</span>, port=<span class="hljs-number">443</span>): Max retries exceeded <span class="hljs-keyword">with</span> url: / (Caused by SSLError(SSLError(<span class="hljs-string">"bad handshake: Error([('SSL routines', 'tls_process_server_certificate', 'certificate verify failed')])"</span>)))
</code></pre>

<p data-nodeid="18803">可以看到，这里直接抛出了 SSLError 错误，原因就是因为我们请求的 URL 的证书是无效的。</p>
<p data-nodeid="18804">那如果我们一定要爬取这个网站怎么办呢？我们可以使用 verify 参数控制是否验证证书，如果将其设置为 False，在请求时就不会再验证证书是否有效。如果不加 verify 参数的话，默认值是 True，会自动验证。</p>
<p data-nodeid="18805">我们改写代码如下：</p>
<pre class="lang-python" data-nodeid="34075"><code data-language="python"><span class="hljs-keyword">import</span> requests

response = requests.get(<span class="hljs-string">'https://static2.scrape.center/'</span>, verify=<span class="hljs-literal">False</span>)
print(response.status_code)
</code></pre>

<p data-nodeid="18807">这样就会打印出请求成功的状态码：</p>
<pre class="lang-python" data-nodeid="18808"><code data-language="python">/usr/local/lib/python3<span class="hljs-number">.7</span>/site-packages/urllib3/connectionpool.py:<span class="hljs-number">857</span>: InsecureRequestWarning: Unverified HTTPS request <span class="hljs-keyword">is</span> being made. Adding certificate verification <span class="hljs-keyword">is</span> strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html<span class="hljs-comment">#ssl-warnings</span>
  InsecureRequestWarning)
<span class="hljs-number">200</span>
</code></pre>
<p data-nodeid="18809">不过我们发现报了一个警告，它建议我们给它指定证书。我们可以通过设置忽略警告的方式来屏蔽这个警告：</p>
<pre class="lang-python" data-nodeid="35038"><code data-language="python"><span class="hljs-keyword">import</span> requests
<span class="hljs-keyword">from</span> requests.packages <span class="hljs-keyword">import</span> urllib3

urllib3.disable_warnings()
response = requests.get(<span class="hljs-string">'https://static2.scrape.center'</span>, verify=<span class="hljs-literal">False</span>)
print(response.status_code)
</code></pre>

<p data-nodeid="18811">或者通过捕获警告到日志的方式忽略警告：</p>
<pre class="lang-python" data-nodeid="36001"><code data-language="python"><span class="hljs-keyword">import</span> logging
<span class="hljs-keyword">import</span> requests
logging.captureWarnings(<span class="hljs-literal">True</span>)
response = requests.get(<span class="hljs-string">'https://static2.scrape.center/'</span>, verify=<span class="hljs-literal">False</span>)
print(response.status_code)
</code></pre>

<p data-nodeid="18813">当然，我们也可以指定一个本地证书用作客户端证书，这可以是单个文件（包含密钥和证书）或一个包含两个文件路径的元组：</p>
<pre class="lang-python" data-nodeid="36964"><code data-language="python"><span class="hljs-keyword">import</span> requests

response = requests.get(<span class="hljs-string">'https://static2.scrape.center/'</span>, cert=(<span class="hljs-string">'/path/server.crt'</span>, <span class="hljs-string">'/path/server.key'</span>))
print(response.status_code)
</code></pre>

<p data-nodeid="18815">当然，上面的代码是演示实例，我们需要有 crt 和 key 文件，并且指定它们的路径。另外注意，本地私有证书的 key 必须是解密状态，加密状态的 key 是不支持的。</p>
<h4 data-nodeid="18816">超时设置</h4>
<p data-nodeid="18817">在本机网络状况不好或者服务器网络响应延迟甚至无响应时，我们可能会等待很久才能收到响应，甚至到最后收不到响应而报错。为了防止服务器不能及时响应，应该设置一个超时时间，即超过了这个时间还没有得到响应，那就报错。这需要用到 timeout 参数。这个时间的计算是发出请求到服务器返回响应的时间。示例如下：</p>
<pre class="lang-python" data-nodeid="18818"><code data-language="python"><span class="hljs-keyword">import</span> requests

r = requests.get(<span class="hljs-string">'https://httpbin.org/get'</span>, timeout=<span class="hljs-number">1</span>)
print(r.status_code)
</code></pre>
<p data-nodeid="18819">通过这样的方式，我们可以将超时时间设置为 1 秒，如果 1 秒内没有响应，那就抛出异常。</p>
<p data-nodeid="18820">实际上，请求分为两个阶段，即连接（connect）和读取（read）。</p>
<p data-nodeid="18821">上面设置的 timeout 将用作连接和读取这二者的 timeout 总和。</p>
<p data-nodeid="18822">如果要分别指定，就可以传入一个元组：</p>
<pre class="lang-python" data-nodeid="18823"><code data-language="python">r = requests.get(<span class="hljs-string">'https://httpbin.org/get'</span>, timeout=(<span class="hljs-number">5</span>, <span class="hljs-number">30</span>))
</code></pre>
<p data-nodeid="18824">如果想永久等待，可以直接将 timeout 设置为 None，或者不设置直接留空，因为默认是 None。这样的话，如果服务器还在运行，但是响应特别慢，那就慢慢等吧，它永远不会返回超时错误的。其用法如下：</p>
<pre class="lang-python" data-nodeid="18825"><code data-language="python">r = requests.get(<span class="hljs-string">'https://httpbin.org/get'</span>, timeout=<span class="hljs-literal">None</span>)
</code></pre>
<p data-nodeid="18826">或直接不加参数：</p>
<pre class="lang-python" data-nodeid="18827"><code data-language="python">r = requests.get(<span class="hljs-string">'https://httpbin.org/get'</span>)
</code></pre>
<h4 data-nodeid="18828">身份认证</h4>
<p data-nodeid="38895" class="">在访问某些设置了身份认证的网站时，例如：<a href="https://static3.scrape.center/" data-nodeid="38899">https://static3.scrape.center/</a>，我们可能会遇到这样的认证窗口，如图所示：</p>


<p data-nodeid="18830"><img src="https://s0.lgstatic.com/i/image3/M01/72/AB/CgpOIF5on8iAM1Y6AAAcAQw7Wy4460.png" alt="" data-nodeid="19078"></p>
<p data-nodeid="18831">如果遇到了这种情况，那就是这个网站启用了基本身份认证，英文叫作 HTTP Basic Access Authentication，它是一种用来允许网页浏览器或其他客户端程序在请求时提供用户名和口令形式的身份凭证的一种登录验证方式。</p>
<p data-nodeid="18832">如果遇到了这种情况，怎么用 reqeusts 来爬取呢，当然也有办法。</p>
<p data-nodeid="18833">我们可以使用 requests 自带的身份认证功能，通过 auth 参数即可设置，示例如下：</p>
<pre class="lang-python" data-nodeid="39863"><code data-language="python"><span class="hljs-keyword">import</span> requests
<span class="hljs-keyword">from</span> requests.auth <span class="hljs-keyword">import</span> HTTPBasicAuth

r = requests.get(<span class="hljs-string">'https://static3.scrape.center/'</span>, auth=HTTPBasicAuth(<span class="hljs-string">'admin'</span>, <span class="hljs-string">'admin'</span>))
print(r.status_code)
</code></pre>

<p data-nodeid="18835">这个示例网站的用户名和密码都是 admin，在这里我们可以直接设置。</p>
<p data-nodeid="18836">如果用户名和密码正确的话，请求时会自动认证成功，返回 200 状态码；如果认证失败，则返回 401 状态码。</p>
<p data-nodeid="18837">当然，如果参数都传一个 HTTPBasicAuth 类，就显得有点烦琐了，所以 requests 提供了一个更简单的写法，可以直接传一个元组，它会默认使用 HTTPBasicAuth 这个类来认证。</p>
<p data-nodeid="18838">所以上面的代码可以直接简写如下：</p>
<pre class="lang-python te-preview-highlight" data-nodeid="40826"><code data-language="python"><span class="hljs-keyword">import</span> requests

r = requests.get(<span class="hljs-string">'https://static3.scrape.center/'</span>, auth=(<span class="hljs-string">'admin'</span>, <span class="hljs-string">'admin'</span>))
print(r.status_code)
</code></pre>

<p data-nodeid="18840">此外，requests 还提供了其他认证方式，如 OAuth 认证，不过此时需要安装 oauth 包，安装命令如下：</p>
<pre class="lang-python" data-nodeid="18841"><code data-language="python">pip3 install requests_oauthlib
</code></pre>
<p data-nodeid="18842">使用 OAuth1 认证的方法如下：</p>
<pre class="lang-python" data-nodeid="18843"><code data-language="python"><span class="hljs-keyword">import</span> requests
<span class="hljs-keyword">from</span> requests_oauthlib <span class="hljs-keyword">import</span> OAuth1

url = <span class="hljs-string">'https://api.twitter.com/1.1/account/verify_credentials.json'</span>
auth = OAuth1(<span class="hljs-string">'YOUR_APP_KEY'</span>, <span class="hljs-string">'YOUR_APP_SECRET'</span>,
              <span class="hljs-string">'USER_OAUTH_TOKEN'</span>, <span class="hljs-string">'USER_OAUTH_TOKEN_SECRET'</span>)
requests.get(url, auth=auth)
</code></pre>
<p data-nodeid="18844">更多详细的功能就可以参考 requests_oauthlib 的官方文档：<a href="https://requests-oauthlib.readthedocs.org/" data-nodeid="19093">https://requests-oauthlib.readthedocs.org/</a>，在此就不再赘述了。</p>
<h4 data-nodeid="18845">代理设置</h4>
<p data-nodeid="18846">某些网站在测试的时候请求几次，能正常获取内容。但是对于大规模且频繁的请求，网站可能会弹出验证码，或者跳转到登录认证页面，更甚者可能会直接封禁客户端的 IP，导致一定时间段内无法访问。</p>
<p data-nodeid="18847">为了防止这种情况发生，我们需要设置代理来解决这个问题，这就需要用到 proxies 参数。可以用这样的方式设置：</p>
<pre class="lang-python" data-nodeid="18848"><code data-language="python"><span class="hljs-keyword">import</span> requests

proxies = {
  <span class="hljs-string">'http'</span>: <span class="hljs-string">'http://10.10.10.10:1080'</span>,
  <span class="hljs-string">'https'</span>: <span class="hljs-string">'http://10.10.10.10:1080'</span>,
}
requests.get(<span class="hljs-string">'https://httpbin.org/get'</span>, proxies=proxies)
</code></pre>
<p data-nodeid="18849">当然，直接运行这个实例或许行不通，因为这个代理可能是无效的，可以直接搜索寻找有效的代理并替换试验一下。</p>
<p data-nodeid="18850">若代理需要使用上文所述的身份认证，可以使用类似&nbsp;<a href="http://user:password@host:port" data-nodeid="19102">http://user:password@host:port</a>&nbsp;这样的语法来设置代理，示例如下：</p>
<pre class="lang-python" data-nodeid="18851"><code data-language="python"><span class="hljs-keyword">import</span> requests

proxies = {<span class="hljs-string">'https'</span>: <span class="hljs-string">'http://user:password@10.10.10.10:1080/'</span>,}
requests.get(<span class="hljs-string">'https://httpbin.org/get'</span>, proxies=proxies)
</code></pre>
<p data-nodeid="18852">除了基本的 HTTP 代理外，requests 还支持 SOCKS 协议的代理。</p>
<p data-nodeid="18853">首先，需要安装 socks 这个库：</p>
<pre class="lang-python" data-nodeid="18854"><code data-language="python">pip3 install <span class="hljs-string">"requests[socks]"</span>
</code></pre>
<p data-nodeid="18855">然后就可以使用 SOCKS 协议代理了，示例如下：</p>
<pre class="lang-python" data-nodeid="18856"><code data-language="python"><span class="hljs-keyword">import</span> requests

proxies = {
    <span class="hljs-string">'http'</span>: <span class="hljs-string">'socks5://user:password@host:port'</span>,
    <span class="hljs-string">'https'</span>: <span class="hljs-string">'socks5://user:password@host:port'</span>
}
requests.get(<span class="hljs-string">'https://httpbin.org/get'</span>, proxies=proxies)
</code></pre>
<h4 data-nodeid="18857">Prepared Request</h4>
<p data-nodeid="18858">我们使用 requests 库的 get 和 post 方法可以直接发送请求，但你有没有想过，这个请求在 requests 内部是怎么实现的呢？</p>
<p data-nodeid="18859">实际上，requests 在发送请求的时候在内部构造了一个 Request 对象，并给这个对象赋予了各种参数，包括 url、headers、data ，等等。然后直接把这个 Request 对象发送出去，请求成功后会再得到一个 Response 对象，再解析即可。</p>
<p data-nodeid="18860">那么这个 Request 是什么类型呢？实际上它就是 Prepared Request。</p>
<p data-nodeid="18861">我们深入一下，不用 get 方法，直接构造一个 Prepared Request 对象来试试，代码如下：</p>
<pre class="lang-python" data-nodeid="18862"><code data-language="python"><span class="hljs-keyword">from</span> requests <span class="hljs-keyword">import</span> Request, Session

url = <span class="hljs-string">'http://httpbin.org/post'</span>
data = {<span class="hljs-string">'name'</span>: <span class="hljs-string">'germey'</span>}
headers = {<span class="hljs-string">'User-Agent'</span>: <span class="hljs-string">'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36'</span>
}
s = Session()
req = Request(<span class="hljs-string">'POST'</span>, url, data=data, headers=headers)
prepped = s.prepare_request(req)
r = s.send(prepped)
print(r.text)
</code></pre>
<p data-nodeid="18863">这里我们引入了 Request，然后用 url、data 和 headers 参数构造了一个 Request 对象，这时需要再调用 Session 的 prepare_request 方法将其转换为一个 Prepared Request 对象，然后调用 send 方法发送，运行结果如下：</p>
<pre class="lang-python" data-nodeid="18864"><code data-language="python">{
  <span class="hljs-string">"args"</span>: {}, 
  <span class="hljs-string">"data"</span>: <span class="hljs-string">""</span>, 
  <span class="hljs-string">"files"</span>: {}, 
  <span class="hljs-string">"form"</span>: {
    <span class="hljs-string">"name"</span>: <span class="hljs-string">"germey"</span>
  }, 
  <span class="hljs-string">"headers"</span>: {
    <span class="hljs-string">"Accept"</span>: <span class="hljs-string">"*/*"</span>, 
    <span class="hljs-string">"Accept-Encoding"</span>: <span class="hljs-string">"gzip, deflate"</span>, 
    <span class="hljs-string">"Content-Length"</span>: <span class="hljs-string">"11"</span>, 
    <span class="hljs-string">"Content-Type"</span>: <span class="hljs-string">"application/x-www-form-urlencoded"</span>, 
    <span class="hljs-string">"Host"</span>: <span class="hljs-string">"httpbin.org"</span>, 
    <span class="hljs-string">"User-Agent"</span>: <span class="hljs-string">"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36"</span>, 
    <span class="hljs-string">"X-Amzn-Trace-Id"</span>: <span class="hljs-string">"Root=1-5e5bd6a9-6513c838f35b06a0751606d8"</span>
  }, 
  <span class="hljs-string">"json"</span>: null, 
  <span class="hljs-string">"origin"</span>: <span class="hljs-string">"167.220.232.237"</span>, 
  <span class="hljs-string">"url"</span>: <span class="hljs-string">"http://httpbin.org/post"</span>
}
</code></pre>
<p data-nodeid="18865">可以看到，我们达到了同样的 POST 请求效果。</p>
<p data-nodeid="18866">有了 Request 这个对象，就可以将请求当作独立的对象来看待，这样在一些场景中我们可以直接操作这个 Request 对象，更灵活地实现请求的调度和各种操作。</p>
<p data-nodeid="18867">更多的用法可以参考 requests 的官方文档：<a href="http://docs.python-requests.org/" data-nodeid="19120">http://docs.python-requests.org/</a>。</p>

# 解析无所不能的正则表达式
<p data-nodeid="4145" class="">在上个课时中，我们学会了如何用 Requests 来获取网页的源代码，得到 HTML 代码。但我们如何从 HTML 代码中获取真正想要的数据呢？</p>
<p data-nodeid="4146">正则表达式就是一个有效的方法。</p>
<p data-nodeid="4147">本课时中，我们将学习正则表达式的相关用法。正则表达式是处理字符串的强大工具，它有自己特定的语法结构。有了它，我们就能实现字符串的检索、替换、匹配验证。</p>
<p data-nodeid="4148">当然，对于爬虫来说，有了它，要从 HTML 里提取想要的信息就非常方便了。</p>
<h3 data-nodeid="4149">实例引入</h3>
<p data-nodeid="4150">说了这么多，可能我们对正则表达式的概念还是比较模糊，下面就用几个实例来看一下正则表达式的用法。</p>
<p data-nodeid="4151">打开开源中国提供的正则表达式测试工具&nbsp;<a href="http://tool.oschina.net/regex/" data-nodeid="4445">http://tool.oschina.net/regex/</a>，输入待匹配的文本，然后选择常用的正则表达式，就可以得出相应的匹配结果了。</p>
<p data-nodeid="4152">例如，输入下面这段待匹配的文本：</p>
<pre class="lang-python" data-nodeid="4153"><code data-language="python">Hello, my phone number <span class="hljs-keyword">is</span> <span class="hljs-number">010</span><span class="hljs-number">-86432100</span> <span class="hljs-keyword">and</span> email <span class="hljs-keyword">is</span> cqc@cuiqingcai.com, <span class="hljs-keyword">and</span> my website <span class="hljs-keyword">is</span> https://cuiqingcai.com.
</code></pre>
<p data-nodeid="4154">这段字符串中包含了一个电话号码和一个电子邮件，接下来就尝试用正则表达式提取出来，如图所示。</p>
<p data-nodeid="4155"><img src="https://s0.lgstatic.com/i/image3/M01/74/26/Cgq2xl5rZiKAcRoMAAKSM5SSmyk124.png" alt="" data-nodeid="4450"></p>
<p data-nodeid="4156">在网页右侧选择 “匹配 Email 地址”，就可以看到下方出现了文本中的 E-mail。如果选择 “匹配网址 URL”，就可以看到下方出现了文本中的 URL。是不是非常神奇？</p>
<p data-nodeid="4157">其实，这里使用了正则表达式的匹配功能，也就是用一定规则将特定的文本提取出来。</p>
<p data-nodeid="4158">比方说，电子邮件是有其特定的组成格式的：一段字符串 + @ 符号 + 某个域名。而 URL的组成格式则是协议类型 + 冒号加双斜线 + 域名和路径。</p>
<p data-nodeid="4159">可以用下面的正则表达式匹配 URL：</p>
<pre data-nodeid="4160"><code>[a-zA-z]+://[^\s]*
</code></pre>
<p data-nodeid="4161">用这个正则表达式去匹配一个字符串，如果这个字符串中包含类似 URL 的文本，那就会被提取出来。</p>
<p data-nodeid="4162">这个看上去乱糟糟的正则表达式其实有特定的语法规则。比如，a-z 匹配任意的小写字母，\s 匹配任意的空白字符，* 匹配前面任意多个字符。这一长串的正则表达式就是这么多匹配规则的组合。</p>
<p data-nodeid="4163">写好正则表达式后，就可以拿它去一个长字符串里匹配查找了。不论这个字符串里面有什么，只要符合我们写的规则，统统可以找出来。对于网页来说，如果想找出网页源代码里有多少 URL，用 URL 的正则表达式去匹配即可。</p>
<p data-nodeid="4164">下表中列出了常用的匹配规则：<br>
<br></p>
<table data-nodeid="4165">
<thead data-nodeid="4166">
<tr data-nodeid="4167">
<th data-nodeid="4169"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">模　　式</span></span></span></th>
<th data-nodeid="4170"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">描　　述</span></span></span></th>
</tr>
</thead>
<tbody data-nodeid="4173">
<tr data-nodeid="4174">
<td data-nodeid="4175"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">\w</span></span></span></td>
<td data-nodeid="4176"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">匹配字母、数字及下划线</span></span></span></td>
</tr>
<tr data-nodeid="4177">
<td data-nodeid="4178"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">\W</span></span></span></td>
<td data-nodeid="4179"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">匹配不是字母、数字及下划线的字符</span></span></span></td>
</tr>
<tr data-nodeid="4180">
<td data-nodeid="4181"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">\s</span></span></span></td>
<td data-nodeid="4182"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">匹配任意空白字符，等价于 [\t\n\r\f]</span></span></span></td>
</tr>
<tr data-nodeid="4183">
<td data-nodeid="4184"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">\S</span></span></span></td>
<td data-nodeid="4185"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">匹配任意非空字符</span></span></span></td>
</tr>
<tr data-nodeid="4186">
<td data-nodeid="4187"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">\d</span></span></span></td>
<td data-nodeid="4188"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">匹配任意数字，等价于 [0~9]</span></span></span></td>
</tr>
<tr data-nodeid="4189">
<td data-nodeid="4190"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">\D</span></span></span></td>
<td data-nodeid="4191"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">匹配任意非数字的字符</span></span></span></td>
</tr>
<tr data-nodeid="4192">
<td data-nodeid="4193"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">\A</span></span></span></td>
<td data-nodeid="4194"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">匹配字符串开头</span></span></span></td>
</tr>
<tr data-nodeid="4195">
<td data-nodeid="4196"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">\Z</span></span></span></td>
<td data-nodeid="4197"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">匹配字符串结尾，如果存在换行，只匹配到换行前的结束字符串</span></span></span></td>
</tr>
<tr data-nodeid="4198">
<td data-nodeid="4199"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">\z</span></span></span></td>
<td data-nodeid="4200"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">匹配字符串结尾，如果存在换行，同时还会匹配换行符</span></span></span></td>
</tr>
<tr data-nodeid="4201">
<td data-nodeid="4202"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">\G</span></span></span></td>
<td data-nodeid="4203"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">匹配最后匹配完成的位置</span></span></span></td>
</tr>
<tr data-nodeid="4204">
<td data-nodeid="4205"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">\n</span></span></span></td>
<td data-nodeid="4206"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">匹配一个换行符</span></span></span></td>
</tr>
<tr data-nodeid="4207">
<td data-nodeid="4208"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">\t</span></span></span></td>
<td data-nodeid="4209"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">匹配一个制表符</span></span></span></td>
</tr>
<tr data-nodeid="4210">
<td data-nodeid="4211"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">^</span></span></span></td>
<td data-nodeid="4212"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">匹配一行字符串的开头</span></span></span></td>
</tr>
<tr data-nodeid="4213">
<td data-nodeid="4214"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">$</span></span></span></td>
<td data-nodeid="4215"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">匹配一行字符串的结尾</span></span></span></td>
</tr>
<tr data-nodeid="4216">
<td data-nodeid="4217"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">.</span></span></span></td>
<td data-nodeid="4218"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">匹配任意字符，除了换行符，当 re.DOTALL 标记被指定时，则可以匹配包括换行符的任意字符</span></span></span></td>
</tr>
<tr data-nodeid="4219">
<td data-nodeid="4220"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">[...]</span></span></span></td>
<td data-nodeid="4221"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">用来表示一组字符，单独列出，比如 [amk] 匹配 a、m 或 k</span></span></span></td>
</tr>
<tr data-nodeid="4222">
<td data-nodeid="4223"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">[^...]</span></span></span></td>
<td data-nodeid="4224"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">不在 [] 中的字符，比如 &nbsp;匹配除了 a、b、c 之外的字符</span></span></span></td>
</tr>
<tr data-nodeid="4225">
<td data-nodeid="4226"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">*</span></span></span></td>
<td data-nodeid="4227"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">匹配 0 个或多个表达式</span></span></span></td>
</tr>
<tr data-nodeid="4228">
<td data-nodeid="4229"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">+</span></span></span></td>
<td data-nodeid="4230"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">匹配 1 个或多个表达式</span></span></span></td>
</tr>
<tr data-nodeid="4231">
<td data-nodeid="4232"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">?</span></span></span></td>
<td data-nodeid="4233"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">匹配 0 个或 1 个前面的正则表达式定义的片段，非贪婪方式</span></span></span></td>
</tr>
<tr data-nodeid="4234">
<td data-nodeid="4235"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">{n}</span></span></span></td>
<td data-nodeid="4236"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">精确匹配 n 个前面的表达式</span></span></span></td>
</tr>
<tr data-nodeid="4237">
<td data-nodeid="4238"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">{n, m}</span></span></span></td>
<td data-nodeid="4239"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">匹配 n 到 m 次由前面正则表达式定义的片段，贪婪方式</span></span></span></td>
</tr>
<tr data-nodeid="4240">
<td data-nodeid="4241"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">a|b</span></span></span></td>
<td data-nodeid="4242"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">匹配 a 或 b</span></span></span></td>
</tr>
<tr data-nodeid="4243">
<td data-nodeid="4244"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">()</span></span></span></td>
<td data-nodeid="4245"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">匹配括号内的表达式，也表示一个组</span></span></span></td>
</tr>
</tbody>
</table>
<p data-nodeid="4246">看完之后，你可能有点晕晕的吧，不用担心，后面我们会详细讲解一些常见规则的用法。</p>
<p data-nodeid="4247">其实正则表达式不是 Python 独有的，它也可以用在其他编程语言中。但是 Python 的 re 库提供了整个正则表达式的实现，利用这个库，可以在 Python 中使用正则表达式。</p>
<p data-nodeid="4248">在 Python 中写正则表达式几乎都用这个库，下面就来了解它的一些常用方法。</p>
<h3 data-nodeid="4249">match</h3>
<p data-nodeid="4250">首先介绍一个常用的匹配方法 —— match，向它传入要匹配的字符串，以及正则表达式，就可以检测这个正则表达式是否匹配字符串。</p>
<p data-nodeid="4251">match 方法会尝试从字符串的起始位置匹配正则表达式，如果匹配，就返回匹配成功的结果；如果不匹配，就返回 None。</p>
<p data-nodeid="4252">示例如下：</p>
<pre class="lang-python" data-nodeid="4253"><code data-language="python"><span class="hljs-keyword">import</span> re

content = <span class="hljs-string">'Hello 123 4567 World_This is a Regex Demo'</span>
print(len(content))
result = re.match(<span class="hljs-string">'^Hello\s\d\d\d\s\d{4}\s\w{10}'</span>, content)
print(result)
print(result.group())
print(result.span())
</code></pre>
<p data-nodeid="4254">运行结果如下：</p>
<pre class="lang-html" data-nodeid="4255"><code data-language="html">41
<span class="hljs-tag">&lt;<span class="hljs-name">_sre.SRE_Match</span> <span class="hljs-attr">object</span>; <span class="hljs-attr">span</span>=<span class="hljs-string">(0,</span> <span class="hljs-attr">25</span>), <span class="hljs-attr">match</span>=<span class="hljs-string">'Hello 123 4567 World_This'</span>&gt;</span>
Hello 123 4567 World_This
(0, 25)
</code></pre>
<p data-nodeid="4256">这里首先声明了一个字符串，其中包含英文字母、空白字符、数字等。接下来，我们写一个正则表达式：</p>
<pre class="lang-python" data-nodeid="4257"><code data-language="python">^Hello\s\d\d\d\s\d{<span class="hljs-number">4</span>}\s\w{<span class="hljs-number">10</span>}
</code></pre>
<p data-nodeid="4258">用它来匹配这个长字符串。开头的 ^ 匹配字符串的开头，也就是以 Hello 开头； \s 匹配空白字符，用来匹配目标字符串的空格；\d 匹配数字，3 个 \d 匹配 123；再写 1 个 \s 匹配空格；后面的 4567，其实依然能用 4 个 \d 来匹配，但是这么写比较烦琐，所以后面可以跟 {4} 代表匹配前面的规则 4 次，也就是匹配 4 个数字；后面再紧接 1 个空白字符，最后\w{10} 匹配 10 个字母及下划线。</p>
<p data-nodeid="4259">我们注意到，这里并没有把目标字符串匹配完，不过依然可以进行匹配，只不过匹配结果短一点而已。</p>
<p data-nodeid="4260">而在 match 方法中，第一个参数传入正则表达式，第二个参数传入要匹配的字符串。</p>
<p data-nodeid="4261">打印输出结果，可以看到结果是 SRE_Match 对象，这证明成功匹配。该对象有两个方法：group 方法可以输出匹配的内容，结果是 Hello 123 4567 World_This，这恰好是正则表达式规则所匹配的内容；span 方法可以输出匹配的范围，结果是 (0, 25)，这就是匹配到的结果字符串在原字符串中的位置范围。</p>
<p data-nodeid="4262">通过上面的例子，我们基本了解了如何在 Python 中使用正则表达式来匹配一段文字。</p>
<h4 data-nodeid="4263">匹配目标</h4>
<p data-nodeid="4264">刚才我们用 match 方法得到了匹配到的字符串内容，但当我们想从字符串中提取一部分内容，该怎么办呢？</p>
<p data-nodeid="4265">就像最前面的实例一样，要从一段文本中提取出邮件或电话号码等内容。我们可以使用 () 括号将想提取的子字符串括起来。() 实际上标记了一个子表达式的开始和结束位置，被标记的每个子表达式会依次对应每一个分组，调用 group 方法传入分组的索引即可获取提取的结果。</p>
<p data-nodeid="4266">示例如下：</p>
<pre class="lang-python" data-nodeid="4267"><code data-language="python"><span class="hljs-keyword">import</span> re

content = <span class="hljs-string">'Hello 1234567 World_This is a Regex Demo'</span>
result = re.match(<span class="hljs-string">'^Hello\s(\d+)\sWorld'</span>, content)
print(result)
print(result.group())
print(result.group(<span class="hljs-number">1</span>))
print(result.span())
</code></pre>
<p data-nodeid="4268">这里我们想把字符串中的 1234567 提取出来，此时可以将数字部分的正则表达式用 () 括起来，然后调用了 group(1) 获取匹配结果。</p>
<p data-nodeid="4269">运行结果如下：</p>
<pre class="lang-html" data-nodeid="4270"><code data-language="html"><span class="hljs-tag">&lt;<span class="hljs-name">_sre.SRE_Match</span> <span class="hljs-attr">object</span>; <span class="hljs-attr">span</span>=<span class="hljs-string">(0,</span> <span class="hljs-attr">19</span>), <span class="hljs-attr">match</span>=<span class="hljs-string">'Hello 1234567 World'</span>&gt;</span>
Hello 1234567 World
1234567
(0, 19)
</code></pre>
<p data-nodeid="4271">可以看到，我们成功得到了 1234567。这里用的是 group(1)，它与 group() 有所不同，后者会输出完整的匹配结果，而前者会输出第一个被 () 包围的匹配结果。假如正则表达式后面还有 () 包括的内容，那么可以依次用 group(2)、group(3) 等来获取。</p>
<h4 data-nodeid="4272">通用匹配</h4>
<p data-nodeid="4273">刚才我们写的正则表达比较复杂，出现空白字符我们就写 \s 匹配，出现数字我们就用 \d 匹配，这样的工作量非常大。</p>
<p data-nodeid="4274">我们还可以用一个万能匹配来减少这些工作，那就是 .*。其中 . 可以匹配任意字符（除换行符），* 代表匹配前面的字符无限次，它们组合在一起就可以匹配任意字符了。有了它，我们就不用挨个字符的匹配了。</p>
<p data-nodeid="4275">接着上面的例子，我们可以改写一下正则表达式：</p>
<pre class="lang-python" data-nodeid="4276"><code data-language="python"><span class="hljs-keyword">import</span> re

content = <span class="hljs-string">'Hello 123 4567 World_This is a Regex Demo'</span>
result = re.match(<span class="hljs-string">'^Hello.*Demo$'</span>, content)
print(result)
print(result.group())
print(result.span())
</code></pre>
<p data-nodeid="4277">这里我们将中间部分直接省略，全部用 .* 来代替，最后加一个结尾字符就好了。</p>
<p data-nodeid="4278">运行结果如下：</p>
<pre class="lang-html" data-nodeid="4279"><code data-language="html"><span class="hljs-tag">&lt;<span class="hljs-name">_sre.SRE_Match</span> <span class="hljs-attr">object</span>; <span class="hljs-attr">span</span>=<span class="hljs-string">(0,</span> <span class="hljs-attr">41</span>), <span class="hljs-attr">match</span>=<span class="hljs-string">'Hello 123 4567 World_This is a Regex Demo'</span>&gt;</span>
Hello 123 4567 World_This is a Regex Demo
(0, 41)
</code></pre>
<p data-nodeid="4280">可以看到，group 方法输出了匹配的全部字符串，也就是说我们写的正则表达式匹配到了目标字符串的全部内容；span 方法输出 (0, 41)，这是整个字符串的长度。</p>
<p data-nodeid="4281">因此，我们可以使用 .* 简化正则表达式的书写。</p>
<h4 data-nodeid="4282">贪婪与非贪婪</h4>
<p data-nodeid="4283">使用上面的通用匹配 .* 时，有时候匹配到的并不是我们想要的结果。</p>
<p data-nodeid="4284">看下面的例子：</p>
<pre class="lang-python" data-nodeid="4285"><code data-language="python"><span class="hljs-keyword">import</span> re

content = <span class="hljs-string">'Hello 1234567 World_This is a Regex Demo'</span>
result = re.match(<span class="hljs-string">'^He.*(\d+).*Demo$'</span>, content)
print(result)
print(result.group(<span class="hljs-number">1</span>))
</code></pre>
<p data-nodeid="4286">这里我们依然想获取中间的数字，所以中间依然写的是 (\d+)。由于数字两侧的内容比较杂乱，所以略写成 .*。最后，组成 ^He.*(\d+).*Demo$，看样子并没有什么问题。</p>
<p data-nodeid="4287">我们看下运行结果：</p>
<pre class="lang-html" data-nodeid="4288"><code data-language="html"><span class="hljs-tag">&lt;<span class="hljs-name">_sre.SRE_Match</span> <span class="hljs-attr">object</span>; <span class="hljs-attr">span</span>=<span class="hljs-string">(0,</span> <span class="hljs-attr">40</span>), <span class="hljs-attr">match</span>=<span class="hljs-string">'Hello 1234567 World_This is a Regex Demo'</span>&gt;</span>
7
</code></pre>
<p data-nodeid="4289">奇怪的事情发生了，我们只得到了 7 这个数字，这是怎么回事呢？</p>
<p data-nodeid="4290">这里就涉及一个贪婪匹配与非贪婪匹配的问题了。在贪婪匹配下，.* 会匹配尽可能多的字符。正则表达式中 .* 后面是 \d+，也就是至少一个数字，并没有指定具体多少个数字，因此，.* 就尽可能匹配多的字符，这里就把 123456 匹配了，给 \d+ 留下一个可满足条件的数字 7，最后得到的内容就只有数字 7 了。</p>
<p data-nodeid="4291">这显然会给我们带来很大的不便。有时候，匹配结果会莫名其妙少了一部分内容。其实，这里只需要使用非贪婪匹配就好了。非贪婪匹配的写法是 .*?，多了一个 ？，那么它可以达到怎样的效果？</p>
<p data-nodeid="4292">我们再用实例看一下：</p>
<pre class="lang-python" data-nodeid="4293"><code data-language="python"><span class="hljs-keyword">import</span> re

content = <span class="hljs-string">'Hello 1234567 World_This is a Regex Demo'</span>
result = re.match(<span class="hljs-string">'^He.*?(\d+).*Demo$'</span>, content)
print(result)
print(result.group(<span class="hljs-number">1</span>))
</code></pre>
<p data-nodeid="4294">这里我们只是将第一个.* 改成了 .*?，转变为非贪婪匹配。</p>
<p data-nodeid="4295">结果如下：</p>
<pre class="lang-html" data-nodeid="4296"><code data-language="html"><span class="hljs-tag">&lt;<span class="hljs-name">_sre.SRE_Match</span> <span class="hljs-attr">object</span>; <span class="hljs-attr">span</span>=<span class="hljs-string">(0,</span> <span class="hljs-attr">40</span>), <span class="hljs-attr">match</span>=<span class="hljs-string">'Hello 1234567 World_This is a Regex Demo'</span>&gt;</span>
1234567
</code></pre>
<p data-nodeid="4297">此时就可以成功获取 1234567 了。原因可想而知，贪婪匹配是尽可能匹配多的字符，非贪婪匹配就是尽可能匹配少的字符。当 .*? 匹配到 Hello 后面的空白字符时，再往后的字符就是数字了，而 \d+ 恰好可以匹配，那么 .*? 就不再进行匹配，交给 \d+ 去匹配后面的数字。这样 .*? 匹配了尽可能少的字符，\d+ 的结果就是 1234567 了。</p>
<p data-nodeid="4298">所以，在做匹配的时候，字符串中间尽量使用非贪婪匹配，也就是用 .*? 来代替 .*，以免出现匹配结果缺失的情况。</p>
<p data-nodeid="4299">但需要注意的是，如果匹配的结果在字符串结尾，.*? 就有可能匹配不到任何内容了，因为它会匹配尽可能少的字符。例如：</p>
<pre class="lang-python" data-nodeid="4300"><code data-language="python"><span class="hljs-keyword">import</span> re

content = <span class="hljs-string">'http://weibo.com/comment/kEraCN'</span>
result1 = re.match(<span class="hljs-string">'http.*?comment/(.*?)'</span>, content)
result2 = re.match(<span class="hljs-string">'http.*?comment/(.*)'</span>, content)
print(<span class="hljs-string">'result1'</span>, result1.group(<span class="hljs-number">1</span>))
print(<span class="hljs-string">'result2'</span>, result2.group(<span class="hljs-number">1</span>))
</code></pre>
<p data-nodeid="4301">运行结果如下：</p>
<pre data-nodeid="9327" class="te-preview-highlight"><code>result1 
result2 kEraCN
</code></pre>



<p data-nodeid="4303" class="">可以观察到，.*? 没有匹配到任何结果，而 .* 则尽量匹配多的内容，成功得到了匹配结果。</p>
<h4 data-nodeid="4304">修饰符</h4>
<p data-nodeid="4305">正则表达式可以包含一些可选标志修饰符来控制匹配的模式。修饰符被指定为一个可选的标志。</p>
<p data-nodeid="4306">我们用实例来看一下：</p>
<pre class="lang-python" data-nodeid="4307"><code data-language="python"><span class="hljs-keyword">import</span> re

content = <span class="hljs-string">'''Hello 1234567 World_This
is a Regex Demo
'''</span>
result = re.match(<span class="hljs-string">'^He.*?(\d+).*?Demo$'</span>, content)
print(result.group(<span class="hljs-number">1</span>))
</code></pre>
<p data-nodeid="4308">和上面的例子相仿，我们在字符串中加了换行符，正则表达式还是一样的，用来匹配其中的数字。看一下运行结果：</p>
<pre class="lang-html" data-nodeid="4309"><code data-language="html">AttributeError Traceback (most recent call last)
<span class="hljs-tag">&lt;<span class="hljs-name">ipython-input-18-c7d232b39645</span>&gt;</span> in <span class="hljs-tag">&lt;<span class="hljs-name">module</span>&gt;</span>()
      5 '''
      6 result = re.match('^He.*?(\d+).*?Demo$', content)
----&gt; 7 print(result.group(1))

AttributeError: 'NoneType' object has no attribute 'group'
</code></pre>
<p data-nodeid="4310">运行直接报错，也就是说正则表达式没有匹配到这个字符串，返回结果为 None，而我们又调用了 group 方法导致 AttributeError。</p>
<p data-nodeid="4311">为什么加了一个换行符，就匹配不到了呢？</p>
<p data-nodeid="4312">这是因为我们匹配的是除换行符之外的任意字符，当遇到换行符时，.*? 就不能匹配了，导致匹配失败。</p>
<p data-nodeid="4313">这里只需加一个修饰符 re.S，即可修正这个错误：</p>
<pre class="lang-python" data-nodeid="4314"><code data-language="python">result = re.match(<span class="hljs-string">'^He.*?(\d+).*?Demo$'</span>, content, re.S)
</code></pre>
<p data-nodeid="4315">这个修饰符的作用是匹配包括换行符在内的所有字符。</p>
<p data-nodeid="4316">此时运行结果如下：</p>
<pre data-nodeid="4317"><code>1234567
</code></pre>
<p data-nodeid="4318">这个 re.S 在网页匹配中经常用到。因为 HTML 节点经常会有换行，加上它，就可以匹配节点与节点之间的换行了。</p>
<p data-nodeid="4319">另外，还有一些修饰符，在必要的情况下也可以使用，如表所示：</p>
<table data-nodeid="4321">
<thead data-nodeid="4322">
<tr data-nodeid="4323">
<th data-nodeid="4325"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">修饰符</span></span></span></th>
<th data-nodeid="4326"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">描　　述</span></span></span></th>
</tr>
</thead>
<tbody data-nodeid="4329">
<tr data-nodeid="4330">
<td data-nodeid="4331"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">re.I</span></span></span></td>
<td data-nodeid="4332"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">使匹配对大小写不敏感</span></span></span></td>
</tr>
<tr data-nodeid="4333">
<td data-nodeid="4334"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">re.L</span></span></span></td>
<td data-nodeid="4335"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">做本地化识别（locale-aware）匹配</span></span></span></td>
</tr>
<tr data-nodeid="4336">
<td data-nodeid="4337"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">re.M</span></span></span></td>
<td data-nodeid="4338"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">多行匹配，影响 ^ 和 $</span></span></span></td>
</tr>
<tr data-nodeid="4339">
<td data-nodeid="4340"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">re.S</span></span></span></td>
<td data-nodeid="4341"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">使匹配包括换行在内的所有字符</span></span></span></td>
</tr>
<tr data-nodeid="4342">
<td data-nodeid="4343"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">re.U</span></span></span></td>
<td data-nodeid="4344"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">根据 Unicode 字符集解析字符。这个标志影响 \w、\W、\b 和 \B</span></span></span></td>
</tr>
<tr data-nodeid="4345">
<td data-nodeid="4346"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">re.X</span></span></span></td>
<td data-nodeid="4347"><span style="color:#3f3f3f"><span class="font" style="font-family:微软雅黑, &quot;Microsoft YaHei&quot;"><span class="size" style="font-size:16px">该标志通过给予你更灵活的格式以便你将正则表达式写得更易于理解</span></span></span></td>
</tr>
</tbody>
</table>
<p data-nodeid="4348">在网页匹配中，较为常用的修饰符有 re.S 和 re.I。</p>
<h4 data-nodeid="4349">转义匹配</h4>
<p data-nodeid="4350">我们知道正则表达式定义了许多匹配模式，如匹配除换行符以外的任意字符，但如果目标字符串里面就包含 .，那该怎么办呢？</p>
<p data-nodeid="4351">这里就需要用到转义匹配了，示例如下：</p>
<pre class="lang-python" data-nodeid="4352"><code data-language="python"><span class="hljs-keyword">import</span> re

content = <span class="hljs-string">'(百度) www.baidu.com'</span>
result = re.match(<span class="hljs-string">'\(百度 \) www\.baidu\.com'</span>, content)
print(result)
</code></pre>
<p data-nodeid="4353">当遇到用于正则匹配模式的特殊字符时，在前面加反斜线转义一下即可。例 . 就可以用 \. 来匹配。</p>
<p data-nodeid="4354">运行结果如下：</p>
<pre class="lang-html" data-nodeid="4355"><code data-language="html"><span class="hljs-tag">&lt;<span class="hljs-name">_sre.SRE_Match</span> <span class="hljs-attr">object</span>; <span class="hljs-attr">span</span>=<span class="hljs-string">(0,</span> <span class="hljs-attr">17</span>), <span class="hljs-attr">match</span>=<span class="hljs-string">'(百度) www.baidu.com'</span>&gt;</span>
</code></pre>
<p data-nodeid="4356">可以看到，这里成功匹配到了原字符串。</p>
<p data-nodeid="4357">这些是写正则表达式常用的几个知识点，熟练掌握它们对后面写正则表达式匹配非常有帮助。</p>
<h3 data-nodeid="4358">search</h3>
<p data-nodeid="4359">前面提到过，match 方法是从字符串的开头开始匹配的，一旦开头不匹配，那么整个匹配就失败了。</p>
<p data-nodeid="4360">我们看下面的例子：</p>
<pre class="lang-python" data-nodeid="4361"><code data-language="python"><span class="hljs-keyword">import</span> re

content = <span class="hljs-string">'Extra stings Hello 1234567 World_This is a Regex Demo Extra stings'</span>
result = re.match(<span class="hljs-string">'Hello.*?(\d+).*?Demo'</span>, content)
print(result)
</code></pre>
<p data-nodeid="4362">这里的字符串以 Extra 开头，但是正则表达式以 Hello 开头，整个正则表达式是字符串的一部分，但是这样匹配是失败的。</p>
<p data-nodeid="4363">运行结果如下：</p>
<pre data-nodeid="4364"><code>None
</code></pre>
<p data-nodeid="4365">因为 match 方法在使用时需要考虑到开头的内容，这在做匹配时并不方便。它更适合用来检测某个字符串是否符合某个正则表达式的规则。</p>
<p data-nodeid="4366">这里有另外一个方法 search，它在匹配时会扫描整个字符串，然后返回第一个成功匹配的结果。也就是说，正则表达式可以是字符串的一部分，在匹配时，search 方法会依次扫描字符串，直到找到第一个符合规则的字符串，然后返回匹配内容，如果搜索完了还没有找到，就返回 None。</p>
<p data-nodeid="4367">我们把上面代码中的 match 方法修改成 search，再看下运行结果：</p>
<pre class="lang-html" data-nodeid="4368"><code data-language="html"><span class="hljs-tag">&lt;<span class="hljs-name">_sre.SRE_Match</span> <span class="hljs-attr">object</span>; <span class="hljs-attr">span</span>=<span class="hljs-string">(13,</span> <span class="hljs-attr">53</span>), <span class="hljs-attr">match</span>=<span class="hljs-string">'Hello 1234567 World_This is a Regex Demo'</span>&gt;</span>
1234567
</code></pre>
<p data-nodeid="4369">这时就得到了匹配结果。</p>
<p data-nodeid="4370">因此，为了匹配方便，我们可以尽量使用 search 方法。</p>
<p data-nodeid="4371">下面再用几个实例来看看 search 方法的用法。</p>
<p data-nodeid="4372">这里有一段待匹配的 HTML 文本，接下来我们写几个正则表达式实例来实现相应信息的提取：</p>
<pre class="lang-html" data-nodeid="4373"><code data-language="html">html = '''<span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">id</span>=<span class="hljs-string">"songs-list"</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">h2</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"title"</span>&gt;</span>经典老歌<span class="hljs-tag">&lt;/<span class="hljs-name">h2</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">p</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"introduction"</span>&gt;</span>
经典老歌列表
<span class="hljs-tag">&lt;/<span class="hljs-name">p</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">ul</span> <span class="hljs-attr">id</span>=<span class="hljs-string">"list"</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"list-group"</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">data-view</span>=<span class="hljs-string">"2"</span>&gt;</span>一路上有你<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">data-view</span>=<span class="hljs-string">"7"</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"/2.mp3"</span> <span class="hljs-attr">singer</span>=<span class="hljs-string">"任贤齐"</span>&gt;</span>沧海一声笑<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">data-view</span>=<span class="hljs-string">"4"</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"active"</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"/3.mp3"</span> <span class="hljs-attr">singer</span>=<span class="hljs-string">"齐秦"</span>&gt;</span>往事随风<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">data-view</span>=<span class="hljs-string">"6"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"/4.mp3"</span> <span class="hljs-attr">singer</span>=<span class="hljs-string">"beyond"</span>&gt;</span>光辉岁月<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">data-view</span>=<span class="hljs-string">"5"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"/5.mp3"</span> <span class="hljs-attr">singer</span>=<span class="hljs-string">"陈慧琳"</span>&gt;</span>记事本<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">data-view</span>=<span class="hljs-string">"5"</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"/6.mp3"</span> <span class="hljs-attr">singer</span>=<span class="hljs-string">"邓丽君"</span>&gt;</span>但愿人长久<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">ul</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>'''
</code></pre>
<p data-nodeid="4374">可以观察到，ul 节点里有许多 li 节点，其中 li 节点中有的包含 a 节点，有的不包含 a 节点，a 节点还有一些相应的属性 —— 超链接和歌手名。</p>
<p data-nodeid="4375">首先，我们尝试提取 class为 active 的 li 节点内部超链接包含的歌手名和歌名，此时需要提取第三个 li 节点下 a 节点的 singer 属性和文本。</p>
<p data-nodeid="4376">此时，正则表达式可以用 li 开头，然后寻找一个标志符 active，中间的部分可以用 .*? 来匹配。</p>
<p data-nodeid="4377">接下来，要提取 singer 这个属性值，所以还需要写入 singer="(.*?)"，这里需要提取的部分用小括号括起来，以便用 group 方法提取出来，它的两侧边界是双引号。</p>
<p data-nodeid="4378">然后还需要匹配 a 节点的文本，其中它的左边界是 &gt;，右边界是 &lt;/a&gt;。目标内容依然用 (.*?) 来匹配，所以最后的正则表达式就变成了：</p>
<pre class="lang-html" data-nodeid="4379"><code data-language="html"><span class="hljs-tag">&lt;<span class="hljs-name">li.*?active.*?singer="(.*?)"</span>&gt;</span>(.*?)<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span>
</code></pre>
<p data-nodeid="4380">然后再调用 search 方法，它会搜索整个 HTML 文本，找到符合正则表达式的第一个内容返回。</p>
<p data-nodeid="4381">另外，由于代码有换行，所以这里第三个参数需要传入 re.S。整个匹配代码如下：</p>
<pre class="lang-python" data-nodeid="4382"><code data-language="python">result = re.search(<span class="hljs-string">'&lt;li.*?active.*?singer="(.*?)"&gt;(.*?)&lt;/a&gt;'</span>, html, re.S) 
<span class="hljs-keyword">if</span> result:
    print(result.group(<span class="hljs-number">1</span>), result.group(<span class="hljs-number">2</span>))
</code></pre>
<p data-nodeid="4383">由于需要获取的歌手和歌名都已经用小括号包围，所以可以用 group 方法获取。</p>
<p data-nodeid="4384">运行结果如下：</p>
<pre data-nodeid="4385"><code>齐秦 往事随风
</code></pre>
<p data-nodeid="4386">可以看到，这正是 class 为 active 的 li 节点内部的超链接包含的歌手名和歌名。</p>
<p data-nodeid="4387">如果正则表达式不加 active（也就是匹配不带 class 为 active 的节点内容），那会怎样呢？我们将正则表达式中的 active 去掉。</p>
<p data-nodeid="4388">代码改写如下：</p>
<pre class="lang-python" data-nodeid="4389"><code data-language="python">result = re.search(<span class="hljs-string">'&lt;li.*?singer="(.*?)"&gt;(.*?)&lt;/a&gt;'</span>, html, re.S)
<span class="hljs-keyword">if</span> result:
    print(result.group(<span class="hljs-number">1</span>), result.group(<span class="hljs-number">2</span>))
</code></pre>
<p data-nodeid="4390">由于 search 方法会返回第一个符合条件的匹配目标，这里结果就变了：</p>
<pre data-nodeid="4391"><code>任贤齐 沧海一声笑
</code></pre>
<p data-nodeid="4392">把 active 标签去掉后，从字符串开头开始搜索，此时符合条件的节点就变成了第二个 li 节点，后面的不再匹配，所以运行结果变成第二个 li 节点中的内容。</p>
<p data-nodeid="4393">注意，在上面的两次匹配中，search 方法的第三个参数都加了 re.S，这使得 .*? 可以匹配换行，所以含有换行的 li 节点被匹配到了。如果我们将其去掉，结果会是什么？</p>
<p data-nodeid="4394">代码如下：</p>
<pre class="lang-python" data-nodeid="4395"><code data-language="python">result = re.search(<span class="hljs-string">'&lt;li.*?singer="(.*?)"&gt;(.*?)&lt;/a&gt;'</span>, html)
<span class="hljs-keyword">if</span> result:
    print(result.group(<span class="hljs-number">1</span>), result.group(<span class="hljs-number">2</span>))
</code></pre>
<p data-nodeid="4396">运行结果如下：</p>
<pre data-nodeid="4397"><code>beyond 光辉岁月
</code></pre>
<p data-nodeid="4398">可以看到，结果变成了第四个 li 节点的内容。这是因为第二个和第三个 li 节点都包含了换行符，去掉 re.S 之后，.*? 已经不能匹配换行符，所以正则表达式不会匹配到第二个和第三个 li 节点，而第四个 li 节点中不包含换行符，所以成功匹配。</p>
<p data-nodeid="4399">由于绝大部分的 HTML 文本都包含了换行符，所以尽量都需要加上 re.S 修饰符，以免出现匹配不到的问题。</p>
<h3 data-nodeid="4400">findall</h3>
<p data-nodeid="4401">前面我们介绍了 search 方法的用法，它可以返回匹配正则表达式的第一个内容，但是如果想要获取匹配正则表达式的所有内容，那该怎么办呢？这时就要借助 findall 方法了。</p>
<p data-nodeid="4402">该方法会搜索整个字符串，然后返回匹配正则表达式的所有内容。</p>
<p data-nodeid="4403">还是上面的 HTML 文本，如果想获取所有 a 节点的超链接、歌手和歌名，就可以将 search 方法换成 findall 方法。如果有返回结果的话，就是列表类型，所以需要遍历一下来依次获取每组内容。</p>
<p data-nodeid="4404">代码如下：</p>
<pre class="lang-python" data-nodeid="4405"><code data-language="python">results = re.findall(<span class="hljs-string">'&lt;li.*?href="(.*?)".*?singer="(.*?)"&gt;(.*?)&lt;/a&gt;'</span>, html, re.S)
print(results)
print(type(results))
<span class="hljs-keyword">for</span> result <span class="hljs-keyword">in</span> results:
    print(result)
    print(result[<span class="hljs-number">0</span>], result[<span class="hljs-number">1</span>], result[<span class="hljs-number">2</span>])
</code></pre>
<p data-nodeid="4406">运行结果如下：</p>
<pre class="lang-html" data-nodeid="4407"><code data-language="html">[('/2.mp3', ' 任贤齐 ', ' 沧海一声笑 '), ('/3.mp3', ' 齐秦 ', ' 往事随风 '), ('/4.mp3', 'beyond', ' 光辉岁月 '), ('/5.mp3', ' 陈慧琳 ', ' 记事本 '), ('/6.mp3', ' 邓丽君 ', ' 但愿人长久 ')]
<span class="hljs-tag">&lt;<span class="hljs-name">class</span> '<span class="hljs-attr">list</span>'&gt;</span>
('/2.mp3', ' 任贤齐 ', ' 沧海一声笑 ')
/2.mp3 任贤齐 沧海一声笑
('/3.mp3', ' 齐秦 ', ' 往事随风 ')
/3.mp3 齐秦 往事随风
('/4.mp3', 'beyond', ' 光辉岁月 ')
/4.mp3 beyond 光辉岁月
('/5.mp3', ' 陈慧琳 ', ' 记事本 ')
/5.mp3 陈慧琳 记事本
('/6.mp3', ' 邓丽君 ', ' 但愿人长久 ')
/6.mp3 邓丽君 但愿人长久
</code></pre>
<p data-nodeid="4408">可以看到，返回的列表中的每个元素都是元组类型，我们用对应的索引依次取出即可。</p>
<p data-nodeid="4409">如果只是获取第一个内容，可以用 search 方法。当需要提取多个内容时，可以用 findall 方法。</p>
<h3 data-nodeid="4410">sub</h3>
<p data-nodeid="4411">除了使用正则表达式提取信息外，有时候还需要借助它来修改文本。比如，想要把一串文本中的所有数字都去掉，如果只用字符串的 replace 方法，那就太烦琐了，这时可以借助 sub 方法。</p>
<p data-nodeid="4412">示例如下：</p>
<pre class="lang-python" data-nodeid="4413"><code data-language="python"><span class="hljs-keyword">import</span> re

content = <span class="hljs-string">'54aK54yr5oiR54ix5L2g'</span>
content = re.sub(<span class="hljs-string">'\d+'</span>, <span class="hljs-string">''</span>, content)
print(content)
</code></pre>
<p data-nodeid="4414">运行结果如下：</p>
<pre data-nodeid="4415"><code>aKyroiRixLg
</code></pre>
<p data-nodeid="4416">这里只需要给第一个参数传入 \d+ 来匹配所有的数字，第二个参数替换成的字符串（如果去掉该参数的话，可以赋值为空），第三个参数是原字符串。</p>
<p data-nodeid="4417">在上面的 HTML 文本中，如果想获取所有 li 节点的歌名，直接用正则表达式来提取可能比较烦琐。比如，可以写成这样子：</p>
<pre class="lang-python" data-nodeid="4418"><code data-language="python">results = re.findall(<span class="hljs-string">'&lt;li.*?&gt;\s*?(&lt;a.*?&gt;)?(\w+)(&lt;/a&gt;)?\s*?&lt;/li&gt;'</span>, html, re.S)
<span class="hljs-keyword">for</span> result <span class="hljs-keyword">in</span> results:
    print(result[<span class="hljs-number">1</span>])
</code></pre>
<p data-nodeid="4419">运行结果如下：</p>
<pre data-nodeid="4420"><code>一路上有你
沧海一声笑
往事随风
光辉岁月
记事本
但愿人长久
</code></pre>
<p data-nodeid="4421">此时借助 sub 方法就比较简单了。可以先用 sub 方法将 a 节点去掉，只留下文本，然后再利用 findall 提取就好了：</p>
<pre class="lang-html" data-nodeid="4422"><code data-language="html">html = re.sub('<span class="hljs-tag">&lt;<span class="hljs-name">a.*?</span>&gt;</span>|<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span>', '', html)
print(html)
results = re.findall('<span class="hljs-tag">&lt;<span class="hljs-name">li.*?</span>&gt;</span>(.*?)<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>', html, re.S)
for result in results:
    print(result.strip())
</code></pre>
<p data-nodeid="4423">运行结果如下：</p>
<pre class="lang-html" data-nodeid="4424"><code data-language="html"><span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">id</span>=<span class="hljs-string">"songs-list"</span>&gt;</span>
    <span class="hljs-tag">&lt;<span class="hljs-name">h2</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"title"</span>&gt;</span> 经典老歌 <span class="hljs-tag">&lt;/<span class="hljs-name">h2</span>&gt;</span>
    <span class="hljs-tag">&lt;<span class="hljs-name">p</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"introduction"</span>&gt;</span>
        经典老歌列表
    <span class="hljs-tag">&lt;/<span class="hljs-name">p</span>&gt;</span>
    <span class="hljs-tag">&lt;<span class="hljs-name">ul</span> <span class="hljs-attr">id</span>=<span class="hljs-string">"list"</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"list-group"</span>&gt;</span>
        <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">data-view</span>=<span class="hljs-string">"2"</span>&gt;</span> 一路上有你 <span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
        <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">data-view</span>=<span class="hljs-string">"7"</span>&gt;</span>
            沧海一声笑
        <span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
        <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">data-view</span>=<span class="hljs-string">"4"</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"active"</span>&gt;</span>
            往事随风
        <span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
        <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">data-view</span>=<span class="hljs-string">"6"</span>&gt;</span> 光辉岁月 <span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
        <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">data-view</span>=<span class="hljs-string">"5"</span>&gt;</span> 记事本 <span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
        <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">data-view</span>=<span class="hljs-string">"5"</span>&gt;</span>
            但愿人长久
        <span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
    <span class="hljs-tag">&lt;/<span class="hljs-name">ul</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
一路上有你
沧海一声笑
往事随风
光辉岁月
记事本
但愿人长久
</code></pre>
<p data-nodeid="4425">可以看到，a 节点经过 sub 方法处理后就没有了，随后我们通过 findall 方法直接提取即可。</p>
<p data-nodeid="4426">通过以上例子，你会发现，在适当的时候，借助 sub 方法可以起到事半功倍的效果。</p>
<h3 data-nodeid="4427">compile</h3>
<p data-nodeid="4428">前面所讲的方法都是用来处理字符串的方法，最后再介绍一下 compile 方法，这个方法可以将正则字符串编译成正则表达式对象，以便在后面的匹配中复用。</p>
<p data-nodeid="4429">示例代码如下：</p>
<pre class="lang-python" data-nodeid="4430"><code data-language="python"><span class="hljs-keyword">import</span> re

content1 = <span class="hljs-string">'2019-12-15 12:00'</span>
content2 = <span class="hljs-string">'2019-12-17 12:55'</span>
content3 = <span class="hljs-string">'2019-12-22 13:21'</span>
pattern = re.compile(<span class="hljs-string">'\d{2}:\d{2}'</span>)
result1 = re.sub(pattern, <span class="hljs-string">''</span>, content1)
result2 = re.sub(pattern, <span class="hljs-string">''</span>, content2)
result3 = re.sub(pattern, <span class="hljs-string">''</span>, content3)
print(result1, result2, result3)
</code></pre>
<p data-nodeid="4431">这里有 3 个日期，我们想分别将 3 个日期中的时间去掉，这时可以借助 sub 方法。该方法的第一个参数是正则表达式，但是我们没有必要重复写 3 个同样的正则表达式。此时可以借助 compile 方法将正则表达式编译成一个正则表达式对象，以便复用。</p>
<p data-nodeid="4432">运行结果如下：</p>
<pre data-nodeid="4433"><code>2019-12-15  2019-12-17  2019-12-22
</code></pre>
<p data-nodeid="4434">另外，compile 还可以传入修饰符，例如 re.S 等修饰符，这样在 search、findall 等方法中就不需要额外传了。所以，compile 方法可以说是给正则表达式做了一层封装，以便我们更好的复用。</p>
<p data-nodeid="4435" class="">到此，正则表达式的基本用法就介绍完了。后面我会通过具体的实例来讲解正则表达式的用法。</p>

# 爬虫解析利器PyQuery的使用
（相同功能的库还有：lxml xpath、beautifulsoap4）
<p>上一课时我们学习了正则表达式的基本用法，然而一旦你的正则表达式写法有问题，我们就无法获取需要的信息。</p>
<p>你可能会思考：每个网页，都有一定的特殊结构和层级关系，而且很多节点都有 id 或 class 作为区分，我们可以借助它们的结构和属性来提取信息吗？</p>
<p>这的确可行。这个课时我会为你介绍一个更加强大的 HTML 解析库：pyquery。利用它，我们可以直接解析 DOM 节点的结构，并通过 DOM 节点的一些属性快速进行内容提取。</p>
<p>接下来，我们就来感受一下 pyquery 的强大之处。</p>
<h3>准备工作</h3>
<p>pyquery 是 Python 的第三方库，我们可以借助于 pip3 来安装，安装命令如下：</p>
<pre><code data-language="python" class="lang-python">pip3 install pyquery
</code></pre>
<p>更详细的安装方法可以参考：<a href="https://cuiqingcai.com/5186.html">https://cuiqingcai.com/5186.html</a>。</p>
<h3>初始化</h3>
<p>我们在解析 HTML 文本的时候，首先需要将其初始化为一个 pyquery 对象。它的初始化方式有多种，比如直接传入字符串、传入 URL、传入文件名，等等。</p>
<p>下面我们来详细介绍一下。</p>
<h4>字符串初始化</h4>
<p>我们可以直接把 HTML 的内容当作参数来初始化 pyquery 对象。我们用一个实例来感受一下：</p>
<pre><code data-language="html" class="lang-html">html = '''
<span class="hljs-tag">&lt;<span class="hljs-name">div</span>&gt;</span>
    <span class="hljs-tag">&lt;<span class="hljs-name">ul</span>&gt;</span>
         <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span>first item<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
         <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link2.html"</span>&gt;</span>second item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
         <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
         <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
         <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link5.html"</span>&gt;</span>fifth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
     <span class="hljs-tag">&lt;/<span class="hljs-name">ul</span>&gt;</span>
 <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
'''
from pyquery import PyQuery as pq
doc = pq(html)
print(doc('li'))
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="html" class="lang-html"><span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span>first item<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link2.html"</span>&gt;</span>second item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link5.html"</span>&gt;</span>fifth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
</code></pre>
<p>这里首先引入 pyquery 这个对象，取别名为 pq，然后声明了一个长 HTML 字符串，并将其当作参数传递给 pyquery 类，这样就成功完成了初始化。</p>
<p>接下来，将初始化的对象传入 CSS 选择器。在这个实例中，我们传入 li 节点，这样就可以选择所有的 li 节点。</p>
<h4>URL 初始化</h4>
<p>初始化的参数不仅可以以字符串的形式传递，还可以传入网页的 URL，此时只需要指定参数为 url 即可：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span> pyquery <span class="hljs-keyword">import</span> PyQuery <span class="hljs-keyword">as</span> pq
doc = pq(url=<span class="hljs-string">'https://cuiqingcai.com'</span>)
print(doc(<span class="hljs-string">'title'</span>))
</code></pre>
<p>运行结果：</p>
<pre><code data-language="html" class="lang-html"><span class="hljs-tag">&lt;<span class="hljs-name">title</span>&gt;</span>静觅丨崔庆才的个人博客<span class="hljs-tag">&lt;/<span class="hljs-name">title</span>&gt;</span>
</code></pre>
<p>这样的话，pyquery 对象会首先请求这个 URL，然后用得到的 HTML 内容完成初始化。这就相当于将网页的源代码以字符串的形式传递给 pyquery 类来初始化。</p>
<p>它与下面的功能是相同的：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span> pyquery <span class="hljs-keyword">import</span> PyQuery <span class="hljs-keyword">as</span> pq
<span class="hljs-keyword">import</span> requests
doc = pq(requests.get(<span class="hljs-string">'https://cuiqingcai.com'</span>).text)
print(doc(<span class="hljs-string">'title'</span>))
</code></pre>
<h4>文件初始化</h4>
<p>当然除了传递一个 URL，我们还可以传递本地的文件名，参数指定为 filename 即可：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span> pyquery <span class="hljs-keyword">import</span> PyQuery <span class="hljs-keyword">as</span> pq
doc = pq(filename=<span class="hljs-string">'demo.html'</span>)
print(doc(<span class="hljs-string">'li'</span>))
</code></pre>
<p>当然，这里需要有一个本地 HTML 文件 demo.html，其内容是待解析的 HTML 字符串。这样它会先读取本地的文件内容，然后将文件内容以字符串的形式传递给 pyquery 类来初始化。</p>
<p>以上 3 种方式均可初始化，当然最常用的初始化方式还是以字符串形式传递。</p>
<h3>基本 CSS 选择器</h3>
<p>我们先用一个实例来感受一下 pyquery 的 CSS 选择器的用法：</p>
<pre><code data-language="html" class="lang-html">html = '''
<span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">id</span>=<span class="hljs-string">"container"</span>&gt;</span>
    <span class="hljs-tag">&lt;<span class="hljs-name">ul</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"list"</span>&gt;</span>
         <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span>first item<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
         <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link2.html"</span>&gt;</span>second item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
         <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
         <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
         <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link5.html"</span>&gt;</span>fifth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
     <span class="hljs-tag">&lt;/<span class="hljs-name">ul</span>&gt;</span>
 <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
'''
from pyquery import PyQuery as pq
doc = pq(html)
print(doc('#container .list li'))
print(type(doc('#container .list li')))
</code></pre>
<p>运行结果：</p>
<pre><code data-language="html" class="lang-html"><span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span>first item<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link2.html"</span>&gt;</span>second item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link5.html"</span>&gt;</span>fifth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">class</span> '<span class="hljs-attr">pyquery.pyquery.PyQuery</span>'&gt;</span>
</code></pre>
<p>在上面的例子中，我们初始化 pyquery 对象之后，传入 CSS 选择器 #container .list li，它的意思是先选取 id 为 container 的节点，然后再选取其内部 class 为 list 的所有 li 节点，最后打印输出。</p>
<p>可以看到，我们成功获取到了符合条件的节点。我们将它的类型打印输出后发现，它的类型依然是 pyquery 类型。</p>
<p>下面，我们直接遍历这些节点，然后调用 text 方法，就可以获取节点的文本内容，代码示例如下：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">for</span> item <span class="hljs-keyword">in</span> doc(<span class="hljs-string">'#container .list li'</span>).items():
    print(item.text())
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="html" class="lang-html">first item
second item
third item
fourth item
fifth item
</code></pre>
<p>怎么样？我们没有再写正则表达式，而是直接通过选择器和 text 方法，就得到了我们想要提取的文本信息，是不是方便多了？</p>
<p>下面我们再来详细了解一下 pyquery 的用法吧，我将为你讲解如何用它查找节点、遍历节点、获取各种信息等操作方法。掌握了这些，我们就能更高效地完成数据提取。</p>
<h3>查找节点</h3>
<p>下面我们介绍一些常用的查询方法。</p>
<h4>子节点</h4>
<p>查找子节点需要用到 find 方法，传入的参数是 CSS 选择器，我们还是以上面的 HTML 为例：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span> pyquery <span class="hljs-keyword">import</span> PyQuery <span class="hljs-keyword">as</span> pq
doc = pq(html)
items = doc(<span class="hljs-string">'.list'</span>)
print(type(items))
print(items)
lis = items.find(<span class="hljs-string">'li'</span>)
print(type(lis))
print(lis)
</code></pre>
<p>运行结果：</p>
<pre><code data-language="html" class="lang-html"><span class="hljs-tag">&lt;<span class="hljs-name">class</span> '<span class="hljs-attr">pyquery.pyquery.PyQuery</span>'&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">ul</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"list"</span>&gt;</span>
    <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span>first item<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
    <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link2.html"</span>&gt;</span>second item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
    <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
    <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
    <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link5.html"</span>&gt;</span>fifth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">ul</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">class</span> '<span class="hljs-attr">pyquery.pyquery.PyQuery</span>'&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span>first item<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link2.html"</span>&gt;</span>second item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link5.html"</span>&gt;</span>fifth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
</code></pre>
<p>首先，我们通过 .list &nbsp;参数选取 class 为 list 的节点，然后调用 find 方法，传入 CSS 选择器，选取其内部的 li 节点，最后打印输出。可以发现，find 方法会将符合条件的所有节点选择出来，结果的类型是 pyquery 类型。</p>
<p>find 的查找范围是节点的所有子孙节点，而如果我们只想查找子节点，那可以用 children 方法：</p>
<pre><code data-language="python" class="lang-python">lis = items.children()
print(type(lis))
print(lis)
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="html" class="lang-html"><span class="hljs-tag">&lt;<span class="hljs-name">class</span> '<span class="hljs-attr">pyquery.pyquery.PyQuery</span>'&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span>first item<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link2.html"</span>&gt;</span>second item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link5.html"</span>&gt;</span>fifth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
</code></pre>
<p>如果要筛选所有子节点中符合条件的节点，比如想筛选出子节点中 class 为 active 的节点，可以向 children 方法传入 CSS 选择器 .active，代码如下：</p>
<pre><code data-language="python" class="lang-python">lis = items.children(<span class="hljs-string">'.active'</span>)
print(lis)
</code></pre>
<p>运行结果：</p>
<pre><code data-language="html" class="lang-html"><span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
</code></pre>
<p>我们看到输出的结果已经做了筛选，留下了 class 为 active 的节点。</p>
<h4>父节点</h4>
<p>我们可以用 parent 方法来获取某个节点的父节点，下面用一个实例来感受一下：</p>
<pre><code data-language="html" class="lang-html">html = '''
<span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"wrap"</span>&gt;</span>
    <span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">id</span>=<span class="hljs-string">"container"</span>&gt;</span>
        <span class="hljs-tag">&lt;<span class="hljs-name">ul</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"list"</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span>first item<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link2.html"</span>&gt;</span>second item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link5.html"</span>&gt;</span>fifth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
         <span class="hljs-tag">&lt;/<span class="hljs-name">ul</span>&gt;</span>
     <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
 <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
'''
from pyquery import PyQuery as pq
doc = pq(html)
items = doc('.list')
container = items.parent()
print(type(container))
print(container)
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="html" class="lang-html"><span class="hljs-tag">&lt;<span class="hljs-name">class</span> '<span class="hljs-attr">pyquery.pyquery.PyQuery</span>'&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">id</span>=<span class="hljs-string">"container"</span>&gt;</span>
    <span class="hljs-tag">&lt;<span class="hljs-name">ul</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"list"</span>&gt;</span>
         <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span>first item<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
         <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link2.html"</span>&gt;</span>second item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
         <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
         <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
         <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link5.html"</span>&gt;</span>fifth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
     <span class="hljs-tag">&lt;/<span class="hljs-name">ul</span>&gt;</span>
 <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
</code></pre>
<p>在上面的例子中我们首先用 .list 选取 class 为 list 的节点，然后调用 parent 方法得到其父节点，其类型依然是 pyquery 类型。</p>
<p>这里的父节点是该节点的直接父节点，也就是说，它不会再去查找父节点的父节点，即祖先节点。</p>
<p>但是如果你想获取某个祖先节点，该怎么办呢？我们可以用 parents 方法：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span> pyquery <span class="hljs-keyword">import</span> PyQuery <span class="hljs-keyword">as</span> pq
doc = pq(html)
items = doc(<span class="hljs-string">'.list'</span>)
parents = items.parents()
print(type(parents))
print(parents)
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="html" class="lang-html"><span class="hljs-tag">&lt;<span class="hljs-name">class</span> '<span class="hljs-attr">pyquery.pyquery.PyQuery</span>'&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"wrap"</span>&gt;</span>
    <span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">id</span>=<span class="hljs-string">"container"</span>&gt;</span>
        <span class="hljs-tag">&lt;<span class="hljs-name">ul</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"list"</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span>first item<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link2.html"</span>&gt;</span>second item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link5.html"</span>&gt;</span>fifth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
         <span class="hljs-tag">&lt;/<span class="hljs-name">ul</span>&gt;</span>
     <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
 <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
 <span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">id</span>=<span class="hljs-string">"container"</span>&gt;</span>
        <span class="hljs-tag">&lt;<span class="hljs-name">ul</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"list"</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span>first item<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link2.html"</span>&gt;</span>second item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link5.html"</span>&gt;</span>fifth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
         <span class="hljs-tag">&lt;/<span class="hljs-name">ul</span>&gt;</span>
     <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
</code></pre>
<p>可以看到，这个例子的输出结果有两个：一个是 class 为 wrap 的节点，一个是 id 为 container 的节点。也就是说，使用 parents 方法会返回所有的祖先节点。</p>
<p>如果你想要筛选某个祖先节点的话，可以向 parents 方法传入 CSS 选择器，这样就会返回祖先节点中符合 CSS 选择器的节点：</p>
<pre><code data-language="python" class="lang-python">parent = items.parents(<span class="hljs-string">'.wrap'</span>)
print(parent)
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="html" class="lang-html"><span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"wrap"</span>&gt;</span>
    <span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">id</span>=<span class="hljs-string">"container"</span>&gt;</span>
        <span class="hljs-tag">&lt;<span class="hljs-name">ul</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"list"</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span>first item<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link2.html"</span>&gt;</span>second item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link5.html"</span>&gt;</span>fifth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
         <span class="hljs-tag">&lt;/<span class="hljs-name">ul</span>&gt;</span>
     <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
 <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
</code></pre>
<p>可以看到，输出结果少了一个节点，只保留了 class 为 wrap 的节点。</p>
<h4>兄弟节点</h4>
<p>前面我们说明了子节点和父节点的用法，还有一种节点叫作兄弟节点。如果要获取兄弟节点，可以使用 siblings 方法。这里还是以上面的 HTML 代码为例：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span> pyquery <span class="hljs-keyword">import</span> PyQuery <span class="hljs-keyword">as</span> pq
doc = pq(html)
li = doc(<span class="hljs-string">'.list .item-0.active'</span>)
print(li.siblings())
</code></pre>
<p>在这个例子中我们首先选择 class 为 list 的节点，内部 class 为 item-0 和 active 的节点，也就是第 3 个 li 节点。很明显，它的兄弟节点有 4 个，那就是第 1、2、4、5 个 li 节点。</p>
<p>我们来运行一下：</p>
<pre><code data-language="html" class="lang-html"><span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link2.html"</span>&gt;</span>second item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span>first item<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link5.html"</span>&gt;</span>fifth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
</code></pre>
<p>可以看到，结果显示的正是我们刚才所说的 4 个兄弟节点。</p>
<p>如果要筛选某个兄弟节点，我们依然可以用 siblings 方法传入 CSS 选择器，这样就会从所有兄弟节点中挑选出符合条件的节点了：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span> pyquery <span class="hljs-keyword">import</span> PyQuery <span class="hljs-keyword">as</span> pq
doc = pq(html)
li = doc(<span class="hljs-string">'.list .item-0.active'</span>)
print(li.siblings(<span class="hljs-string">'.active'</span>))
</code></pre>
<p>在这个例子中我们筛选 class 为 active 的节点，从刚才的结果中可以观察到，class 为 active 兄弟节点的是第 4 个 li 节点，所以结果应该是1个。</p>
<p>我们再看一下运行结果：</p>
<pre><code data-language="html" class="lang-html"><span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
</code></pre>
<h3>遍历</h3>
<p>通过刚才的例子我们可以观察到，pyquery 的选择结果既可能是多个节点，也可能是单个节点，类型都是 pyquery 类型，并没有返回列表。</p>
<p>对于单个节点来说，可以直接打印输出，也可以直接转成字符串：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span> pyquery <span class="hljs-keyword">import</span> PyQuery <span class="hljs-keyword">as</span> pq
doc = pq(html)
li = doc(<span class="hljs-string">'.item-0.active'</span>)
print(li)
print(str(li))
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="html" class="lang-html"><span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
</code></pre>
<p>对于有多个节点的结果，我们就需要用遍历来获取了。例如，如果要把每一个 li 节点进行遍历，需要调用 items 方法：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span> pyquery <span class="hljs-keyword">import</span> PyQuery <span class="hljs-keyword">as</span> pq
doc = pq(html)
lis = doc(<span class="hljs-string">'li'</span>).items()
print(type(lis))
<span class="hljs-keyword">for</span> li <span class="hljs-keyword">in</span> lis:
    print(li, type(li))
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="html" class="lang-html"><span class="hljs-tag">&lt;<span class="hljs-name">class</span> '<span class="hljs-attr">generator</span>'&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span>first item<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">class</span> '<span class="hljs-attr">pyquery.pyquery.PyQuery</span>'&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link2.html"</span>&gt;</span>second item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">class</span> '<span class="hljs-attr">pyquery.pyquery.PyQuery</span>'&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">class</span> '<span class="hljs-attr">pyquery.pyquery.PyQuery</span>'&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">class</span> '<span class="hljs-attr">pyquery.pyquery.PyQuery</span>'&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link5.html"</span>&gt;</span>fifth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">class</span> '<span class="hljs-attr">pyquery.pyquery.PyQuery</span>'&gt;</span>
</code></pre>
<p>可以发现，调用 items 方法后，会得到一个生成器，遍历一下，就可以逐个得到 li 节点对象了，它的类型也是 pyquery 类型。每个 li 节点还可以调用前面所说的方法进行选择，比如继续查询子节点，寻找某个祖先节点等，非常灵活。</p>
<h4>获取信息</h4>
<p>提取到节点之后，我们的最终目的当然是提取节点所包含的信息了。比较重要的信息有两类，一是获取属性，二是获取文本，下面分别进行说明。</p>
<p><strong>获取属性</strong></p>
<p>提取到某个 pyquery 类型的节点后，就可以调用 attr 方法来获取属性：</p>
<pre><code data-language="html" class="lang-html">html = '''
<span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"wrap"</span>&gt;</span>
    <span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">id</span>=<span class="hljs-string">"container"</span>&gt;</span>
        <span class="hljs-tag">&lt;<span class="hljs-name">ul</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"list"</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span>first item<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link2.html"</span>&gt;</span>second item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link5.html"</span>&gt;</span>fifth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
         <span class="hljs-tag">&lt;/<span class="hljs-name">ul</span>&gt;</span>
     <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
 <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
'''
from pyquery import PyQuery as pq
doc = pq(html)
a = doc('.item-0.active a')
print(a, type(a))
print(a.attr('href'))
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="html" class="lang-html"><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span> <span class="hljs-tag">&lt;<span class="hljs-name">class</span> '<span class="hljs-attr">pyquery.pyquery.PyQuery</span>'&gt;</span>
link3.html
</code></pre>
<p>在这个例子中我们首先选中 class 为 item-0 和 active 的 li 节点内的 a 节点，它的类型是 pyquery 类型。</p>
<p>然后调用 attr 方法。在这个方法中传入属性的名称，就可以得到属性值了。</p>
<p>此外，也可以通过调用 attr 属性来获取属性值，用法如下：</p>
<pre><code data-language="python" class="lang-python">print(a.attr.href)
</code></pre>
<p>结果：</p>
<pre><code>link3.html
</code></pre>
<p>这两种方法的结果完全一样。</p>
<p>如果选中的是多个元素，然后调用 attr 方法，会出现怎样的结果呢？我们用实例来测试一下：</p>
<pre><code data-language="python" class="lang-python">a = doc(<span class="hljs-string">'a'</span>)
print(a, type(a))
print(a.attr(<span class="hljs-string">'href'</span>))
print(a.attr.href)
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="html" class="lang-html"><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link2.html"</span>&gt;</span>second item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link5.html"</span>&gt;</span>fifth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span> <span class="hljs-tag">&lt;<span class="hljs-name">class</span> '<span class="hljs-attr">pyquery.pyquery.PyQuery</span>'&gt;</span>
link2.html
link2.html
</code></pre>
<p>照理来说，我们选中的 a 节点应该有 4 个，打印结果也应该是 4 个，但是当我们调用 attr 方法时，返回结果却只有第 1 个。这是因为，当返回结果包含多个节点时，调用 attr 方法，只会得到第 1 个节点的属性。</p>
<p>那么，遇到这种情况时，如果想获取所有的 a 节点的属性，就要用到前面所说的遍历了：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span> pyquery <span class="hljs-keyword">import</span> PyQuery <span class="hljs-keyword">as</span> pq
doc = pq(html)
a = doc(<span class="hljs-string">'a'</span>)
<span class="hljs-keyword">for</span> item <span class="hljs-keyword">in</span> a.items():
    print(item.attr(<span class="hljs-string">'href'</span>))
</code></pre>
<p>运行结果：</p>
<pre><code data-language="html" class="lang-html">link2.html
link3.html
link4.html
link5.html
</code></pre>
<p>因此，在进行属性获取时，先要观察返回节点是一个还是多个，如果是多个，则需要遍历才能依次获取每个节点的属性。</p>
<p><strong>获取文本</strong></p>
<p>获取节点之后的另一个主要操作就是获取其内部文本了，此时可以调用 text 方法来实现：</p>
<pre><code data-language="html" class="lang-html">html = '''
<span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"wrap"</span>&gt;</span>
    <span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">id</span>=<span class="hljs-string">"container"</span>&gt;</span>
        <span class="hljs-tag">&lt;<span class="hljs-name">ul</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"list"</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span>first item<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link2.html"</span>&gt;</span>second item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link5.html"</span>&gt;</span>fifth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
         <span class="hljs-tag">&lt;/<span class="hljs-name">ul</span>&gt;</span>
     <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
 <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
'''
from pyquery import PyQuery as pq
doc = pq(html)
a = doc('.item-0.active a')
print(a)
print(a.text())
</code></pre>
<p>运行结果：</p>
<pre><code data-language="html" class="lang-html"><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span>
third item
</code></pre>
<p>这里我们首先选中一个 a 节点，然后调用 text 方法，就可以获取其内部的文本信息了。text 会忽略节点内部包含的所有 HTML，只返回纯文字内容。</p>
<p>但如果你想要获取这个节点内部的 HTML 文本，就要用 html 方法了：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span> pyquery <span class="hljs-keyword">import</span> PyQuery <span class="hljs-keyword">as</span> pq
doc = pq(html)
li = doc(<span class="hljs-string">'.item-0.active'</span>)
print(li)
print(li.html())
</code></pre>
<p>这里我们选中第 3 个 li 节点，然后调用 html 方法，它返回的结果应该是 li 节点内的所有 HTML 文本。</p>
<p>运行结果：</p>
<pre><code data-language="html" class="lang-html"><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span>
</code></pre>
<p>这里同样有一个问题，如果我们选中的结果是多个节点，text 或 html 方法会返回什么内容？我们用实例来看一下：</p>
<pre><code data-language="html" class="lang-html">html = '''
<span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"wrap"</span>&gt;</span>
    <span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">id</span>=<span class="hljs-string">"container"</span>&gt;</span>
        <span class="hljs-tag">&lt;<span class="hljs-name">ul</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"list"</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link2.html"</span>&gt;</span>second item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link5.html"</span>&gt;</span>fifth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
         <span class="hljs-tag">&lt;/<span class="hljs-name">ul</span>&gt;</span>
     <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
 <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
'''
from pyquery import PyQuery as pq
doc = pq(html)
li = doc('li')
print(li.html())
print(li.text())
print(type(li.text())
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="html" class="lang-html"><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link2.html"</span>&gt;</span>second item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span>
second item third item fourth item fifth item
<span class="hljs-tag">&lt;<span class="hljs-name">class'str'</span>&gt;</span>
</code></pre>
<p>结果比较出乎意料，html 方法返回的是第 1 个 li 节点的内部 HTML 文本，而 text 则返回了所有的 li 节点内部的纯文本，中间用一个空格分割开，即返回结果是一个字符串。</p>
<p>这个地方值得注意，如果你想要得到的结果是多个节点，并且需要获取每个节点的内部 HTML 文本，则需要遍历每个节点。而 text 方法不需要遍历就可以获取，它将所有节点取文本之后合并成一个字符串。</p>
<h3>节点操作</h3>
<p>pyquery 提供了一系列方法来对节点进行动态修改，比如为某个节点添加一个 class，移除某个节点等，这些操作有时会为提取信息带来极大的便利。</p>
<p>由于节点操作的方法太多，下面举几个典型的例子来说明它的用法。</p>
<h4>addClass 和 removeClass</h4>
<p>我们先用一个实例来感受一下：</p>
<pre><code data-language="html" class="lang-html">html = '''
<span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"wrap"</span>&gt;</span>
    <span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">id</span>=<span class="hljs-string">"container"</span>&gt;</span>
        <span class="hljs-tag">&lt;<span class="hljs-name">ul</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"list"</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span>first item<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link2.html"</span>&gt;</span>second item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link5.html"</span>&gt;</span>fifth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
         <span class="hljs-tag">&lt;/<span class="hljs-name">ul</span>&gt;</span>
     <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
 <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
'''
from pyquery import PyQuery as pq
doc = pq(html)
li = doc('.item-0.active')
print(li)
li.removeClass('active')
print(li)
li.addClass('active')
print(li)
</code></pre>
<p>首先选中第 3 个 li 节点，然后调用 removeClass 方法，将 li 节点的 active 这个 class 移除，第 2 步调用 addClass 方法，将 class 添加回来。每执行一次操作，就打印输出当前 li 节点的内容。</p>
<p>运行结果如下：</p>
<pre><code data-language="html" class="lang-html"><span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
</code></pre>
<p>可以看到，一共输出了 3 次。第 2 次输出时，li 节点的 active 这个 class 被移除了，第 3 次 class 又添加回来了。</p>
<p>所以说，addClass 和 removeClass 方法可以动态改变节点的 class 属性。</p>
<h4>attr、text、html</h4>
<p>当然，除了操作 class 这个属性外，也可以用 attr 方法对属性进行操作。此外，我们还可以用 text 和 html 方法来改变节点内部的内容。示例如下：</p>
<pre><code data-language="html" class="lang-html">html = '''
<span class="hljs-tag">&lt;<span class="hljs-name">ul</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"list"</span>&gt;</span>
     <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;/<span class="hljs-name">ul</span>&gt;</span>
'''
from pyquery import PyQuery as pq
doc = pq(html)
li = doc('.item-0.active')
print(li)
li.attr('name', 'link')
print(li)
li.text('changed item')
print(li)
li.html('<span class="hljs-tag">&lt;<span class="hljs-name">span</span>&gt;</span>changed item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span>')
print(li)
</code></pre>
<p>这里我们首先选中 li 节点，然后调用 attr 方法来修改属性。该方法的第 1 个参数为属性名，第 2 个参数为属性值。最后调用 text 和 html 方法来改变节点内部的内容。3 次操作后，分别打印输出当前的 li 节点。</p>
<p>运行结果如下：</p>
<pre><code data-language="html" class="lang-html"><span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span> <span class="hljs-attr">name</span>=<span class="hljs-string">"link"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span> <span class="hljs-attr">name</span>=<span class="hljs-string">"link"</span>&gt;</span>changed item<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
<span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span> <span class="hljs-attr">name</span>=<span class="hljs-string">"link"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span>&gt;</span>changed item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
</code></pre>
<p>我们发现，调用 attr 方法后，li 节点多了一个原本不存在的属性 name，其值为 link。接着调用 text 方法传入文本，li 节点内部的文本全被改为传入的字符串文本。最后，调用 html 方法传入 HTML 文本，li 节点内部又变为传入的 HTML 文本了。</p>
<p>所以说，使用 attr 方法时如果只传入第 1 个参数的属性名，则是获取这个属性值；如果传入第 2 个参数，可以用来修改属性值。使用 text 和 html 方法时如果不传参数，则是获取节点内纯文本和 HTML 文本，如果传入参数，则进行赋值。</p>
<h4>remove</h4>
<p>顾名思义，remove 方法就是移除，它有时会为信息的提取带来非常大的便利。下面有一段 HTML 文本：</p>
<pre><code data-language="html" class="lang-html">html = '''
<span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"wrap"</span>&gt;</span>
    Hello, World
    <span class="hljs-tag">&lt;<span class="hljs-name">p</span>&gt;</span>This is a paragraph.<span class="hljs-tag">&lt;/<span class="hljs-name">p</span>&gt;</span>
 <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
'''
from pyquery import PyQuery as pq
doc = pq(html)
wrap = doc('.wrap')
print(wrap.text())
</code></pre>
<p>现在我们想提取“Hello, World”这个字符串，该怎样操作呢？</p>
<p>这里先直接尝试提取 class 为 wrap 的节点的内容，看看是不是我们想要的。</p>
<p>运行结果如下：</p>
<pre><code data-language="python" class="lang-python">Hello, World This <span class="hljs-keyword">is</span> a paragraph.
</code></pre>
<p>这个结果还包含了内部的 p 节点的内容，也就是说 text 把所有的纯文本全提取出来了。</p>
<p>如果我们想去掉 p 节点内部的文本，可以选择再把 p 节点内的文本提取一遍，然后从整个结果中移除这个子串，但这个做法明显比较烦琐。</p>
<p>这时 remove 方法就可以派上用场了，我们可以接着这么做：</p>
<pre><code data-language="python" class="lang-python">wrap.find(<span class="hljs-string">'p'</span>).remove()
print(wrap.text())
</code></pre>
<p>首先选中 p 节点，然后调用 remove 方法将其移除，这时 wrap 内部就只剩下“Hello, World”这句话了，最后利用 text 方法提取即可。</p>
<p>其实还有很多其他节点操作的方法，比如 append、empty 和 prepend 等方法，详细的用法可以参考官方文档：<a href="http://pyquery.readthedocs.io/en/latest/api.html">http://pyquery.readthedocs.io/en/latest/api.html</a>。</p>
<h3>伪类选择器</h3>
<p>CSS 选择器之所以强大，还有一个很重要的原因，那就是它支持多种多样的伪类选择器，例如选择第一个节点、最后一个节点、奇偶数节点、包含某一文本的节点等。示例如下：</p>
<pre><code data-language="html" class="lang-html">html = '''
<span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"wrap"</span>&gt;</span>
    <span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">id</span>=<span class="hljs-string">"container"</span>&gt;</span>
        <span class="hljs-tag">&lt;<span class="hljs-name">ul</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"list"</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span>first item<span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link2.html"</span>&gt;</span>second item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link3.html"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">span</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"bold"</span>&gt;</span>third item<span class="hljs-tag">&lt;/<span class="hljs-name">span</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-1 active"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link4.html"</span>&gt;</span>fourth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
             <span class="hljs-tag">&lt;<span class="hljs-name">li</span> <span class="hljs-attr">class</span>=<span class="hljs-string">"item-0"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">a</span> <span class="hljs-attr">href</span>=<span class="hljs-string">"link5.html"</span>&gt;</span>fifth item<span class="hljs-tag">&lt;/<span class="hljs-name">a</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">li</span>&gt;</span>
         <span class="hljs-tag">&lt;/<span class="hljs-name">ul</span>&gt;</span>
     <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
 <span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span>
'''
from pyquery import PyQuery as pq
doc = pq(html)
li = doc('li:first-child')
print(li)
li = doc('li:last-child')
print(li)
li = doc('li:nth-child(2)')
print(li)
li = doc('li:gt(2)')
print(li)
li = doc('li:nth-child(2n)')
print(li)
li = doc('li:contains(second)')
print(li)
</code></pre>
<p>在这个例子中我们使用了 CSS3 的伪类选择器，依次选择了第 1 个 li 节点、最后一个 li 节点、第 2 个 li 节点、第 3 个 li 之后的 li 节点、偶数位置的 li 节点、包含 second 文本的 li 节点。</p>
<p>关于 CSS 选择器的更多用法，可以参考&nbsp;<a href="http://www.w3school.com.cn/css/index.asp">http://www.w3school.com.cn/css/index.asp</a>。</p>
<p>到此为止，pyquery 的常用用法就介绍完了。如果想查看更多的内容，可以参考 pyquery 的官方文档：<a href="http://pyquery.readthedocs.io">http://pyquery.readthedocs.io</a>。相信一旦你拥有了它，解析网页将不再是难事。</p>

# 高效存储MongoDB的用法
<p>上节课我们学习了如何用 pyquery 提取 HTML 中的信息，但是当我们成功提取了数据之后，该往哪里存放呢？</p>
<p>用文本文件当然是可以的，但文本存储不方便检索。有没有既方便存，又方便检索的存储方式呢？</p>
<p>当然有，本课时我将为你介绍一个文档型数据库 —— MongoDB。</p>
<p>MongoDB 是由 C++ 语言编写的非关系型数据库，是一个基于分布式文件存储的开源数据库系统，其内容存储形式类似 JSON 对象，它的字段值可以包含其他文档、数组及文档数组，非常灵活。</p>
<p>在这个课时中，我们就来看看 Python 3 下 MongoDB 的存储操作。</p>
<h3>准备工作</h3>
<p>在开始之前，请确保你已经安装好了 MongoDB 并启动了其服务，同时安装好了 Python 的 PyMongo 库。</p>
<p>MongoDB 的安装方式可以参考：<a href="https://cuiqingcai.com/5205.html">https://cuiqingcai.com/5205.html</a>，安装好之后，我们需要把 MongoDB 服务启动起来。</p>
<blockquote>
<p>注意：这里我们为了学习，仅使用 MongoDB 最基本的单机版，MongoDB 还有主从复制、副本集、分片集群等集群架构，可用性可靠性更好，如有需要可以自行搭建相应的集群进行使用。</p>
</blockquote>
<p>启动完成之后，它会默认在本地 localhost 的 27017 端口上运行。</p>
<p>接下来我们需要安装 PyMongo 这个库，它是 Python 用来操作 MongoDB 的第三方库，直接用 pip3 安装即可：<code data-backticks="1">pip3 install pymongo</code>。</p>
<p>更详细的安装方式可以参考：<a href="https://cuiqingcai.com/5230.html">https://cuiqingcai.com/5230.html</a>。</p>
<p>安装完成之后，我们就可以使用 PyMongo 来将数据存储到 MongoDB 了。</p>
<h3>连接 MongoDB</h3>
<p>连接 MongoDB 时，我们需要使用 PyMongo 库里面的 MongoClient。一般来说，我们只需要向其传入 MongoDB 的 IP 及端口即可，其中第一个参数为地址 host，第二个参数为端口 port（如果不给它传递参数，则默认是 27017）：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span> pymongo
client = pymongo.MongoClient(host=<span class="hljs-string">'localhost'</span>, port=<span class="hljs-number">27017</span>)
</code></pre>
<p>这样我们就可以创建 MongoDB 的连接对象了。</p>
<p>另外，MongoClient 的第一个参数 host 还可以直接传入 MongoDB 的连接字符串，它以 mongodb 开头，例如：</p>
<pre><code data-language="python" class="lang-python">client = MongoClient(<span class="hljs-string">'mongodb://localhost:27017/'</span>)
</code></pre>
<p>这样也可以达到同样的连接效果。</p>
<h3>指定数据库</h3>
<p>MongoDB 中可以建立多个数据库，接下来我们需要指定操作其中一个数据库。这里我们以 test 数据库作为下一步需要在程序中指定使用的例子：</p>
<pre><code data-language="python" class="lang-python">db = client.test
</code></pre>
<p>这里调用 client 的 test 属性即可返回 test 数据库。当然，我们也可以这样指定：</p>
<pre><code data-language="pyton" class="lang-pyton">db = client['test']
</code></pre>
<p>这两种方式是等价的。</p>
<h3>指定集合</h3>
<p>MongoDB 的每个数据库又包含许多集合（collection），它们类似于关系型数据库中的表。</p>
<p>下一步需要指定要操作的集合，这里我们指定一个名称为 students 的集合。与指定数据库类似，指定集合也有两种方式：</p>
<pre><code data-language="python" class="lang-python">collection = db.students
</code></pre>
<p>或是</p>
<pre><code data-language="python" class="lang-python">collection = db[<span class="hljs-string">'students'</span>]
</code></pre>
<p>这样我们便声明了一个 Collection 对象。</p>
<h3>插入数据</h3>
<p>接下来，便可以插入数据了。我们对 students 这个集合新建一条学生数据，这条数据以字典形式表示：</p>
<pre><code data-language="python" class="lang-python">student = {
    <span class="hljs-string">'id'</span>: <span class="hljs-string">'20170101'</span>,
    <span class="hljs-string">'name'</span>: <span class="hljs-string">'Jordan'</span>,
    <span class="hljs-string">'age'</span>: <span class="hljs-number">20</span>,
    <span class="hljs-string">'gender'</span>: <span class="hljs-string">'male'</span>
}
</code></pre>
<p>新建的这条数据里指定了学生的学号、姓名、年龄和性别。接下来，我们直接调用 collection 的 insert 方法即可插入数据，代码如下：</p>
<pre><code data-language="python" class="lang-python">result = collection.insert(student)
print(result)
</code></pre>
<p>在 MongoDB 中，每条数据其实都有一个 _id 属性来唯一标识。如果没有显式指明该属性，MongoDB 会自动产生一个 ObjectId 类型的 _id 属性。insert() 方法会在执行后返回_id 值。</p>
<p>运行结果如下：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-number">5932</span>a68615c2606814c91f3d
</code></pre>
<p>当然，我们也可以同时插入多条数据，只需要以列表形式传递即可，示例如下：</p>
<pre><code data-language="python" class="lang-python">student1 = {
    <span class="hljs-string">'id'</span>: <span class="hljs-string">'20170101'</span>,
    <span class="hljs-string">'name'</span>: <span class="hljs-string">'Jordan'</span>,
    <span class="hljs-string">'age'</span>: <span class="hljs-number">20</span>,
    <span class="hljs-string">'gender'</span>: <span class="hljs-string">'male'</span>
}

student2 = {
    <span class="hljs-string">'id'</span>: <span class="hljs-string">'20170202'</span>,
    <span class="hljs-string">'name'</span>: <span class="hljs-string">'Mike'</span>,
    <span class="hljs-string">'age'</span>: <span class="hljs-number">21</span>,
    <span class="hljs-string">'gender'</span>: <span class="hljs-string">'male'</span>
}

result = collection.insert([student1, student2])
print(result)
</code></pre>
<p>返回结果是对应的_id 的集合：</p>
<pre><code data-language="python" class="lang-python">[ObjectId(<span class="hljs-string">'5932a80115c2606a59e8a048'</span>), ObjectId(<span class="hljs-string">'5932a80115c2606a59e8a049'</span>)]
</code></pre>
<p>实际上，在 PyMongo 中，官方已经不推荐使用 insert 方法了。但是如果你要继续使用也没有什么问题。目前，官方推荐使用 insert_one 和 insert_many 方法来分别插入单条记录和多条记录，示例如下：</p>
<pre><code data-language="python" class="lang-python">student = {
    <span class="hljs-string">'id'</span>: <span class="hljs-string">'20170101'</span>,
    <span class="hljs-string">'name'</span>: <span class="hljs-string">'Jordan'</span>,
    <span class="hljs-string">'age'</span>: <span class="hljs-number">20</span>,
    <span class="hljs-string">'gender'</span>: <span class="hljs-string">'male'</span>
}

result = collection.insert_one(student)
print(result)
print(result.inserted_id)
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="python" class="lang-python">&lt;pymongo.results.InsertOneResult object at <span class="hljs-number">0x10d68b558</span>&gt;
<span class="hljs-number">5932</span>ab0f15c2606f0c1cf6c5
</code></pre>
<p>与 insert 方法不同，这次返回的是 InsertOneResult 对象，我们可以调用其 inserted_id 属性获取_id。</p>
<p>对于 insert_many 方法，我们可以将数据以列表形式传递，示例如下：</p>
<pre><code data-language="python" class="lang-python">student1 = {
    <span class="hljs-string">'id'</span>: <span class="hljs-string">'20170101'</span>,
    <span class="hljs-string">'name'</span>: <span class="hljs-string">'Jordan'</span>,
    <span class="hljs-string">'age'</span>: <span class="hljs-number">20</span>,
    <span class="hljs-string">'gender'</span>: <span class="hljs-string">'male'</span>
}

student2 = {
    <span class="hljs-string">'id'</span>: <span class="hljs-string">'20170202'</span>,
    <span class="hljs-string">'name'</span>: <span class="hljs-string">'Mike'</span>,
    <span class="hljs-string">'age'</span>: <span class="hljs-number">21</span>,
    <span class="hljs-string">'gender'</span>: <span class="hljs-string">'male'</span>
}

result = collection.insert_many([student1, student2])
print(result)
print(result.inserted_ids)
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="python" class="lang-python">&lt;pymongo.results.InsertManyResult object at <span class="hljs-number">0x101dea558</span>&gt;
[ObjectId(<span class="hljs-string">'5932abf415c2607083d3b2ac'</span>), ObjectId(<span class="hljs-string">'5932abf415c2607083d3b2ad'</span>)]
</code></pre>
<p>该方法返回的类型是 InsertManyResult，调用 inserted_ids 属性可以获取插入数据的 _id 列表。</p>
<h3>查询</h3>
<p>插入数据后，我们可以利用 find_one 或 find 方法进行查询，其中 find_one 查询得到的是单个结果，find 则返回一个生成器对象。示例如下：</p>
<pre><code data-language="python" class="lang-python">result = collection.find_one({<span class="hljs-string">'name'</span>: <span class="hljs-string">'Mike'</span>})
print(type(result))
print(result)
</code></pre>
<p>这里我们查询 name 为 Mike 的数据，它的返回结果是字典类型，运行结果如下：</p>
<pre><code data-language="python" class="lang-python">&lt;class 'dict'&gt;
{'_id': ObjectId('5932a80115c2606a59e8a049'), 'id': '20170202', 'name': 'Mike', 'age': 21, 'gender': 'male'}
</code></pre>
<p>可以发现，它多了 _id 属性，这就是 MongoDB 在插入过程中自动添加的。</p>
<p>此外，我们也可以根据 ObjectId 来查询，此时需要调用 bson 库里面的 objectid：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span> bson.objectid <span class="hljs-keyword">import</span> ObjectId

result = collection.find_one({<span class="hljs-string">'_id'</span>: ObjectId(<span class="hljs-string">'593278c115c2602667ec6bae'</span>)})
print(result)
</code></pre>
<p>其查询结果依然是字典类型，具体如下：</p>
<pre><code data-language="python" class="lang-python">{<span class="hljs-string">'_id'</span>: ObjectId(<span class="hljs-string">'593278c115c2602667ec6bae'</span>), <span class="hljs-string">'id'</span>: <span class="hljs-string">'20170101'</span>, <span class="hljs-string">'name'</span>: <span class="hljs-string">'Jordan'</span>, <span class="hljs-string">'age'</span>: <span class="hljs-number">20</span>, <span class="hljs-string">'gender'</span>: <span class="hljs-string">'male'</span>}
</code></pre>
<p>如果查询结果不存在，则会返回 None。</p>
<p>对于多条数据的查询，我们可以使用 find 方法。例如，这里查找年龄为 20 的数据，示例如下：</p>
<pre><code data-language="python" class="lang-python">results = collection.find({<span class="hljs-string">'age'</span>: <span class="hljs-number">20</span>})
print(results)
<span class="hljs-keyword">for</span> result <span class="hljs-keyword">in</span> results:
    print(result)
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="python" class="lang-python">&lt;pymongo.cursor.Cursor object at <span class="hljs-number">0x1032d5128</span>&gt;
{<span class="hljs-string">'_id'</span>: ObjectId(<span class="hljs-string">'593278c115c2602667ec6bae'</span>), <span class="hljs-string">'id'</span>: <span class="hljs-string">'20170101'</span>, <span class="hljs-string">'name'</span>: <span class="hljs-string">'Jordan'</span>, <span class="hljs-string">'age'</span>: <span class="hljs-number">20</span>, <span class="hljs-string">'gender'</span>: <span class="hljs-string">'male'</span>}
{<span class="hljs-string">'_id'</span>: ObjectId(<span class="hljs-string">'593278c815c2602678bb2b8d'</span>), <span class="hljs-string">'id'</span>: <span class="hljs-string">'20170102'</span>, <span class="hljs-string">'name'</span>: <span class="hljs-string">'Kevin'</span>, <span class="hljs-string">'age'</span>: <span class="hljs-number">20</span>, <span class="hljs-string">'gender'</span>: <span class="hljs-string">'male'</span>}
{<span class="hljs-string">'_id'</span>: ObjectId(<span class="hljs-string">'593278d815c260269d7645a8'</span>), <span class="hljs-string">'id'</span>: <span class="hljs-string">'20170103'</span>, <span class="hljs-string">'name'</span>: <span class="hljs-string">'Harden'</span>, <span class="hljs-string">'age'</span>: <span class="hljs-number">20</span>, <span class="hljs-string">'gender'</span>: <span class="hljs-string">'male'</span>}
</code></pre>
<p>返回结果是 Cursor 类型，它相当于一个生成器，我们需要遍历获取的所有结果，其中每个结果都是字典类型。</p>
<p>如果要查询年龄大于 20 的数据，则写法如下：</p>
<pre><code data-language="python" class="lang-python">results = collection.find({<span class="hljs-string">'age'</span>: {<span class="hljs-string">'$gt'</span>: <span class="hljs-number">20</span>}})
</code></pre>
<p>这里查询的条件键值已经不是单纯的数字了，而是一个字典，其键名为比较符号 $gt，意思是大于，键值为 20。</p>
<p>我将比较符号归纳为下表：</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/77/5E/CgpOIF5x81GAEHGaAACf0FUPMUU926.png" alt=""></p>
<p>另外，还可以进行正则匹配查询。例如，查询名字以 M 开头的学生数据，示例如下：</p>
<pre><code data-language="python" class="lang-python">results = collection.find({<span class="hljs-string">'name'</span>: {<span class="hljs-string">'$regex'</span>: <span class="hljs-string">'^M.*'</span>}})
</code></pre>
<p>这里使用 $regex 来指定正则匹配，^M.* 代表以 M 开头的正则表达式。</p>
<p>我将一些功能符号归类为下表：</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/77/5E/Cgq2xl5x83iAAn4RAAEsbDKOSTc291.png" alt=""><br>
关于这些操作的更详细用法，可以在 MongoDB 官方文档找到：&nbsp;<a href="https://docs.mongodb.com/manual/reference/operator/query/">https://docs.mongodb.com/manual/reference/operator/query/</a>。</p>
<h3>计数</h3>
<p>要统计查询结果有多少条数据，可以调用 count 方法。我们以统计所有数据条数为例：</p>
<pre><code data-language="python" class="lang-python">count = collection.find().count()
print(count)
</code></pre>
<p>我们还可以统计符合某个条件的数据：</p>
<pre><code data-language="python" class="lang-python">count = collection.find({<span class="hljs-string">'age'</span>: <span class="hljs-number">20</span>}).count()
print(count)
</code></pre>
<p>运行结果是一个数值，即符合条件的数据条数。</p>
<h3>排序</h3>
<p>排序时，我们可以直接调用 sort 方法，并在其中传入排序的字段及升降序标志。示例如下：</p>
<pre><code data-language="python" class="lang-python">results = collection.find().sort(<span class="hljs-string">'name'</span>, pymongo.ASCENDING)
print([result[<span class="hljs-string">'name'</span>] <span class="hljs-keyword">for</span> result <span class="hljs-keyword">in</span> results])
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="python" class="lang-python">[<span class="hljs-string">'Harden'</span>, <span class="hljs-string">'Jordan'</span>, <span class="hljs-string">'Kevin'</span>, <span class="hljs-string">'Mark'</span>, <span class="hljs-string">'Mike'</span>]
</code></pre>
<p>这里我们调用 pymongo.ASCENDING 指定升序。如果要降序排列，可以传入 pymongo.DESCENDING。</p>
<h3>偏移</h3>
<p>在某些情况下，我们可能只需要取某几个元素，这时可以利用 skip 方法偏移几个位置，比如偏移 2，就代表忽略前两个元素，得到第 3 个及以后的元素：</p>
<pre><code data-language="python" class="lang-python">results = collection.find().sort(<span class="hljs-string">'name'</span>, pymongo.ASCENDING).skip(<span class="hljs-number">2</span>)
print([result[<span class="hljs-string">'name'</span>] <span class="hljs-keyword">for</span> result <span class="hljs-keyword">in</span> results])
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="python" class="lang-python">[<span class="hljs-string">'Kevin'</span>, <span class="hljs-string">'Mark'</span>, <span class="hljs-string">'Mike'</span>]
</code></pre>
<p>另外，我们还可以用 limit 方法指定要取的结果个数，示例如下：</p>
<pre><code data-language="python" class="lang-python">results = collection.find().sort(<span class="hljs-string">'name'</span>, pymongo.ASCENDING).skip(<span class="hljs-number">2</span>).limit(<span class="hljs-number">2</span>)
print([result[<span class="hljs-string">'name'</span>] <span class="hljs-keyword">for</span> result <span class="hljs-keyword">in</span> results])
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="python" class="lang-python">[<span class="hljs-string">'Kevin'</span>, <span class="hljs-string">'Mark'</span>]
</code></pre>
<p>如果不使用 limit 方法，原本会返回 3 个结果，加了限制后，就会截取两个结果返回。</p>
<p>值得注意的是，在数据量非常庞大的时候，比如在查询千万、亿级别的数据库时，最好不要使用大的偏移量，因为这样很可能导致内存溢出。此时可以使用类似如下操作来查询：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span> bson.objectid <span class="hljs-keyword">import</span> ObjectId
collection.find({<span class="hljs-string">'_id'</span>: {<span class="hljs-string">'$gt'</span>: ObjectId(<span class="hljs-string">'593278c815c2602678bb2b8d'</span>)}})
</code></pre>
<p>这时需要记录好上次查询的 _id。</p>
<h3>更新</h3>
<p>对于数据更新，我们可以使用 update 方法，指定更新的条件和更新后的数据即可。例如：</p>
<pre><code data-language="python" class="lang-python">condition = {<span class="hljs-string">'name'</span>: <span class="hljs-string">'Kevin'</span>}
student = collection.find_one(condition)
student[<span class="hljs-string">'age'</span>] = <span class="hljs-number">25</span>
result = collection.update(condition, student)
print(result)
</code></pre>
<p>这里我们要更新 name 为 Kevin 的数据的年龄：首先指定查询条件，然后将数据查询出来，修改年龄后调用 update 方法将原条件和修改后的数据传入。</p>
<p>运行结果如下：</p>
<pre><code data-language="python" class="lang-python">{<span class="hljs-string">'ok'</span>: <span class="hljs-number">1</span>, <span class="hljs-string">'nModified'</span>: <span class="hljs-number">1</span>, <span class="hljs-string">'n'</span>: <span class="hljs-number">1</span>, <span class="hljs-string">'updatedExisting'</span>: <span class="hljs-literal">True</span>}
</code></pre>
<p>返回结果是字典形式，ok 代表执行成功，nModified 代表影响的数据条数。</p>
<p>另外，我们也可以使用 $set 操作符对数据进行更新，代码如下：</p>
<pre><code data-language="python" class="lang-python">result = collection.update(condition, {<span class="hljs-string">'$set'</span>: student})
</code></pre>
<p>这样可以只更新 student 字典内存在的字段。如果原先还有其他字段，则不会更新，也不会删除。而如果不用 $set 的话，则会把之前的数据全部用 student 字典替换；如果原本存在其他字段，则会被删除。</p>
<p>另外，update 方法其实也是官方不推荐使用的方法。这里也分为 update_one 方法和 update_many 方法，用法更加严格，它们的第 2 个参数需要使用 $ 类型操作符作为字典的键名，示例如下：</p>
<pre><code data-language="python" class="lang-python">condition = {<span class="hljs-string">'name'</span>: <span class="hljs-string">'Kevin'</span>}
student = collection.find_one(condition)
student[<span class="hljs-string">'age'</span>] = <span class="hljs-number">26</span>
result = collection.update_one(condition, {<span class="hljs-string">'$set'</span>: student})
print(result)
print(result.matched_count, result.modified_count)
</code></pre>
<p>上面的例子中调用了 update_one 方法，使得第 2 个参数不能再直接传入修改后的字典，而是需要使用 {'$set': student} 这样的形式，其返回结果是 UpdateResult 类型。然后分别调用 matched_count 和 modified_count 属性，可以获得匹配的数据条数和影响的数据条数。</p>
<p>运行结果如下：</p>
<pre><code data-language="python" class="lang-python">&lt;pymongo.results.UpdateResult object at <span class="hljs-number">0x10d17b678</span>&gt;
<span class="hljs-number">1</span> <span class="hljs-number">0</span>
</code></pre>
<p>我们再看一个例子：</p>
<pre><code data-language="python" class="lang-python">condition = {<span class="hljs-string">'age'</span>: {<span class="hljs-string">'$gt'</span>: <span class="hljs-number">20</span>}}
result = collection.update_one(condition, {<span class="hljs-string">'$inc'</span>: {<span class="hljs-string">'age'</span>: <span class="hljs-number">1</span>}})
print(result)
print(result.matched_count, result.modified_count)
</code></pre>
<p>这里指定查询条件为年龄大于 20，然后更新条件为 {'$inc': {'age': 1}}，表示年龄加 1，执行之后会将第一条符合条件的数据年龄加 1。</p>
<p>运行结果如下：</p>
<pre><code data-language="python" class="lang-python">&lt;pymongo.results.UpdateResult object at <span class="hljs-number">0x10b8874c8</span>&gt;
<span class="hljs-number">1</span> <span class="hljs-number">1</span>
</code></pre>
<p>可以看到匹配条数为 1 条，影响条数也为 1 条。</p>
<p>如果调用 update_many 方法，则会将所有符合条件的数据都更新，示例如下：</p>
<pre><code data-language="python" class="lang-python">condition = {<span class="hljs-string">'age'</span>: {<span class="hljs-string">'$gt'</span>: <span class="hljs-number">20</span>}}
result = collection.update_many(condition, {<span class="hljs-string">'$inc'</span>: {<span class="hljs-string">'age'</span>: <span class="hljs-number">1</span>}})
print(result)
print(result.matched_count, result.modified_count)
</code></pre>
<p>这时匹配条数就不再为 1 条了，运行结果如下：</p>
<pre><code data-language="python" class="lang-python">&lt;pymongo.results.UpdateResult object at <span class="hljs-number">0x10c6384c8</span>&gt;
<span class="hljs-number">3</span> <span class="hljs-number">3</span>
</code></pre>
<p>可以看到，这时所有匹配到的数据都会被更新。</p>
<h3>删除</h3>
<p>删除操作比较简单，直接调用 remove 方法指定删除的条件即可，此时符合条件的所有数据均会被删除。</p>
<p>示例如下：</p>
<pre><code data-language="python" class="lang-python">result = collection.remove({<span class="hljs-string">'name'</span>: <span class="hljs-string">'Kevin'</span>})
print(result)
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="python" class="lang-python">{<span class="hljs-string">'ok'</span>: <span class="hljs-number">1</span>, <span class="hljs-string">'n'</span>: <span class="hljs-number">1</span>}
</code></pre>
<p>另外，这里依然存在两个新的推荐方法 —— delete_one 和 delete_many，示例如下：</p>
<pre><code data-language="python" class="lang-python">result = collection.delete_one({<span class="hljs-string">'name'</span>: <span class="hljs-string">'Kevin'</span>})
print(result)
print(result.deleted_count)
result = collection.delete_many({<span class="hljs-string">'age'</span>: {<span class="hljs-string">'$lt'</span>: <span class="hljs-number">25</span>}})
print(result.deleted_count)
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="python" class="lang-python">&lt;pymongo.results.DeleteResult object at <span class="hljs-number">0x10e6ba4c8</span>&gt;
<span class="hljs-number">1</span>
<span class="hljs-number">4</span>
</code></pre>
<p>delete_one 即删除第一条符合条件的数据，delete_many 即删除所有符合条件的数据。它们的返回结果都是 DeleteResult 类型，可以调用 deleted_count 属性获取删除的数据条数。</p>
<h3>其他操作</h3>
<p>另外，PyMongo 还提供了一些组合方法，如 find_one_and_delete、find_one_and_replace 和 find_one_and_update，它们分别用于查找后删除、替换和更新操作，其使用方法与上述方法基本一致。</p>
<p>另外，我们还可以对索引进行操作，相关方法有 create_index、create_indexes 和 drop_index 等。</p>
<p>关于 PyMongo 的详细用法，可以参见官方文档：<a href="http://api.mongodb.com/python/current/api/pymongo/collection.html">http://api.mongodb.com/python/current/api/pymongo/collection.html</a>。</p>
<p>另外，还有对数据库和集合本身等的一些操作，这里不再一一讲解，可以参见官方文档：<a href="http://api.mongodb.com/python/current/api/pymongo/">http://api.mongodb.com/python/current/api/pymongo/</a>。</p>

# Requests+PyQuery+PyMongo基本案例实战
<p data-nodeid="55363" class="te-preview-highlight">在前面我们已经学习了多进程、requests、正则表达式、pyquery、PyMongo 等的基本用法，但我们还没有完整地实现一个爬取案例。本课时，我们就来实现一个完整的网站爬虫案例，把前面学习的知识点串联起来，同时加深对这些知识点的理解。</p>
<h3 data-nodeid="55364">准备工作</h3>
<p data-nodeid="55365">在本节课开始之前，我们需要做好如下的准备工作：</p>
<ul data-nodeid="55366">
<li data-nodeid="55367">
<p data-nodeid="55368">安装好 Python3（最低为 3.6 版本），并能成功运行 Python3 程序。</p>
</li>
<li data-nodeid="55369">
<p data-nodeid="55370">了解 Python 多进程的基本原理。</p>
</li>
<li data-nodeid="55371">
<p data-nodeid="55372">了解 Python HTTP 请求库 requests 的基本用法。</p>
</li>
<li data-nodeid="55373">
<p data-nodeid="55374">了解正则表达式的用法和 Python 中正则表达式库 re 的基本用法。</p>
</li>
<li data-nodeid="55375">
<p data-nodeid="55376">了解 Python HTML 解析库 pyquery 的基本用法。</p>
</li>
<li data-nodeid="55377">
<p data-nodeid="55378">了解 MongoDB 并安装和启动 MongoDB 服务。</p>
</li>
<li data-nodeid="55379">
<p data-nodeid="55380">了解 Python 的 MongoDB 操作库 PyMongo 的基本用法。</p>
</li>
</ul>
<p data-nodeid="55381">以上内容在前面的课时中均有讲解，如果你还没有准备好，那么我建议你可以再复习一下这些内容。</p>
<h3 data-nodeid="55382">爬取目标</h3>
<p data-nodeid="56617" class="">这节课我们以一个基本的静态网站作为案例进行爬取，需要爬取的链接为：<a href="https://static1.scrape.center/" data-nodeid="56621">https://static1.scrape.center/</a>，这个网站里面包含了一些电影信息，界面如下：</p>


<p data-nodeid="55384"><img src="https://s0.lgstatic.com/i/image3/M01/78/4E/CgpOIF5zioaAJ2pmAAVK9IN6YAk404.png" alt="" data-nodeid="55553"></p>
<p data-nodeid="55385">首页是一个影片列表，每栏里都包含了这部电影的封面、名称、分类、上映时间、评分等内容，同时列表页还支持翻页，点击相应的页码我们就能进入到对应的新列表页。</p>
<p data-nodeid="55386">如果我们点开其中一部电影，会进入电影的详情页面，比如我们点开第一部《霸王别姬》，会得到如下页面：</p>
<p data-nodeid="55387"><img src="https://s0.lgstatic.com/i/image3/M01/78/4E/Cgq2xl5zioeATkf_AAZdwA77BpU974.png" alt="" data-nodeid="55557"></p>
<p data-nodeid="55388">这里显示的内容更加丰富、包括剧情简介、导演、演员等信息。</p>
<p data-nodeid="55389">我们这节课要完成的目标是：</p>
<ul data-nodeid="55390">
<li data-nodeid="55391">
<p data-nodeid="55392">用 requests 爬取这个站点每一页的电影列表，顺着列表再爬取每个电影的详情页。</p>
</li>
<li data-nodeid="55393">
<p data-nodeid="55394">用 pyquery 和正则表达式提取每部电影的名称、封面、类别、上映时间、评分、剧情简介等内容。</p>
</li>
<li data-nodeid="55395">
<p data-nodeid="55396">把以上爬取的内容存入 MongoDB 数据库。</p>
</li>
<li data-nodeid="55397">
<p data-nodeid="55398">使用多进程实现爬取的加速。</p>
</li>
</ul>
<p data-nodeid="55399">那么我们现在就开始吧。</p>
<h3 data-nodeid="55400">爬取列表页</h3>
<p data-nodeid="58293" class="">爬取的第一步肯定要从列表页入手，我们首先观察一下列表页的结构和翻页规则。在浏览器中访问&nbsp;<a href="https://static1.scrape.center/" data-nodeid="58297">https://static1.scrape.center/</a>，然后打开浏览器开发者工具，观察每一个电影信息区块对应的 HTML，以及进入到详情页的 URL 是怎样的，如图所示：</p>


<p data-nodeid="55402"><img src="https://s0.lgstatic.com/i/image3/M01/78/4E/CgpOIF5zioeAQINcAAPHrIebZSk802.png" alt="" data-nodeid="55572"></p>
<p data-nodeid="55403">可以看到每部电影对应的区块都是一个 div 节点，它的 class 属性都有 el-card 这个值。每个列表页有 10 个这样的 div 节点，也就对应着 10 部电影的信息。</p>
<p data-nodeid="55404">我们再分析下从列表页是怎么进入到详情页的，我们选中电影的名称，看下结果：</p>
<p data-nodeid="55405"><img src="https://s0.lgstatic.com/i/image3/M01/78/4E/Cgq2xl5zioiAOdUKAAPy8ODV7Vw336.png" alt="" data-nodeid="55576"></p>
<p data-nodeid="61657" class="">可以看到这个名称实际上是一个 h2 节点，其内部的文字就是电影的标题。h2 节点的外面包含了一个 a 节点，这个 a 节点带有 href 属性，这就是一个超链接，其中 href 的值为 /detail/1，这是一个相对网站的根 URL&nbsp;<a href="https://static1.scrape.center/" data-nodeid="61661">https://static1.scrape.center/</a>&nbsp;路径，加上网站的根 URL 就构成了&nbsp;<a href="https://static1.scrape.center/detail/1" data-nodeid="61665">https://static1.scrape.center/detail/1</a>，也就是这部电影详情页的 URL。这样我们只需要提取这个 href 属性就能构造出详情页的 URL 并接着爬取了。</p>




<p data-nodeid="55407">接下来我们来分析下翻页的逻辑，我们拉到页面的最下方，可以看到分页页码，如图所示：</p>
<p data-nodeid="55408"><img src="https://s0.lgstatic.com/i/image3/M01/78/4E/CgpOIF5zioiAc8WPAARcToG65Dw195.png" alt="" data-nodeid="55588"></p>
<p data-nodeid="55409">页面显示一共有 100 条数据，10 页的内容，因此页码最多是 10。接着我们点击第 2 页，如图所示：</p>
<p data-nodeid="55410"><img src="https://s0.lgstatic.com/i/image3/M01/78/4E/Cgq2xl5ziomAckUSAAQX_VVjG7U401.png" alt="" data-nodeid="55591"></p>
<p data-nodeid="63337" class="">可以看到网页的 URL 变成了&nbsp;<a href="http://%20https://static1.center/page/2" data-nodeid="63341">https://static1.scrape.center/page/2</a>，相比根 URL 多了 &nbsp;/page/2 &nbsp;这部分内容。网页的结构还是和原来一模一样，所以我们可以和第 1 页一样处理。</p>


<p data-nodeid="55412">接着我们查看第 3 页、第 4 页等内容，可以发现有这么一个规律，每一页的 URL 最后分别变成了 /page/3、/page/4。所以，/page 后面跟的就是列表页的页码，当然第 1 页也是一样，我们在根 URL 后面加上 /page/1 也是能访问的，只不过网站做了一下处理，默认的页码是 1，所以显示第 1 页的内容。</p>
<p data-nodeid="55413">好，分析到这里，逻辑基本就清晰了。</p>
<p data-nodeid="55414">如果我们要完成列表页的爬取，可以这么实现：</p>
<ul data-nodeid="55415">
<li data-nodeid="55416">
<p data-nodeid="55417">遍历页码构造 10 页的索引页 URL。</p>
</li>
<li data-nodeid="55418">
<p data-nodeid="55419">从每个索引页分析提取出每个电影的详情页 URL。</p>
</li>
</ul>
<p data-nodeid="55420">现在我们写代码来实现一下吧。</p>
<p data-nodeid="55421">首先，我们需要先定义一些基础的变量，并引入一些必要的库，写法如下：</p>
<pre class="lang-python" data-nodeid="64175"><code data-language="python"><span class="hljs-keyword">import</span> requests
<span class="hljs-keyword">import</span> logging
<span class="hljs-keyword">import</span> re
<span class="hljs-keyword">import</span> pymongo
<span class="hljs-keyword">from</span> pyquery <span class="hljs-keyword">import</span> PyQuery <span class="hljs-keyword">as</span> pq
<span class="hljs-keyword">from</span> urllib.parse <span class="hljs-keyword">import</span> urljoin

logging.basicConfig(level=logging.INFO,
                    format=<span class="hljs-string">'%(asctime)s - %(levelname)s: %(message)s'</span>)

BASE_URL = <span class="hljs-string">'https://static1.scrape.center'</span>
TOTAL_PAGE = <span class="hljs-number">10</span>
</code></pre>

<p data-nodeid="55423">这里我们引入了 requests 用来爬取页面，logging 用来输出信息，re 用来实现正则表达式解析，pyquery 用来直接解析网页，pymongo 用来实现 MongoDB 存储，urljoin 用来做 URL 的拼接。</p>
<p data-nodeid="55424">接着我们定义日志输出级别和输出格式，完成之后再定义 BASE_URL 为当前站点的根 URL，TOTAL_PAGE 为需要爬取的总页码数量。</p>
<p data-nodeid="55425">定义好了之后，我们来实现一个页面爬取的方法吧，实现如下：</p>
<pre class="lang-python" data-nodeid="55426"><code data-language="python"><span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">scrape_page</span>(<span class="hljs-params">url</span>):</span>
    logging.info(<span class="hljs-string">'scraping %s...'</span>, url)
    <span class="hljs-keyword">try</span>:
        response = requests.get(url)
        <span class="hljs-keyword">if</span> response.status_code == <span class="hljs-number">200</span>:
            <span class="hljs-keyword">return</span> response.text
        logging.error(<span class="hljs-string">'get invalid status code %s while scraping %s'</span>, response.status_code, url)
    <span class="hljs-keyword">except</span> requests.RequestException:
        logging.error(<span class="hljs-string">'error occurred while scraping %s'</span>, url, exc_info=<span class="hljs-literal">True</span>)
</code></pre>
<p data-nodeid="55427">考虑到我们不仅要爬取列表页，还要爬取详情页，所以在这里我们定义一个较通用的爬取页面的方法，叫作 scrape_page，它接收一个 url 参数，返回页面的 html 代码。</p>
<p data-nodeid="55428">这里我们首先判断状态码是不是 200，如果是，则直接返回页面的 HTML 代码，如果不是，则会输出错误日志信息。另外，这里实现了 requests 的异常处理，如果出现了爬取异常，则会输出对应的错误日志信息。这时我们将 logging 的 error 方法的 exc_info 参数设置为 True 则可以打印出 Traceback 错误堆栈信息。</p>
<p data-nodeid="55429">好了，有了 scrape_page 方法之后，我们给这个方法传入一个 url，正常情况下它就可以返回页面的 HTML 代码了。</p>
<p data-nodeid="55430">在这个基础上，我们来定义列表页的爬取方法吧，实现如下：</p>
<pre class="lang-python" data-nodeid="55431"><code data-language="python"><span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">scrape_index</span>(<span class="hljs-params">page</span>):</span>
    index_url = <span class="hljs-string">f'<span class="hljs-subst">{BASE_URL}</span>/page/<span class="hljs-subst">{page}</span>'</span>
    <span class="hljs-keyword">return</span> scrape_page(index_url)
</code></pre>
<p data-nodeid="55432">方法名称叫作 scrape_index，这个方法会接收一个 page 参数，即列表页的页码，我们在方法里面实现列表页的 URL 拼接，然后调用 scrape_page 方法爬取即可得到列表页的 HTML 代码了。</p>
<p data-nodeid="55433">获取了 HTML 代码后，下一步就是解析列表页，并得到每部电影的详情页的 URL 了，实现如下：</p>
<pre class="lang-python" data-nodeid="55434"><code data-language="python"><span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">parse_index</span>(<span class="hljs-params">html</span>):</span>
    doc = pq(html)
    links = doc(<span class="hljs-string">'.el-card .name'</span>)
    <span class="hljs-keyword">for</span> link <span class="hljs-keyword">in</span> links.items():
        href = link.attr(<span class="hljs-string">'href'</span>)
        detail_url = urljoin(BASE_URL, href)
        logging.info(<span class="hljs-string">'get detail url %s'</span>, detail_url)
        <span class="hljs-keyword">yield</span> detail_url
</code></pre>
<p data-nodeid="65850" class="">在这里我们定义了 parse_index 方法，它接收一个 html 参数，即列表页的 HTML 代码。接着我们用 pyquery 新建一个 PyQuery 对象，完成之后再用 .el-card .name 选择器选出来每个电影名称对应的超链接节点。我们遍历这些节点，通过调用 attr 方法并传入 href 获得详情页的 URL 路径，得到的 href 就是我们在上文所说的类似 &nbsp;/detail/1 &nbsp;这样的结果。由于这并不是一个完整的 URL，所以我们需要借助 urljoin 方法把 BASE_URL 和 href 拼接起来，获得详情页的完整 URL，得到的结果就是类似&nbsp;<a href="https://static1.scrape.center/detail/1" data-nodeid="65858">https://static1.scrape.center/detail/1</a>&nbsp;这样完整的 URL 了，最后 yield 返回即可。</p>


<p data-nodeid="55436">这样我们通过调用 parse_index 方法传入列表页的 HTML 代码就可以获得该列表页所有电影的详情页 URL 了。</p>
<p data-nodeid="55437">好，接下来我们把上面的方法串联调用一下，实现如下：</p>
<pre class="lang-python" data-nodeid="55438"><code data-language="python"><span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">main</span>():</span>
    <span class="hljs-keyword">for</span> page <span class="hljs-keyword">in</span> range(<span class="hljs-number">1</span>, TOTAL_PAGE + <span class="hljs-number">1</span>):
        index_html = scrape_index(page)
        detail_urls = parse_index(index_html)
        logging.info(<span class="hljs-string">'detail urls %s'</span>, list(detail_urls))

<span class="hljs-keyword">if</span> __name__ == <span class="hljs-string">'__main__'</span>:
    main()
</code></pre>
<p data-nodeid="55439">这里我们定义了 main 方法来完成上面所有方法的调用，首先使用 range 方法遍历一下页码，得到的 page 是 1~10，接着把 page 变量传给 scrape_index 方法，得到列表页的 HTML，赋值为 index_html 变量。接下来再将 index_html 变量传给 parse_index 方法，得到列表页所有电影的详情页 URL，赋值为 detail_urls，结果是一个生成器，我们调用 list 方法就可以将其输出出来。</p>
<p data-nodeid="55440">好，我们运行一下上面的代码，结果如下：</p>
<pre class="lang-python" data-nodeid="85851"><code data-language="python"><span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">22</span>:<span class="hljs-number">39</span>:<span class="hljs-number">50</span>,<span class="hljs-number">505</span> - INFO: scraping https://static1.scrape.center/page/<span class="hljs-number">1.</span>..
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">22</span>:<span class="hljs-number">39</span>:<span class="hljs-number">51</span>,<span class="hljs-number">949</span> - INFO: get detail url https://static1.scrape.center/detail/<span class="hljs-number">1</span>
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">22</span>:<span class="hljs-number">39</span>:<span class="hljs-number">51</span>,<span class="hljs-number">950</span> - INFO: get detail url https://static1.scrape.center/detail/<span class="hljs-number">2</span>
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">22</span>:<span class="hljs-number">39</span>:<span class="hljs-number">51</span>,<span class="hljs-number">950</span> - INFO: get detail url https://static1.scrape.center/detail/<span class="hljs-number">3</span>
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">22</span>:<span class="hljs-number">39</span>:<span class="hljs-number">51</span>,<span class="hljs-number">950</span> - INFO: get detail url https://static1.scrape.center/detail/<span class="hljs-number">4</span>
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">22</span>:<span class="hljs-number">39</span>:<span class="hljs-number">51</span>,<span class="hljs-number">950</span> - INFO: get detail url https://static1.scrape.center/detail/<span class="hljs-number">5</span>
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">22</span>:<span class="hljs-number">39</span>:<span class="hljs-number">51</span>,<span class="hljs-number">950</span> - INFO: get detail url https://static1.scrape.center/detail/<span class="hljs-number">6</span>
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">22</span>:<span class="hljs-number">39</span>:<span class="hljs-number">51</span>,<span class="hljs-number">950</span> - INFO: get detail url https://static1.scrape.center/detail/<span class="hljs-number">7</span>
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">22</span>:<span class="hljs-number">39</span>:<span class="hljs-number">51</span>,<span class="hljs-number">950</span> - INFO: get detail url https://static1.scrape.center/detail/<span class="hljs-number">8</span>
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">22</span>:<span class="hljs-number">39</span>:<span class="hljs-number">51</span>,<span class="hljs-number">950</span> - INFO: get detail url https://static1.scrape.center/detail/<span class="hljs-number">9</span>
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">22</span>:<span class="hljs-number">39</span>:<span class="hljs-number">51</span>,<span class="hljs-number">950</span> - INFO: get detail url https://static1.scrape.center/detail/<span class="hljs-number">10</span>
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">22</span>:<span class="hljs-number">39</span>:<span class="hljs-number">51</span>,<span class="hljs-number">951</span> - INFO: detail urls [<span class="hljs-string">'https://static1.scrape.center/detail/1'</span>, <span class="hljs-string">'https://static1.scrape.center/detail/2'</span>, <span class="hljs-string">'https://static1.scrape.center/detail/3'</span>, <span class="hljs-string">'https://static1.scrape.center/detail/4'</span>, <span class="hljs-string">'https://static1.scrape.center/detail/5'</span>, <span class="hljs-string">'https://static1.scrape.center/detail/6'</span>, <span class="hljs-string">'https://static1.scrape.center/detail/7'</span>, <span class="hljs-string">'https://static1.scrape.center/detail/8'</span>, <span class="hljs-string">'https://static1.scrape.center/detail/9'</span>, <span class="hljs-string">'https://static1.scrape.center/detail/10'</span>]
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">22</span>:<span class="hljs-number">39</span>:<span class="hljs-number">51</span>,<span class="hljs-number">951</span> - INFO: scraping https://static1.scrape.center/page/<span class="hljs-number">2.</span>..
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">22</span>:<span class="hljs-number">39</span>:<span class="hljs-number">52</span>,<span class="hljs-number">842</span> - INFO: get detail url https://static1.scrape.center/detail/<span class="hljs-number">11</span>
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">22</span>:<span class="hljs-number">39</span>:<span class="hljs-number">52</span>,<span class="hljs-number">842</span> - INFO: get detail url https://static1.scrape.center/detail/<span class="hljs-number">12</span>
...
</code></pre>
























<p data-nodeid="55442">由于输出内容比较多，这里只贴了一部分。</p>
<p data-nodeid="55443">可以看到，在这个过程中程序首先爬取了第 1 页列表页，然后得到了对应详情页的每个 URL，接着再接着爬第 2 页、第 3 页，一直到第 10 页，依次输出了每一页的详情页 URL。这样，我们就成功获取到所有电影详情页 URL 啦。</p>
<h3 data-nodeid="55444">爬取详情页</h3>
<p data-nodeid="55445">现在我们已经成功获取所有详情页 URL 了，那么下一步当然就是解析详情页并提取出我们想要的信息了。</p>
<p data-nodeid="55446">我们首先观察一下详情页的 HTML 代码吧，如图所示：</p>
<p data-nodeid="55447"><img src="https://s0.lgstatic.com/i/image3/M01/78/4E/Cgq2xl5zioqAFeI4AAXOH43947I062.png" alt="" data-nodeid="55660"></p>
<p data-nodeid="55448">经过分析，我们想要提取的内容和对应的节点信息如下：</p>
<ul data-nodeid="55449">
<li data-nodeid="55450">
<p data-nodeid="55451">封面：是一个 img 节点，其 class 属性为 cover。</p>
</li>
<li data-nodeid="55452">
<p data-nodeid="55453">名称：是一个 h2 节点，其内容便是名称。</p>
</li>
<li data-nodeid="55454">
<p data-nodeid="55455">类别：是 span 节点，其内容便是类别内容，其外侧是 button 节点，再外侧则是 class 为 categories 的 div 节点。</p>
</li>
<li data-nodeid="55456">
<p data-nodeid="55457">上映时间：是 span 节点，其内容包含了上映时间，其外侧是包含了 class 为 info 的 div 节点。但注意这个 div 前面还有一个 class 为 info 的 div 节点，我们可以使用其内容来区分，也可以使用 nth-child 或 nth-of-type 这样的选择器来区分。另外提取结果中还多了「上映」二字，我们可以用正则表达式把日期提取出来。</p>
</li>
<li data-nodeid="55458">
<p data-nodeid="55459">评分：是一个 p 节点，其内容便是评分，p 节点的 class 属性为 score。</p>
</li>
<li data-nodeid="55460">
<p data-nodeid="55461">剧情简介：是一个 p 节点，其内容便是剧情简介，其外侧是 class 为 drama 的 div 节点。</p>
</li>
</ul>
<p data-nodeid="55462">看上去有点复杂，但是不用担心，有了 pyquery 和正则表达式，我们可以轻松搞定。</p>
<p data-nodeid="55463">接着我们来实现一下代码吧。</p>
<p data-nodeid="55464">刚才我们已经成功获取了详情页的 URL，接下来我们要定义一个详情页的爬取方法，实现如下：</p>
<pre class="lang-python" data-nodeid="55465"><code data-language="python"><span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">scrape_detail</span>(<span class="hljs-params">url</span>):</span>
    <span class="hljs-keyword">return</span> scrape_page(url)
</code></pre>
<p data-nodeid="55466">这里定义了一个 scrape_detail 方法，它接收一个 url 参数，并通过调用 scrape_page 方法获得网页源代码。由于我们刚才已经实现了 scrape_page 方法，所以在这里我们不用再写一遍页面爬取的逻辑了，直接调用即可，这就做到了代码复用。</p>
<p data-nodeid="55467">另外你可能会问，这个 scrape_detail 方法里面只调用了 scrape_page 方法，没有别的功能，那爬取详情页直接用 scrape_page 方法不就好了，还有必要再单独定义 scrape_detail 方法吗？</p>
<p data-nodeid="55468">答案是有必要，单独定义一个 scrape_detail 方法在逻辑上会显得更清晰，而且以后如果我们想要对 scrape_detail 方法进行改动，比如添加日志输出或是增加预处理，都可以在 scrape_detail 里面实现，而不用改动 scrape_page 方法，灵活性会更好。</p>
<p data-nodeid="55469">好了，详情页的爬取方法已经实现了，接着就是详情页的解析了，实现如下：</p>
<pre class="lang-python" data-nodeid="55470"><code data-language="python"><span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">parse_detail</span>(<span class="hljs-params">html</span>):</span>
    doc = pq(html)
    cover = doc(<span class="hljs-string">'img.cover'</span>).attr(<span class="hljs-string">'src'</span>)
    name = doc(<span class="hljs-string">'a &gt; h2'</span>).text()
    categories = [item.text() <span class="hljs-keyword">for</span> item <span class="hljs-keyword">in</span> doc(<span class="hljs-string">'.categories button span'</span>).items()]
    published_at = doc(<span class="hljs-string">'.info:contains(上映)'</span>).text()
    published_at = re.search(<span class="hljs-string">'(\d{4}-\d{2}-\d{2})'</span>, published_at).group(<span class="hljs-number">1</span>) \
        <span class="hljs-keyword">if</span> published_at <span class="hljs-keyword">and</span> re.search(<span class="hljs-string">'\d{4}-\d{2}-\d{2}'</span>, published_at) <span class="hljs-keyword">else</span> <span class="hljs-literal">None</span>
    drama = doc(<span class="hljs-string">'.drama p'</span>).text()
    score = doc(<span class="hljs-string">'p.score'</span>).text()
    score = float(score) <span class="hljs-keyword">if</span> score <span class="hljs-keyword">else</span> <span class="hljs-literal">None</span>
    <span class="hljs-keyword">return</span> {
        <span class="hljs-string">'cover'</span>: cover,
        <span class="hljs-string">'name'</span>: name,
        <span class="hljs-string">'categories'</span>: categories,
        <span class="hljs-string">'published_at'</span>: published_at,
        <span class="hljs-string">'drama'</span>: drama,
        <span class="hljs-string">'score'</span>: score
    }
</code></pre>
<p data-nodeid="55471">这里我们定义了 parse_detail 方法用于解析详情页，它接收一个 html 参数，解析其中的内容，并以字典的形式返回结果。每个字段的解析情况如下所述：</p>
<ul data-nodeid="55472">
<li data-nodeid="55473">
<p data-nodeid="55474">cover：封面，直接选取 class 为 cover 的 img 节点，并调用 attr 方法获取 src 属性的内容即可。</p>
</li>
<li data-nodeid="55475">
<p data-nodeid="55476">name：名称，直接选取 a 节点的直接子节点 h2 节点，并调用 text 方法提取其文本内容即可得到名称。</p>
</li>
<li data-nodeid="55477">
<p data-nodeid="55478">categories：类别，由于类别是多个，所以这里首先用 .categories button span 选取了 class 为 categories 的节点内部的 span 节点，其结果是多个，所以这里进行了遍历，取出了每个 span 节点的文本内容，得到的便是列表形式的类别。</p>
</li>
<li data-nodeid="55479">
<p data-nodeid="55480">published_at：上映时间，由于 pyquery 支持使用 :contains 直接指定包含的文本内容并进行提取，且每个上映时间信息都包含了「上映」二字，所以我们这里就直接使用 :contains(上映) 提取了 class 为 info 的 div 节点。提取之后，得到的结果类似「1993-07-26 上映」这样，但我们并不想要「上映」这两个字，所以我们又调用了正则表达式把日期单独提取出来了。当然这里也可以直接使用 strip 或 replace 方法把多余的文字去掉，但我们为了练习正则表达式的用法，使用了正则表达式来提取。</p>
</li>
<li data-nodeid="55481">
<p data-nodeid="55482">drama：直接提取 class 为 drama 的节点内部的 p 节点的文本即可。</p>
</li>
<li data-nodeid="55483">
<p data-nodeid="55484">score：直接提取 class 为 score 的 p 节点的文本即可，但由于提取结果是字符串，所以我们需要把它转成浮点数，即 float 类型。</p>
</li>
</ul>
<p data-nodeid="55485">上述字段提取完毕之后，构造一个字典返回即可。</p>
<p data-nodeid="55486">这样，我们就成功完成了详情页的提取和分析了。</p>
<p data-nodeid="55487">最后，我们将 main 方法稍微改写一下，增加这两个方法的调用，改写如下：</p>
<pre class="lang-python" data-nodeid="55488"><code data-language="python"><span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">main</span>():</span>
    <span class="hljs-keyword">for</span> page <span class="hljs-keyword">in</span> range(<span class="hljs-number">1</span>, TOTAL_PAGE + <span class="hljs-number">1</span>):
        index_html = scrape_index(page)
        detail_urls = parse_index(index_html)
        <span class="hljs-keyword">for</span> detail_url <span class="hljs-keyword">in</span> detail_urls:
            detail_html = scrape_detail(detail_url)
            data = parse_detail(detail_html)
            logging.info(<span class="hljs-string">'get detail data %s'</span>, data)
</code></pre>
<p data-nodeid="55489">这里我们首先遍历了 detail_urls，获取了每个详情页的 URL，然后依次调用了 scrape_detail 和 parse_detail 方法，最后得到了每个详情页的提取结果，赋值为 data 并输出。</p>
<p data-nodeid="55490">运行结果如下：</p>
<pre class="lang-python" data-nodeid="90849"><code data-language="python"><span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">23</span>:<span class="hljs-number">37</span>:<span class="hljs-number">35</span>,<span class="hljs-number">936</span> - INFO: scraping https://static1.scrape.center/page/<span class="hljs-number">1.</span>..
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">23</span>:<span class="hljs-number">37</span>:<span class="hljs-number">36</span>,<span class="hljs-number">833</span> - INFO: get detail url https://static1.scrape.center/detail/<span class="hljs-number">1</span>
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">23</span>:<span class="hljs-number">37</span>:<span class="hljs-number">36</span>,<span class="hljs-number">833</span> - INFO: scraping https://static1.scrape.center/detail/<span class="hljs-number">1.</span>..
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">23</span>:<span class="hljs-number">37</span>:<span class="hljs-number">39</span>,<span class="hljs-number">985</span> - INFO: get detail data {<span class="hljs-string">'cover'</span>: <span class="hljs-string">'https://p0.meituan.net/movie/ce4da3e03e655b5b88ed31b5cd7896cf62472.jpg@464w_644h_1e_1c'</span>, <span class="hljs-string">'name'</span>: <span class="hljs-string">'霸王别姬 - Farewell My Concubine'</span>, <span class="hljs-string">'categories'</span>: [<span class="hljs-string">'剧情'</span>, <span class="hljs-string">'爱情'</span>], <span class="hljs-string">'published_at'</span>: <span class="hljs-string">'1993-07-26'</span>, <span class="hljs-string">'drama'</span>: <span class="hljs-string">'影片借一出《霸王别姬》的京戏，牵扯出三个人之间一段随时代风云变幻的爱恨情仇。段小楼（张丰毅 饰）与程蝶衣（张国荣 饰）是一对打小一起长大的师兄弟，两人一个演生，一个饰旦，一向配合天衣无缝，尤其一出《霸王别姬》，更是誉满京城，为此，两人约定合演一辈子《霸王别姬》。但两人对戏剧与人生关系的理解有本质不同，段小楼深知戏非人生，程蝶衣则是人戏不分。段小楼在认为该成家立业之时迎娶了名妓菊仙（巩俐 饰），致使程蝶衣认定菊仙是可耻的第三者，使段小楼做了叛徒，自此，三人围绕一出《霸王别姬》生出的爱恨情仇战开始随着时代风云的变迁不断升级，终酿成悲剧。'</span>, <span class="hljs-string">'score'</span>: <span class="hljs-number">9.5</span>}
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">23</span>:<span class="hljs-number">37</span>:<span class="hljs-number">39</span>,<span class="hljs-number">985</span> - INFO: get detail url https://static1.scrape.center/detail/<span class="hljs-number">2</span>
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">23</span>:<span class="hljs-number">37</span>:<span class="hljs-number">39</span>,<span class="hljs-number">985</span> - INFO: scraping https://static1.scrape.center/detail/<span class="hljs-number">2.</span>..
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">23</span>:<span class="hljs-number">37</span>:<span class="hljs-number">41</span>,<span class="hljs-number">061</span> - INFO: get detail data {<span class="hljs-string">'cover'</span>: <span class="hljs-string">'https://p1.meituan.net/movie/6bea9af4524dfbd0b668eaa7e187c3df767253.jpg@464w_644h_1e_1c'</span>, <span class="hljs-string">'name'</span>: <span class="hljs-string">'这个杀手不太冷 - Léon'</span>, <span class="hljs-string">'categories'</span>: [<span class="hljs-string">'剧情'</span>, <span class="hljs-string">'动作'</span>, <span class="hljs-string">'犯罪'</span>], <span class="hljs-string">'published_at'</span>: <span class="hljs-string">'1994-09-14'</span>, <span class="hljs-string">'drama'</span>: <span class="hljs-string">'里昂（让·雷诺 饰）是名孤独的职业杀手，受人雇佣。一天，邻居家小姑娘马蒂尔德（纳塔丽·波特曼 饰）敲开他的房门，要求在他那里暂避杀身之祸。原来邻居家的主人是警方缉毒组的眼线，只因贪污了一小包毒品而遭恶警（加里·奥德曼 饰）杀害全家的惩罚。马蒂尔德 得到里昂的留救，幸免于难，并留在里昂那里。里昂教小女孩使枪，她教里昂法文，两人关系日趋亲密，相处融洽。 女孩想着去报仇，反倒被抓，里昂及时赶到，将女孩救回。混杂着哀怨情仇的正邪之战渐次升级，更大的冲突在所难免……'</span>, <span class="hljs-string">'score'</span>: <span class="hljs-number">9.5</span>}
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-08</span> <span class="hljs-number">23</span>:<span class="hljs-number">37</span>:<span class="hljs-number">41</span>,<span class="hljs-number">062</span> - INFO: get detail url https://static1.scrape.center/detail/<span class="hljs-number">3</span>
...
</code></pre>






<p data-nodeid="55492">由于内容较多，这里省略了后续内容。</p>
<p data-nodeid="55493">可以看到，我们已经成功提取出每部电影的基本信息，包括封面、名称、类别，等等。</p>
<h3 data-nodeid="55494">保存到 MongoDB</h3>
<p data-nodeid="55495">成功提取到详情页信息之后，下一步我们就要把数据保存起来了。在上一课时我们学习了 MongoDB 的相关操作，接下来我们就把数据保存到 MongoDB 吧。</p>
<p data-nodeid="55496">在这之前，请确保现在有一个可以正常连接和使用的 MongoDB 数据库。</p>
<p data-nodeid="55497">将数据导入 MongoDB 需要用到 PyMongo 这个库，这个在最开始已经引入过了。那么接下来我们定义一下 MongoDB 的连接配置，实现如下：</p>
<pre class="lang-python" data-nodeid="55498"><code data-language="python">MONGO_CONNECTION_STRING = <span class="hljs-string">'mongodb://localhost:27017'</span>
MONGO_DB_NAME = <span class="hljs-string">'movies'</span>
MONGO_COLLECTION_NAME = <span class="hljs-string">'movies'</span>

client = pymongo.MongoClient(MONGO_CONNECTION_STRING)
db = client[<span class="hljs-string">'movies'</span>]
collection = db[<span class="hljs-string">'movies'</span>]
</code></pre>
<p data-nodeid="55499">在这里我们声明了几个变量，介绍如下：</p>
<ul data-nodeid="55500">
<li data-nodeid="55501">
<p data-nodeid="55502">MONGO_CONNECTION_STRING：MongoDB 的连接字符串，里面定义了 MongoDB 的基本连接信息，如 host、port，还可以定义用户名密码等内容。</p>
</li>
<li data-nodeid="55503">
<p data-nodeid="55504">MONGO_DB_NAME：MongoDB 数据库的名称。</p>
</li>
<li data-nodeid="55505">
<p data-nodeid="55506">MONGO_COLLECTION_NAME：MongoDB 的集合名称。</p>
</li>
</ul>
<p data-nodeid="55507">这里我们用 MongoClient 声明了一个连接对象，然后依次声明了存储的数据库和集合。</p>
<p data-nodeid="55508">接下来，我们再实现一个将数据保存到 MongoDB 的方法，实现如下：</p>
<pre class="lang-python" data-nodeid="55509"><code data-language="python"><span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">save_data</span>(<span class="hljs-params">data</span>):</span>
    collection.update_one({
        <span class="hljs-string">'name'</span>: data.get(<span class="hljs-string">'name'</span>)
    }, {
        <span class="hljs-string">'$set'</span>: data
    }, upsert=<span class="hljs-literal">True</span>)
</code></pre>
<p data-nodeid="55510">在这里我们声明了一个 save_data 方法，它接收一个 data 参数，也就是我们刚才提取的电影详情信息。在方法里面，我们调用了 update_one 方法，第 1 个参数是查询条件，即根据 name 进行查询；第 2 个参数是 data 对象本身，也就是所有的数据，这里我们用 $set 操作符表示更新操作；第 3 个参数很关键，这里实际上是 upsert 参数，如果把这个设置为 True，则可以做到存在即更新，不存在即插入的功能，更新会根据第一个参数设置的 name 字段，所以这样可以防止数据库中出现同名的电影数据。</p>
<blockquote data-nodeid="55511">
<p data-nodeid="55512">注：实际上电影可能有同名，但该场景下的爬取数据没有同名情况，当然这里更重要的是实现 MongoDB 的去重操作。</p>
</blockquote>
<p data-nodeid="55513">好的，那么接下来我们将 main 方法稍微改写一下就好了，改写如下：</p>
<pre class="lang-python" data-nodeid="55514"><code data-language="python"><span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">main</span>():</span>
    <span class="hljs-keyword">for</span> page <span class="hljs-keyword">in</span> range(<span class="hljs-number">1</span>, TOTAL_PAGE + <span class="hljs-number">1</span>):
        index_html = scrape_index(page)
        detail_urls = parse_index(index_html)
        <span class="hljs-keyword">for</span> detail_url <span class="hljs-keyword">in</span> detail_urls:
            detail_html = scrape_detail(detail_url)
            data = parse_detail(detail_html)
            logging.info(<span class="hljs-string">'get detail data %s'</span>, data)
            logging.info(<span class="hljs-string">'saving data to mongodb'</span>)
            save_data(data)
            logging.info(<span class="hljs-string">'data saved successfully'</span>)
</code></pre>
<p data-nodeid="55515">这里增加了 save_data 方法的调用，并加了一些日志信息。</p>
<p data-nodeid="55516">重新运行，我们看下输出结果：</p>
<pre class="lang-python" data-nodeid="95014"><code data-language="python"><span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-09</span> <span class="hljs-number">01</span>:<span class="hljs-number">10</span>:<span class="hljs-number">27</span>,<span class="hljs-number">094</span> - INFO: scraping https://static1.scrape.center/page/<span class="hljs-number">1.</span>..
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-09</span> <span class="hljs-number">01</span>:<span class="hljs-number">10</span>:<span class="hljs-number">28</span>,<span class="hljs-number">019</span> - INFO: get detail url https://static1.scrape.center/detail/<span class="hljs-number">1</span>
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-09</span> <span class="hljs-number">01</span>:<span class="hljs-number">10</span>:<span class="hljs-number">28</span>,<span class="hljs-number">019</span> - INFO: scraping https://static1.scrape.center/detail/<span class="hljs-number">1.</span>..
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-09</span> <span class="hljs-number">01</span>:<span class="hljs-number">10</span>:<span class="hljs-number">29</span>,<span class="hljs-number">183</span> - INFO: get detail data {<span class="hljs-string">'cover'</span>: <span class="hljs-string">'https://p0.meituan.net/movie/ce4da3e03e655b5b88ed31b5cd7896cf62472.jpg@464w_644h_1e_1c'</span>, <span class="hljs-string">'name'</span>: <span class="hljs-string">'霸王别姬 - Farewell My Concubine'</span>, <span class="hljs-string">'categories'</span>: [<span class="hljs-string">'剧情'</span>, <span class="hljs-string">'爱情'</span>], <span class="hljs-string">'published_at'</span>: <span class="hljs-string">'1993-07-26'</span>, <span class="hljs-string">'drama'</span>: <span class="hljs-string">'影片借一出《霸王别姬》的京戏，牵扯出三个人之间一段随时代风云变幻的爱恨情仇。段小楼（张丰毅 饰）与程蝶衣（张国荣 饰）是一对打小一起长大的师兄弟，两人一个演生，一个饰旦，一向配合天衣无缝，尤其一出《霸王别姬》，更是誉满京城，为此，两人约定合演一辈子《霸王别姬》。但两人对戏剧与人生关系的理解有本质不同，段小楼深知戏非人生，程蝶衣则是人戏不分。段小楼在认为该成家立业之时迎娶了名妓菊仙（巩俐 饰），致使程蝶衣认定菊仙是可耻的第三者，使段小楼做了叛徒，自此，三人围绕一出《霸王别姬》生出的爱恨情仇战开始随着时代风云的变迁不断升级，终酿成悲剧。'</span>, <span class="hljs-string">'score'</span>: <span class="hljs-number">9.5</span>}
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-09</span> <span class="hljs-number">01</span>:<span class="hljs-number">10</span>:<span class="hljs-number">29</span>,<span class="hljs-number">183</span> - INFO: saving data to mongodb
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-09</span> <span class="hljs-number">01</span>:<span class="hljs-number">10</span>:<span class="hljs-number">29</span>,<span class="hljs-number">288</span> - INFO: data saved successfully
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-09</span> <span class="hljs-number">01</span>:<span class="hljs-number">10</span>:<span class="hljs-number">29</span>,<span class="hljs-number">288</span> - INFO: get detail url https://static1.scrape.center/detail/<span class="hljs-number">2</span>
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-09</span> <span class="hljs-number">01</span>:<span class="hljs-number">10</span>:<span class="hljs-number">29</span>,<span class="hljs-number">288</span> - INFO: scraping https://static1.scrape.center/detail/<span class="hljs-number">2.</span>..
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-09</span> <span class="hljs-number">01</span>:<span class="hljs-number">10</span>:<span class="hljs-number">30</span>,<span class="hljs-number">250</span> - INFO: get detail data {<span class="hljs-string">'cover'</span>: <span class="hljs-string">'https://p1.meituan.net/movie/6bea9af4524dfbd0b668eaa7e187c3df767253.jpg@464w_644h_1e_1c'</span>, <span class="hljs-string">'name'</span>: <span class="hljs-string">'这个杀手不太冷 - Léon'</span>, <span class="hljs-string">'categories'</span>: [<span class="hljs-string">'剧情'</span>, <span class="hljs-string">'动作'</span>, <span class="hljs-string">'犯罪'</span>], <span class="hljs-string">'published_at'</span>: <span class="hljs-string">'1994-09-14'</span>, <span class="hljs-string">'drama'</span>: <span class="hljs-string">'里昂（让·雷诺 饰）是名孤独的职业杀手，受人雇佣。一天，邻居家小姑娘马蒂尔德（纳塔丽·波特曼 饰）敲开他的房门，要求在他那里暂避杀身之祸。原来邻居家的主人是警方缉毒组的眼线，只因贪污了一小包毒品而遭恶警（加里·奥德曼 饰）杀害全家的惩罚。马蒂尔德 得到里昂的留救，幸免于难，并留在里昂那里。里昂教小女孩使枪，她教里昂法文，两人关系日趋亲密，相处融洽。 女孩想着去报仇，反倒被抓，里昂及时赶到，将女孩救回。混杂着哀怨情仇的正邪之战渐次升级，更大的冲突在所难免……'</span>, <span class="hljs-string">'score'</span>: <span class="hljs-number">9.5</span>}
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-09</span> <span class="hljs-number">01</span>:<span class="hljs-number">10</span>:<span class="hljs-number">30</span>,<span class="hljs-number">250</span> - INFO: saving data to mongodb
<span class="hljs-number">2020</span><span class="hljs-number">-03</span><span class="hljs-number">-09</span> <span class="hljs-number">01</span>:<span class="hljs-number">10</span>:<span class="hljs-number">30</span>,<span class="hljs-number">253</span> - INFO: data saved successfully
...
</code></pre>





<p data-nodeid="55518">在运行结果中我们可以发现，这里输出了存储 MongoDB 成功的信息。</p>
<p data-nodeid="55519">运行完毕之后我们可以使用 MongoDB 客户端工具（例如 Robo 3T ）可视化地查看已经爬取到的数据，结果如下：</p>
<p data-nodeid="55520"><img src="https://s0.lgstatic.com/i/image3/M01/78/4E/CgpOIF5zio2ANIa3AAK44uJ0lis128.png" alt="" data-nodeid="55757"></p>
<p data-nodeid="55521">这样，所有的电影就被我们成功爬取下来啦！不多不少，正好 100 条。</p>
<h3 data-nodeid="55522">多进程加速</h3>
<p data-nodeid="55523">由于整个的爬取是单进程的，而且只能逐条爬取，速度稍微有点慢，有没有方法来对整个爬取过程进行加速呢？</p>
<p data-nodeid="55524">在前面我们讲了多进程的基本原理和使用方法，下面我们就来实践一下多进程的爬取吧。</p>
<p data-nodeid="55525">由于一共有 10 页详情页，并且这 10 页内容是互不干扰的，所以我们可以一页开一个进程来爬取。由于这 10 个列表页页码正好可以提前构造成一个列表，所以我们可以选用多进程里面的进程池 Pool 来实现这个过程。</p>
<p data-nodeid="55526">这里我们需要改写下 main 方法的调用，实现如下：</p>
<pre class="lang-python" data-nodeid="55527"><code data-language="python"><span class="hljs-keyword">import</span> multiprocessing

<span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">main</span>(<span class="hljs-params">page</span>):</span>
    index_html = scrape_index(page)
    detail_urls = parse_index(index_html)
    <span class="hljs-keyword">for</span> detail_url <span class="hljs-keyword">in</span> detail_urls:
        detail_html = scrape_detail(detail_url)
        data = parse_detail(detail_html)
        logging.info(<span class="hljs-string">'get detail data %s'</span>, data)
        logging.info(<span class="hljs-string">'saving data to mongodb'</span>)
        save_data(data)
        logging.info(<span class="hljs-string">'data saved successfully'</span>)

<span class="hljs-keyword">if</span> __name__ == <span class="hljs-string">'__main__'</span>:
    pool = multiprocessing.Pool()
    pages = range(<span class="hljs-number">1</span>, TOTAL_PAGE + <span class="hljs-number">1</span>)
    pool.map(main, pages)
    pool.close()
    pool.join()
</code></pre>
<p data-nodeid="55528">这里我们首先给 main 方法添加一个参数 page，用以表示列表页的页码。接着我们声明了一个进程池，并声明 pages 为所有需要遍历的页码，即 1~10。最后调用 map 方法，第 1 个参数就是需要被调用的方法，第 2 个参数就是 pages，即需要遍历的页码。</p>
<p data-nodeid="55529">这样 pages 就会被依次遍历。把 1~10 这 10 个页码分别传递给 main 方法，并把每次的调用变成一个进程，加入到进程池中执行，进程池会根据当前运行环境来决定运行多少进程。比如我的机器的 CPU 有 8 个核，那么进程池的大小会默认设定为 8，这样就会同时有 8 个进程并行执行。</p>
<p data-nodeid="55530">运行输出结果和之前类似，但是可以明显看到加了多进程执行之后，爬取速度快了非常多。我们可以清空一下之前的 MongoDB 数据，可以发现数据依然可以被正常保存到 MongoDB 数据库中。</p>
<h3 data-nodeid="55531">总结</h3>
<p data-nodeid="55532">到现在为止，我们就完成了全站电影数据的爬取并实现了存储和优化。</p>
<p data-nodeid="55533">这节课我们用到的库有 requests、pyquery、PyMongo、multiprocessing、re、logging 等，通过这个案例实战，我们把前面学习到的知识都串联了起来，其中的一些实现方法可以好好思考和体会，也希望这个案例能够让你对爬虫的实现有更实际的了解。</p>
