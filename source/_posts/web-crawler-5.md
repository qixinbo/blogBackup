---
title: 52讲轻松搞定网络爬虫笔记5
tags: [Web Crawler]
categories: data analysis
date: 2023-1-15
---

# 资料
[52讲轻松搞定网络爬虫](https://kaiwu.lagou.com/course/courseInfo.htm?courseId=46#/sale)


# 代理的基本原理和用法
<p>我们在做爬虫的过程中经常会遇到这样的情况，最初爬虫正常运行，正常抓取数据，一切看起来都是那么的美好，然而一杯茶的功夫可能就会出现错误，比如 403 Forbidden，这时候打开网页一看，可能会看到 “您的 IP 访问频率太高” 这样的提示，或者跳出一个验证码让我们输入，输入之后才可能解封，但是输入之后过一会儿就又这样了。</p>
<p>出现这种现象的原因是网站采取了一些反爬虫的措施，比如服务器会检测某个 IP 在单位时间内的请求次数，如果超过了这个阈值，那么会直接拒绝服务，返回一些错误信息，这种情况可以称之为封 IP，于是乎就成功把我们的爬虫禁掉了。</p>
<p>既然服务器检测的是某个 IP 单位时间的请求次数，那么我们借助某种方式来伪装我们的 IP，让服务器识别不出是由我们本机发起的请求，不就可以成功防止封 IP 了吗？所以这时候代理就派上用场了。</p>
<p>本课时我们先来看下代理的基本原理和使用代理处理反爬虫的方法。</p>
<h3>基本原理</h3>
<p>代理实际上指的就是代理服务器，英文叫作 proxy server，它的功能是代理网络用户去获取网络信息。形象地说，它是网络信息的中转站。在我们正常请求一个网站时，是发送了请求给 Web 服务器，Web 服务器把响应传回给我们。如果设置了代理服务器，实际上就是在本机和服务器之间搭建了一个桥，此时本机不是直接向 Web 服务器发起请求，而是向代理服务器发出请求，请求会发送给代理服务器，然后由代理服务器再发送给 Web 服务器，接着由代理服务器再把 Web 服务器返回的响应转发给本机。这样我们同样可以正常访问网页，但这个过程中 Web 服务器识别出的真实 IP 就不再是我们本机的 IP 了，就成功实现了 IP 伪装，这就是代理的基本原理。</p>
<h3>代理的作用</h3>
<p>那么，代理有什么作用呢？我们可以简单列举如下。</p>
<ul>
<li>突破自身 IP 访问限制，访问一些平时不能访问的站点。</li>
<li>访问一些单位或团体内部资源，如使用教育网内地址段免费代理服务器，就可以用于对教育网开放的各类 FTP 下载上传，以及各类资料查询共享等服务。</li>
<li>提高访问速度，通常代理服务器都设置一个较大的硬盘缓冲区，当有外界的信息通过时，也将其保存到缓冲区中，当其他用户再访问相同的信息时， 则直接由缓冲区中取出信息，传给用户，以提高访问速度。</li>
<li>隐藏真实 IP，上网者也可以通过这种方法隐藏自己的 IP，免受攻击，对于爬虫来说，我们用代理就是为了隐藏自身 IP，防止自身的 IP 被封锁。</li>
</ul>
<h3>爬虫代理</h3>
<p>对于爬虫来说，由于爬虫爬取速度过快，在爬取过程中可能遇到同一个 IP 访问过于频繁的问题，此时网站就会让我们输入验证码登录或者直接封锁 IP，这样会给爬取带来极大的不便。</p>
<p>使用代理隐藏真实的 IP，让服务器误以为是代理服务器在请求自己。这样在爬取过程中通过不断更换代理，就不会被封锁，可以达到很好的爬取效果。</p>
<h3>代理分类</h3>
<p>代理分类时，既可以根据协议区分，也可以根据其匿名程度区分，下面分别总结如下：</p>
<h4>根据协议区分</h4>
<p>根据代理的协议，代理可以分为如下类别：</p>
<ul>
<li>FTP 代理服务器，主要用于访问 FTP 服务器，一般有上传、下载以及缓存功能，端口一般为 21、2121 等。</li>
<li>HTTP 代理服务器，主要用于访问网页，一般有内容过滤和缓存功能，端口一般为 80、8080、3128 等。</li>
<li>SSL/TLS 代理，主要用于访问加密网站，一般有 SSL 或 TLS 加密功能（最高支持 128 位加密强度），端口一般为 443。</li>
<li>RTSP 代理，主要用于 Realplayer 访问 Real 流媒体服务器，一般有缓存功能，端口一般为 554。</li>
<li>Telnet 代理，主要用于 telnet 远程控制（黑客入侵计算机时常用于隐藏身份），端口一般为 23。</li>
<li>POP3/SMTP 代理，主要用于 POP3/SMTP 方式收发邮件，一般有缓存功能，端口一般为 110/25。</li>
<li>SOCKS 代理，只是单纯传递数据包，不关心具体协议和用法，所以速度快很多，一般有缓存功能，端口一般为 1080。SOCKS 代理协议又分为 SOCKS4 和 SOCKS5，SOCKS4 协议只支持 TCP，而 SOCKS5 协议支持 TCP 和 UDP，还支持各种身份验证机制、服务器端域名解析等。简单来说，SOCK4 能做到的 SOCKS5 都可以做到，但 SOCKS5 能做到的 SOCK4 不一定能做到。</li>
</ul>
<h4>根据匿名程度区分</h4>
<p>根据代理的匿名程度，代理可以分为如下类别。</p>
<ul>
<li>高度匿名代理，高度匿名代理会将数据包原封不动的转发，在服务端看来就好像真的是一个普通客户端在访问，而记录的 IP 是代理服务器的 IP。</li>
<li>普通匿名代理，普通匿名代理会在数据包上做一些改动，服务端上有可能发现这是个代理服务器，也有一定几率追查到客户端的真实 IP。代理服务器通常会加入的 HTTP 头有 HTTP_VIA 和 HTTP_X_FORWARDED_FOR。</li>
<li>透明代理，透明代理不但改动了数据包，还会告诉服务器客户端的真实 IP。这种代理除了能用缓存技术提高浏览速度，能用内容过滤提高安全性之外，并无其他显著作用，最常见的例子是内网中的硬件防火墙。</li>
<li>间谍代理，间谍代理指组织或个人创建的，用于记录用户传输的数据，然后进行研究、监控等目的的代理服务器。</li>
</ul>
<h3>常见代理类型</h3>
<ul>
<li>使用网上的免费代理，最好使用高匿代理，使用前抓取下来筛选一下可用代理，也可以进一步维护一个代理池。</li>
<li>使用付费代理服务，互联网上存在许多代理商，可以付费使用，质量比免费代理好很多。</li>
<li>ADSL 拨号，拨一次号换一次 IP，稳定性高，也是一种比较有效的解决方案。</li>
<li>蜂窝代理，即用 4G 或 5G 网卡等制作的代理，由于蜂窝网络用作代理的情形较少，因此整体被封锁的几率会较低，但搭建蜂窝代理的成本较高。</li>
</ul>
<h3>代理设置</h3>
<p>在前面我们介绍了多种请求库，如 Requests、Selenium、Pyppeteer 等。我们接下来首先贴近实战，了解一下代理怎么使用，为后面了解代理池打下基础。</p>
<p>下面我们来梳理一下这些库的代理的设置方法。</p>
<p>做测试之前，我们需要先获取一个可用代理。搜索引擎搜索 “代理” 关键字，就可以看到许多代理服务网站，网站上会有很多免费或付费代理，比如免费代理“快代理”：<a href="https://www.kuaidaili.com/free/">https://www.kuaidaili.com/free/</a>。但是这些免费代理大多数情况下都是不好用的，所以比较靠谱的方法是购买付费代理。付费代理各大代理商家都有套餐，数量不用多，稳定可用即可，我们可以自行选购。</p>
<p>如果本机有相关代理软件的话，软件一般会在本机创建 HTTP 或 SOCKS 代理服务，本机直接使用此代理也可以。</p>
<p>在这里，我的本机安装了一部代理软件，它会在本地的 7890 端口上创建 HTTP 代理服务，即代理为127.0.0.1:7890，另外还会在 7891 端口创建 SOCKS 代理服务，即代理为 127.0.0.1:7891。</p>
<p>我只要设置了这个代理，就可以成功将本机 IP 切换到代理软件连接的服务器的 IP 了。下面的示例里，我将使用上述代理来演示其设置方法，你也可以自行替换成自己的可用代理。设置代理后测试的网址是：<a href="http://httpbin.org/get">http://httpbin.org/get</a>，我们访问该网址可以得到请求的相关信息，其中 origin 字段就是客户端的 IP，我们可以根据它来判断代理是否设置成功，即是否成功伪装了 IP。</p>
<h3>requests 设置代理</h3>
<p>对于 requests 来说，代理设置非常简单，我们只需要传入 proxies 参数即可。</p>
<p>我在这里以我本机的代理为例，来看下 requests 的 HTTP 代理的设置，代码如下：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span>&nbsp;requests
proxy&nbsp;=&nbsp;<span class="hljs-string">'127.0.0.1:7890'</span>
proxies&nbsp;=&nbsp;{
&nbsp;&nbsp;&nbsp;<span class="hljs-string">'http'</span>:&nbsp;<span class="hljs-string">'http://'</span>&nbsp;+&nbsp;proxy,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">'https'</span>:&nbsp;<span class="hljs-string">'https://'</span>&nbsp;+&nbsp;proxy,
}
<span class="hljs-keyword">try</span>:
&nbsp;&nbsp;&nbsp;response&nbsp;=&nbsp;requests.get(<span class="hljs-string">'https://httpbin.org/get'</span>,&nbsp;proxies=proxies)
&nbsp;&nbsp;&nbsp;print(response.text)
<span class="hljs-keyword">except</span>&nbsp;requests.exceptions.ConnectionError&nbsp;<span class="hljs-keyword">as</span>&nbsp;e:
&nbsp;&nbsp;&nbsp;print(<span class="hljs-string">'Error'</span>,&nbsp;e.args)
运行结果：
{
&nbsp;<span class="hljs-string">"args"</span>:&nbsp;{},
&nbsp;<span class="hljs-string">"headers"</span>:&nbsp;{
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"Accept"</span>:&nbsp;<span class="hljs-string">"*/*"</span>,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"Accept-Encoding"</span>:&nbsp;<span class="hljs-string">"gzip,&nbsp;deflate"</span>,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"Host"</span>:&nbsp;<span class="hljs-string">"httpbin.org"</span>,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"User-Agent"</span>:&nbsp;<span class="hljs-string">"python-requests/2.22.0"</span>,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"X-Amzn-Trace-Id"</span>:&nbsp;<span class="hljs-string">"Root=1-5e8f358d-87913f68a192fb9f87aa0323"</span>
&nbsp;},
&nbsp;<span class="hljs-string">"origin"</span>:&nbsp;<span class="hljs-string">"210.173.1.204"</span>,
&nbsp;<span class="hljs-string">"url"</span>:&nbsp;<span class="hljs-string">"https://httpbin.org/get"</span>
}
</code></pre>
<p>可以发现，我们通过一个字典的形式就设置好了 HTTP 代理，它分为两个类别，有 HTTP 和 HTTPS，如果我们访问的链接是 HTTP 协议，那就用 http 字典名指定的代理，如果是 HTTPS 协议，那就用 https 字典名指定的代理。</p>
<p>其运行结果的 origin 如是代理服务器的 IP，则证明代理已经设置成功。</p>
<p>如果代理需要认证，同样在代理的前面加上用户名密码即可，代理的写法就变成如下所示：</p>
<pre><code data-language="python" class="lang-python">proxy&nbsp;=&nbsp;<span class="hljs-string">'username:password@127.0.0.1:7890'</span>
</code></pre>
<p>这里只需要将 username 和 password 替换即可。</p>
<p>如果需要使用 SOCKS 代理，则可以使用如下方式来设置：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span>&nbsp;requests
proxy&nbsp;=&nbsp;<span class="hljs-string">'127.0.0.1:7891'</span>
proxies&nbsp;=&nbsp;{
&nbsp;&nbsp;&nbsp;<span class="hljs-string">'http'</span>:&nbsp;<span class="hljs-string">'socks5://'</span>&nbsp;+&nbsp;proxy,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">'https'</span>:&nbsp;<span class="hljs-string">'socks5://'</span>&nbsp;+&nbsp;proxy
}
<span class="hljs-keyword">try</span>:
&nbsp;&nbsp;&nbsp;response&nbsp;=&nbsp;requests.get(<span class="hljs-string">'https://httpbin.org/get'</span>,&nbsp;proxies=proxies)
&nbsp;&nbsp;&nbsp;print(response.text)
<span class="hljs-keyword">except</span>&nbsp;requests.exceptions.ConnectionError&nbsp;<span class="hljs-keyword">as</span>&nbsp;e:
&nbsp;&nbsp;&nbsp;print(<span class="hljs-string">'Error'</span>,&nbsp;e.args)
</code></pre>
<p>在这里，我们需要额外安装一个包，这个包叫作 requests[socks]，安装命令如下所示：</p>
<pre><code data-language="python" class="lang-python">pip3&nbsp;install&nbsp;<span class="hljs-string">"requests[socks]"</span>
</code></pre>
<p>运行结果是完全相同的：</p>
<pre><code data-language="python" class="lang-python">{
&nbsp;<span class="hljs-string">"args"</span>:&nbsp;{},
&nbsp;<span class="hljs-string">"headers"</span>:&nbsp;{
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"Accept"</span>:&nbsp;<span class="hljs-string">"*/*"</span>,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"Accept-Encoding"</span>:&nbsp;<span class="hljs-string">"gzip,&nbsp;deflate"</span>,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"Host"</span>:&nbsp;<span class="hljs-string">"httpbin.org"</span>,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"User-Agent"</span>:&nbsp;<span class="hljs-string">"python-requests/2.22.0"</span>,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"X-Amzn-Trace-Id"</span>:&nbsp;<span class="hljs-string">"Root=1-5e8f364a-589d3cf2500fafd47b5560f2"</span>
&nbsp;},
&nbsp;<span class="hljs-string">"origin"</span>:&nbsp;<span class="hljs-string">"210.173.1.204"</span>,
&nbsp;<span class="hljs-string">"url"</span>:&nbsp;<span class="hljs-string">"https://httpbin.org/get"</span>
}
</code></pre>
<p>另外，还有一种设置方式即使用 socks 模块，也需要像上文一样安装 socks 库。这种设置方法如下所示：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span>&nbsp;requests
<span class="hljs-keyword">import</span>&nbsp;socks
<span class="hljs-keyword">import</span>&nbsp;socket
socks.set_default_proxy(socks.SOCKS5,&nbsp;<span class="hljs-string">'127.0.0.1'</span>,&nbsp;<span class="hljs-number">7891</span>)
socket.socket&nbsp;=&nbsp;socks.socksocket
<span class="hljs-keyword">try</span>:
&nbsp;&nbsp;&nbsp;response&nbsp;=&nbsp;requests.get(<span class="hljs-string">'https://httpbin.org/get'</span>)
&nbsp;&nbsp;&nbsp;print(response.text)
<span class="hljs-keyword">except</span>&nbsp;requests.exceptions.ConnectionError&nbsp;<span class="hljs-keyword">as</span>&nbsp;e:
&nbsp;&nbsp;&nbsp;print(<span class="hljs-string">'Error'</span>,&nbsp;e.args)
</code></pre>
<p>使用这种方法也可以设置 SOCKS 代理，运行结果完全相同。相比第一种方法，此方法是全局设置。我们可以在不同情况下选用不同的方法。</p>
<h3>Selenium 设置代理</h3>
<p>Selenium 同样可以设置代理，在这里以 Chrome 为例来介绍下其设置方法。</p>
<p>对于无认证的代理，设置方法如下：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver
proxy&nbsp;=&nbsp;<span class="hljs-string">'127.0.0.1:7890'</span>
options&nbsp;=&nbsp;webdriver.ChromeOptions()
options.add_argument(<span class="hljs-string">'--proxy-server=http://'</span>&nbsp;+&nbsp;proxy)
browser&nbsp;=&nbsp;webdriver.Chrome(options=options)
browser.get(<span class="hljs-string">'https://httpbin.org/get'</span>)
print(browser.page_source)
browser.close()
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="python" class="lang-python">{
&nbsp;<span class="hljs-string">"args"</span>:&nbsp;{},
&nbsp;<span class="hljs-string">"headers"</span>:&nbsp;{
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"Accept"</span>:&nbsp;<span class="hljs-string">"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9"</span>,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"Accept-Encoding"</span>:&nbsp;<span class="hljs-string">"gzip,&nbsp;deflate"</span>,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"Accept-Language"</span>:&nbsp;<span class="hljs-string">"zh-CN,zh;q=0.9"</span>,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"Host"</span>:&nbsp;<span class="hljs-string">"httpbin.org"</span>,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"Upgrade-Insecure-Requests"</span>:&nbsp;<span class="hljs-string">"1"</span>,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"User-Agent"</span>:&nbsp;<span class="hljs-string">"Mozilla/5.0&nbsp;(Macintosh;&nbsp;Intel&nbsp;Mac&nbsp;OS&nbsp;X&nbsp;10_15_4)&nbsp;AppleWebKit/537.36&nbsp;(KHTML,&nbsp;like&nbsp;Gecko)&nbsp;Chrome/80.0.3987.149&nbsp;Safari/537.36"</span>,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"X-Amzn-Trace-Id"</span>:&nbsp;<span class="hljs-string">"Root=1-5e8f39cd-60930018205fd154a9af39cc"</span>
&nbsp;},
&nbsp;<span class="hljs-string">"origin"</span>:&nbsp;<span class="hljs-string">"210.173.1.204"</span>,
&nbsp;<span class="hljs-string">"url"</span>:&nbsp;<span class="hljs-string">"http://httpbin.org/get"</span>
}
</code></pre>
<p>代理设置成功，origin 同样为代理 IP 的地址。</p>
<p>如果代理是认证代理，则设置方法相对比较麻烦，设置方法如下所示：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver
<span class="hljs-keyword">from</span>&nbsp;selenium.webdriver.chrome.options&nbsp;<span class="hljs-keyword">import</span>&nbsp;Options
<span class="hljs-keyword">import</span>&nbsp;zipfile
&nbsp;
ip&nbsp;=&nbsp;<span class="hljs-string">'127.0.0.1'</span>
port&nbsp;=&nbsp;<span class="hljs-number">7890</span>
username&nbsp;=&nbsp;<span class="hljs-string">'foo'</span>
password&nbsp;=&nbsp;<span class="hljs-string">'bar'</span>
&nbsp;
manifest_json&nbsp;=&nbsp;<span class="hljs-string">"""{"version":"1.0.0","manifest_version":&nbsp;2,"name":"Chrome&nbsp;Proxy","permissions":&nbsp;["proxy","tabs","unlimitedStorage","storage","&lt;all_urls&gt;","webRequest","webRequestBlocking"],"background":&nbsp;{"scripts":&nbsp;["background.js"]
&nbsp;&nbsp;&nbsp;}
}
"""</span>
background_js&nbsp;=&nbsp;<span class="hljs-string">"""
var&nbsp;config&nbsp;=&nbsp;{
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;mode:&nbsp;"fixed_servers",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;rules:&nbsp;{
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;singleProxy:&nbsp;{
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;scheme:&nbsp;"http",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;host:&nbsp;"%(ip)&nbsp;s",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;port:&nbsp;%(port)&nbsp;s
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}
&nbsp;
chrome.proxy.settings.set({value:&nbsp;config,&nbsp;scope:&nbsp;"regular"},&nbsp;function()&nbsp;{});
&nbsp;
function&nbsp;callbackFn(details)&nbsp;{
&nbsp;&nbsp;&nbsp;return&nbsp;{
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;authCredentials:&nbsp;{username:&nbsp;"%(username)&nbsp;s",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;password:&nbsp;"%(password)&nbsp;s"
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}
&nbsp;&nbsp;&nbsp;}
}
&nbsp;
chrome.webRequest.onAuthRequired.addListener(
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;callbackFn,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{urls:&nbsp;["&lt;all_urls&gt;"]},
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;['blocking']
)
"""</span>&nbsp;%&nbsp;{<span class="hljs-string">'ip'</span>:&nbsp;ip,&nbsp;<span class="hljs-string">'port'</span>:&nbsp;port,&nbsp;<span class="hljs-string">'username'</span>:&nbsp;username,&nbsp;<span class="hljs-string">'password'</span>:&nbsp;password}
&nbsp;
plugin_file&nbsp;=&nbsp;<span class="hljs-string">'proxy_auth_plugin.zip'</span>
<span class="hljs-keyword">with</span>&nbsp;zipfile.ZipFile(plugin_file,&nbsp;<span class="hljs-string">'w'</span>)&nbsp;<span class="hljs-keyword">as</span>&nbsp;zp:
&nbsp;&nbsp;&nbsp;zp.writestr(<span class="hljs-string">"manifest.json"</span>,&nbsp;manifest_json)
&nbsp;&nbsp;&nbsp;zp.writestr(<span class="hljs-string">"background.js"</span>,&nbsp;background_js)
options&nbsp;=&nbsp;Options()
options.add_argument(<span class="hljs-string">"--start-maximized"</span>)
options.add_extension(plugin_file)
browser&nbsp;=&nbsp;webdriver.Chrome(options=options)
browser.get(<span class="hljs-string">'https://httpbin.org/get'</span>)
print(browser.page_source)
browser.close()
</code></pre>
<p>这里需要在本地创建一个 manifest.json 配置文件和 background.js 脚本来设置认证代理。运行代码之后本地会生成一个 proxy_auth_plugin.zip 文件来保存当前配置。</p>
<p>运行结果和上例一致，origin 同样为代理 IP。</p>
<p>SOCKS 代理的设置也比较简单，把对应的协议修改为 socks5 即可，如无密码认证的代理设置方法为：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver

proxy&nbsp;=&nbsp;<span class="hljs-string">'127.0.0.1:7891'</span>
options&nbsp;=&nbsp;webdriver.ChromeOptions()
options.add_argument(<span class="hljs-string">'--proxy-server=socks5://'</span>&nbsp;+&nbsp;proxy)
browser&nbsp;=&nbsp;webdriver.Chrome(options=options)
browser.get(<span class="hljs-string">'https://httpbin.org/get'</span>)
print(browser.page_source)
browser.close()
</code></pre>
<p>运行结果是一样的。</p>
<h3>aiohttp 设置代理</h3>
<p>对于 aiohttp 来说，我们可以通过 proxy 参数直接设置即可，HTTP 代理设置如下：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span>&nbsp;asyncio
<span class="hljs-keyword">import</span>&nbsp;aiohttp

proxy&nbsp;=&nbsp;<span class="hljs-string">'http://127.0.0.1:7890'</span>

<span class="hljs-keyword">async</span>&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">main</span><span class="hljs-params">()</span>:</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">async</span>&nbsp;<span class="hljs-keyword">with</span>&nbsp;aiohttp.ClientSession()&nbsp;<span class="hljs-keyword">as</span>&nbsp;session:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">async</span>&nbsp;<span class="hljs-keyword">with</span>&nbsp;session.get(<span class="hljs-string">'https://httpbin.org/get'</span>,&nbsp;proxy=proxy)&nbsp;<span class="hljs-keyword">as</span>&nbsp;response:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;print(<span class="hljs-keyword">await</span>&nbsp;response.text())

<span class="hljs-keyword">if</span>&nbsp;__name__&nbsp;==&nbsp;<span class="hljs-string">'__main__'</span>:
&nbsp;&nbsp;&nbsp;asyncio.get_event_loop().run_until_complete(main())
</code></pre>
<p>如果代理有用户名密码，像 requests 一样，把 proxy 修改为如下内容：</p>
<pre><code data-language="python" class="lang-python">proxy&nbsp;=&nbsp;<span class="hljs-string">'http://username:password@127.0.0.1:7890'</span>
</code></pre>
<p>这里只需要将 username 和 password 替换即可。</p>
<p>对于 SOCKS 代理，我们需要安装一个支持库，叫作 aiohttp-socks，安装命令如下：</p>
<pre><code data-language="python" class="lang-python">pip3&nbsp;install&nbsp;aiohttp-socks
</code></pre>
<p>可以借助于这个库的 ProxyConnector 来设置 SOCKS 代理，代码如下：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span>&nbsp;asyncio
<span class="hljs-keyword">import</span>&nbsp;aiohttp
<span class="hljs-keyword">from</span>&nbsp;aiohttp_socks&nbsp;<span class="hljs-keyword">import</span>&nbsp;ProxyConnector
&nbsp;
connector&nbsp;=&nbsp;ProxyConnector.from_url(<span class="hljs-string">'socks5://127.0.0.1:7891'</span>)
&nbsp;
<span class="hljs-keyword">async</span>&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">main</span><span class="hljs-params">()</span>:</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">async</span>&nbsp;<span class="hljs-keyword">with</span>&nbsp;aiohttp.ClientSession(connector=connector)&nbsp;<span class="hljs-keyword">as</span>&nbsp;session:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">async</span>&nbsp;<span class="hljs-keyword">with</span>&nbsp;session.get(<span class="hljs-string">'https://httpbin.org/get'</span>)&nbsp;<span class="hljs-keyword">as</span>&nbsp;response:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;print(<span class="hljs-keyword">await</span>&nbsp;response.text())

<span class="hljs-keyword">if</span>&nbsp;__name__&nbsp;==&nbsp;<span class="hljs-string">'__main__'</span>:
&nbsp;&nbsp;&nbsp;asyncio.get_event_loop().run_until_complete(main())
</code></pre>
<p>运行结果是一样的。</p>
<p>另外这个库还支持设置 SOCKS4、HTTP 代理以及对应的代理认证，可以参考其官方介绍。</p>
<h3>Pyppeteer 设置代理</h3>
<p>对于 Pyppeteer 来说，由于其默认使用的是类似 Chrome 的 Chromium 浏览器，因此设置方法和 Selenium 的 Chrome 是一样的，如 HTTP 无认证代理设置方法都是通过 args 来设置，实现如下：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span>&nbsp;asyncio
<span class="hljs-keyword">from</span>&nbsp;pyppeteer&nbsp;<span class="hljs-keyword">import</span>&nbsp;launch

proxy&nbsp;=&nbsp;<span class="hljs-string">'127.0.0.1:7890'</span>

<span class="hljs-keyword">async</span>&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">main</span><span class="hljs-params">()</span>:</span>
&nbsp;&nbsp;&nbsp;browser&nbsp;=&nbsp;<span class="hljs-keyword">await</span>&nbsp;launch({<span class="hljs-string">'args'</span>:&nbsp;[<span class="hljs-string">'--proxy-server=http://'</span>&nbsp;+&nbsp;proxy],&nbsp;<span class="hljs-string">'headless'</span>:&nbsp;<span class="hljs-literal">False</span>})
&nbsp;&nbsp;&nbsp;page&nbsp;=&nbsp;<span class="hljs-keyword">await</span>&nbsp;browser.newPage()
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">await</span>&nbsp;page.goto(<span class="hljs-string">'https://httpbin.org/get'</span>)
&nbsp;&nbsp;&nbsp;print(<span class="hljs-keyword">await</span>&nbsp;page.content())
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">await</span>&nbsp;browser.close()

<span class="hljs-keyword">if</span>&nbsp;__name__&nbsp;==&nbsp;<span class="hljs-string">'__main__'</span>:
&nbsp;&nbsp;&nbsp;asyncio.get_event_loop().run_until_complete(main())
</code></pre>
<p>运行结果：</p>
<pre><code data-language="python" class="lang-python">{
&nbsp;<span class="hljs-string">"args"</span>:&nbsp;{},
&nbsp;<span class="hljs-string">"headers"</span>:&nbsp;{
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"Accept"</span>:&nbsp;<span class="hljs-string">"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8"</span>,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"Accept-Encoding"</span>:&nbsp;<span class="hljs-string">"gzip,&nbsp;deflate,&nbsp;br"</span>,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"Accept-Language"</span>:&nbsp;<span class="hljs-string">"zh-CN,zh;q=0.9"</span>,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"Host"</span>:&nbsp;<span class="hljs-string">"httpbin.org"</span>,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"Upgrade-Insecure-Requests"</span>:&nbsp;<span class="hljs-string">"1"</span>,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"User-Agent"</span>:&nbsp;<span class="hljs-string">"Mozilla/5.0&nbsp;(Macintosh;&nbsp;Intel&nbsp;Mac&nbsp;OS&nbsp;X&nbsp;10_15_4)&nbsp;AppleWebKit/537.36&nbsp;(KHTML,&nbsp;like&nbsp;Gecko)&nbsp;Chrome/69.0.3494.0&nbsp;Safari/537.36"</span>,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"X-Amzn-Trace-Id"</span>:&nbsp;<span class="hljs-string">"Root=1-5e8f442c-12b1ed7865b049007267a66c"</span>
&nbsp;},
&nbsp;<span class="hljs-string">"origin"</span>:&nbsp;<span class="hljs-string">"210.173.1.204"</span>,
&nbsp;<span class="hljs-string">"url"</span>:&nbsp;<span class="hljs-string">"https://httpbin.org/get"</span>
}
</code></pre>
<p>同样可以看到设置成功。</p>
<p>对于 SOCKS 代理，也是一样的，只需要将协议修改为 socks5 即可，代码实现如下：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span>&nbsp;asyncio
<span class="hljs-keyword">from</span>&nbsp;pyppeteer&nbsp;<span class="hljs-keyword">import</span>&nbsp;launch

proxy&nbsp;=&nbsp;<span class="hljs-string">'127.0.0.1:7891'</span>

<span class="hljs-keyword">async</span>&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">main</span><span class="hljs-params">()</span>:</span>
&nbsp;&nbsp;&nbsp;browser&nbsp;=&nbsp;<span class="hljs-keyword">await</span>&nbsp;launch({<span class="hljs-string">'args'</span>:&nbsp;[<span class="hljs-string">'--proxy-server=socks5://'</span>&nbsp;+&nbsp;proxy],&nbsp;<span class="hljs-string">'headless'</span>:&nbsp;<span class="hljs-literal">False</span>})
&nbsp;&nbsp;&nbsp;page&nbsp;=&nbsp;<span class="hljs-keyword">await</span>&nbsp;browser.newPage()
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">await</span>&nbsp;page.goto(<span class="hljs-string">'https://httpbin.org/get'</span>)
&nbsp;&nbsp;&nbsp;print(<span class="hljs-keyword">await</span>&nbsp;page.content())
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">await</span>&nbsp;browser.close()

<span class="hljs-keyword">if</span>&nbsp;__name__&nbsp;==&nbsp;<span class="hljs-string">'__main__'</span>:
&nbsp;&nbsp;&nbsp;asyncio.get_event_loop().run_until_complete(main())
</code></pre>
<p>运行结果也是一样的。</p>
<h3>总结</h3>
<p>以上总结了各个库的代理使用方式，以后如果遇到封 IP 的问题，我们就可以轻松通过加代理的方式来解决啦。</p>
<p>本节代码：<a href="https://github.com/Python3WebSpider/ProxyTest">https://github.com/Python3WebSpider/ProxyTest</a>。</p>

# 代理池的搭建和使用
<p>我们在上一课时了解了利用代理可以解决目标网站封 IP 的问题，但是如何实时高效地获取到大量可用的代理又是一个问题。</p>
<p>首先在互联网上有大量公开的免费代理，当然我们也可以购买付费的代理 IP，但是代理不论是免费的还是付费的，都不能保证是可用的，因为可能此 IP 已被其他人使用来爬取同样的目标站点而被封禁，或者代理服务器突然发生故障或网络繁忙。一旦我们选用了一个不可用的代理，这势必会影响爬虫的工作效率。</p>
<p>所以，我们需要提前做筛选，将不可用的代理剔除掉，保留可用代理。那么这个怎么来实现呢？这里就需要借助于一个叫作代理池的东西了。</p>
<p>接下来本课时我们就介绍一下如何搭建一个高效易用的代理池。</p>
<h3>准备工作</h3>
<p>在这里代理池的存储我们需要借助于 Redis，因此这个需要额外安装。总体来说，本课时需要的环境如下：</p>
<ul>
<li>安装并成功运行和连接一个 Redis 数据库，安装方法见：<a href="https://cuiqingcai.com/5219.html">https://cuiqingcai.com/5219.html</a>。</li>
<li>安装好 Python3（至少为 Python 3.6 版本），并能成功运行 Python 程序。</li>
</ul>
<p>安装好一些必要的库，包括 aiohttp、requests、redis-py、pyquery、Flask 等。</p>
<p>建议使用 Python 虚拟环境安装，参考安装命令如下：</p>
<ul>
<li>pip3 install - r <a href="https://raw.githubusercontent.com/Python3WebSpider/ProxyPool/master/requirements.txt">https://raw.githubusercontent.com/Python3WebSpider/ProxyPool/master/requirements.txt</a></li>
</ul>
<p>做好了如上准备工作，我们便可以开始实现或运行本课时所讲的代理池了。</p>
<h3>代理池的目标</h3>
<p>我们需要做到下面的几个目标，来实现易用高效的代理池。</p>
<ul>
<li>基本模块分为 4 块：存储模块、获取模块、检测模块、接口模块。</li>
<li>存储模块：负责存储抓取下来的代理。首先要保证代理不重复，要标识代理的可用情况，还要动态实时处理每个代理，所以一种比较高效和方便的存储方式就是使用 Redis 的 Sorted Set，即有序集合。</li>
<li>获取模块：需要定时在各大代理网站抓取代理。代理可以是免费公开代理也可以是付费代理，代理的形式都是 IP 加端口，此模块尽量从不同来源获取，尽量抓取高匿代理，抓取成功之后将可用代理保存到数据库中。</li>
<li>检测模块：需要定时检测数据库中的代理。这里需要设置一个检测链接，最好是爬取哪个网站就检测哪个网站，这样更加有针对性，如果要做一个通用型的代理，那可以设置百度等链接来检测。另外，我们需要标识每一个代理的状态，如设置分数标识，100 分代表可用，分数越少代表越不可用。检测一次，如果代理可用，我们可以将分数标识立即设置为 100 满分，也可以在原基础上加 1 分；如果代理不可用，可以将分数标识减 1 分，当分数减到一定阈值后，代理就直接从数据库移除。通过这样的标识分数，我们就可以辨别代理的可用情况，选用的时候会更有针对性。</li>
<li>接口模块：需要用 API 来提供对外服务的接口。其实我们可以直接连接数据库来获取对应的数据，但是这样就需要知道数据库的连接信息，并且要配置连接，而比较安全和方便的方式就是提供一个 Web API 接口，我们通过访问接口即可拿到可用代理。另外，由于可用代理可能有多个，那么我们可以设置一个随机返回某个可用代理的接口，这样就能保证每个可用代理都可以取到，实现负载均衡。</li>
</ul>
<p>以上内容是设计代理的一些基本思路。接下来我们设计整体的架构，然后用代码实现代理池。</p>
<h3>代理池的架构</h3>
<p>根据上文的描述，代理池的架构如图所示。</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/0E/BE/Ciqah16ULCeAIIqyAABXJVAIMbk947.png" alt=""></p>
<p>代理池分为 4 个模块：存储模块、获取模块、检测模块、接口模块。</p>
<ul>
<li>存储模块使用 Redis 的有序集合，用来做代理的去重和状态标识，同时它也是中心模块和基础模块，将其他模块串联起来。</li>
<li>获取模块定时从代理网站获取代理，将获取的代理传递给存储模块，并保存到数据库。</li>
<li>检测模块定时通过存储模块获取所有代理，并对代理进行检测，根据不同的检测结果对代理设置不同的标识。</li>
<li>接口模块通过 Web API 提供服务接口，接口通过连接数据库并通过 Web 形式返回可用的代理。</li>
</ul>
<h3>代理池的实现</h3>
<p>接下来我们分别用代码来实现一下这四个模块。</p>
<blockquote>
<p>注：完整的代理池代码量较大，因此本课时的代码不必一步步跟着编写，最后去了解源码即可。</p>
</blockquote>
<h4>存储模块</h4>
<p>这里我们使用 Redis 的有序集合，集合的每一个元素都是不重复的，对于代理池来说，集合的元素就变成了一个个代理，也就是 IP 加端口的形式，如 60.207.237.111:8888，这样的一个代理就是集合的一个元素。另外，有序集合的每一个元素都有一个分数字段，分数是可以重复的，可以是浮点数类型，也可以是整数类型。该集合会根据每一个元素的分数对集合进行排序，数值小的排在前面，数值大的排在后面，这样就可以实现集合元素的排序了。</p>
<p>对于代理池来说，这个分数可以作为判断一个代理是否可用的标志，100 为最高分，代表最可用，0 为最低分，代表最不可用。如果要获取可用代理，可以从代理池中随机获取分数最高的代理，注意是随机，这样可以保证每个可用代理都会被调用到。</p>
<p>分数是我们判断代理稳定性的重要标准，设置分数规则如下所示。</p>
<ul>
<li>分数 100 为可用，检测器会定时循环检测每个代理可用情况，一旦检测到有可用的代理就立即置为 100，检测到不可用就将分数减 1，分数减至 0 后代理移除。</li>
<li>新获取的代理的分数为 10，如果测试可行，分数立即置为 100，不可行则分数减 1，分数减至 0 后代理移除。</li>
</ul>
<p>这只是一种解决方案，当然可能还有更合理的方案。之所以设置此方案有如下几个原因。</p>
<ul>
<li>在检测到代理可用时，分数立即置为 100，这样可以保证所有可用代理有更大的机会被获取到。你可能会问，为什么不将分数加 1 而是直接设为最高 100 呢？设想一下，有的代理是从各大免费公开代理网站获取的，常常一个代理并没有那么稳定，平均 5 次请求可能有两次成功，3 次失败，如果按照这种方式来设置分数，那么这个代理几乎不可能达到一个高的分数，也就是说即便它有时是可用的，但是筛选的分数最高，那这样的代理几乎不可能被取到。如果想追求代理稳定性，可以用上述方法，这种方法可确保分数最高的代理一定是最稳定可用的。所以，这里我们采取 “可用即设置 100” 的方法，确保只要可用的代理都可以被获取到。</li>
<li>在检测到代理不可用时，分数减 1，分数减至 0 后，代理移除。这样一个有效代理如果要被移除需要连续不断失败 100 次，也就是说当一个可用代理如果尝试了 100 次都失败了，就一直减分直到移除，一旦成功就重新置回 100。尝试机会越多，则这个代理拯救回来的机会越多，这样就不容易将曾经的一个可用代理丢弃，因为代理不可用的原因很可能是网络繁忙或者其他人用此代理请求太过频繁，所以在这里将分数为 100。</li>
<li>新获取的代理的分数设置为 10，代理如果不可用，分数就减 1，分数减到 0，代理就移除，如果代理可用，分数就置为 100。由于很多代理是从免费网站获取的，所以新获取的代理无效的比例非常高，可能可用的代理不足 10%。所以在这里我们将分数设置为 10，检测的机会没有可用代理的 100 次那么多，这也可以适当减少开销。</li>
</ul>
<p>上述代理分数的设置思路不一定是最优思路，但据个人实测，它的实用性还是比较强的。</p>
<p>在这里首先给出存储模块的实现代码，见：<a href="https://github.com/Python3WebSpider/ProxyPool/tree/master/proxypool/storages">https://github.com/Python3WebSpider/ProxyPool/tree/master/proxypool/storages</a>，建议直接对照源码阅读。</p>
<p>在代码中，我们定义了一个类来操作数据库的有序集合，定义一些方法来实现分数的设置、代理的获取等。其核心实现代码实现如下所示：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span>&nbsp;redis
<span class="hljs-keyword">from</span>&nbsp;proxypool.exceptions&nbsp;<span class="hljs-keyword">import</span>&nbsp;PoolEmptyException
<span class="hljs-keyword">from</span>&nbsp;proxypool.schemas.proxy&nbsp;<span class="hljs-keyword">import</span>&nbsp;Proxy
<span class="hljs-keyword">from</span>&nbsp;proxypool.setting&nbsp;<span class="hljs-keyword">import</span>&nbsp;REDIS_HOST,&nbsp;REDIS_PORT,&nbsp;REDIS_PASSWORD,&nbsp;REDIS_KEY,&nbsp;PROXY_SCORE_MAX,&nbsp;PROXY_SCORE_MIN,&nbsp;\
&nbsp;&nbsp;&nbsp;PROXY_SCORE_INIT
<span class="hljs-keyword">from</span>&nbsp;random&nbsp;<span class="hljs-keyword">import</span>&nbsp;choice
<span class="hljs-keyword">from</span>&nbsp;typing&nbsp;<span class="hljs-keyword">import</span>&nbsp;List
<span class="hljs-keyword">from</span>&nbsp;loguru&nbsp;<span class="hljs-keyword">import</span>&nbsp;logger
<span class="hljs-keyword">from</span>&nbsp;proxypool.utils.proxy&nbsp;<span class="hljs-keyword">import</span>&nbsp;is_valid_proxy,&nbsp;convert_proxy_or_proxies
REDIS_CLIENT_VERSION&nbsp;=&nbsp;redis.__version__
IS_REDIS_VERSION_2&nbsp;=&nbsp;REDIS_CLIENT_VERSION.startswith(<span class="hljs-string">'2.'</span>)
<span class="hljs-class"><span class="hljs-keyword">class</span>&nbsp;<span class="hljs-title">RedisClient</span><span class="hljs-params">(object)</span>:</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;redis&nbsp;connection&nbsp;client&nbsp;of&nbsp;proxypool
&nbsp;&nbsp;&nbsp;"""</span>

&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">__init__</span><span class="hljs-params">(self,&nbsp;host=REDIS_HOST,&nbsp;port=REDIS_PORT,&nbsp;password=REDIS_PASSWORD,&nbsp;**kwargs)</span>:</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;init&nbsp;redis&nbsp;client
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:param&nbsp;host:&nbsp;redis&nbsp;host
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:param&nbsp;port:&nbsp;redis&nbsp;port
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:param&nbsp;password:&nbsp;redis&nbsp;password
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.db&nbsp;=&nbsp;redis.StrictRedis(host=host,&nbsp;port=port,&nbsp;password=password,&nbsp;decode_responses=<span class="hljs-literal">True</span>,&nbsp;**kwargs)

&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">add</span><span class="hljs-params">(self,&nbsp;proxy:&nbsp;Proxy,&nbsp;score=PROXY_SCORE_INIT)</span>&nbsp;-&gt;&nbsp;int:</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;add&nbsp;proxy&nbsp;and&nbsp;set&nbsp;it&nbsp;to&nbsp;init&nbsp;score
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:param&nbsp;proxy:&nbsp;proxy,&nbsp;ip:port,&nbsp;like&nbsp;8.8.8.8:88
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:param&nbsp;score:&nbsp;int&nbsp;score
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:return:&nbsp;result
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">if</span>&nbsp;<span class="hljs-keyword">not</span>&nbsp;is_valid_proxy(<span class="hljs-string">f'<span class="hljs-subst">{proxy.host}</span>:<span class="hljs-subst">{proxy.port}</span>'</span>):
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.info(<span class="hljs-string">f'invalid&nbsp;proxy&nbsp;<span class="hljs-subst">{proxy}</span>,&nbsp;throw&nbsp;it'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">if</span>&nbsp;<span class="hljs-keyword">not</span>&nbsp;self.exists(proxy):
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">if</span>&nbsp;IS_REDIS_VERSION_2:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;self.db.zadd(REDIS_KEY,&nbsp;score,&nbsp;proxy.string())
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;self.db.zadd(REDIS_KEY,&nbsp;{proxy.string():&nbsp;score})

&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">random</span><span class="hljs-params">(self)</span>&nbsp;-&gt;&nbsp;Proxy:</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;get&nbsp;random&nbsp;proxy
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;firstly&nbsp;try&nbsp;to&nbsp;get&nbsp;proxy&nbsp;with&nbsp;max&nbsp;score
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if&nbsp;not&nbsp;exists,&nbsp;try&nbsp;to&nbsp;get&nbsp;proxy&nbsp;by&nbsp;rank
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if&nbsp;not&nbsp;exists,&nbsp;raise&nbsp;error
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:return:&nbsp;proxy,&nbsp;like&nbsp;8.8.8.8:8
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-comment">#&nbsp;try&nbsp;to&nbsp;get&nbsp;proxy&nbsp;with&nbsp;max&nbsp;score</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;proxies&nbsp;=&nbsp;self.db.zrangebyscore(REDIS_KEY,&nbsp;PROXY_SCORE_MAX,&nbsp;PROXY_SCORE_MAX)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">if</span>&nbsp;len(proxies):
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;convert_proxy_or_proxies(choice(proxies))
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-comment">#&nbsp;else&nbsp;get&nbsp;proxy&nbsp;by&nbsp;rank</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;proxies&nbsp;=&nbsp;self.db.zrevrange(REDIS_KEY,&nbsp;PROXY_SCORE_MIN,&nbsp;PROXY_SCORE_MAX)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">if</span>&nbsp;len(proxies):
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;convert_proxy_or_proxies(choice(proxies))
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-comment">#&nbsp;else&nbsp;raise&nbsp;error</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">raise</span>&nbsp;PoolEmptyException

&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">decrease</span><span class="hljs-params">(self,&nbsp;proxy:&nbsp;Proxy)</span>&nbsp;-&gt;&nbsp;int:</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;decrease&nbsp;score&nbsp;of&nbsp;proxy,&nbsp;if&nbsp;small&nbsp;than&nbsp;PROXY_SCORE_MIN,&nbsp;delete&nbsp;it
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:param&nbsp;proxy:&nbsp;proxy
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:return:&nbsp;new&nbsp;score
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;score&nbsp;=&nbsp;self.db.zscore(REDIS_KEY,&nbsp;proxy.string())
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-comment">#&nbsp;current&nbsp;score&nbsp;is&nbsp;larger&nbsp;than&nbsp;PROXY_SCORE_MIN</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">if</span>&nbsp;score&nbsp;<span class="hljs-keyword">and</span>&nbsp;score&nbsp;&gt;&nbsp;PROXY_SCORE_MIN:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.info(<span class="hljs-string">f'<span class="hljs-subst">{proxy.string()}</span>&nbsp;current&nbsp;score&nbsp;<span class="hljs-subst">{score}</span>,&nbsp;decrease&nbsp;1'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">if</span>&nbsp;IS_REDIS_VERSION_2:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;self.db.zincrby(REDIS_KEY,&nbsp;proxy.string(),&nbsp;<span class="hljs-number">-1</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;self.db.zincrby(REDIS_KEY,&nbsp;<span class="hljs-number">-1</span>,&nbsp;proxy.string())
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-comment">#&nbsp;otherwise&nbsp;delete&nbsp;proxy</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">else</span>:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.info(<span class="hljs-string">f'<span class="hljs-subst">{proxy.string()}</span>&nbsp;current&nbsp;score&nbsp;<span class="hljs-subst">{score}</span>,&nbsp;remove'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;self.db.zrem(REDIS_KEY,&nbsp;proxy.string())

&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">exists</span><span class="hljs-params">(self,&nbsp;proxy:&nbsp;Proxy)</span>&nbsp;-&gt;&nbsp;bool:</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if&nbsp;proxy&nbsp;exists
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:param&nbsp;proxy:&nbsp;proxy
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:return:&nbsp;if&nbsp;exists,&nbsp;bool
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;<span class="hljs-keyword">not</span>&nbsp;self.db.zscore(REDIS_KEY,&nbsp;proxy.string())&nbsp;<span class="hljs-keyword">is</span>&nbsp;<span class="hljs-literal">None</span>

&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">max</span><span class="hljs-params">(self,&nbsp;proxy:&nbsp;Proxy)</span>&nbsp;-&gt;&nbsp;int:</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;set&nbsp;proxy&nbsp;to&nbsp;max&nbsp;score
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:param&nbsp;proxy:&nbsp;proxy
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:return:&nbsp;new&nbsp;score
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.info(<span class="hljs-string">f'<span class="hljs-subst">{proxy.string()}</span>&nbsp;is&nbsp;valid,&nbsp;set&nbsp;to&nbsp;<span class="hljs-subst">{PROXY_SCORE_MAX}</span>'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">if</span>&nbsp;IS_REDIS_VERSION_2:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;self.db.zadd(REDIS_KEY,&nbsp;PROXY_SCORE_MAX,&nbsp;proxy.string())
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;self.db.zadd(REDIS_KEY,&nbsp;{proxy.string():&nbsp;PROXY_SCORE_MAX})

&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">count</span><span class="hljs-params">(self)</span>&nbsp;-&gt;&nbsp;int:</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;get&nbsp;count&nbsp;of&nbsp;proxies
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:return:&nbsp;count,&nbsp;int
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;self.db.zcard(REDIS_KEY)

&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">all</span><span class="hljs-params">(self)</span>&nbsp;-&gt;&nbsp;List[Proxy]:</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;get&nbsp;all&nbsp;proxies
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:return:&nbsp;list&nbsp;of&nbsp;proxies
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;convert_proxy_or_proxies(self.db.zrangebyscore(REDIS_KEY,&nbsp;PROXY_SCORE_MIN,&nbsp;PROXY_SCORE_MAX))

&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">batch</span><span class="hljs-params">(self,&nbsp;start,&nbsp;end)</span>&nbsp;-&gt;&nbsp;List[Proxy]:</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;get&nbsp;batch&nbsp;of&nbsp;proxies
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:param&nbsp;start:&nbsp;start&nbsp;index
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:param&nbsp;end:&nbsp;end&nbsp;index
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:return:&nbsp;list&nbsp;of&nbsp;proxies
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;convert_proxy_or_proxies(self.db.zrevrange(REDIS_KEY,&nbsp;start,&nbsp;end&nbsp;-&nbsp;<span class="hljs-number">1</span>))
<span class="hljs-keyword">if</span>&nbsp;__name__&nbsp;==&nbsp;<span class="hljs-string">'__main__'</span>:
&nbsp;&nbsp;&nbsp;conn&nbsp;=&nbsp;RedisClient()
&nbsp;&nbsp;&nbsp;result&nbsp;=&nbsp;conn.random()
&nbsp;&nbsp;&nbsp;print(result)
</code></pre>
<p>首先我们定义了一些常量，如 PROXY_SCORE_MAX、PROXY_SCORE_MIN、PROXY_SCORE_INIT 分别代表最大分数、最小分数、初始分数。REDIS_HOST、REDIS_PORT、REDIS_PASSWORD 分别代表了 Redis 的连接信息，即地址、端口、密码。REDIS_KEY 是有序集合的键名，我们可以通过它来获取代理存储所使用的有序集合。</p>
<p>RedisClient 这个类可以用来操作 Redis 的有序集合，其中定义了一些方法来对集合中的元素进行处理，它的主要功能如下所示。</p>
<ul>
<li>__init__ 方法是初始化的方法，其参数是 Redis 的连接信息，默认的连接信息已经定义为常量，在 __init__ 方法中初始化了一个 StrictRedis 的类，建立 Redis 连接。</li>
<li>add 方法向数据库添加代理并设置分数，默认的分数是 PROXY_SCORE_INIT 也就是 10，返回结果是添加的结果。</li>
<li>random 方法是随机获取代理的方法，首先获取 100 分的代理，然后随机选择一个返回。如果不存在 100 分的代理，则此方法按照排名来获取，选取前 100 名，然后随机选择一个返回，否则抛出异常。</li>
<li>decrease 方法是在代理检测无效的时候设置分数减 1 的方法，代理传入后，此方法将代理的分数减 1，如果分数达到最低值，那么代理就删除。</li>
<li>exists 方法可判断代理是否存在集合中。</li>
<li>max 方法将代理的分数设置为 PROXY_SCORE_MAX，即 100，也就是当代理有效时的设置。</li>
<li>count 方法返回当前集合的元素个数。</li>
<li>all 方法返回所有的代理列表，供检测使用。</li>
</ul>
<p>定义好了这些方法，我们可以在后续的模块中调用此类来连接和操作数据库。如想要获取随机可用的代理，只需要调用 random 方法即可，得到的就是随机的可用代理。</p>
<h4>获取模块</h4>
<p>获取模块主要是为了从各大网站抓取代理并调用存储模块进行保存，代码实现见：<a href="https://github.com/Python3WebSpider/ProxyPool/tree/master/proxypool/crawlers">https://github.com/Python3WebSpider/ProxyPool/tree/master/proxypool/crawlers</a>。</p>
<p>获取模块的逻辑相对简单，比如我们可以定义一些抓取代理的方法，示例如下：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span>&nbsp;proxypool.crawlers.base&nbsp;<span class="hljs-keyword">import</span>&nbsp;BaseCrawler
<span class="hljs-keyword">from</span>&nbsp;proxypool.schemas.proxy&nbsp;<span class="hljs-keyword">import</span>&nbsp;Proxy
<span class="hljs-keyword">import</span>&nbsp;re
MAX_PAGE&nbsp;=&nbsp;<span class="hljs-number">5</span>
BASE_URL&nbsp;=&nbsp;<span class="hljs-string">'http://www.ip3366.net/free/?stype=1&amp;page={page}'</span>
<span class="hljs-class"><span class="hljs-keyword">class</span>&nbsp;<span class="hljs-title">IP3366Crawler</span><span class="hljs-params">(BaseCrawler)</span>:</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;ip3366&nbsp;crawler,&nbsp;http://www.ip3366.net/
&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;urls&nbsp;=&nbsp;[BASE_URL.format(page=i)&nbsp;<span class="hljs-keyword">for</span>&nbsp;i&nbsp;<span class="hljs-keyword">in</span>&nbsp;range(<span class="hljs-number">1</span>,&nbsp;<span class="hljs-number">8</span>)]

&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">parse</span><span class="hljs-params">(self,&nbsp;html)</span>:</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;parse&nbsp;html&nbsp;file&nbsp;to&nbsp;get&nbsp;proxies
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:return:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ip_address&nbsp;=&nbsp;re.compile(<span class="hljs-string">'&lt;tr&gt;\s*&lt;td&gt;(.*?)&lt;/td&gt;\s*&lt;td&gt;(.*?)&lt;/td&gt;'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-comment">#&nbsp;\s&nbsp;*&nbsp;匹配空格，起到换行作用</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;re_ip_address&nbsp;=&nbsp;ip_address.findall(html)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">for</span>&nbsp;address,&nbsp;port&nbsp;<span class="hljs-keyword">in</span>&nbsp;re_ip_address:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;proxy&nbsp;=&nbsp;Proxy(host=address.strip(),&nbsp;port=int(port.strip()))
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">yield</span>&nbsp;proxy
</code></pre>
<p>我们在这里定义了一个代理 Crawler 类，用来抓取某一网站的代理，这里是抓取的 IP3366 的公开代理，通过 parse 方法来解析页面的源码并构造一个个 Proxy 对象返回即可。</p>
<p>另外在其父类 BaseCrawler 里面定义了通用的页面抓取方法，它可以读取子类里面定义的 urls 全局变量并进行爬取，然后调用子类的 parse 方法来解析页面，代码实现如下：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span>&nbsp;retrying&nbsp;<span class="hljs-keyword">import</span>&nbsp;retry
<span class="hljs-keyword">import</span>&nbsp;requests
<span class="hljs-keyword">from</span>&nbsp;loguru&nbsp;<span class="hljs-keyword">import</span>&nbsp;logger
<span class="hljs-class"><span class="hljs-keyword">class</span>&nbsp;<span class="hljs-title">BaseCrawler</span><span class="hljs-params">(object)</span>:</span>
&nbsp;&nbsp;&nbsp;urls&nbsp;=&nbsp;[]

&nbsp;&nbsp;&nbsp;@retry(stop_max_attempt_number=<span class="hljs-number">3</span>,&nbsp;retry_on_result=<span class="hljs-keyword">lambda</span>&nbsp;x:&nbsp;x&nbsp;<span class="hljs-keyword">is</span>&nbsp;<span class="hljs-literal">None</span>)
&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">fetch</span><span class="hljs-params">(self,&nbsp;url,&nbsp;**kwargs)</span>:</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">try</span>:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;response&nbsp;=&nbsp;requests.get(url,&nbsp;**kwargs)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">if</span>&nbsp;response.status_code&nbsp;==&nbsp;<span class="hljs-number">200</span>:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;response.text
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">except</span>&nbsp;requests.ConnectionError:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>

&nbsp;&nbsp;&nbsp;@logger.catch
&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">crawl</span><span class="hljs-params">(self)</span>:</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;crawl&nbsp;main&nbsp;method
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">for</span>&nbsp;url&nbsp;<span class="hljs-keyword">in</span>&nbsp;self.urls:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.info(<span class="hljs-string">f'fetching&nbsp;<span class="hljs-subst">{url}</span>'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;html&nbsp;=&nbsp;self.fetch(url)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">for</span>&nbsp;proxy&nbsp;<span class="hljs-keyword">in</span>&nbsp;self.parse(html):
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.info(<span class="hljs-string">f'fetched&nbsp;proxy&nbsp;<span class="hljs-subst">{proxy.string()}</span>&nbsp;from&nbsp;<span class="hljs-subst">{url}</span>'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">yield</span>&nbsp;proxy
</code></pre>
<p>所以，我们如果要扩展一个代理的 Crawler，只需要继承 BaseCrawler 并实现 parse 方法即可，扩展性较好。</p>
<p>因此，这一个个的 Crawler 就可以针对各个不同的代理网站进行代理的抓取。最后有一个统一的方法将 Crawler 汇总起来，遍历调用即可。</p>
<p>如何汇总呢？在这里我们可以检测代码只要定义有 BaseCrawler 的子类就算一个有效的代理 Crawler，可以直接通过遍历 Python 文件包的方式来获取，代码实现如下：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span>&nbsp;pkgutil
<span class="hljs-keyword">from</span>&nbsp;.base&nbsp;<span class="hljs-keyword">import</span>&nbsp;BaseCrawler
<span class="hljs-keyword">import</span>&nbsp;inspect
<span class="hljs-comment">#&nbsp;load&nbsp;classes&nbsp;subclass&nbsp;of&nbsp;BaseCrawler</span>
classes&nbsp;=&nbsp;[]
<span class="hljs-keyword">for</span>&nbsp;loader,&nbsp;name,&nbsp;is_pkg&nbsp;<span class="hljs-keyword">in</span>&nbsp;pkgutil.walk_packages(__path__):
&nbsp;&nbsp;&nbsp;module&nbsp;=&nbsp;loader.find_module(name).load_module(name)
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">for</span>&nbsp;name,&nbsp;value&nbsp;<span class="hljs-keyword">in</span>&nbsp;inspect.getmembers(module):
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;globals()[name]&nbsp;=&nbsp;value
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">if</span>&nbsp;inspect.isclass(value)&nbsp;<span class="hljs-keyword">and</span>&nbsp;issubclass(value,&nbsp;BaseCrawler)&nbsp;<span class="hljs-keyword">and</span>&nbsp;value&nbsp;<span class="hljs-keyword">is</span>&nbsp;<span class="hljs-keyword">not</span>&nbsp;BaseCrawler:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;classes.append(value)
__all__&nbsp;=&nbsp;__ALL__&nbsp;=&nbsp;classes
</code></pre>
<p>在这里我们调用了 walk_packages 方法，遍历了整个 crawlers 模块下的类，并判断了它是 BaseCrawler 的子类，那就将其添加到结果中，并返回。</p>
<p>最后只要将 classes 遍历并依次实例化，调用其 crawl 方法即可完成代理的爬取和提取，代码实现见：<a href="https://github.com/Python3WebSpider/ProxyPool/blob/master/proxypool/processors/getter.py">https://github.com/Python3WebSpider/ProxyPool/blob/master/proxypool/processors/getter.py</a>。</p>
<h4>检测模块</h4>
<p>我们已经成功将各个网站的代理获取下来了，现在就需要一个检测模块来对所有代理进行多轮检测。代理检测可用，分数就设置为 100，代理不可用，分数减 1，这样就可以实时改变每个代理的可用情况。如要获取有效代理只需要获取分数高的代理即可。</p>
<p>由于代理的数量非常多，为了提高代理的检测效率，我们在这里使用异步请求库 aiohttp 来进行检测。</p>
<p>requests 作为一个同步请求库，我们在发出一个请求之后，程序需要等待网页加载完成之后才能继续执行。也就是这个过程会阻塞等待响应，如果服务器响应非常慢，比如一个请求等待十几秒，那么我们使用 requests 完成一个请求就会需要十几秒的时间，程序也不会继续往下执行，而在这十几秒的时间里程序其实完全可以去做其他的事情，比如调度其他的请求或者进行网页解析等。</p>
<p>对于响应速度比较快的网站来说，requests 同步请求和 aiohttp 异步请求的效果差距没那么大。可对于检测代理来说，检测一个代理一般需要十多秒甚至几十秒的时间，这时候使用 aiohttp 异步请求库的优势就大大体现出来了，效率可能会提高几十倍不止。</p>
<p>所以，我们的代理检测使用异步请求库 aiohttp，实现示例如下所示：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span>&nbsp;asyncio
<span class="hljs-keyword">import</span>&nbsp;aiohttp
<span class="hljs-keyword">from</span>&nbsp;loguru&nbsp;<span class="hljs-keyword">import</span>&nbsp;logger
<span class="hljs-keyword">from</span>&nbsp;proxypool.schemas&nbsp;<span class="hljs-keyword">import</span>&nbsp;Proxy
<span class="hljs-keyword">from</span>&nbsp;proxypool.storages.redis&nbsp;<span class="hljs-keyword">import</span>&nbsp;RedisClient
<span class="hljs-keyword">from</span>&nbsp;proxypool.setting&nbsp;<span class="hljs-keyword">import</span>&nbsp;TEST_TIMEOUT,&nbsp;TEST_BATCH,&nbsp;TEST_URL,&nbsp;TEST_VALID_STATUS
<span class="hljs-keyword">from</span>&nbsp;aiohttp&nbsp;<span class="hljs-keyword">import</span>&nbsp;ClientProxyConnectionError,&nbsp;ServerDisconnectedError,&nbsp;ClientOSError,&nbsp;ClientHttpProxyError
<span class="hljs-keyword">from</span>&nbsp;asyncio&nbsp;<span class="hljs-keyword">import</span>&nbsp;TimeoutError
EXCEPTIONS&nbsp;=&nbsp;(
&nbsp;&nbsp;&nbsp;ClientProxyConnectionError,
&nbsp;&nbsp;&nbsp;ConnectionRefusedError,
&nbsp;&nbsp;&nbsp;TimeoutError,
&nbsp;&nbsp;&nbsp;ServerDisconnectedError,
&nbsp;&nbsp;&nbsp;ClientOSError,
&nbsp;&nbsp;&nbsp;ClientHttpProxyError
)
<span class="hljs-class"><span class="hljs-keyword">class</span>&nbsp;<span class="hljs-title">Tester</span><span class="hljs-params">(object)</span>:</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;tester&nbsp;for&nbsp;testing&nbsp;proxies&nbsp;in&nbsp;queue
&nbsp;&nbsp;&nbsp;"""</span>

&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">__init__</span><span class="hljs-params">(self)</span>:</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;init&nbsp;redis
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.redis&nbsp;=&nbsp;RedisClient()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.loop&nbsp;=&nbsp;asyncio.get_event_loop()

&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">async</span>&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">test</span><span class="hljs-params">(self,&nbsp;proxy:&nbsp;Proxy)</span>:</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;test&nbsp;single&nbsp;proxy
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:param&nbsp;proxy:&nbsp;Proxy&nbsp;object
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:return:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">async</span>&nbsp;<span class="hljs-keyword">with</span>&nbsp;aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=<span class="hljs-literal">False</span>))&nbsp;<span class="hljs-keyword">as</span>&nbsp;session:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">try</span>:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.debug(<span class="hljs-string">f'testing&nbsp;<span class="hljs-subst">{proxy.string()}</span>'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">async</span>&nbsp;<span class="hljs-keyword">with</span>&nbsp;session.get(TEST_URL,&nbsp;proxy=<span class="hljs-string">f'http://<span class="hljs-subst">{proxy.string()}</span>'</span>,&nbsp;timeout=TEST_TIMEOUT,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;allow_redirects=<span class="hljs-literal">False</span>)&nbsp;<span class="hljs-keyword">as</span>&nbsp;response:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">if</span>&nbsp;response.status&nbsp;<span class="hljs-keyword">in</span>&nbsp;TEST_VALID_STATUS:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.redis.max(proxy)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.debug(<span class="hljs-string">f'proxy&nbsp;<span class="hljs-subst">{proxy.string()}</span>&nbsp;is&nbsp;valid,&nbsp;set&nbsp;max&nbsp;score'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">else</span>:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.redis.decrease(proxy)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.debug(<span class="hljs-string">f'proxy&nbsp;<span class="hljs-subst">{proxy.string()}</span>&nbsp;is&nbsp;invalid,&nbsp;decrease&nbsp;score'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">except</span>&nbsp;EXCEPTIONS:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.redis.decrease(proxy)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.debug(<span class="hljs-string">f'proxy&nbsp;<span class="hljs-subst">{proxy.string()}</span>&nbsp;is&nbsp;invalid,&nbsp;decrease&nbsp;score'</span>)

&nbsp;&nbsp;&nbsp;@logger.catch
&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">run</span><span class="hljs-params">(self)</span>:</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;test&nbsp;main&nbsp;method
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:return:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-comment">#&nbsp;event&nbsp;loop&nbsp;of&nbsp;aiohttp</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.info(<span class="hljs-string">'stating&nbsp;tester...'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;count&nbsp;=&nbsp;self.redis.count()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.debug(<span class="hljs-string">f'<span class="hljs-subst">{count}</span>&nbsp;proxies&nbsp;to&nbsp;test'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">for</span>&nbsp;i&nbsp;<span class="hljs-keyword">in</span>&nbsp;range(<span class="hljs-number">0</span>,&nbsp;count,&nbsp;TEST_BATCH):
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-comment">#&nbsp;start&nbsp;end&nbsp;end&nbsp;offset</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;start,&nbsp;end&nbsp;=&nbsp;i,&nbsp;min(i&nbsp;+&nbsp;TEST_BATCH,&nbsp;count)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.debug(<span class="hljs-string">f'testing&nbsp;proxies&nbsp;from&nbsp;<span class="hljs-subst">{start}</span>&nbsp;to&nbsp;<span class="hljs-subst">{end}</span>&nbsp;indices'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;proxies&nbsp;=&nbsp;self.redis.batch(start,&nbsp;end)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tasks&nbsp;=&nbsp;[self.test(proxy)&nbsp;<span class="hljs-keyword">for</span>&nbsp;proxy&nbsp;<span class="hljs-keyword">in</span>&nbsp;proxies]
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-comment">#&nbsp;run&nbsp;tasks&nbsp;using&nbsp;event&nbsp;loop</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.loop.run_until_complete(asyncio.wait(tasks))
<span class="hljs-keyword">if</span>&nbsp;__name__&nbsp;==&nbsp;<span class="hljs-string">'__main__'</span>:
&nbsp;&nbsp;&nbsp;tester&nbsp;=&nbsp;Tester()
&nbsp;&nbsp;&nbsp;tester.run()
</code></pre>
<p>这里定义了一个类 Tester，__init__ 方法中建立了一个 RedisClient 对象，供该对象中其他方法使用。接下来定义了一个 test 方法，这个方法用来检测单个代理的可用情况，其参数就是被检测的代理。注意，test 方法前面加了 async 关键词，代表这个方法是异步的。方法内部首先创建了 aiohttp 的 ClientSession 对象，可以直接调用该对象的 get 方法来访问页面。</p>
<p>测试的链接在这里定义为常量 TEST_URL。如果针对某个网站有抓取需求，建议将 TEST_URL 设置为目标网站的地址，因为在抓取的过程中，代理本身可能是可用的，但是该代理的 IP 已经被目标网站封掉了。例如，某些代理可以正常访问百度等页面，但是对知乎来说可能就被封了，所以我们可以将 TEST_URL 设置为知乎的某个页面的链接，当请求失败、代理被封时，分数自然会减下来，失效的代理就不会被取到了。</p>
<p>如果想做一个通用的代理池，则不需要专门设置 TEST_URL，可以将其设置为一个不会封 IP 的网站，也可以设置为百度这类响应稳定的网站。</p>
<p>我们还定义了 TEST_VALID_STATUS 变量，这个变量是一个列表形式，包含了正常的状态码，如可以定义成 [200]。当然某些目标网站可能会出现其他的状态码，你可以自行配置。</p>
<p>程序在获取 Response 后需要判断响应的状态，如果状态码在 TEST_VALID_STATUS 列表里，则代表代理可用，可以调用 RedisClient 的 max 方法将代理分数设为 100，否则调用 decrease 方法将代理分数减 1，如果出现异常也同样将代理分数减 1。</p>
<p>另外，我们设置了批量测试的最大值为 TEST_BATCH，也就是一批测试最多 TEST_BATCH 个，这可以避免代理池过大时一次性测试全部代理导致内存开销过大的问题。当然也可以用信号量来实现并发控制。</p>
<p>随后，在 run 方法里面获取了所有的代理列表，使用 aiohttp 分配任务，启动运行。这样在不断的运行过程中，代理池中无效的代理的分数会一直被减 1，直至被清除，有效的代理则会一直保持 100 分，供随时取用。</p>
<p>这样，测试模块的逻辑就完成了。</p>
<h4>接口模块</h4>
<p>通过上述 3 个模块，我们已经可以做到代理的获取、检测和更新，数据库就会以有序集合的形式存储各个代理及其对应的分数，分数 100 代表可用，分数越小代表越不可用。</p>
<p>但是我们怎样方便地获取可用代理呢？可以用 RedisClient 类直接连接 Redis，然后调用 random 方法。这样做没问题，效率很高，但是会有几个弊端。</p>
<ul>
<li>如果其他人使用这个代理池，他需要知道 Redis 连接的用户名和密码信息，这样很不安全。</li>
<li>如果代理池需要部署在远程服务器上运行，而远程服务器的 Redis 只允许本地连接，那么我们就不能远程直连 Redis 来获取代理。</li>
<li>如果爬虫所在的主机没有连接 Redis 模块，或者爬虫不是由 Python 语言编写的，那么我们就无法使用 RedisClient 来获取代理。</li>
<li>如果 RedisClient 类或者数据库结构有更新，那么爬虫端必须同步这些更新，这样非常麻烦。</li>
</ul>
<p>综上考虑，为了使代理池可以作为一个独立服务运行，我们最好增加一个接口模块，并以 Web API 的形式暴露可用代理。</p>
<p>这样一来，获取代理只需要请求接口即可，以上的几个缺点弊端也可以避免。</p>
<p>我们使用一个比较轻量级的库 Flask 来实现这个接口模块，实现示例如下所示：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">from</span>&nbsp;flask&nbsp;<span class="hljs-keyword">import</span>&nbsp;Flask,&nbsp;g
<span class="hljs-keyword">from</span>&nbsp;proxypool.storages.redis&nbsp;<span class="hljs-keyword">import</span>&nbsp;RedisClient
<span class="hljs-keyword">from</span>&nbsp;proxypool.setting&nbsp;<span class="hljs-keyword">import</span>&nbsp;API_HOST,&nbsp;API_PORT,&nbsp;API_THREADED
__all__&nbsp;=&nbsp;[<span class="hljs-string">'app'</span>]
app&nbsp;=&nbsp;Flask(__name__)
<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">get_conn</span><span class="hljs-params">()</span>:</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;get&nbsp;redis&nbsp;client&nbsp;object
&nbsp;&nbsp;&nbsp;:return:
&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">if</span>&nbsp;<span class="hljs-keyword">not</span>&nbsp;hasattr(g,&nbsp;<span class="hljs-string">'redis'</span>):
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;g.redis&nbsp;=&nbsp;RedisClient()
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;g.redis
<span class="hljs-meta">@app.route('/')</span>
<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">index</span><span class="hljs-params">()</span>:</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;get&nbsp;home&nbsp;page,&nbsp;you&nbsp;can&nbsp;define&nbsp;your&nbsp;own&nbsp;templates
&nbsp;&nbsp;&nbsp;:return:
&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;<span class="hljs-string">'&lt;h2&gt;Welcome&nbsp;to&nbsp;Proxy&nbsp;Pool&nbsp;System&lt;/h2&gt;'</span>
<span class="hljs-meta">@app.route('/random')</span>
<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">get_proxy</span><span class="hljs-params">()</span>:</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;get&nbsp;a&nbsp;random&nbsp;proxy
&nbsp;&nbsp;&nbsp;:return:&nbsp;get&nbsp;a&nbsp;random&nbsp;proxy
&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;conn&nbsp;=&nbsp;get_conn()
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;conn.random().string()
<span class="hljs-meta">@app.route('/count')</span>
<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">get_count</span><span class="hljs-params">()</span>:</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;get&nbsp;the&nbsp;count&nbsp;of&nbsp;proxies
&nbsp;&nbsp;&nbsp;:return:&nbsp;count,&nbsp;int
&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;conn&nbsp;=&nbsp;get_conn()
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;str(conn.count())
<span class="hljs-keyword">if</span>&nbsp;__name__&nbsp;==&nbsp;<span class="hljs-string">'__main__'</span>:
&nbsp;&nbsp;&nbsp;app.run(host=API_HOST,&nbsp;port=API_PORT,&nbsp;threaded=API_THREADED)
</code></pre>
<p>在这里，我们声明了一个 Flask 对象，定义了 3 个接口，分别是首页、随机代理页、获取数量页。</p>
<p>运行之后，Flask 会启动一个 Web 服务，我们只需要访问对应的接口即可获取到可用代理。</p>
<h4>调度模块</h4>
<p>调度模块就是调用以上所定义的 3 个模块，将这 3 个模块通过多进程的形式运行起来，示例如下所示：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span>&nbsp;time
<span class="hljs-keyword">import</span>&nbsp;multiprocessing
<span class="hljs-keyword">from</span>&nbsp;proxypool.processors.server&nbsp;<span class="hljs-keyword">import</span>&nbsp;app
<span class="hljs-keyword">from</span>&nbsp;proxypool.processors.getter&nbsp;<span class="hljs-keyword">import</span>&nbsp;Getter
<span class="hljs-keyword">from</span>&nbsp;proxypool.processors.tester&nbsp;<span class="hljs-keyword">import</span>&nbsp;Tester
<span class="hljs-keyword">from</span>&nbsp;proxypool.setting&nbsp;<span class="hljs-keyword">import</span>&nbsp;CYCLE_GETTER,&nbsp;CYCLE_TESTER,&nbsp;API_HOST,&nbsp;API_THREADED,&nbsp;API_PORT,&nbsp;ENABLE_SERVER,&nbsp;\
&nbsp;&nbsp;&nbsp;ENABLE_GETTER,&nbsp;ENABLE_TESTER,&nbsp;IS_WINDOWS
<span class="hljs-keyword">from</span>&nbsp;loguru&nbsp;<span class="hljs-keyword">import</span>&nbsp;logger
<span class="hljs-keyword">if</span>&nbsp;IS_WINDOWS:
&nbsp;&nbsp;&nbsp;multiprocessing.freeze_support()
tester_process,&nbsp;getter_process,&nbsp;server_process&nbsp;=&nbsp;<span class="hljs-literal">None</span>,&nbsp;<span class="hljs-literal">None</span>,&nbsp;<span class="hljs-literal">None</span>
<span class="hljs-class"><span class="hljs-keyword">class</span>&nbsp;<span class="hljs-title">Scheduler</span><span class="hljs-params">()</span>:</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;scheduler
&nbsp;&nbsp;&nbsp;"""</span>

&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">run_tester</span><span class="hljs-params">(self,&nbsp;cycle=CYCLE_TESTER)</span>:</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;run&nbsp;tester
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">if</span>&nbsp;<span class="hljs-keyword">not</span>&nbsp;ENABLE_TESTER:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.info(<span class="hljs-string">'tester&nbsp;not&nbsp;enabled,&nbsp;exit'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tester&nbsp;=&nbsp;Tester()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;loop&nbsp;=&nbsp;<span class="hljs-number">0</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">while</span>&nbsp;<span class="hljs-literal">True</span>:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.debug(<span class="hljs-string">f'tester&nbsp;loop&nbsp;<span class="hljs-subst">{loop}</span>&nbsp;start...'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tester.run()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;loop&nbsp;+=&nbsp;<span class="hljs-number">1</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;time.sleep(cycle)

&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">run_getter</span><span class="hljs-params">(self,&nbsp;cycle=CYCLE_GETTER)</span>:</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;run&nbsp;getter
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">if</span>&nbsp;<span class="hljs-keyword">not</span>&nbsp;ENABLE_GETTER:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.info(<span class="hljs-string">'getter&nbsp;not&nbsp;enabled,&nbsp;exit'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;getter&nbsp;=&nbsp;Getter()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;loop&nbsp;=&nbsp;<span class="hljs-number">0</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">while</span>&nbsp;<span class="hljs-literal">True</span>:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.debug(<span class="hljs-string">f'getter&nbsp;loop&nbsp;<span class="hljs-subst">{loop}</span>&nbsp;start...'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;getter.run()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;loop&nbsp;+=&nbsp;<span class="hljs-number">1</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;time.sleep(cycle)

&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">run_server</span><span class="hljs-params">(self)</span>:</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;run&nbsp;server&nbsp;for&nbsp;api
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">if</span>&nbsp;<span class="hljs-keyword">not</span>&nbsp;ENABLE_SERVER:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.info(<span class="hljs-string">'server&nbsp;not&nbsp;enabled,&nbsp;exit'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;app.run(host=API_HOST,&nbsp;port=API_PORT,&nbsp;threaded=API_THREADED)

&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">run</span><span class="hljs-params">(self)</span>:</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">global</span>&nbsp;tester_process,&nbsp;getter_process,&nbsp;server_process
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">try</span>:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.info(<span class="hljs-string">'starting&nbsp;proxypool...'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">if</span>&nbsp;ENABLE_TESTER:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tester_process&nbsp;=&nbsp;multiprocessing.Process(target=self.run_tester)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.info(<span class="hljs-string">f'starting&nbsp;tester,&nbsp;pid&nbsp;<span class="hljs-subst">{tester_process.pid}</span>...'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tester_process.start()

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">if</span>&nbsp;ENABLE_GETTER:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;getter_process&nbsp;=&nbsp;multiprocessing.Process(target=self.run_getter)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.info(<span class="hljs-string">f'starting&nbsp;getter,&nbsp;pid<span class="hljs-subst">{getter_process.pid}</span>...'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;getter_process.start()

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">if</span>&nbsp;ENABLE_SERVER:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;server_process&nbsp;=&nbsp;multiprocessing.Process(target=self.run_server)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.info(<span class="hljs-string">f'starting&nbsp;server,&nbsp;pid<span class="hljs-subst">{server_process.pid}</span>...'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;server_process.start()

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tester_process.join()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;getter_process.join()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;server_process.join()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">except</span>&nbsp;KeyboardInterrupt:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.info(<span class="hljs-string">'received&nbsp;keyboard&nbsp;interrupt&nbsp;signal'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tester_process.terminate()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;getter_process.terminate()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;server_process.terminate()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">finally</span>:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-comment">#&nbsp;must&nbsp;call&nbsp;join&nbsp;method&nbsp;before&nbsp;calling&nbsp;is_alive</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tester_process.join()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;getter_process.join()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;server_process.join()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.info(<span class="hljs-string">f'tester&nbsp;is&nbsp;<span class="hljs-subst">{<span class="hljs-string">"alive"</span>&nbsp;<span class="hljs-keyword">if</span>&nbsp;tester_process.is_alive()&nbsp;<span class="hljs-keyword">else</span>&nbsp;<span class="hljs-string">"dead"</span>}</span>'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.info(<span class="hljs-string">f'getter&nbsp;is&nbsp;<span class="hljs-subst">{<span class="hljs-string">"alive"</span>&nbsp;<span class="hljs-keyword">if</span>&nbsp;getter_process.is_alive()&nbsp;<span class="hljs-keyword">else</span>&nbsp;<span class="hljs-string">"dead"</span>}</span>'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.info(<span class="hljs-string">f'server&nbsp;is&nbsp;<span class="hljs-subst">{<span class="hljs-string">"alive"</span>&nbsp;<span class="hljs-keyword">if</span>&nbsp;server_process.is_alive()&nbsp;<span class="hljs-keyword">else</span>&nbsp;<span class="hljs-string">"dead"</span>}</span>'</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.info(<span class="hljs-string">'proxy&nbsp;terminated'</span>)
<span class="hljs-keyword">if</span>&nbsp;__name__&nbsp;==&nbsp;<span class="hljs-string">'__main__'</span>:
&nbsp;&nbsp;&nbsp;scheduler&nbsp;=&nbsp;Scheduler()
&nbsp;&nbsp;&nbsp;scheduler.run()
</code></pre>
<p>3 个常量 ENABLE_TESTER、ENABLE_GETTER、ENABLE_SERVER 都是布尔类型，表示测试模块、获取模块、接口模块的开关，如果都为 True，则代表模块开启。</p>
<p>启动入口是 run 方法，这个方法分别判断 3 个模块的开关。如果开关开启，启动时程序就新建一个 Process 进程，设置好启动目标，然后调用 start 方法运行，这样 3 个进程就可以并行执行，互不干扰。</p>
<p>3 个调度方法结构也非常清晰。比如，run_tester 方法用来调度测试模块，首先声明一个 Tester 对象，然后进入死循环不断循环调用其 run 方法，执行完一轮之后就休眠一段时间，休眠结束之后重新再执行。在这里，休眠时间也定义为一个常量，如 20 秒，即每隔 20 秒进行一次代理检测。</p>
<p>最后，只需要调用 Scheduler 的 run 方法即可启动整个代理池。</p>
<p>以上内容便是整个代理池的架构和相应实现逻辑。</p>
<h3>运行</h3>
<p>接下来我们将代码整合一下，将代理运行起来，运行之后的输出结果如下所示：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-number">2020</span><span class="hljs-number">-04</span><span class="hljs-number">-13</span>&nbsp;<span class="hljs-number">02</span>:<span class="hljs-number">52</span>:<span class="hljs-number">06.510</span>&nbsp;|&nbsp;INFO&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;proxypool.storages.redis:decrease:<span class="hljs-number">73</span>&nbsp;-&nbsp;<span class="hljs-number">60.186</span><span class="hljs-number">.146</span><span class="hljs-number">.193</span>:<span class="hljs-number">9000</span>&nbsp;current&nbsp;score&nbsp;<span class="hljs-number">10.0</span>,&nbsp;decrease&nbsp;<span class="hljs-number">1</span>
<span class="hljs-number">2020</span><span class="hljs-number">-04</span><span class="hljs-number">-13</span>&nbsp;<span class="hljs-number">02</span>:<span class="hljs-number">52</span>:<span class="hljs-number">06.517</span>&nbsp;|&nbsp;DEBUG&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;proxypool.processors.tester:test:<span class="hljs-number">52</span>&nbsp;-&nbsp;proxy&nbsp;<span class="hljs-number">60.186</span><span class="hljs-number">.146</span><span class="hljs-number">.193</span>:<span class="hljs-number">9000</span>&nbsp;<span class="hljs-keyword">is</span>&nbsp;invalid,&nbsp;decrease&nbsp;score
<span class="hljs-number">2020</span><span class="hljs-number">-04</span><span class="hljs-number">-13</span>&nbsp;<span class="hljs-number">02</span>:<span class="hljs-number">52</span>:<span class="hljs-number">06.524</span>&nbsp;|&nbsp;INFO&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;proxypool.storages.redis:decrease:<span class="hljs-number">73</span>&nbsp;-&nbsp;<span class="hljs-number">60.186</span><span class="hljs-number">.151</span><span class="hljs-number">.147</span>:<span class="hljs-number">9000</span>&nbsp;current&nbsp;score&nbsp;<span class="hljs-number">10.0</span>,&nbsp;decrease&nbsp;<span class="hljs-number">1</span>
<span class="hljs-number">2020</span><span class="hljs-number">-04</span><span class="hljs-number">-13</span>&nbsp;<span class="hljs-number">02</span>:<span class="hljs-number">52</span>:<span class="hljs-number">06.532</span>&nbsp;|&nbsp;DEBUG&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;proxypool.processors.tester:test:<span class="hljs-number">52</span>&nbsp;-&nbsp;proxy&nbsp;<span class="hljs-number">60.186</span><span class="hljs-number">.151</span><span class="hljs-number">.147</span>:<span class="hljs-number">9000</span>&nbsp;<span class="hljs-keyword">is</span>&nbsp;invalid,&nbsp;decrease&nbsp;score
<span class="hljs-number">2020</span><span class="hljs-number">-04</span><span class="hljs-number">-13</span>&nbsp;<span class="hljs-number">02</span>:<span class="hljs-number">52</span>:<span class="hljs-number">07.159</span>&nbsp;|&nbsp;INFO&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;proxypool.storages.redis:max:<span class="hljs-number">96</span>&nbsp;-&nbsp;<span class="hljs-number">60.191</span><span class="hljs-number">.11</span><span class="hljs-number">.246</span>:<span class="hljs-number">3128</span>&nbsp;<span class="hljs-keyword">is</span>&nbsp;valid,&nbsp;set&nbsp;to&nbsp;<span class="hljs-number">100</span>
<span class="hljs-number">2020</span><span class="hljs-number">-04</span><span class="hljs-number">-13</span>&nbsp;<span class="hljs-number">02</span>:<span class="hljs-number">52</span>:<span class="hljs-number">07.167</span>&nbsp;|&nbsp;DEBUG&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;proxypool.processors.tester:test:<span class="hljs-number">46</span>&nbsp;-&nbsp;proxy&nbsp;<span class="hljs-number">60.191</span><span class="hljs-number">.11</span><span class="hljs-number">.246</span>:<span class="hljs-number">3128</span>&nbsp;<span class="hljs-keyword">is</span>&nbsp;valid,&nbsp;set&nbsp;max&nbsp;score
<span class="hljs-number">2020</span><span class="hljs-number">-04</span><span class="hljs-number">-13</span>&nbsp;<span class="hljs-number">02</span>:<span class="hljs-number">52</span>:<span class="hljs-number">17.271</span>&nbsp;|&nbsp;INFO&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;proxypool.storages.redis:decrease:<span class="hljs-number">73</span>&nbsp;-&nbsp;<span class="hljs-number">59.62</span><span class="hljs-number">.7</span><span class="hljs-number">.130</span>:<span class="hljs-number">9000</span>&nbsp;current&nbsp;score&nbsp;<span class="hljs-number">10.0</span>,&nbsp;decrease&nbsp;<span class="hljs-number">1</span>
<span class="hljs-number">2020</span><span class="hljs-number">-04</span><span class="hljs-number">-13</span>&nbsp;<span class="hljs-number">02</span>:<span class="hljs-number">52</span>:<span class="hljs-number">17.280</span>&nbsp;|&nbsp;DEBUG&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;proxypool.processors.tester:test:<span class="hljs-number">52</span>&nbsp;-&nbsp;proxy&nbsp;<span class="hljs-number">59.62</span><span class="hljs-number">.7</span><span class="hljs-number">.130</span>:<span class="hljs-number">9000</span>&nbsp;<span class="hljs-keyword">is</span>&nbsp;invalid,&nbsp;decrease&nbsp;score
<span class="hljs-number">2020</span><span class="hljs-number">-04</span><span class="hljs-number">-13</span>&nbsp;<span class="hljs-number">02</span>:<span class="hljs-number">52</span>:<span class="hljs-number">17.288</span>&nbsp;|&nbsp;INFO&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;proxypool.storages.redis:decrease:<span class="hljs-number">73</span>&nbsp;-&nbsp;<span class="hljs-number">60.167</span><span class="hljs-number">.103</span><span class="hljs-number">.74</span>:<span class="hljs-number">1133</span>&nbsp;current&nbsp;score&nbsp;<span class="hljs-number">10.0</span>,&nbsp;decrease&nbsp;<span class="hljs-number">1</span>
<span class="hljs-number">2020</span><span class="hljs-number">-04</span><span class="hljs-number">-13</span>&nbsp;<span class="hljs-number">02</span>:<span class="hljs-number">52</span>:<span class="hljs-number">17.295</span>&nbsp;|&nbsp;DEBUG&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;proxypool.processors.tester:test:<span class="hljs-number">52</span>&nbsp;-&nbsp;proxy&nbsp;<span class="hljs-number">60.167</span><span class="hljs-number">.103</span><span class="hljs-number">.74</span>:<span class="hljs-number">1133</span>&nbsp;<span class="hljs-keyword">is</span>&nbsp;invalid,&nbsp;decrease&nbsp;score
<span class="hljs-number">2020</span><span class="hljs-number">-04</span><span class="hljs-number">-13</span>&nbsp;<span class="hljs-number">02</span>:<span class="hljs-number">52</span>:<span class="hljs-number">17.302</span>&nbsp;|&nbsp;INFO&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;proxypool.storages.redis:decrease:<span class="hljs-number">73</span>&nbsp;-&nbsp;<span class="hljs-number">60.162</span><span class="hljs-number">.71</span><span class="hljs-number">.113</span>:<span class="hljs-number">9000</span>&nbsp;current&nbsp;score&nbsp;<span class="hljs-number">10.0</span>,&nbsp;decrease&nbsp;<span class="hljs-number">1</span>
<span class="hljs-number">2020</span><span class="hljs-number">-04</span><span class="hljs-number">-13</span>&nbsp;<span class="hljs-number">02</span>:<span class="hljs-number">52</span>:<span class="hljs-number">17.309</span>&nbsp;|&nbsp;DEBUG&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;proxypool.processors.tester:test:<span class="hljs-number">52</span>&nbsp;-&nbsp;proxy&nbsp;<span class="hljs-number">60.162</span><span class="hljs-number">.71</span><span class="hljs-number">.113</span>:<span class="hljs-number">9000</span>&nbsp;<span class="hljs-keyword">is</span>&nbsp;invalid,&nbsp;decrease&nbsp;score
</code></pre>
<p>以上是代理池的控制台输出，可以看到可用代理设置为 100，不可用代理分数减 1。</p>
<p>接下来我们再打开浏览器，当前配置了运行在 5555 端口，所以打开：<a href="http://127.0.0.1:5555">http://127.0.0.1:5555</a>，即可看到其首页，如图所示。</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/0E/BE/Ciqah16UK_WAX8YSAAEMqmBH7rI997.png" alt=""></p>
<p>再访问 <a href="http://127.0.0.1:5555/random">http://127.0.0.1:5555/random</a>，即可获取随机可用代理，如图所示。<br>
<img src="https://s0.lgstatic.com/i/image3/M01/87/D4/Cgq2xl6UK_WARL3GAADK8PRJ5EQ293.png" alt=""></p>
<p>我们只需要访问此接口即可获取一个随机可用代理，这非常方便。</p>
<p>获取代理的代码如下所示：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span>&nbsp;requests
&nbsp;
PROXY_POOL_URL&nbsp;=&nbsp;<span class="hljs-string">'http://localhost:5555/random'</span>
&nbsp;
<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">get_proxy</span><span class="hljs-params">()</span>:</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">try</span>:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;response&nbsp;=&nbsp;requests.get(PROXY_POOL_URL)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">if</span>&nbsp;response.status_code&nbsp;==&nbsp;<span class="hljs-number">200</span>:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;response.text
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">except</span>&nbsp;ConnectionError:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;<span class="hljs-literal">None</span>
</code></pre>
<p>这样便可以获取到一个随机代理了，它是字符串类型，此代理可以按照上一课时所示的方法设置，如 requests 的使用方法如下所示：</p>
<pre><code data-language="python" class="lang-python"><span class="hljs-keyword">import</span>&nbsp;requests
&nbsp;
proxy&nbsp;=&nbsp;get_proxy()
proxies&nbsp;=&nbsp;{
&nbsp;&nbsp;&nbsp;<span class="hljs-string">'http'</span>:&nbsp;<span class="hljs-string">'http://'</span>&nbsp;+&nbsp;proxy,
&nbsp;&nbsp;&nbsp;<span class="hljs-string">'https'</span>:&nbsp;<span class="hljs-string">'https://'</span>&nbsp;+&nbsp;proxy,
}
<span class="hljs-keyword">try</span>:
&nbsp;&nbsp;&nbsp;response&nbsp;=&nbsp;requests.get(<span class="hljs-string">'http://httpbin.org/get'</span>,&nbsp;proxies=proxies)
&nbsp;&nbsp;&nbsp;print(response.text)
<span class="hljs-keyword">except</span>&nbsp;requests.exceptions.ConnectionError&nbsp;<span class="hljs-keyword">as</span>&nbsp;e:
&nbsp;&nbsp;&nbsp;print(<span class="hljs-string">'Error'</span>,&nbsp;e.args)
</code></pre>
<p>有了代理池之后，我们再取出代理即可有效防止 IP 被封禁的情况。</p>
<h3>总结</h3>
<p>本课时代码地址为：<a href="https://github.com/Python3WebSpider/ProxyPool">https://github.com/Python3WebSpider/ProxyPool</a>，代码量相比之前的案例复杂了很多，逻辑也相对完善。另外代码库中还提供了 Docker 和 Kubernetes 的运行和部署操作，可以帮助我们更加快捷地运行代理池，如果你感兴趣可以了解下。</p>

# 验证码反爬虫的基本原理
<p>我们在浏览网站的时候经常会遇到各种各样的验证码，在多数情况下这些验证码会出现在登录账号的时候，也可能会出现在访问页面的过程中，严格来说，这些行为都算验证码反爬虫。</p>
<p>本课时我们就来介绍下验证码反爬虫的基本原理及常见的验证码和解决方案。</p>
<h3>验证码</h3>
<p>验证码，全称叫作 Completely Automated Public Turing test to tell Computers and Humans Apart，意思是全自动区分计算机和人类的图灵测试，取了它们关键词的首字母变成了 CAPTCHA，它是一种用来区分用户是计算机还是人的公共全自动程序。</p>
<p>它有什么用呢？当然很多用处，如：</p>
<ul>
<li>网站注册的时候加上验证码，可以一定程度上防止恶意大批量注册。</li>
<li>网站登录的时候加上验证码，可以一定程度上防止恶意密码爆破。</li>
<li>网站在发表评论的时候加上验证码，可以在一定程度上防止恶意灌水。</li>
<li>网站在投票的时候加上验证码，可以在一定程度上防止恶意刷票。</li>
<li>网站在被频繁访问的时候或者浏览行为不正常的时候，一般可能是遇到了爬虫，可以一定程度上防止爬虫的爬取。</li>
</ul>
<p>总的来说呢，以上的行为都可以称之为验证码反爬虫行为。使用验证码可以防止各种可以用程序模拟的行为。有了验证码，机器要想完全自动化执行就会遇到一些麻烦，当然这个麻烦的大小就取决于验证码的破解难易程度了。</p>
<h3>验证码反爬虫</h3>
<p>那为什么会出现验证码呢？在大多数情形下是因为网站的访问频率过高或者行为异常，或者是为了直接限制某些自动化行为。归类如下：</p>
<ul>
<li>很多情况下，比如登录和注册，这些验证码几乎是必现的，它的目的就是为了限制恶意注册、恶意爆破等行为，这也算反爬的一种手段。</li>
<li>一些网站遇到访问频率过高的行为的时候，可能会直接弹出一个登录窗口，要求我们登录才能继续访问，此时的验证码就直接和登录表单绑定在一起了，这就算检测到异常之后利用强制登录的方式进行反爬。</li>
<li>一些较为常规的网站如果遇到访问频率稍高的情形的时候，会主动弹出一个验证码让用户识别并提交，验证当前访问网站的是不是真实的人，用来限制一些机器的行为，实现反爬虫。</li>
</ul>
<p>这几种情形都能在一定程度上限制程序的一些自动化行为，因此都可以称之为反爬虫。</p>
<h3>验证码反爬虫的原理</h3>
<p>在模块一的时候，我们已经讲到过 Session 的基本概念了，它是存在于服务端的，用于保存当前用户的会话信息，这个信息对于验证码的机制非常重要。</p>
<p>服务端是可以往 Session 对象里面存一些值的，比如我们要生成一个图形验证码，比如 1234 这四个数字的图形验证码。</p>
<p>首先客户端要显示某个验证码，这个验证码相关的信息肯定要从服务器端来获取。比如说请求了这个生成验证码的接口，我们要生成一个图形验证码，内容为 1234，这时候服务端会将 1234 这四个数字保存到 Session 对象里面，然后把 1234 这个结果返回给客户端，或者直接把生成好的验证码图形返回也是可以的，客户端会将其呈现出来，用户就能看到验证码的内容了。</p>
<p>用户看到验证码之后呢，就会在表单里面输入验证码的内容，点击提交按钮的时候，这些信息就会又发送给服务器，服务器拿着提交的信息和 Session 里面保存的验证码信息后进行对比，如果一致，那就代表验证码输入正确，校验成功，然后就继续放行恢复正常状态。如果不一致，那就代表校验失败，会继续进行校验。</p>
<p>目前市面上大多数的验证码都是基于这个机制来实现的，归类如下：</p>
<ul>
<li>对于图形验证码，服务器会把图形的内容保存到 Session，然后将验证码图返回或者客户端自行显示，等用户提交表单之后校验 Session 里验证码的值和用户提交的值。</li>
<li>对于行为验证码，服务器会做一些计算，把一些 Key、Token 等信息也储存在 Session 里面，用户首先要完成客户端的校验，如果校验成功才能提交表单，当客户端的校验完成之后，客户端会把验证之后计算产生的 Key、Token、Code 等信息发送到服务端，服务端会再做一次校验，如果服务端也校验通过了，那就算真正的通过了。</li>
<li>对于手机验证码，服务器会预先生成一个验证码的信息，然后会把这个验证码的结果还有要发送的手机号发送给短信发送服务商，让服务商下发验证码给用户，用户再把这个码提交给服务器，服务器判断 Session 里面的验证码和提交的验证码是否一致即可。</li>
</ul>
<p>还有很多其他的验证码，其原理基本都是一致的。</p>
<h3>常见验证码</h3>
<p>下面我们再来看看市面上的一些常见的验证码，并简单介绍一些识别思路。</p>
<h4>图形验证码</h4>
<p>最基本的验证码就是图形验证码了，比如下图。</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/10/16/Ciqah16W5UyAEhSHAAAX1y2xT2g229.png" alt=""></p>
<p>一般来说，识别思路有这么几种：</p>
<ul>
<li>利用 OCR 识别，比如 Tesserocr 等库，或者直接调用 OCR 接口，如百度、腾讯的，识别效果相比 Tesserocr 更好。</li>
<li>打码平台，把验证码发送给打码平台，平台内实现了一些强大的识别算法或者平台背后有人来专门做识别，速度快，省心。</li>
<li>深度学习训练，这类验证码也可以使用 CNN 等深度学习模型来训练分类算法，但是如果种类繁多或者写法各异的话，其识别精度会有一些影响。</li>
</ul>
<h4>行为验证码</h4>
<p>现在我们能见到非常多类型的行为验证码，可以说是十分流行了，比如极验、腾讯、网易盾等等都有类似的验证码服务，另外验证的方式也多种多样，如滑动、拖动、点选、逻辑判断等等，如图所示。</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/89/2C/Cgq2xl6W5UyAGwElAADQXzjdcRk446.png" alt=""></p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/02/E8/CgoCgV6W5U2AP1KFAALydwxJhow955.png" alt=""></p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/10/16/Ciqah16W5U2AO2ACAAK0QZRq5dQ995.png" alt=""></p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/89/2C/Cgq2xl6W5U2AUskeAAFkApuai3Y580.png" alt=""></p>
<p>这里推荐的识别方案有以下几种：</p>
<ul>
<li>打码平台，这里面很多验证码都是与坐标相关的，我们可以直接将验证码截图发送给打码平台，打码平台背后会有人帮我们找到对应的位置坐标，获取位置坐标之后就可以来模拟了。这时候模拟的方法有两种，一种是模拟行为，使用 Selenium 等实现，模拟完成之后通常能登录或者解锁某个 Session 封锁状态，获取有效 Cookies 即可。另一种是在 JavaScript 层级上模拟，这种难度更高，模拟完了可以直接获取验证码提交的一些 Token 值等内容。</li>
<li>深度学习，利用一些图像标注加深度学习的方法同样可以识别验证码，其实主要还是识别位置，有了位置之后同样可以模拟。</li>
</ul>
<h4>短信、扫码验证码</h4>
<p>另外我们可能遇到一些类似短信、扫码的验证码，这种操作起来就会更加麻烦，一些解决思路如下：</p>
<ul>
<li>手机号可以不用自己的，可以从某些平台来获取，平台维护了一套手机短信收发系统，填入手机号，并通过 API 获取短信验证码即可。</li>
<li>另外也可以购买一些专业的收码设备或者安装一些监听短信的软件，它会有一些机制把一些手机短信信息导出到某个接口或文本或数据库，然后再提取即可。</li>
<li>对于扫码验证的情况，如果不用自己的账号，可以把码发送到打码平台，让对方用自己的账号扫码处理，但这种情况多数需要定制，可以去跟平台沟通。另外的方案就涉及到逆向和破解相关的内容了，一般需要逆向手机 App 内的扫码和解析逻辑，然后再模拟，这里就不再展开讲了。</li>
</ul>
<p>基本上验证码都是类似的，其中有一些列举不全，但是基本类别都能大致归类。</p>
<p>以上我们就介绍了验证码反爬虫的基本原理和一些验证码识别的思路。在后面的课时我会介绍使用打码平台和深度学习的方式来识别验证码的方案。</p>

# 学会用打码平台处理验证码
<p data-nodeid="15448" class="">在前一课时我们介绍了多种多样的验证码，有图形文字的、有模拟点选的、有拖动滑动的，但其实归根结底都需要人来对某种情形做一些判断，然后把结果返回并提交。如果此时提交的验证码结果是正确的，并且通过了一些验证码的检测，就能成功突破这个验证码了。</p>
<p data-nodeid="15449">那么，既然验证码就是让人来识别的，那么机器怎么办呢？如果我们也不会什么算法，怎么去解这些验证码呢？此时如果有一个帮助我们来识别验证码的工具或平台就好了，让工具或平台把验证码识别的结果返回给我们，我们拿着结果提交，那不就好了吗？</p>
<p data-nodeid="15450">有这种工具或平台吗？还真有专门的打码平台帮助我们来识别各种各样的验证码，平台内部对算法和人力做了集成，可以 7x24 小时来识别各种验证码，包括识别图形、坐标点、缺口等各种验证码，返回对应的结果或坐标，正好可以解决我们的问题。</p>
<p data-nodeid="15451">本课时我们就来介绍利用打码平台来识别验证码的流程。</p>
<h3 data-nodeid="15452">学习目标</h3>
<p data-nodeid="16033" class="">本课时我们以一种点选验证码为例来讲解打码平台的使用方法，验证码的链接为：<a href="https://captcha3.scrape.center/" data-nodeid="16037">https://captcha3.scrape.center/</a>，这个网站在每次登录的时候都会弹出一个验证码，其验证码效果图如下所示。</p>


<p data-nodeid="15454"><img src="https://s0.lgstatic.com/i/image3/M01/8A/25/Cgq2xl6ZWNWAUhQaAAMYtARC5aI058.png" alt="" data-nodeid="15533"></p>
<p data-nodeid="15455">这个验证码上面显示了几个汉字，同时在图中也显示了几个汉字，我们需要按照顺序依次点击汉字在图中的位置，点击完成之后确认提交，即可完成验证。</p>
<p data-nodeid="15456">这种验证码如果我们没有任何图像识别算法基础的话，是很难去识别的，所以这里我们可以借助打码平台来帮助我们识别汉字的位置。</p>
<h3 data-nodeid="15457">准备工作</h3>
<p data-nodeid="15458">我们使用的 Python 库是 Selenium，使用的浏览器为 Chrome。</p>
<p data-nodeid="15459">在本课时开始之前请确保已经正确安装好 Selenium 库、Chrome 浏览器，并配置好 ChromeDriver，相关流程可以参考 Selenium 那一课时的介绍。</p>
<p data-nodeid="15460">另外本课时使用的打码平台是超级鹰，链接为：<a href="https://www.chaojiying.com/" data-nodeid="15542">https://www.chaojiying.com/</a>，在使用之前请你自己注册账号并获取一些题分供测试，另外还可以了解平台可识别的验证码的类别。</p>
<h3 data-nodeid="15461">打码平台</h3>
<p data-nodeid="15462">打码平台能提供的服务种类一般都非常广泛，可识别的验证码类型也非常多，其中就包括点触验证码。</p>
<p data-nodeid="15463">超级鹰平台同样支持简单的图形验证码识别。超级鹰平台提供了如下一些服务。</p>
<ul data-nodeid="15464">
<li data-nodeid="15465">
<p data-nodeid="15466">英文数字：提供最多 20 位英文数字的混合识别；</p>
</li>
<li data-nodeid="15467">
<p data-nodeid="15468">中文汉字：提供最多 7 个汉字的识别；</p>
</li>
<li data-nodeid="15469">
<p data-nodeid="15470">纯英文：提供最多 12 位的英文识别；</p>
</li>
<li data-nodeid="15471">
<p data-nodeid="15472">纯数字：提供最多 11 位的数字识别；</p>
</li>
<li data-nodeid="15473">
<p data-nodeid="15474">任意特殊字符：提供不定长汉字英文数字、拼音首字母、计算题、成语混合、集装箱号等字符的识别；</p>
</li>
<li data-nodeid="15475">
<p data-nodeid="15476">坐标选择识别：如复杂计算题、选择题四选一、问答题、点击相同的字、物品、动物等返回多个坐标的识别。</p>
</li>
</ul>
<p data-nodeid="15477">具体如有变动以官网为准：<a href="https://www.chaojiying.com/price.html" data-nodeid="15556">https://www.chaojiying.com/price.html</a>。</p>
<p data-nodeid="15478">这里需要处理的就是坐标多选识别的情况。我们先将验证码图片提交给平台，平台会返回识别结果在图片中的坐标位置，然后我们再解析坐标模拟点击。</p>
<p data-nodeid="15479">下面我们就用程序来实现。</p>
<h3 data-nodeid="15480">获取 API</h3>
<p data-nodeid="15481">在官方网站下载对应的 Python API，链接为：<a href="https://www.chaojiying.com/api-14.html" data-nodeid="15564">https://www.chaojiying.com/api-14.html</a>。API 是 Python 2 版本的，是用 requests 库来实现的。我们可以简单更改几个地方，即可将其修改为 Python 3 版本。</p>
<p data-nodeid="15482">修改之后的 API 如下所示：</p>
<pre class="lang-python" data-nodeid="15483"><code data-language="python"><span class="hljs-keyword">import</span>&nbsp;requests
<span class="hljs-keyword">from</span>&nbsp;hashlib&nbsp;<span class="hljs-keyword">import</span>&nbsp;md5
<span class="hljs-class"><span class="hljs-keyword">class</span>&nbsp;<span class="hljs-title">Chaojiying</span>(<span class="hljs-params">object</span>):</span>

&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">__init__</span>(<span class="hljs-params">self,&nbsp;username,&nbsp;password,&nbsp;soft_id</span>):</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.username&nbsp;=&nbsp;username
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.password&nbsp;=&nbsp;md5(password.encode(<span class="hljs-string">'utf-8'</span>)).hexdigest()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.soft_id&nbsp;=&nbsp;soft_id
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.base_params&nbsp;=&nbsp;{
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">'user'</span>:&nbsp;self.username,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">'pass2'</span>:&nbsp;self.password,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">'softid'</span>:&nbsp;self.soft_id,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.headers&nbsp;=&nbsp;{
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">'Connection'</span>:&nbsp;<span class="hljs-string">'Keep-Alive'</span>,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">'User-Agent'</span>:&nbsp;<span class="hljs-string">'Mozilla/4.0&nbsp;(compatible;&nbsp;MSIE&nbsp;8.0;&nbsp;Windows&nbsp;NT&nbsp;5.1;&nbsp;Trident/4.0)'</span>,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}

&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">post_pic</span>(<span class="hljs-params">self,&nbsp;im,&nbsp;codetype</span>):</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;im:&nbsp;图片字节
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;codetype:&nbsp;题目类型&nbsp;参考&nbsp;http://www.chaojiying.com/price.html
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params&nbsp;=&nbsp;{
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">'codetype'</span>:&nbsp;codetype,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params.update(self.base_params)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;files&nbsp;=&nbsp;{<span class="hljs-string">'userfile'</span>:&nbsp;(<span class="hljs-string">'ccc.jpg'</span>,&nbsp;im)}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;r&nbsp;=&nbsp;requests.post(<span class="hljs-string">'http://upload.chaojiying.net/Upload/Processing.php'</span>,&nbsp;data=params,&nbsp;files=files,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;headers=self.headers)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;r.json()

&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">report_error</span>(<span class="hljs-params">self,&nbsp;im_id</span>):</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;im_id:报错题目的图片ID
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params&nbsp;=&nbsp;{
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">'id'</span>:&nbsp;im_id,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params.update(self.base_params)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;r&nbsp;=&nbsp;requests.post(<span class="hljs-string">'http://upload.chaojiying.net/Upload/ReportError.php'</span>,&nbsp;data=params,&nbsp;headers=self.headers)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;r.json()
</code></pre>
<p data-nodeid="15484">这里定义了一个 Chaojiying 类，其构造函数接收三个参数，分别是超级鹰的用户名、密码以及软件 ID，保存以备使用。</p>
<p data-nodeid="15485">最重要的一个方法叫作 post_pic，它需要传入图片对象和验证码类型的代号。该方法会将图片对象和相关信息发给超级鹰的后台进行识别，然后将识别成功的 JSON 返回。</p>
<p data-nodeid="15486">另一个方法叫作 report_error，它是发生错误时的回调。如果验证码识别错误，调用此方法会返回相应的题分。</p>
<p data-nodeid="16817" class="">接下来，我们以 <a href="https://captcha3.scrape.center/" data-nodeid="16821">https://captcha3.scrape.center/</a> 为例来演示下识别的过程。</p>


<h3 data-nodeid="15488">初始化</h3>
<p data-nodeid="15489">首先我们引入一些必要的包，然后初始化一些变量，如 WebDriver、Chaojiying 对象等，代码实现如下所示：</p>
<pre class="lang-python" data-nodeid="17209"><code data-language="python"><span class="hljs-keyword">import</span>&nbsp;time
<span class="hljs-keyword">from</span>&nbsp;io&nbsp;<span class="hljs-keyword">import</span>&nbsp;BytesIO
<span class="hljs-keyword">from</span>&nbsp;PIL&nbsp;<span class="hljs-keyword">import</span>&nbsp;Image
<span class="hljs-keyword">from</span>&nbsp;selenium&nbsp;<span class="hljs-keyword">import</span>&nbsp;webdriver
<span class="hljs-keyword">from</span>&nbsp;selenium.webdriver&nbsp;<span class="hljs-keyword">import</span>&nbsp;ActionChains
<span class="hljs-keyword">from</span>&nbsp;selenium.webdriver.common.by&nbsp;<span class="hljs-keyword">import</span>&nbsp;By
<span class="hljs-keyword">from</span>&nbsp;selenium.webdriver.support.ui&nbsp;<span class="hljs-keyword">import</span>&nbsp;WebDriverWait
<span class="hljs-keyword">from</span>&nbsp;selenium.webdriver.support&nbsp;<span class="hljs-keyword">import</span>&nbsp;expected_conditions&nbsp;<span class="hljs-keyword">as</span>&nbsp;EC
<span class="hljs-keyword">from</span>&nbsp;chaojiying&nbsp;<span class="hljs-keyword">import</span>&nbsp;Chaojiying
USERNAME&nbsp;=&nbsp;<span class="hljs-string">'admin'</span>
PASSWORD&nbsp;=&nbsp;<span class="hljs-string">'admin'</span>
CHAOJIYING_USERNAME&nbsp;=&nbsp;<span class="hljs-string">''</span>
CHAOJIYING_PASSWORD&nbsp;=&nbsp;<span class="hljs-string">''</span>
CHAOJIYING_SOFT_ID&nbsp;=&nbsp;<span class="hljs-number">893590</span>
CHAOJIYING_KIND&nbsp;=&nbsp;<span class="hljs-number">9102</span>
<span class="hljs-keyword">if</span>&nbsp;<span class="hljs-keyword">not</span>&nbsp;CHAOJIYING_USERNAME&nbsp;<span class="hljs-keyword">or</span>&nbsp;<span class="hljs-keyword">not</span>&nbsp;CHAOJIYING_PASSWORD:
&nbsp;&nbsp;&nbsp;print(<span class="hljs-string">'请设置用户名和密码'</span>)
&nbsp;&nbsp;&nbsp;exit(<span class="hljs-number">0</span>)
<span class="hljs-class"><span class="hljs-keyword">class</span>&nbsp;<span class="hljs-title">CrackCaptcha</span>():</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">__init__</span>(<span class="hljs-params">self</span>):</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.url&nbsp;=&nbsp;<span class="hljs-string">'https://captcha3.scrape.center/'</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.browser&nbsp;=&nbsp;webdriver.Chrome()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.wait&nbsp;=&nbsp;WebDriverWait(self.browser,&nbsp;<span class="hljs-number">20</span>)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.username&nbsp;=&nbsp;USERNAME
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.password&nbsp;=&nbsp;PASSWORD
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.chaojiying&nbsp;=&nbsp;Chaojiying(CHAOJIYING_USERNAME,&nbsp;CHAOJIYING_PASSWORD,&nbsp;CHAOJIYING_SOFT_ID)
</code></pre>

<p data-nodeid="15491" class="te-preview-highlight">这里的 USERNAME、PASSWORD 是示例网站的用户名和密码，都设置为 admin 即可。另外 CHAOJIYING_USERNAME、CHAOJIYING_PASSWORD 就是超级鹰打码平台的用户名和密码，可以自行设置成自己的。</p>
<p data-nodeid="15492">另外这里定义了一个 CrackCaptcha 类，初始化了浏览器对象和打码平台的操作对象。</p>
<p data-nodeid="15493">接下来我们用 Selenium 模拟呼出验证码开始验证就好啦。</p>
<h3 data-nodeid="15494">获取验证码</h3>
<p data-nodeid="15495">接下来的步骤就是完善相关表单，模拟点击呼出验证码了，代码实现如下所示：</p>
<pre class="lang-python" data-nodeid="15496"><code data-language="python"><span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">open</span>(<span class="hljs-params">self</span>):</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;打开网页输入用户名密码
&nbsp;&nbsp;&nbsp;:return:&nbsp;None
&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;self.browser.get(self.url)
&nbsp;&nbsp;&nbsp;<span class="hljs-comment">#&nbsp;填入用户名密码</span>
&nbsp;&nbsp;&nbsp;username&nbsp;=&nbsp;self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,&nbsp;<span class="hljs-string">'input[type="text"]'</span>)))
&nbsp;&nbsp;&nbsp;password&nbsp;=&nbsp;self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,&nbsp;<span class="hljs-string">'input[type="password"]'</span>)))
&nbsp;&nbsp;&nbsp;username.send_keys(self.username)
&nbsp;&nbsp;&nbsp;password.send_keys(self.password)
<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">get_captcha_button</span>(<span class="hljs-params">self</span>):</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;获取初始验证按钮
&nbsp;&nbsp;&nbsp;:return:
&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;button&nbsp;=&nbsp;self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,&nbsp;<span class="hljs-string">'button[type="button"]'</span>)))
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;button
</code></pre>
<p data-nodeid="15497">这里我们调用了 open 方法负责填写表单，get_captcha_button 方法获取验证码按钮，之后触发点击，这时候就可以看到页面已经把验证码呈现出来了。</p>
<p data-nodeid="15498">有了验证码的图片，我们下一步要做的就是把验证码的具体内容获取下来，然后发送给打码平台识别。</p>
<p data-nodeid="15499">那怎么获取验证码的图片呢？我们可以先获取验证码图片的位置和大小，从网页截图里截取相应的验证码图片即可，代码实现如下所示：</p>
<pre class="lang-python" data-nodeid="15500"><code data-language="python"><span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">get_captcha_element</span>(<span class="hljs-params">self</span>):</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;获取验证图片对象
&nbsp;&nbsp;&nbsp;:return:&nbsp;图片对象
&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-comment">#&nbsp;验证码图片加载出来</span>
&nbsp;&nbsp;&nbsp;self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,&nbsp;<span class="hljs-string">'img.geetest_item_img'</span>)))
&nbsp;&nbsp;&nbsp;<span class="hljs-comment">#&nbsp;验证码完整节点</span>
&nbsp;&nbsp;&nbsp;element&nbsp;=&nbsp;self.wait.until(EC.presence_of_element_located((By.CLASS_NAME,&nbsp;<span class="hljs-string">'geetest_panel_box'</span>)))
&nbsp;&nbsp;&nbsp;print(<span class="hljs-string">'成功获取验证码节点'</span>)
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;element
<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">get_captcha_position</span>(<span class="hljs-params">self</span>):</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;获取验证码位置
&nbsp;&nbsp;&nbsp;:return:&nbsp;验证码位置元组
&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;element&nbsp;=&nbsp;self.get_captcha_element()
&nbsp;&nbsp;&nbsp;time.sleep(<span class="hljs-number">2</span>)
&nbsp;&nbsp;&nbsp;location&nbsp;=&nbsp;element.location
&nbsp;&nbsp;&nbsp;size&nbsp;=&nbsp;element.size
&nbsp;&nbsp;&nbsp;top,&nbsp;bottom,&nbsp;left,&nbsp;right&nbsp;=&nbsp;location[<span class="hljs-string">'y'</span>],&nbsp;location[<span class="hljs-string">'y'</span>]&nbsp;+&nbsp;size[<span class="hljs-string">'height'</span>],&nbsp;location[<span class="hljs-string">'x'</span>],&nbsp;location[<span class="hljs-string">'x'</span>]&nbsp;+&nbsp;size[
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hljs-string">'width'</span>]
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;(top,&nbsp;bottom,&nbsp;left,&nbsp;right)
<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">get_screenshot</span>(<span class="hljs-params">self</span>):</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;获取网页截图
&nbsp;&nbsp;&nbsp;:return:&nbsp;截图对象
&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;screenshot&nbsp;=&nbsp;self.browser.get_screenshot_as_png()
&nbsp;&nbsp;&nbsp;screenshot&nbsp;=&nbsp;Image.open(BytesIO(screenshot))
&nbsp;&nbsp;&nbsp;screenshot.save(<span class="hljs-string">'screenshot.png'</span>)
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;screenshot
<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">get_captcha_image</span>(<span class="hljs-params">self,&nbsp;name=<span class="hljs-string">'captcha.png'</span></span>):</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;获取验证码图片
&nbsp;&nbsp;&nbsp;:return:&nbsp;图片对象
&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;top,&nbsp;bottom,&nbsp;left,&nbsp;right&nbsp;=&nbsp;self.get_captcha_position()
&nbsp;&nbsp;&nbsp;print(<span class="hljs-string">'验证码位置'</span>,&nbsp;top,&nbsp;bottom,&nbsp;left,&nbsp;right)
&nbsp;&nbsp;&nbsp;screenshot&nbsp;=&nbsp;self.get_screenshot()
&nbsp;&nbsp;&nbsp;captcha&nbsp;=&nbsp;screenshot.crop((left,&nbsp;top,&nbsp;right,&nbsp;bottom))
&nbsp;&nbsp;&nbsp;captcha.save(name)
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;captcha
</code></pre>
<p data-nodeid="15501">这里 get_captcha_image 方法即为从网页截图中截取对应的验证码图片，其中验证码图片的相对位置坐标由 get_captcha_position 方法返回得到。所以就是利用了先截图再裁切的方法获取了验证码。</p>
<p data-nodeid="15502">注意：如果你的屏幕是高清屏如 Mac 的 Retina 屏幕的话，可能需要适当调整下屏幕分辨率或者对获取到的验证码位置做一些倍数偏移计算。</p>
<p data-nodeid="15503">最后我们得到的验证码是 Image 对象，其结果样例如图所示。</p>
<p data-nodeid="15504"><img src="https://s0.lgstatic.com/i/image3/M01/03/E0/CgoCgV6ZWNWAT2cwAANbWOcLXsY268.png" alt="" data-nodeid="15609"></p>
<h3 data-nodeid="15505">识别验证码</h3>
<p data-nodeid="15506">现在我们有了验证码图了，下一步就是把图发送给打码平台了。</p>
<p data-nodeid="15507">我们调用 Chaojiying 对象的 post_pic 方法，即可把图片发送给超级鹰后台，这里发送的图像是字节流格式，代码实现如下所示：</p>
<pre class="lang-python" data-nodeid="15508"><code data-language="python">image&nbsp;=&nbsp;self.get_touclick_image()
bytes_array&nbsp;=&nbsp;BytesIO()
image.save(bytes_array,&nbsp;format=<span class="hljs-string">'PNG'</span>)
<span class="hljs-comment">#&nbsp;识别验证码</span>
result&nbsp;=&nbsp;self.chaojiying.post_pic(bytes_array.getvalue(),&nbsp;CHAOJIYING_KIND)
print(result)
</code></pre>
<p data-nodeid="15509">运行之后，result 变量就是超级鹰后台的识别结果。可能运行需要等待几秒，它会返回一个 JSON 格式的字符串。</p>
<p data-nodeid="15510">如果识别成功，典型的返回结果如下所示：</p>
<pre class="lang-python" data-nodeid="15511"><code data-language="python">{<span class="hljs-string">'err_no'</span>:&nbsp;<span class="hljs-number">0</span>,&nbsp;<span class="hljs-string">'err_str'</span>:&nbsp;<span class="hljs-string">'OK'</span>,&nbsp;<span class="hljs-string">'pic_id'</span>:&nbsp;<span class="hljs-string">'6002001380949200001'</span>,&nbsp;<span class="hljs-string">'pic_str'</span>:&nbsp;<span class="hljs-string">'132,127|56,77'</span>,&nbsp;<span class="hljs-string">'md5'</span>:&nbsp;<span class="hljs-string">'1f8e1d4bef8b11484cb1f1f34299865b'</span>}
其中，pic_str&nbsp;就是识别的文字的坐标，是以字符串形式返回的，每个坐标都以&nbsp;|&nbsp;分隔。接下来我们只需要将其解析，然后模拟点击，代码实现如下所示：
<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">get_points</span>(<span class="hljs-params">self,&nbsp;captcha_result</span>):</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;解析识别结果
&nbsp;&nbsp;&nbsp;:param&nbsp;captcha_result:&nbsp;识别结果
&nbsp;&nbsp;&nbsp;:return:&nbsp;转化后的结果
&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;groups&nbsp;=&nbsp;captcha_result.get(<span class="hljs-string">'pic_str'</span>).split(<span class="hljs-string">'|'</span>)
&nbsp;&nbsp;&nbsp;locations&nbsp;=&nbsp;[[int(number)&nbsp;<span class="hljs-keyword">for</span>&nbsp;number&nbsp;<span class="hljs-keyword">in</span>&nbsp;group.split(<span class="hljs-string">','</span>)]&nbsp;<span class="hljs-keyword">for</span>&nbsp;group&nbsp;<span class="hljs-keyword">in</span>&nbsp;groups]
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">return</span>&nbsp;locations
<span class="hljs-function"><span class="hljs-keyword">def</span>&nbsp;<span class="hljs-title">touch_click_words</span>(<span class="hljs-params">self,&nbsp;locations</span>):</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-string">"""
&nbsp;&nbsp;&nbsp;点击验证图片
&nbsp;&nbsp;&nbsp;:param&nbsp;locations:&nbsp;点击位置
&nbsp;&nbsp;&nbsp;:return:&nbsp;None
&nbsp;&nbsp;&nbsp;"""</span>
&nbsp;&nbsp;&nbsp;<span class="hljs-keyword">for</span>&nbsp;location&nbsp;<span class="hljs-keyword">in</span>&nbsp;locations:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ActionChains(self.browser).move_to_element_with_offset(self.get_captcha_element(),&nbsp;location[<span class="hljs-number">0</span>],&nbsp;location[<span class="hljs-number">1</span>]).click().perform()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;time.sleep(<span class="hljs-number">1</span>)
</code></pre>
<p data-nodeid="15512">这里用 get_points 方法将识别结果变成列表的形式。touch_click_words 方法则通过调用 move_to_element_with_offset 方法依次传入解析后的坐标，点击即可。</p>
<p data-nodeid="15513">这样我们就模拟完成坐标的点选了，运行效果如下所示。</p>
<p data-nodeid="15514"><img src="https://s0.lgstatic.com/i/image3/M01/11/0F/Ciqah16ZWNaAJNOKAAM9-CV9h3A564.png" alt="" data-nodeid="15634"></p>
<p data-nodeid="15515">最后再模拟点击提交验证的按钮，等待验证通过就会自动登录啦，后续实现在此不再赘述。</p>
<p data-nodeid="15516">如何判断登录是否成功呢？同样可以使用 Selenium 的判定条件，比如判断页面里面出现了某个文字就代表登录成功了，代码如下：</p>
<pre class="lang-python" data-nodeid="15517"><code data-language="python"><span class="hljs-comment">#&nbsp;判定是否成功</span>
success&nbsp;=&nbsp;self.wait.until(EC.text_to_be_present_in_element((By.TAG_NAME,&nbsp;<span class="hljs-string">'h2'</span>),&nbsp;<span class="hljs-string">'登录成功'</span>))
</code></pre>
<p data-nodeid="15518">比如这里我们判定了点击确认按钮，页面会不会跳转到提示成功的页面，成功的页面包含一个 h2 节点，包含“登录成功”四个字，就代表登录成功啦。</p>
<p data-nodeid="15519">这样我们就借助在线验证码平台完成了点触验证码的识别。此方法是一种通用方法，我们也可以用此方法来识别图文、数字、算术等各种各样的验证码。</p>
<h3 data-nodeid="15520">结语</h3>
<p data-nodeid="15521" class="">本课时我们通过在线打码平台辅助完成了验证码的识别。这种识别方法非常强大，几乎任意的验证码都可以识别。如果遇到难题，借助打码平台无疑是一个极佳的选择。</p>

# 更智能的深度学习处理验证码
<p>我们在前面讲解了如何使用打码平台来识别验证码，简单高效。但是也有一些缺点，比如效率可能没那么高，准确率也不一定能做到完全可控，并且需要付出一定的费用。</p>
<p>本课时我们就来介绍使用深度学习来识别验证码的方法，训练好对应的模型就能更好地对验证码进行识别，并且准确率可控，节省一定的成本。</p>
<p>本课时我们以深度学习识别滑块验证码为例来讲解深度学习对于此类验证码识别的实现。</p>
<p>滑块验证码是怎样的呢？如图所示，验证码是一张矩形图，图片左侧会出现一个滑块，右侧会出现一个缺口，下侧会出现一个滑轨。左侧的滑块会随着滑轨的拖动而移动，如果能将左侧滑块匹配滑动到右侧缺口处，就算完成了验证。</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/12/3A/Ciqah16dWvaAFI32AAHHgbnOPXI004.png" alt="1.png"></p>
<p>由于这种验证码交互形式比较友好，且安全性、美观度上也会更高，像这种类似的验证码也变得越来越流行。另外不仅仅是“极验”，其他很多验证码服务商也推出了类似的验证码服务，如“网易易盾”等，上图所示的就是“网易易盾”的滑动验证码。</p>
<p>没错，这种滑动验证码的出现确实让很多网站变得更安全。但是做爬虫的可就苦恼了，如果想采用自动化的方法来绕过这种滑动验证码，关键点在于以下两点：</p>
<ul>
<li>找出目标缺口的位置。</li>
<li>模拟人的滑动轨迹将滑块滑动到缺口处。</li>
</ul>
<p>那么问题来了，第一步怎么做呢？</p>
<p>接下来我们就来看看如何利用深度学习来实现吧。</p>
<h4>目标检测</h4>
<p>我们的目标就是输入一张图，输出缺口的的位置，所以只需要将这个问题归结成一个深度学习的“目标检测”问题就好了。</p>
<p>首先在开始之前简单说下目标检测。什么叫目标检测？顾名思义，就是把我们想找的东西找出来。比如给一张“狗”的图片，如图所示：</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/05/0B/CgoCgV6dWxOAPxb8AAehD7cahYQ261.png" alt="2.png"></p>
<p>我们想知道这只狗在哪，它的舌头在哪，找到了就把它们框选出来，这就是目标检测。</p>
<p>经过目标检测算法处理之后，我们期望得到的图片是这样的：</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/12/3A/Ciqah16dWx2AUveZAAfLKj2mbM0144.png" alt="3.png"></p>
<p>可以看到这只狗和它的舌头就被框选出来了，这样就完成了一个不错的目标检测。</p>
<p>当前做目标检测的算法主要有两个方向，有一阶段式和两阶段式，英文分别叫作 One stage 和 Two stage，简述如下。</p>
<ul>
<li>Two Stage：算法首先生成一系列目标所在位置的候选框，然后再对这些框选出来的结果进行样本分类，即先找出来在哪，然后再分出来是什么，俗话说叫“看两眼”，这种算法有 R-CNN、Fast R-CNN、Faster R-CNN 等，这些算法架构相对复杂，但准确率上有优势。</li>
<li>One Stage：不需要产生候选框，直接将目标定位和分类的问题转化为回归问题，俗话说叫“看一眼”，这种算法有 YOLO、SSD，这些算法虽然准确率上不及 Two stage，但架构相对简单，检测速度更快。</li>
</ul>
<p>所以这次我们选用 One Stage 的有代表性的目标检测算法 YOLO 来实现滑动验证码缺口的识别。</p>
<p>YOLO，英文全称叫作 You Only Look Once，取了它们的首字母就构成了算法名，目前 YOLO 算法最新的版本是 V3 版本，这里算法的具体流程我们就不过多介绍了，如果你感兴趣可以搜一下相关资料了解下，另外也可以了解下 YOLO V1～V3 版本的不同和改进之处，这里列几个参考链接。</p>
<ul>
<li>YOLO V3 论文：<a href="https://pjreddie.com/media/files/papers/YOLOv3.pdf">https://pjreddie.com/media/files/papers/YOLOv3.pdf</a></li>
<li>YOLO V3 介绍：<a href="https://zhuanlan.zhihu.com/p/34997279">https://zhuanlan.zhihu.com/p/34997279</a></li>
<li>YOLO V1-V3 对比介绍：<a href="https://www.cnblogs.com/makefile/p/yolov3.html">https://www.cnblogs.com/makefile/p/yolov3.html</a></li>
</ul>
<h4>数据准备</h4>
<p>回归我们本课时的主题，我们要做的是缺口的位置识别，那么第一步应该做什么呢？</p>
<p>我们的目标是要训练深度学习模型，那我们总得需要让模型知道要学点什么东西吧，这次我们做缺口识别，那么我们需要让模型学的就是找到这个缺口在哪里。由于一张验证码图片只有一个缺口，要分类就是一类，所以我们只需要找到缺口位置就行了。</p>
<p>好，那模型要学如何找出缺口的位置，就需要我们提供样本数据让模型来学习才行。样本数据怎样的呢？样本数据就得有带缺口的验证码图片以及我们自己标注的缺口位置。只有把这两部分都告诉模型，模型才能去学习。等模型学好了，当我们再给个新的验证码时，就能检测出缺口在哪里了，这就是一个成功的模型。</p>
<p>OK，那我们就开始准备数据和缺口标注结果吧。</p>
<p>数据这里用的是网易盾的验证码，验证码图片可以自行收集，写个脚本批量保存下来就行。标注的工具可以使用 LabelImg，GitHub 链接为：<a href="https://github.com/tzutalin/labelImg">https://github.com/tzutalin/labelImg</a>，利用它我们可以方便地进行检测目标位置的标注和类别的标注，如这里验证码和标注示例如下：</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/8B/50/Cgq2xl6dW4OAffxPAAmrTFP3fXg210.png" alt="4.png"></p>
<p>标注完了会生成一系列 xml 文件，你需要解析 xml 文件把位置的坐标和类别等处理一下，转成训练模型需要的数据。</p>
<p>在这里我已经整理好了我的数据集，完整 GitHub 链接为：<a href="https://github.com/Python3WebSpider/DeepLearningSlideCaptcha">https://github.com/Python3WebSpider/DeepLearningSlideCaptcha</a>，我标注了 200 多张图片，然后处理了 xml 文件，变成训练 YOLO 模型需要的数据格式，验证码图片和标注结果见 data/captcha 文件夹。</p>
<p>如果要训练自己的数据，数据格式准备见：<a href="https://github.com/eriklindernoren/PyTorch-YOLOv3#train-on-custom-dataset">https://github.com/eriklindernoren/PyTorch-YOLOv3#train-on-custom-dataset</a></p>
<h4>初始化</h4>
<p>上一步我已经把标注好的数据处理好了，可以直接拿来训练了。</p>
<p>由于 YOLO 模型相对比较复杂，所以这个项目我就直接基于开源的 PyTorch-YOLOV3 项目来进行修改了，模型使用的深度学习框架为 PyTorch，具体的 YOLO V3 模型的实现这里不再阐述了。</p>
<p>另外推荐使用 GPU 训练，不然拿 CPU 直接训练速度会很慢。我的 GPU 是 P100，几乎十几秒就训练完一轮。</p>
<p>下面就直接把代码克隆下来吧。</p>
<p>由于本项目我把训练好的模型也放上去了，使用了 Git LFS，所以克隆时间较长，克隆命令如下：</p>
<pre><code data-language="js" class="lang-js">git clone https:<span class="hljs-comment">//github.com/Python3WebSpider/DeepLearningSlideCaptcha.git</span>
</code></pre>
<p>如果想加速克隆，可以暂时先跳过大文件模型下载，可以执行命令：</p>
<pre><code data-language="js" class="lang-js">GIT_LFS_SKIP_SMUDGE=<span class="hljs-number">1</span> git clone https:<span class="hljs-comment">//github.com/Python3WebSpider/DeepLearningSlideCaptcha.git</span>
</code></pre>
<h4>环境安装</h4>
<p>代码克隆下载之后，我们还需要下载一些预训练模型。</p>
<p>YOLOV3 的训练要加载预训练模型才能有不错的训练效果，预训练模型下载命令如下：</p>
<pre><code data-language="js" class="lang-js">bash prepare.sh
</code></pre>
<p>执行这个脚本，就能下载 YOLO V3 模型的一些权重文件，包括 yolov3 和 weights，还有 darknet 的 weights，在训练之前我们需要用这些权重文件初始化 YOLO V3 模型。</p>
<blockquote>
<p>注意：Windows 下建议使用 Git Bash 来运行上述命令。</p>
</blockquote>
<p>另外还需要安装一些必须的库，如 PyTorch、TensorBoard 等，建议使用 Python 虚拟环境，运行命令如下：</p>
<pre><code data-language="js" class="lang-js">pip3 install -r requirements.txt
</code></pre>
<p>这些库都安装好了之后，就可以开始训练了。</p>
<h4>训练</h4>
<p>本项目已经提供了标注好的数据集，在 data/captcha，可以直接使用。</p>
<p>当前数据训练脚本：</p>
<pre><code data-language="js" class="lang-js">bash train.sh
</code></pre>
<p>实测 P100 训练时长约 15 秒一个 epoch，大约几分钟即可训练出较好效果。</p>
<p>训练差不多了，我们便可以使用 TensorBoard 来看看 loss 和 mAP 的变化，运行 TensorBoard：</p>
<pre><code data-language="js" class="lang-js">tensorboard --logdir=<span class="hljs-string">'logs'</span> --port=<span class="hljs-number">6006</span> --host <span class="hljs-number">0.0</span><span class="hljs-number">.0</span><span class="hljs-number">.0</span>
</code></pre>
<p>loss_1 变化如下：</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/05/0D/CgoCgV6dXTKAfgtfAAA6KqSjdxM383.png" alt="5.png"></p>
<p>val_mAP 变化如下：</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/12/3C/Ciqah16dXTqAbsXfAABBNhdjyNM911.png" alt="6.png"></p>
<p>可以看到 loss 从最初的非常高下降到了很低，准确率也逐渐接近 100%。</p>
<p>另外训练过程中还能看到如下的输出结果：</p>
<pre><code data-language="js" class="lang-js">---- [Epoch <span class="hljs-number">99</span>/<span class="hljs-number">100</span>, Batch <span class="hljs-number">27</span>/<span class="hljs-number">29</span>] ----
+------------+--------------+--------------+--------------+
| Metrics    | YOLO Layer <span class="hljs-number">0</span> | YOLO Layer <span class="hljs-number">1</span> | YOLO Layer <span class="hljs-number">2</span> |
+------------+--------------+--------------+--------------+
| grid_size  | <span class="hljs-number">14</span>           | <span class="hljs-number">28</span>           | <span class="hljs-number">56</span>           |
| loss       | <span class="hljs-number">0.028268</span>     | <span class="hljs-number">0.046053</span>     | <span class="hljs-number">0.043745</span>     |
| x          | <span class="hljs-number">0.002108</span>     | <span class="hljs-number">0.005267</span>     | <span class="hljs-number">0.008111</span>     |
| y          | <span class="hljs-number">0.004561</span>     | <span class="hljs-number">0.002016</span>     | <span class="hljs-number">0.009047</span>     |
| w          | <span class="hljs-number">0.001284</span>     | <span class="hljs-number">0.004618</span>     | <span class="hljs-number">0.000207</span>     |
| h          | <span class="hljs-number">0.000594</span>     | <span class="hljs-number">0.000528</span>     | <span class="hljs-number">0.000946</span>     |
| conf       | <span class="hljs-number">0.019700</span>     | <span class="hljs-number">0.033624</span>     | <span class="hljs-number">0.025432</span>     |
| cls        | <span class="hljs-number">0.000022</span>     | <span class="hljs-number">0.000001</span>     | <span class="hljs-number">0.000002</span>     |
| cls_acc    | <span class="hljs-number">100.00</span>%      | <span class="hljs-number">100.00</span>%      | <span class="hljs-number">100.00</span>%      |
| recall50   | <span class="hljs-number">1.000000</span>     | <span class="hljs-number">1.000000</span>     | <span class="hljs-number">1.000000</span>     |
| recall75   | <span class="hljs-number">1.000000</span>     | <span class="hljs-number">1.000000</span>     | <span class="hljs-number">1.000000</span>     |
| precision  | <span class="hljs-number">1.000000</span>     | <span class="hljs-number">0.800000</span>     | <span class="hljs-number">0.666667</span>     |
| conf_obj   | <span class="hljs-number">0.994271</span>     | <span class="hljs-number">0.999249</span>     | <span class="hljs-number">0.997762</span>     |
| conf_noobj | <span class="hljs-number">0.000126</span>     | <span class="hljs-number">0.000158</span>     | <span class="hljs-number">0.000140</span>     |
+------------+--------------+--------------+--------------+
Total loss <span class="hljs-number">0.11806630343198776</span>
</code></pre>
<p>这里显示了训练过程中各个指标的变化情况，如 loss、recall、precision、confidence 等，分别代表训练过程的损失（越小越好）、召回率（能识别出的结果占应该识别出结果的比例，越高越好）、精确率（识别出的结果中正确的比率，越高越好）、置信度（模型有把握识别对的概率，越高越好），可以作为参考。</p>
<h4>测试</h4>
<p>训练完毕之后会在 checkpoints 文件夹生成 pth 文件，可直接使用模型来预测生成标注结果。</p>
<p>如果你没有训练自己的模型的话，这里我已经把训练好的模型放上去了，可以直接使用我训练好的模型来测试。如之前跳过了 Git LFS 文件下载，则可以使用如下命令下载 Git LFS 文件：</p>
<pre><code data-language="js" class="lang-js">git lfs pull
</code></pre>
<p>此时 checkpoints 文件夹会生成训练好的 pth 文件。</p>
<p>测试脚本：</p>
<pre><code data-language="js" class="lang-js">sh detect.sh
</code></pre>
<p>该脚本会读取 captcha 下的 test 文件夹所有图片，并将处理后的结果输出到 result 文件夹。</p>
<p>运行结果样例：</p>
<pre><code data-language="js" class="lang-js">Performing object detection:
        + Batch <span class="hljs-number">0</span>, Inference Time: <span class="hljs-number">0</span>:<span class="hljs-number">00</span>:<span class="hljs-number">00.044223</span>
        + Batch <span class="hljs-number">1</span>, Inference Time: <span class="hljs-number">0</span>:<span class="hljs-number">00</span>:<span class="hljs-number">00.028566</span>
        + Batch <span class="hljs-number">2</span>, Inference Time: <span class="hljs-number">0</span>:<span class="hljs-number">00</span>:<span class="hljs-number">00.029764</span>
        + Batch <span class="hljs-number">3</span>, Inference Time: <span class="hljs-number">0</span>:<span class="hljs-number">00</span>:<span class="hljs-number">00.032430</span>
        + Batch <span class="hljs-number">4</span>, Inference Time: <span class="hljs-number">0</span>:<span class="hljs-number">00</span>:<span class="hljs-number">00.033373</span>
        + Batch <span class="hljs-number">5</span>, Inference Time: <span class="hljs-number">0</span>:<span class="hljs-number">00</span>:<span class="hljs-number">00.027861</span>
        + Batch <span class="hljs-number">6</span>, Inference Time: <span class="hljs-number">0</span>:<span class="hljs-number">00</span>:<span class="hljs-number">00.031444</span>
        + Batch <span class="hljs-number">7</span>, Inference Time: <span class="hljs-number">0</span>:<span class="hljs-number">00</span>:<span class="hljs-number">00.032110</span>
        + Batch <span class="hljs-number">8</span>, Inference Time: <span class="hljs-number">0</span>:<span class="hljs-number">00</span>:<span class="hljs-number">00.029131</span>

Saving images:
(<span class="hljs-number">0</span>) Image: <span class="hljs-string">'data/captcha/test/captcha_4497.png'</span>
        + Label: target, <span class="hljs-attr">Conf</span>: <span class="hljs-number">0.99999</span>
(<span class="hljs-number">1</span>) Image: <span class="hljs-string">'data/captcha/test/captcha_4498.png'</span>
        + Label: target, <span class="hljs-attr">Conf</span>: <span class="hljs-number">0.99999</span>
(<span class="hljs-number">2</span>) Image: <span class="hljs-string">'data/captcha/test/captcha_4499.png'</span>
        + Label: target, <span class="hljs-attr">Conf</span>: <span class="hljs-number">0.99997</span>
(<span class="hljs-number">3</span>) Image: <span class="hljs-string">'data/captcha/test/captcha_4500.png'</span>
        + Label: target, <span class="hljs-attr">Conf</span>: <span class="hljs-number">0.99999</span>
(<span class="hljs-number">4</span>) Image: <span class="hljs-string">'data/captcha/test/captcha_4501.png'</span>
        + Label: target, <span class="hljs-attr">Conf</span>: <span class="hljs-number">0.99997</span>
(<span class="hljs-number">5</span>) Image: <span class="hljs-string">'data/captcha/test/captcha_4502.png'</span>
        + Label: target, <span class="hljs-attr">Conf</span>: <span class="hljs-number">0.99999</span>
(<span class="hljs-number">6</span>) Image: <span class="hljs-string">'data/captcha/test/captcha_4503.png'</span>
        + Label: target, <span class="hljs-attr">Conf</span>: <span class="hljs-number">0.99997</span>
(<span class="hljs-number">7</span>) Image: <span class="hljs-string">'data/captcha/test/captcha_4504.png'</span>
        + Label: target, <span class="hljs-attr">Conf</span>: <span class="hljs-number">0.99998</span>
(<span class="hljs-number">8</span>) Image: <span class="hljs-string">'data/captcha/test/captcha_4505.png'</span>
        + Label: target, <span class="hljs-attr">Conf</span>: <span class="hljs-number">0.99998</span>
</code></pre>
<p>拿几个样例结果看下：</p>
<p><img src="https://s0.lgstatic.com/i/image3/M01/05/0E/CgoCgV6dXamAbOLDAALPfVGbpFE467.png" alt="7.png"><br>
<img src="https://s0.lgstatic.com/i/image3/M01/8B/53/Cgq2xl6dXbGAOPAzAAKfofXm6vo290.png" alt="8.png"><br>
<img src="https://s0.lgstatic.com/i/image3/M01/8B/53/Cgq2xl6dXbeAKiqQAALeTg7slfU302.png" alt="9.png"></p>
<p>这里我们可以看到，利用训练好的模型我们就成功识别出缺口的位置了，另外程序还会打印输出这个边框的中心点和宽高信息。</p>
<p>有了这个边界信息，我们再利用某些手段拖动滑块即可通过验证了，比如可以模拟加速减速过程，或者可以录制人的轨迹再执行都是可以的，由于本课时更多是介绍深度学习识别相关内容，所以关于拖动轨迹不再展开讲解。</p>
<h4>总结</h4>
<p>本课时我们介绍了使用深度学习识别滑动验证码缺口的方法，包括标注、训练、测试等环节都进行了阐述。有了它，我们就能轻松方便地对缺口进行识别了。</p>
<p>代码：<a href="https://github.com/Python3WebSpider/DeepLearningSlideCaptcha">https://github.com/Python3WebSpider/DeepLearningSlideCaptcha</a></p>

