---
title: 52讲轻松搞定网络爬虫笔记6
tags: [Web Crawler]
categories: data analysis
date: 2023-1-15
---

# 资料
[52讲轻松搞定网络爬虫](https://kaiwu.lagou.com/course/courseInfo.htm?courseId=46#/sale)


# 你有权限吗？解析模拟登录基本原理
<p>在很多情况下，一些网站的页面或资源我们通常需要登录才能看到。比如访问 GitHub 的个人设置页面，如果不登录是无法查看的；比如 12306 买票提交订单的页面，如果不登录是无法提交订单的；再比如要发一条微博，如果不登录是无法发送的。</p>
<p>我们之前学习的案例都是爬取的无需登录即可访问的站点，但是诸如上面例子的情况非常非常多，那假如我们想要用爬虫来访问这些页面，比如用爬虫修改 GitHub 的个人设置，用爬虫提交购票订单，用爬虫发微博，能做到吗？</p>
<p>答案是可以，这里就需要用到一些模拟登录相关的技术了。那么本课时我们就先来了解模拟登录的一些基本原理和实现吧。</p>
<h3>网站登录验证的实现</h3>
<p>我们要实现模拟登录，那就得首先了解网站登录验证的实现。</p>
<p>登录一般需要两个内容，用户名和密码，有的网站可能是手机号和验证码，有的是微信扫码，有的是 OAuth 验证等等，但根本上来说，都是把一些可供认证的信息提交给了服务器。</p>
<p>比如这里我们就拿用户名和密码来举例吧。用户在一个网页表单里面输入了内容，然后点击登录按钮的一瞬间，浏览器客户端就会向服务器发送一个登录请求，这个请求里面肯定就包含了用户名和密码信息，这时候，服务器需要处理这些信息，然后返回给客户端一个类似“凭证”的东西，有了这个“凭证”以后呢，客户端拿着这个“凭证”再去访问某些需要登录才能查看的页面，服务器自然就能“放行”了，然后返回对应的内容或执行对应的操作就好了。</p>
<p>形象地说，我们以登录发微博和买票坐火车这两件事来类比。发微博就好像要坐火车，没票是没法坐火车的吧，要坐火车怎么办呢？当然是先买票了，我们拿钱去火车站买了票，有了票之后，进站口查验一下，没问题就自然能去坐火车了，这个票就是坐火车的“凭证”。</p>
<p>发微博也一样，我们有用户名和密码，请求下服务器，获得一个“凭证”，这就相当于买到了火车票，然后在发微博的时候拿着这个“凭证”去请求服务器，服务器校验没问题，自然就把微博发出去了。</p>
<p>那么问题来了，这个“凭证“”到底是怎么生成和验证的呢？目前比较流行的实现方式有两种，一种是基于 Session + Cookies 的验证，一种是基于 JWT（JSON Web Token）的验证，下面我们来介绍下。</p>
<h3>Session 和 Cookies</h3>
<p>我们在模块一了解了 Session 和 Cookies 的基本概念。简而言之，Session 就是存在服务端的，里面保存了用户此次访问的会话信息，Cookies 则是保存在用户本地浏览器的，它会在每次用户访问网站的时候发送给服务器，Cookies 会作为 Request Headers 的一部分发送给服务器，服务器根据 Cookies 里面包含的信息判断找出其 Session 对象，不同的 Session 对象里面维持了不同访问用户的状态，服务器可以根据这些信息决定返回 Response 的内容。</p>
<p>我们以用户登录的情形来举例，其实不同的网站对于用户的登录状态的实现可能是不同的，但是 Session 和 Cookies 一定是相互配合工作的。</p>
<p>梳理如下：</p>
<ul>
<li>比如，Cookies 里面可能只存了 Session ID 相关信息，服务器能根据 Cookies 找到对应的 Session，用户登录之后，服务器会在对应的 Session 里面标记一个字段，代表已登录状态或者其他信息（如角色、登录时间）等等，这样用户每次访问网站的时候都带着 Cookies 来访问，服务器就能找到对应的 Session，然后看一下 Session 里面的状态是登录状态，就可以返回对应的结果或执行某些操作。</li>
<li>当然 Cookies 里面也可能直接存了某些凭证信息。比如说用户在发起登录请求之后，服务器校验通过，返回给客户端的 Response Headers 里面可能带有 Set-Cookie 字段，里面可能就包含了类似凭证的信息，这样客户端会执行 Set Cookie 的操作，将这些信息保存到 Cookies 里面，以后再访问网页时携带这些 Cookies 信息，服务器拿着这里面的信息校验，自然也能实现登录状态检测了。</li>
</ul>
<p>以上两种情况几乎能涵盖大部分的 Session 和 Cookies 登录验证的实现，具体的实现逻辑因服务器而异，但 Session 和 Cookies 一定是需要相互配合才能实现的。</p>
<h3>JWT</h3>
<p>Web 开发技术是一直在发展的，近几年前后端分离的趋势越来越火，很多 Web 网站都采取了前后端分离的技术来实现。而且传统的基于 Session 和 Cookies 的校验也存在一定问题，比如服务器需要维护登录用户的 Session 信息，而且不太方便分布式部署，也不太适合前后端分离的项目。</p>
<p>所以，JWT 技术应运而生。JWT，英文全称叫作 JSON Web Token，是为了在网络应用环境间传递声明而执行的一种基于 JSON 的开放标准。实际上就是每次登录的时候通过一个 Token 字符串来校验登录状态。</p>
<p>JWT 的声明一般被用来在身份提供者和服务提供者间传递被认证的用户身份信息，以便于从资源服务器获取资源，也可以增加一些额外的其他业务逻辑所必须的声明信息，所以这个 Token 也可直接被用于认证，也可传递一些额外信息。</p>
<p>有了 JWT，一些认证就不需要借助于 Session 和 Cookies 了，服务器也无需维护 Session 信息，减少了服务器的开销。服务器只需要有一个校验 JWT 的功能就好了，同时也可以做到分布式部署和跨语言的支持。</p>
<p>JWT 通常就是一个加密的字符串，它也有自己的标准，类似下面的这种格式：</p>
<pre><code>eyJ0eXAxIjoiMTIzNCIsImFsZzIiOiJhZG1pbiIsInR5cCI6IkpXVCIsImFsZyI6IkhTMjU2In0.eyJVc2VySWQiOjEyMywiVXNlck5hbWUiOiJhZG1pbiIsImV4cCI6MTU1MjI4Njc0Ni44Nzc0MDE4fQ.pEgdmFAy73walFonEm2zbxg46Oth3dlT02HR9iVzXa8
</code></pre>
<p>可以发现中间有两个“.”来分割开，可以把它看成是一个三段式的加密字符串。它由三部分构成，分别是 Header、Payload、Signature。</p>
<ul>
<li>Header，声明了 JWT 的签名算法，如 RSA、SHA256 等等，也可能包含 JWT 编号或类型等数据，然后整个信息 Base64 编码即可。</li>
<li>Payload，通常用来存放一些业务需要但不敏感的信息，如 UserID 等，另外它也有很多默认的字段，如 JWT 签发者、JWT 接受者、JWT 过期时间等等，Base64 编码即可。</li>
<li>Signature，这个就是一个签名，是把 Header、Payload 的信息用秘钥 secret 加密后形成的，这个 secret 是保存在服务器端的，不能被轻易泄露。这样的话，即使一些 Payload 的信息被篡改，服务器也能通过 Signature 判断出来是非法请求，拒绝服务。</li>
</ul>
<p>这三部分通过“.”组合起来就形成了 JWT 的字符串，就是用户的访问凭证。</p>
<p>所以这个登录认证流程也很简单了，用户拿着用户名密码登录，然后服务器生成 JWT 字符串返回给客户端，客户端每次请求都带着这个 JWT 就行了，服务器会自动判断其有效情况，如果有效，那自然就返回对应的数据。但 JWT 的传输就多种多样了，可以放在 Request Headers，也可以放在 URL 里，甚至有的网站也放在 Cookies 里，但总而言之，能传给服务器校验就好了。</p>
<p>好，到此为止呢，我们就已经了解了网站登录验证的实现了。</p>
<h3>模拟登录</h3>
<p>好，了解了网站登录验证的实现后，模拟登录自然就有思路了。下面我们也是分两种认证方式来说明。</p>
<h4>Session 和 Cookies</h4>
<p>基于 Session 和 Cookies 的模拟登录，如果我们要用爬虫实现的话，其实最主要的就是把 Cookies 的信息维护好，因为爬虫就相当于客户端浏览器，我们模拟好浏览器做的事情就好了。</p>
<p>那一般情况下，模拟登录一般可以怎样实现呢，我们结合之前所讲的技术来总结一下：</p>
<ul>
<li>第一，如果我们已经在浏览器里面登录了自己的账号，我们要想用爬虫模拟的话，可以直接把 Cookies 复制过来交给爬虫就行了，这也是最省事省力的方式。这就相当于，我们用浏览器手动操作登录了，然后把 Cookies 拿过来放到代码里面，爬虫每次请求的时候把 Cookies 放到 Request Headers 里面，就相当于完全模拟了浏览器的操作，服务器会通过 Cookies 校验登录状态，如果没问题，自然可以执行某些操作或返回某些内容了。</li>
<li>第二，如果我们不想有任何手工操作，可以直接使用爬虫来模拟登录过程。登录的过程其实多数也是一个 POST 请求，我们用爬虫提交用户名密码等信息给服务器，服务器返回的 Response Headers 里面可能带了 Set-Cookie 的字段，我们只需要把这些 Cookies 保存下来就行了。所以，最主要的就是把这个过程中的 Cookies 维护好就行了。当然这里可能会遇到一些困难，比如登录过程还伴随着各种校验参数，不好直接模拟请求，也可能网站设置 Cookies 的过程是通过 JavaScript 实现的，所以可能还得仔细分析下其中的一些逻辑，尤其是我们用 requests 这样的请求库进行模拟登录的时候，遇到的问题可能比较多。</li>
<li>第三，我们也可以用一些简单的方式来实现模拟登录，即把人在浏览器中手工登录的过程自动化实现，比如我们用 Selenium 或 Pyppeteer 来驱动浏览器模拟执行一些操作，如填写用户名、密码和表单提交等操作，等待登录成功之后，通过 Selenium 或 Pyppeteer 获取当前浏览器的 Cookies 保存起来即可。然后后续的请求可以携带 Cookies 的内容请求，同样也能实现模拟登录。</li>
</ul>
<p>以上介绍的就是一些常用的爬虫模拟登录的方案，其目的就是维护好客户端的 Cookies 信息，然后每次请求都携带好 Cookies 信息就能实现模拟登录了。</p>
<h4>JWT</h4>
<p>基于 JWT 的真实情况也比较清晰了，由于 JWT 的这个字符串就是用户访问的凭证，那么模拟登录只需要做到下面几步即可：</p>
<ul>
<li>第一，模拟网站登录操作的请求，比如携带用户名和密码信息请求登录接口，获取服务器返回结果，这个结果中通常包含 JWT 字符串的信息，保存下来即可。</li>
<li>第二，后续的请求携带 JWT 访问即可，一般情况在 JWT 不过期的情况下都能正常访问和执行对应的操作。携带方式多种多样，因网站而异。</li>
<li>第三，如果 JWT 过期了，可能需要重复步骤一，重新获取 JWT。</li>
</ul>
<p>当然这个模拟登录的过程也肯定带有其他的一些加密参数，需要根据实际情况具体分析。</p>
<h3>优化方案</h3>
<p>如果爬虫要求爬取的数据量比较大或爬取速度比较快，而网站又有单账号并发限制或者访问状态检测并反爬的话，可能我们的账号就会无法访问或者面临封号的风险了。这时候一般怎么办呢？</p>
<p>我们可以使用分流的方案来解决，比如某个网站一分钟之内检测一个账号只能访问三次或者超过三次就封号的话，我们可以建立一个账号池，用多个账号来随机访问或爬取，这样就能数倍提高爬虫的并发量或者降低被封的风险了。</p>
<p>比如在访问某个网站的时候，我们可以准备 100 个账号，然后 100 个账号都模拟登录，把对应的 Cookies 或 JWT 存下来，每次访问的时候随机取一个来访问，由于账号多，所以每个账号被取用的概率也就降下来了，这样就能避免单账号并发过大的问题，也降低封号风险。</p>
<p>以上，我们就介绍完了模拟登录的基本原理和实现以及优化方案，希望你可以好好理解。</p>

# 模拟登录爬取实战案例
<p data-nodeid="21526">在上一课时我们了解了网站登录验证和模拟登录的基本原理。网站登录验证主要有两种实现，一种是基于 Session + Cookies 的登录验证，另一种是基于 JWT 的登录验证，那么本课时我们就通过两个实例来分别讲解这两种登录验证的分析和模拟登录流程。</p>
<h4 data-nodeid="21527">准备工作</h4>
<p data-nodeid="21528">在本课时开始之前，请你确保已经做好了如下准备工作：</p>
<ul data-nodeid="21529">
<li data-nodeid="21530">
<p data-nodeid="21531">安装好了 Python （最好 3.6 及以上版本）并能成功运行 Python 程序；</p>
</li>
<li data-nodeid="21532">
<p data-nodeid="21533">安装好了 requests 请求库并学会了其基本用法；</p>
</li>
<li data-nodeid="21534">
<p data-nodeid="21535">安装好了 Selenium 库并学会了其基本用法。</p>
</li>
</ul>
<p data-nodeid="21536">下面我们就以两个案例为例来分别讲解模拟登录的实现。</p>
<h4 data-nodeid="21537">案例介绍</h4>
<p data-nodeid="21538">这里有两个需要登录才能抓取的网站，链接为 <a href="https://login2.scrape.center/" data-nodeid="21621">https://login2.scrape.center/</a> 和 <a href="https://login3.scrape.center/" data-nodeid="21625">https://login3.scrape.center/</a>，前者是基于 Session + Cookies 认证的网站，后者是基于 JWT 认证的网站。</p>
<p data-nodeid="21539">首先看下第一个网站，打开后会看到如图所示的页面。<br>
<img src="https://s0.lgstatic.com/i/image3/M01/16/03/Ciqah16lJAqAS2wvAADcb_q6Bz8267.png" alt="image.png" data-nodeid="21631"><br>
它直接跳转到了登录页面，这里用户名和密码都是 admin，我们输入之后登录。</p>
<p data-nodeid="21540">登录成功之后，我们便看到了熟悉的电影网站的展示页面，如图所示。<br>
<img src="https://s0.lgstatic.com/i/image3/M01/16/03/Ciqah16lJCiAMPHfAAUyMc8cA9g219.png" alt="image (1).png" data-nodeid="21638"></p>
<p data-nodeid="21541">这个网站是基于传统的 MVC 模式开发的，因此也比较适合 Session + Cookies 的认证。</p>
<p data-nodeid="21542">第二个网站打开后同样会跳到登录页面，如图所示。</p>
<p data-nodeid="21543"><img src="https://s0.lgstatic.com/i/image3/M01/16/03/Ciqah16lJDKAbqUOAADYULf8h7E835.png" alt="image (2).png" data-nodeid="21643"><br>
用户名和密码是一样的，都输入 admin 即可登录。</p>
<p data-nodeid="21544">登录之后会跳转到首页，展示了一些书籍信息，如图所示。<br>
<img src="https://s0.lgstatic.com/i/image3/M01/08/D4/CgoCgV6lJDyAWsKlABEdKYqGoMg374.png" alt="image (3).png" data-nodeid="21650"><br>
这个页面是前后端分离式的页面，数据的加载都是通过 Ajax 请求后端 API 接口获取，登录的校验是基于 JWT 的，同时后端每个 API 都会校验 JWT 是否是有效的，如果无效则不会返回数据。</p>
<p data-nodeid="21545">接下来我们就分析这两个案例并实现模拟登录吧。</p>
<h4 data-nodeid="21546">案例一</h4>
<p data-nodeid="21547">对于案例一，我们如果要模拟登录，就需要先分析下登录过程究竟发生了什么，首先我们打开 <a href="https://login2.scrape.center/" data-nodeid="21658">https://login2.scrape.center/</a>，然后执行登录操作，查看其登录过程中发生的请求，如图所示。</p>
<p data-nodeid="21548"><img src="https://s0.lgstatic.com/i/image3/M01/16/03/Ciqah16lJEiACmWwAAP81rGCv5M937.png" alt="image (4).png" data-nodeid="21662"><br>
这里我们可以看到其登录的瞬间是发起了一个 POST 请求，目标 URL 为 <a href="https://login2.scrape.center/login" data-nodeid="21667">https://login2.scrape.center/login</a>，通过表单提交的方式提交了登录数据，包括 username 和 password 两个字段，返回的状态码是 302，Response Headers 的 location 字段是根页面，同时 Response Headers 还包含了 set-cookie 信息，设置了 Session ID。</p>
<p data-nodeid="21549">由此我们可以发现，要实现模拟登录，我们只需要模拟这个请求就好了，登录完成之后获取 Response 设置的 Cookies，将 Cookies 保存好，以后后续的请求带上 Cookies 就可以正常访问了。</p>
<p data-nodeid="21550">好，那么我们接下来用代码实现一下吧。</p>
<p data-nodeid="21551">requests 默认情况下每次请求都是独立互不干扰的，比如我们第一次先调用了 post 方法模拟登录，然后紧接着再调用 get 方法请求下主页面，其实这是两个完全独立的请求，第一次请求获取的 Cookies 并不能传给第二次请求，因此说，常规的顺序调用是不能起到模拟登录的效果的。</p>
<p data-nodeid="21552">我们先来看一个无效的代码：</p>
<pre class="lang-js" data-nodeid="21553"><code data-language="js"><span class="hljs-keyword">import</span> requests
<span class="hljs-keyword">from</span> urllib.parse <span class="hljs-keyword">import</span> urljoin

BASE_URL = <span class="hljs-string">'https://login2.scrape.center/'</span>
LOGIN_URL = urljoin(BASE_URL, <span class="hljs-string">'/login'</span>)
INDEX_URL = urljoin(BASE_URL, <span class="hljs-string">'/page/1'</span>)
USERNAME = <span class="hljs-string">'admin'</span>
PASSWORD = <span class="hljs-string">'admin'</span>

response_login = requests.post(LOGIN_URL, data={
   <span class="hljs-string">'username'</span>: USERNAME,
   <span class="hljs-string">'password'</span>: PASSWORD
})

response_index = requests.get(INDEX_URL)
print(<span class="hljs-string">'Response Status'</span>, response_index.status_code)
print(<span class="hljs-string">'Response URL'</span>, response_index.url)
</code></pre>
<p data-nodeid="21554">这里我们先定义了几个基本的 URL 和用户名、密码，接下来分别用 requests 请求了登录的 URL 进行模拟登录，然后紧接着请求了首页来获取页面内容，但是能正常获取数据吗？</p>
<p data-nodeid="21555">由于 requests 可以自动处理重定向，我们最后把 Response 的 URL 打印出来，如果它的结果是 INDEX_URL，那么就证明模拟登录成功并成功爬取到了首页的内容。如果它跳回到了登录页面，那就说明模拟登录失败。</p>
<p data-nodeid="21556">我们通过结果来验证一下，运行结果如下：</p>
<pre class="lang-js" data-nodeid="21557"><code data-language="js">Response Status <span class="hljs-number">200</span>
Response URL https:<span class="hljs-comment">//login2.scrape.center/login?next=/page/1</span>
</code></pre>
<p data-nodeid="21558">这里可以看到，其最终的页面 URL 是登录页面的 URL，另外这里也可以通过 response 的 text 属性来验证页面源码，其源码内容就是登录页面的源码内容，由于内容较多，这里就不再输出比对了。</p>
<p data-nodeid="21559">总之，这个现象说明我们并没有成功完成模拟登录，这是因为 requests 直接调用 post、get 等方法，每次请求都是一个独立的请求，都相当于是新开了一个浏览器打开这些链接，这两次请求对应的 Session 并不是同一个，因此这里我们模拟了第一个 Session 登录，而这并不能影响第二个 Session 的状态，因此模拟登录也就无效了。<br>
那么怎样才能实现正确的模拟登录呢？</p>
<p data-nodeid="21560">我们知道 Cookies 里面是保存了 Session ID 信息的，刚才也观察到了登录成功后 Response Headers 里面是有 set-cookie 字段，实际上这就是让浏览器生成了 Cookies。</p>
<p data-nodeid="21561">Cookies 里面包含了 Session ID 的信息，所以只要后续的请求携带这些 Cookies，服务器便能通过 Cookies 里的 Session ID 信息找到对应的 Session，因此服务端对于这两次请求就会使用同一个 Session 了。而因为第一次我们已经完成了模拟登录，所以第一次模拟登录成功后，Session 里面就记录了用户的登录信息，第二次访问的时候，由于是同一个 Session，服务器就能知道用户当前是登录状态，就可以返回正确的结果而不再是跳转到登录页面了。</p>
<p data-nodeid="21562">所以，这里的关键就在于两次请求的 Cookies 的传递。所以这里我们可以把第一次模拟登录后的 Cookies 保存下来，在第二次请求的时候加上这个 Cookies 就好了，所以代码可以改写如下：</p>
<pre class="lang-js" data-nodeid="21563"><code data-language="js"><span class="hljs-keyword">import</span> requests
<span class="hljs-keyword">from</span> urllib.parse <span class="hljs-keyword">import</span> urljoin

BASE_URL = <span class="hljs-string">'https://login2.scrape.center/'</span>
LOGIN_URL = urljoin(BASE_URL, <span class="hljs-string">'/login'</span>)
INDEX_URL = urljoin(BASE_URL, <span class="hljs-string">'/page/1'</span>)
USERNAME = <span class="hljs-string">'admin'</span>
PASSWORD = <span class="hljs-string">'admin'</span>

response_login = requests.post(LOGIN_URL, data={
   <span class="hljs-string">'username'</span>: USERNAME,
   <span class="hljs-string">'password'</span>: PASSWORD
}, allow_redirects=False)

cookies = response_login.cookies
print(<span class="hljs-string">'Cookies'</span>, cookies)

response_index = requests.get(INDEX_URL, cookies=cookies)
print(<span class="hljs-string">'Response Status'</span>, response_index.status_code)
print(<span class="hljs-string">'Response URL'</span>, response_index.url)
</code></pre>
<p data-nodeid="21564">由于 requests 可以自动处理重定向，所以模拟登录的过程我们要加上 allow_redirects 参数并设置为 False，使其不自动处理重定向，这里登录之后返回的 Response 我们赋值为 response_login，这样通过调用 response_login 的 cookies 就可以获取到网站的 Cookies 信息了，这里 requests 自动帮我们解析了 Response Headers 的 set-cookie 字段并设置了 Cookies，所以我们不需要手动解析 Response Headers 的内容了，直接使用 response_login 对象的 cookies 属性即可获取 Cookies。</p>
<p data-nodeid="21565">好，接下来我们再次用 requests 的 get 方法来请求网站的 INDEX_URL，不过这里和之前不同，get 方法多加了一个参数 cookies，这就是第一次模拟登录完之后获取的 Cookies，这样第二次请求就能携带第一次模拟登录获取的 Cookies 信息了，此时网站会根据 Cookies 里面的 Session ID 信息查找到同一个 Session，校验其已经是登录状态，然后返回正确的结果。</p>
<p data-nodeid="21566">这里我们还是输出了最终的 URL，如果其是 INDEX_URL，那就代表模拟登录成功并获取到了有效数据，否则就代表模拟登录失败。</p>
<p data-nodeid="21567">我们看下运行结果：</p>
<pre class="lang-js" data-nodeid="21568"><code data-language="js">Cookies &lt;RequestsCookieJar[<span class="xml"><span class="hljs-tag">&lt;<span class="hljs-name">Cookie</span> <span class="hljs-attr">sessionid</span>=<span class="hljs-string">psnu8ij69f0ltecd5wasccyzc6ud41tc</span> <span class="hljs-attr">for</span> <span class="hljs-attr">login2.scrape.center</span>/&gt;</span></span>]&gt;
Response Status <span class="hljs-number">200</span>
Response URL https:<span class="hljs-comment">//login2.scrape.center/page/1</span>
</code></pre>
<p data-nodeid="21569">这下就没有问题了，这次我们发现其 URL 就是 INDEX_URL，模拟登录成功了！同时还可以进一步输出 response_index 的 text 属性看下是否获取成功。</p>
<p data-nodeid="21570">接下来后续的爬取用同样的方式爬取即可。</p>
<p data-nodeid="21571">但是我们发现其实这种实现方式比较烦琐，每次还需要处理 Cookies 并进行一次传递，有没有更简便的方法呢？</p>
<p data-nodeid="21572">有的，我们可以直接借助于 requests 内置的 Session 对象来帮我们自动处理 Cookies，使用了 Session 对象之后，requests 会将每次请求后需要设置的 Cookies 自动保存好，并在下次请求时自动携带上去，就相当于帮我们维持了一个 Session 对象，这样就更方便了。</p>
<p data-nodeid="21573">所以，刚才的代码可以简化如下：</p>
<pre class="lang-js" data-nodeid="21574"><code data-language="js"><span class="hljs-keyword">import</span> requests
<span class="hljs-keyword">from</span> urllib.parse <span class="hljs-keyword">import</span> urljoin

BASE_URL = <span class="hljs-string">'https://login2.scrape.center/'</span>
LOGIN_URL = urljoin(BASE_URL, <span class="hljs-string">'/login'</span>)
INDEX_URL = urljoin(BASE_URL, <span class="hljs-string">'/page/1'</span>)
USERNAME = <span class="hljs-string">'admin'</span>
PASSWORD = <span class="hljs-string">'admin'</span>

session = requests.Session()

response_login = session.post(LOGIN_URL, data={
   <span class="hljs-string">'username'</span>: USERNAME,
   <span class="hljs-string">'password'</span>: PASSWORD
})

cookies = session.cookies
print(<span class="hljs-string">'Cookies'</span>, cookies)

response_index = session.get(INDEX_URL)
print(<span class="hljs-string">'Response Status'</span>, response_index.status_code)
print(<span class="hljs-string">'Response URL'</span>, response_index.url)
</code></pre>
<p data-nodeid="21575">可以看到，这里我们无需再关心 Cookies 的处理和传递问题，我们声明了一个 Session 对象，然后每次调用请求的时候都直接使用 Session 对象的 post 或 get 方法就好了。</p>
<p data-nodeid="21576">运行效果是完全一样的，结果如下：</p>
<pre class="lang-js" data-nodeid="21577"><code data-language="js">Cookies &lt;RequestsCookieJar[<span class="xml"><span class="hljs-tag">&lt;<span class="hljs-name">Cookie</span> <span class="hljs-attr">sessionid</span>=<span class="hljs-string">ssngkl4i7en9vm73bb36hxif05k10k13</span> <span class="hljs-attr">for</span> <span class="hljs-attr">login2.scrape.center</span>/&gt;</span></span>]&gt;

Response Status <span class="hljs-number">200</span>

Response URL https:<span class="hljs-comment">//login2.scrape.center/page/1</span>
</code></pre>
<p data-nodeid="21578">因此，为了简化写法，这里建议直接使用 Session 对象来进行请求，这样我们就无需关心 Cookies 的操作了，实现起来会更加方便。</p>
<p data-nodeid="21579">这个案例整体来说比较简单，但是如果碰上复杂一点的网站，如带有验证码，带有加密参数等等，直接用 requests 并不好处理模拟登录，如果登录不了，那岂不是整个页面都没法爬了吗？那么有没有其他的方式来解决这个问题呢？当然是有的，比如说，我们可以使用 Selenium 来通过模拟浏览器的方式实现模拟登录，然后获取模拟登录成功后的 Cookies，再把获取的 Cookies 交由 requests 等来爬取就好了。</p>
<p data-nodeid="21580">这里我们还是以刚才的页面为例，我们可以把模拟登录这块交由 Selenium 来实现，后续的爬取交由 requests 来实现，代码实现如下：</p>
<pre class="lang-js" data-nodeid="21581"><code data-language="js">from urllib.parse import urljoin
from selenium import webdriver
import requests
import time

BASE_URL = 'https://login2.scrape.center/'
LOGIN_URL = urljoin(BASE_URL, '/login')
INDEX_URL = urljoin(BASE_URL, '/page/1')
USERNAME = 'admin'
PASSWORD = 'admin'

browser = webdriver.Chrome()
browser.get(BASE_URL)
browser.find_element_by_css_selector('input[name="username"]').send_keys(USERNAME)
browser.find_element_by_css_selector('input[name="password"]').send_keys(PASSWORD)
browser.find_element_by_css_selector('input[type="submit"]').click()
time.sleep(10)

# get cookies from selenium
cookies = browser.get_cookies()
print('Cookies', cookies)
browser.close()

# set cookies to requests
session = requests.Session()
for cookie in cookies:
   session.cookies.set(cookie['name'], cookie['value'])

response_index = session.get(INDEX_URL)
print('Response Status', response_index.status_code)
print('Response URL', response_index.url)
</code></pre>
<p data-nodeid="21582">这里我们使用 Selenium 先打开了 Chrome 浏览器，然后跳转到了登录页面，随后模拟输入了用户名和密码，接着点击了登录按钮，这时候我们可以发现浏览器里面就提示登录成功，然后成功跳转到了主页面。</p>
<p data-nodeid="21583">这时候，我们通过调用 get_cookies 方法便能获取到当前浏览器所有的 Cookies，这就是模拟登录成功之后的 Cookies，用这些 Cookies 我们就能访问其他的数据了。</p>
<p data-nodeid="21584">接下来，我们声明了 requests 的 Session 对象，然后遍历了刚才的 Cookies 并设置到 Session 对象的 cookies 上面去，接着再拿着这个 Session 对象去请求 INDEX_URL，也就能够获取到对应的信息而不会跳转到登录页面了。</p>
<p data-nodeid="21585">运行结果如下：</p>
<pre class="lang-js" data-nodeid="21586"><code data-language="js">Cookies [{<span class="hljs-string">'domain'</span>: <span class="hljs-string">'login2.scrape.center'</span>, <span class="hljs-string">'expiry'</span>: <span class="hljs-number">1589043753.553155</span>, <span class="hljs-string">'httpOnly'</span>: True, <span class="hljs-string">'name'</span>: <span class="hljs-string">'sessionid'</span>, <span class="hljs-string">'path'</span>: <span class="hljs-string">'/'</span>, <span class="hljs-string">'sameSite'</span>: <span class="hljs-string">'Lax'</span>, <span class="hljs-string">'secure'</span>: False, <span class="hljs-string">'value'</span>: <span class="hljs-string">'rdag7ttjqhvazavpxjz31y0tmze81zur'</span>}]

Response Status <span class="hljs-number">200</span>

Response URL https:<span class="hljs-comment">//login2.scrape.center/page/1</span>
</code></pre>
<p data-nodeid="21587">可以看到这里的模拟登录和后续的爬取也成功了。所以说，如果碰到难以模拟登录的过程，我们也可以使用 Selenium 或 Pyppeteer 等模拟浏览器操作的方式来实现，其目的就是取到登录后的 Cookies，有了 Cookies 之后，我们再用这些 Cookies 爬取其他页面就好了。</p>
<p data-nodeid="21588">所以这里我们也可以发现，对于基于 Session + Cookies 验证的网站，模拟登录的核心要点就是获取 Cookies，这个 Cookies 可以被保存下来或传递给其他的程序继续使用。甚至说可以将 Cookies 持久化存储或传输给其他终端来使用。另外，为了提高 Cookies 利用率或降低封号几率，可以搭建一个 Cookies 池实现 Cookies 的随机取用。</p>
<h4 data-nodeid="21589">案例二</h4>
<p data-nodeid="21590">对于案例二这种基于 JWT 的网站，其通常都是采用前后端分离式的，前后端的数据传输依赖于 Ajax，登录验证依赖于 JWT 本身这个 token 的值，如果 JWT 这个 token 是有效的，那么服务器就能返回想要的数据。</p>
<p data-nodeid="21591">下面我们先来在浏览器里面操作登录，观察下其网络请求过程，如图所示。</p>
<p data-nodeid="21592"><img src="https://s0.lgstatic.com/i/image3/M01/08/D5/CgoCgV6lJUyAToz_AARRgha8y4I294.png" alt="image (5).png" data-nodeid="21730"><br>
这里我们发现登录时其请求的 URL 为 <a href="https://login3.scrape.center/api/login" data-nodeid="21735">https://login3.scrape.center/api/login</a>，是通过 Ajax 请求的，同时其 Request Body 是 JSON 格式的数据，而不是 Form Data，返回状态码为 200。</p>
<p data-nodeid="21593">然后再看下返回结果，如图所示。</p>
<p data-nodeid="21594"><img src="https://s0.lgstatic.com/i/image3/M01/08/D5/CgoCgV6lJViAPFkQAAOlxLLBXVk095.png" alt="image (6).png" data-nodeid="21740"><br>
可以看到返回结果是一个 JSON 格式的数据，包含一个 token 字段，其结果为：</p>
<pre data-nodeid="21595"><code>eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxLCJ1c2VybmFtZSI6ImFkbWluIiwiZXhwIjoxNTg3ODc3OTQ2LCJlbWFpbCI6ImFkbWluQGFkbWluLmNvbSIsIm9yaWdfaWF0IjoxNTg3ODM0NzQ2fQ.ujEXXAZcCDyIfRLs44i_jdfA3LIp5Jc74n-Wq2udCR8
</code></pre>
<p data-nodeid="21596">这就是我们上一课时所讲的 JWT 的内容，格式是三段式的，通过“.”来分隔。</p>
<p data-nodeid="21597">那么有了这个 JWT 之后，后续的数据怎么获取呢？下面我们再来观察下后续的请求内容，如图所示。</p>
<p data-nodeid="21598"><img src="https://s0.lgstatic.com/i/image3/M01/16/04/Ciqah16lJcWAeoAVAAXnl2HHZic036.png" alt="image (7).png" data-nodeid="21747"><br>
这里我们可以发现，后续获取数据的 Ajax 请求中的 Request Headers 里面就多了一个 Authorization 字段，其结果为 jwt 然后加上刚才的 JWT 的内容，返回结果就是 JSON 格式的数据。</p>
<p data-nodeid="21599"><img src="https://s0.lgstatic.com/i/image3/M01/16/04/Ciqah16lJc2ATGC8AAZ4n3K84ns927.png" alt="image (8).png" data-nodeid="21752"><br>
没有问题，那模拟登录的整个思路就简单了：<br>
模拟请求登录结果，带上必要的登录信息，获取 JWT 的结果。</p>
<p data-nodeid="21600">后续的请求在 Request Headers 里面加上 Authorization 字段，值就是 JWT 对应的内容。<br>
好，接下来我们用代码实现如下：</p>
<pre class="lang-js" data-nodeid="21601"><code data-language="js"><span class="hljs-keyword">import</span> requests
<span class="hljs-keyword">from</span> urllib.parse <span class="hljs-keyword">import</span> urljoin

BASE_URL = <span class="hljs-string">'https://login3.scrape.center/'</span>
LOGIN_URL = urljoin(BASE_URL, <span class="hljs-string">'/api/login'</span>)
INDEX_URL = urljoin(BASE_URL, <span class="hljs-string">'/api/book'</span>)
USERNAME = <span class="hljs-string">'admin'</span>
PASSWORD = <span class="hljs-string">'admin'</span>

response_login = requests.post(LOGIN_URL, json={
   <span class="hljs-string">'username'</span>: USERNAME,
   <span class="hljs-string">'password'</span>: PASSWORD
})
data = response_login.json()
print(<span class="hljs-string">'Response JSON'</span>, data)
jwt = data.get(<span class="hljs-string">'token'</span>)
print(<span class="hljs-string">'JWT'</span>, jwt)

headers = {
   <span class="hljs-string">'Authorization'</span>: f<span class="hljs-string">'jwt {jwt}'</span>
}
response_index = requests.get(INDEX_URL, params={
   <span class="hljs-string">'limit'</span>: <span class="hljs-number">18</span>,
   <span class="hljs-string">'offset'</span>: <span class="hljs-number">0</span>
}, headers=headers)
print(<span class="hljs-string">'Response Status'</span>, response_index.status_code)
print(<span class="hljs-string">'Response URL'</span>, response_index.url)
print(<span class="hljs-string">'Response Data'</span>, response_index.json())
</code></pre>
<p data-nodeid="21602">这里我们同样是定义了登录接口和获取数据的接口，分别为 LOGIN_URL 和 INDEX_URL，接着通过 post 请求进行了模拟登录，这里提交的数据由于是 JSON 格式，所以这里使用 json 参数来传递。接着获取了返回结果中包含的 JWT 的结果。第二步就可以构造 Request Headers，然后设置 Authorization 字段并传入 JWT 即可，这样就能成功获取数据了。</p>
<p data-nodeid="21603">运行结果如下：</p>
<pre class="lang-js" data-nodeid="21604"><code data-language="js">Response <span class="hljs-built_in">JSON</span> {<span class="hljs-string">'token'</span>: <span class="hljs-string">'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxLCJ1c2VybmFtZSI6ImFkbWluIiwiZXhwIjoxNTg3ODc4NzkxLCJlbWFpbCI6ImFkbWluQGFkbWluLmNvbSIsIm9yaWdfaWF0IjoxNTg3ODM1NTkxfQ.iUnu3Yhdi_a-Bupb2BLgCTUd5yHL6jgPhkBPorCPvm4'</span>}

JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxLCJ1c2VybmFtZSI6ImFkbWluIiwiZXhwIjoxNTg3ODc4NzkxLCJlbWFpbCI6ImFkbWluQGFkbWluLmNvbSIsIm9yaWdfaWF0IjoxNTg3ODM1NTkxfQ.iUnu3Yhdi_a-Bupb2BLgCTUd5yHL6jgPhkBPorCPvm4

Response Status <span class="hljs-number">200</span>
Response URL https:<span class="hljs-comment">//login3.scrape.center/api/book/?limit=18&amp;offset=0</span>
Response Data {<span class="hljs-string">'count'</span>: <span class="hljs-number">9200</span>, <span class="hljs-string">'results'</span>: [{<span class="hljs-string">'id'</span>: <span class="hljs-string">'27135877'</span>, <span class="hljs-string">'name'</span>: <span class="hljs-string">'校园市场：布局未来消费群，决战年轻人市场'</span>, <span class="hljs-string">'authors'</span>: [<span class="hljs-string">'单兴华'</span>, <span class="hljs-string">'李烨'</span>], <span class="hljs-string">'cover'</span>: <span class="hljs-string">'https://img9.doubanio.com/view/subject/l/public/s29539805.jpg'</span>, <span class="hljs-string">'score'</span>: <span class="hljs-string">'5.5'</span>},
...
{<span class="hljs-string">'id'</span>: <span class="hljs-string">'30289316'</span>, <span class="hljs-string">'name'</span>: <span class="hljs-string">'就算這樣,還是喜歡你,笠原先生'</span>, <span class="hljs-string">'authors'</span>: [<span class="hljs-string">'おまる'</span>], <span class="hljs-string">'cover'</span>: <span class="hljs-string">'https://img3.doubanio.com/view/subject/l/public/s29875002.jpg'</span>, <span class="hljs-string">'score'</span>: <span class="hljs-string">'7.5'</span>}]}
</code></pre>
<p data-nodeid="21605">可以看到，这里成功输出了 JWT 的内容，同时最终也获取到了对应的数据，模拟登录成功！</p>
<p data-nodeid="21606">类似的思路，如果我们遇到 JWT 认证的网站，也可以通过类似的方式来实现模拟登录。当然可能某些页面比较复杂，需要具体情况具体分析。</p>
<h4 data-nodeid="21607">总结</h4>
<p data-nodeid="21608">以上我们就通过两个示例来演示了模拟登录爬取的过程，以后遇到这种情形的时候就可以用类似的思路解决了。</p>
<p data-nodeid="21609" class="te-preview-highlight">代码：<a href="https://github.com/Python3WebSpider/ScrapeLogin2" data-nodeid="21773">https://github.com/Python3WebSpider/ScrapeLogin2</a>、<a href="https://github.com/Python3WebSpider/ScrapeLogin3" data-nodeid="21777">https://github.com/Python3WebSpider/ScrapeLogin3</a>。</p>


# 令人抓狂的JavaScript混淆技术
<p>我们在爬取网站的时候，经常会遇到各种各样类似加密的情形，比如：</p>
<ul>
<li>某个网站的 URL 带有一些看不懂的长串加密参数，想要抓取就必须要懂得这些参数是怎么构造的，否则我们连完整的 URL 都构造不出来，更不用说爬取了。</li>
<li>分析某个网站的 Ajax 接口的时候，可以看到接口的一些参数也是加密的，或者 Request Headers 里面也可能带有一些加密参数，如果不知道这些参数的具体构造逻辑就无法直接用程序来模拟这些 Ajax 请求。</li>
<li>翻看网站的 JavaScript 源代码，可以发现很多压缩了或者看不太懂的字符，比如 JavaScript 文件名被编码，JavaScript 的文件内容被压缩成几行，JavaScript 变量也被修改成单个字符或者一些十六进制的字符，导致我们不好轻易根据 JavaScript 找出某些接口的加密逻辑。</li>
</ul>
<p>这些情况，基本上都是网站为了保护其本身的一些数据不被轻易抓取而采取的一些措施，我们可以把它归为两大类：</p>
<ul>
<li>接口加密技术；</li>
<li>JavaScript 压缩、混淆和加密技术。</li>
</ul>
<p>本课时我们就来了解下这两类技术的实现原理。</p>
<h3>数据保护</h3>
<p>当今大数据时代，数据已经变得越来越重要，网页和 App 现在是主流的数据载体，如果其数据的接口没有设置任何保护措施，在爬虫工程师解决了一些基本的反爬如封 IP、验证码的问题之后，那么数据还是可以被轻松抓取到。</p>
<p>那么，有没有可能在接口或 JavaScript 层面也加上一层防护呢？答案是可以的。</p>
<h4>接口加密技术</h4>
<p>网站运营商首先想到防护措施可能是对某些数据接口进行加密，比如说对某些 URL 的一些参数加上校验码或者把一些 ID 信息进行编码，使其变得难以阅读或构造；或者对某些接口请求加上一些 token、sign 等签名，这样这些请求发送到服务器时，服务器会通过客户端发来的一些请求信息以及双方约定好的秘钥等来对当前的请求进行校验，如果校验通过，才返回对应数据结果。</p>
<p>比如说客户端和服务端约定一种接口校验逻辑，客户端在每次请求服务端接口的时候都会附带一个 sign 参数，这个 sign 参数可能是由当前时间信息、请求的 URL、请求的数据、设备的 ID、双方约定好的秘钥经过一些加密算法构造而成的，客户端会实现这个加密算法构造 sign，然后每次请求服务器的时候附带上这个参数。服务端会根据约定好的算法和请求的数据对 sign 进行校验，如果校验通过，才返回对应的数据，否则拒绝响应。</p>
<h4>JavaScript 压缩、混淆和加密技术</h4>
<p>接口加密技术看起来的确是一个不错的解决方案，但单纯依靠它并不能很好地解决问题。为什么呢？</p>
<p>对于网页来说，其逻辑是依赖于 JavaScript 来实现的，JavaScript 有如下特点：</p>
<ul>
<li>JavaScript 代码运行于客户端，也就是它必须要在用户浏览器端加载并运行。</li>
<li>JavaScript 代码是公开透明的，也就是说浏览器可以直接获取到正在运行的 JavaScript 的源码。</li>
</ul>
<p>由于这两个原因，导致 JavaScript 代码是不安全的，任何人都可以读、分析、复制、盗用，甚至篡改。</p>
<p>所以说，对于上述情形，客户端 JavaScript 对于某些加密的实现是很容易被找到或模拟的，了解了加密逻辑后，模拟参数的构造和请求也就是轻而易举了，所以如果 JavaScript 没有做任何层面的保护的话，接口加密技术基本上对数据起不到什么防护作用。</p>
<p>如果你不想让自己的数据被轻易获取，不想他人了解 JavaScript 逻辑的实现，或者想降低被不怀好意的人甚至是黑客攻击。那么你就需要用到 JavaScript 压缩、混淆和加密技术了。</p>
<p>这里压缩、混淆、加密技术简述如下。</p>
<ul>
<li>代码压缩：即去除 JavaScript 代码中的不必要的空格、换行等内容，使源码都压缩为几行内容，降低代码可读性，当然同时也能提高网站的加载速度。</li>
<li>代码混淆：使用变量替换、字符串阵列化、控制流平坦化、多态变异、僵尸函数、调试保护等手段，使代码变得难以阅读和分析，达到最终保护的目的。但这不影响代码原有功能。是理想、实用的 JavaScript 保护方案。</li>
<li>代码加密：可以通过某种手段将 JavaScript 代码进行加密，转成人无法阅读或者解析的代码，如将代码完全抽象化加密，如 eval 加密。另外还有更强大的加密技术，可以直接将 JavaScript 代码用 C/C++ 实现，JavaScript 调用其编译后形成的文件来执行相应的功能，如 Emscripten 还有 WebAssembly。</li>
</ul>
<p>下面我们对上面的技术分别予以介绍。</p>
<h3>接口加密技术</h3>
<p>数据一般都是通过服务器提供的接口来获取的，网站或 App 可以请求某个数据接口获取到对应的数据，然后再把获取的数据展示出来。</p>
<p>但有些数据是比较宝贵或私密的，这些数据肯定是需要一定层面上的保护。所以不同接口的实现也就对应着不同的安全防护级别，我们这里来总结下。</p>
<h4>完全开放的接口</h4>
<p>有些接口是没有设置任何防护的，谁都可以调用和访问，而且没有任何时空限制和频率限制。任何人只要知道了接口的调用方式就能无限制地调用。</p>
<p>这种接口的安全性是非常非常低的，如果接口的调用方式一旦泄露或被抓包获取到，任何人都可以无限制地对数据进行操作或访问。此时如果接口里面包含一些重要的数据或隐私数据，就能轻易被篡改或窃取了。</p>
<h4>接口参数加密</h4>
<p>为了提升接口的安全性，客户端会和服务端约定一种接口校验方式，一般来说会使用到各种加密和编码算法，如 Base64、Hex 编码，MD5、AES、DES、RSA 等加密。</p>
<p>比如客户端和服务器双方约定一个 sign 用作接口的签名校验，其生成逻辑是客户端将 URL Path 进行 MD5 加密然后拼接上 URL 的某个参数再进行 Base64 编码，最后得到一个字符串 sign，这个 sign 会通过 Request URL 的某个参数或 Request Headers 发送给服务器。服务器接收到请求后，对 URL Path 同样进行 MD5 加密，然后拼接上 URL 的某个参数，也进行 Base64 编码得到了一个 sign，然后比对生成的 sign 和客户端发来的 sign 是否是一致的，如果是一致的，那就返回正确的结果，否则拒绝响应。这就是一个比较简单的接口参数加密的实现。如果有人想要调用这个接口的话，必须要定义好 sign 的生成逻辑，否则是无法正常调用接口的。</p>
<p>以上就是一个基本的接口参数加密逻辑的实现。</p>
<p>当然上面的这个实现思路比较简单，这里还可以增加一些时间戳信息增加时效性判断，或增加一些非对称加密进一步提高加密的复杂程度。但不管怎样，只要客户端和服务器约定好了加密和校验逻辑，任何形式加密算法都是可以的。</p>
<p>这里要实现接口参数加密就需要用到一些加密算法，客户端和服务器肯定也都有对应的 SDK 实现这些加密算法，如 JavaScript 的 crypto-js，Python 的 hashlib、Crypto 等等。</p>
<p>但还是如上文所说，如果是网页的话，客户端实现加密逻辑如果是用 JavaScript 来实现，其源代码对用户是完全可见的，如果没有对 JavaScript 做任何保护的话，是很容易弄清楚客户端加密的流程的。</p>
<p>因此，我们需要对 JavaScript 利用压缩、混淆、加密的方式来对客户端的逻辑进行一定程度上的保护。</p>
<h3>JavaScript 压缩、混淆、加密</h3>
<p>下面我们再来介绍下 JavaScript 的压缩、混淆和加密技术。</p>
<h4>JavaScript 压缩</h4>
<p>这个非常简单，JavaScript 压缩即去除 JavaScript 代码中的不必要的空格、换行等内容或者把一些可能公用的代码进行处理实现共享，最后输出的结果都被压缩为几行内容，代码可读性变得很差，同时也能提高网站加载速度。</p>
<p>如果仅仅是去除空格换行这样的压缩方式，其实几乎是没有任何防护作用的，因为这种压缩方式仅仅是降低了代码的直接可读性。如果我们有一些格式化工具可以轻松将 JavaScript 代码变得易读，比如利用 IDE、在线工具或 Chrome 浏览器都能还原格式化的代码。</p>
<p>目前主流的前端开发技术大多都会利用 Webpack 进行打包，Webpack 会对源代码进行编译和压缩，输出几个打包好的 JavaScript 文件，其中我们可以看到输出的 JavaScript 文件名带有一些不规则字符串，同时文件内容可能只有几行内容，变量名都是一些简单字母表示。这其中就包含 JavaScript 压缩技术，比如一些公共的库输出成 bundle 文件，一些调用逻辑压缩和转义成几行代码，这些都属于 JavaScript 压缩。另外其中也包含了一些很基础的 JavaScript 混淆技术，比如把变量名、方法名替换成一些简单字符，降低代码可读性。</p>
<p>但整体来说，JavaScript 压缩技术只能在很小的程度上起到防护作用，要想真正提高防护效果还得依靠 JavaScript 混淆和加密技术。</p>
<h4>JavaScript 混淆</h4>
<p>JavaScript 混淆完全是在 JavaScript 上面进行的处理，它的目的就是使得 JavaScript 变得难以阅读和分析，大大降低代码可读性，是一种很实用的 JavaScript 保护方案。</p>
<p>JavaScript 混淆技术主要有以下几种：</p>
<ul>
<li>变量混淆</li>
</ul>
<p>将带有含意的变量名、方法名、常量名随机变为无意义的类乱码字符串，降低代码可读性，如转成单个字符或十六进制字符串。</p>
<ul>
<li>字符串混淆</li>
</ul>
<p>将字符串阵列化集中放置、并可进行 MD5 或 Base64 加密存储，使代码中不出现明文字符串，这样可以避免使用全局搜索字符串的方式定位到入口点。</p>
<ul>
<li>属性加密</li>
</ul>
<p>针对 JavaScript 对象的属性进行加密转化，隐藏代码之间的调用关系。</p>
<ul>
<li>控制流平坦化</li>
</ul>
<p>打乱函数原有代码执行流程及函数调用关系，使代码逻变得混乱无序。</p>
<ul>
<li>僵尸代码</li>
</ul>
<p>随机在代码中插入无用的僵尸代码、僵尸函数，进一步使代码混乱。</p>
<ul>
<li>调试保护</li>
</ul>
<p>基于调试器特性，对当前运行环境进行检验，加入一些强制调试 debugger 语句，使其在调试模式下难以顺利执行 JavaScript 代码。</p>
<ul>
<li>多态变异</li>
</ul>
<p>使 JavaScript 代码每次被调用时，将代码自身即立刻自动发生变异，变化为与之前完全不同的代码，即功能完全不变，只是代码形式变异，以此杜绝代码被动态分析调试。</p>
<ul>
<li>锁定域名</li>
</ul>
<p>使 JavaScript 代码只能在指定域名下执行。</p>
<ul>
<li>反格式化</li>
</ul>
<p>如果对 JavaScript 代码进行格式化，则无法执行，导致浏览器假死。</p>
<ul>
<li>特殊编码</li>
</ul>
<p>将 JavaScript 完全编码为人不可读的代码，如表情符号、特殊表示内容等等。</p>
<p>总之，以上方案都是 JavaScript 混淆的实现方式，可以在不同程度上保护 JavaScript 代码。</p>
<p>在前端开发中，现在 JavaScript 混淆主流的实现是 javascript-obfuscator 这个库，利用它我们可以非常方便地实现页面的混淆，它与 Webpack 结合起来，最终可以输出压缩和混淆后的 JavaScript 代码，使得可读性大大降低，难以逆向。</p>
<p>下面我们会介绍下 javascript-obfuscator 对代码混淆的实现，了解了实现，那么自然我们就对混淆的机理有了更加深刻的认识。</p>
<p>javascript-obfuscator 的官网地址为：<a href="https://obfuscator.io/">https://obfuscator.io/</a>，其官方介绍内容如下：</p>
<blockquote>
<p>A free and efficient obfuscator for JavaScript (including ES2017). Make your code harder to copy and prevent people from stealing your work.</p>
</blockquote>
<p>它是支持 ES8 的免费、高效的 JavaScript 混淆库，它可以使得你的 JavaScript 代码经过混淆后难以被复制、盗用，混淆后的代码具有和原来的代码一模一样的功能。</p>
<p>怎么使用呢？首先，我们需要安装好 Node.js，可以使用 npm 命令。</p>
<p>然后新建一个文件夹，比如 js-obfuscate，随后进入该文件夹，初始化工作空间：</p>
<pre><code data-language="java" class="lang-java">npm init
</code></pre>
<p>这里会提示我们输入一些信息，创建一个 package.json 文件，这就完成了项目初始化了。</p>
<p>接下来我们来安装 javascript-obfuscator 这个库：</p>
<pre><code data-language="js" class="lang-js">npm install --save-dev javascript-obfuscator
</code></pre>
<p>接下来我们就可以编写代码来实现混淆了，如新建一个 main.js 文件，内容如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">const</span> code = <span class="hljs-string">`
let x = '1' + 1
console.log('x', x)
`</span>

<span class="hljs-keyword">const</span> options = {
   <span class="hljs-attr">compact</span>: <span class="hljs-literal">false</span>,
   <span class="hljs-attr">controlFlowFlattening</span>: <span class="hljs-literal">true</span>

}

<span class="hljs-keyword">const</span> obfuscator = <span class="hljs-built_in">require</span>(<span class="hljs-string">'javascript-obfuscator'</span>)
<span class="hljs-function"><span class="hljs-keyword">function</span> <span class="hljs-title">obfuscate</span>(<span class="hljs-params">code, options</span>) </span>{
   <span class="hljs-keyword">return</span> obfuscator.obfuscate(code, options).getObfuscatedCode()
}
<span class="hljs-built_in">console</span>.log(obfuscate(code, options))
</code></pre>
<p>在这里我们定义了两个变量，一个是 code，即需要被混淆的代码，另一个是混淆选项，是一个 Object。接下来我们引入了 javascript-obfuscator 库，然后定义了一个方法，传入 code 和 options，来获取混淆后的代码，最后控制台输出混淆后的代码。</p>
<p>代码逻辑比较简单，我们来执行一下代码：</p>
<pre><code>node main.js
</code></pre>
<p>输出结果如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> _0x53bf = [<span class="hljs-string">'log'</span>];
(<span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x1d84fe, _0x3aeda0</span>) </span>{
   <span class="hljs-keyword">var</span> _0x10a5a = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x2f0a52</span>) </span>{
       <span class="hljs-keyword">while</span> (--_0x2f0a52) {
           _0x1d84fe[<span class="hljs-string">'push'</span>](_0x1d84fe[<span class="hljs-string">'shift'</span>]());
      }
  };
   _0x10a5a(++_0x3aeda0);
}(_0x53bf, <span class="hljs-number">0x172</span>));
<span class="hljs-keyword">var</span> _0x480a = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x4341e5, _0x5923b4</span>) </span>{
   _0x4341e5 = _0x4341e5 - <span class="hljs-number">0x0</span>;
   <span class="hljs-keyword">var</span> _0xb3622e = _0x53bf[_0x4341e5];
   <span class="hljs-keyword">return</span> _0xb3622e;
};
<span class="hljs-keyword">let</span> x = <span class="hljs-string">'1'</span> + <span class="hljs-number">0x1</span>;
<span class="hljs-built_in">console</span>[_0x480a(<span class="hljs-string">'0x0'</span>)](<span class="hljs-string">'x'</span>, x);
</code></pre>
<p>看到了吧，这么简单的两行代码，被我们混淆成了这个样子，其实这里我们就是设定了一个“控制流扁平化”的选项。</p>
<p>整体看来，代码的可读性大大降低，也大大加大了 JavaScript 调试的难度。</p>
<p>好，接下来我们来跟着 javascript-obfuscator 走一遍，就能具体知道 JavaScript 混淆到底有多少方法了。</p>
<h5>代码压缩</h5>
<p>这里 javascript-obfuscator 也提供了代码压缩的功能，使用其参数 compact 即可完成 JavaScript 代码的压缩，输出为一行内容。默认是 true，如果定义为 false，则混淆后的代码会分行显示。</p>
<p>示例如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">const</span> code = <span class="hljs-string">`
let x = '1' + 1
console.log('x', x)
`</span>
<span class="hljs-keyword">const</span> options = {
   <span class="hljs-attr">compact</span>: <span class="hljs-literal">false</span>
}
</code></pre>
<p>这里我们先把代码压缩 compact 选项设置为 false，运行结果如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">let</span> x = <span class="hljs-string">'1'</span> + <span class="hljs-number">0x1</span>;
<span class="hljs-built_in">console</span>[<span class="hljs-string">'log'</span>](<span class="hljs-string">'x'</span>, x);
</code></pre>
<p>如果不设置 compact 或把 compact 设置为 true，结果如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> _0x151c=[<span class="hljs-string">'log'</span>];(<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x1ce384,_0x20a7c7</span>)</span>{<span class="hljs-keyword">var</span> _0x25fc92=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x188aec</span>)</span>{<span class="hljs-keyword">while</span>(--_0x188aec){_0x1ce384[<span class="hljs-string">'push'</span>](_0x1ce384[<span class="hljs-string">'shift'</span>]());}};_0x25fc92(++_0x20a7c7);}(_0x151c,<span class="hljs-number">0x1b7</span>));<span class="hljs-keyword">var</span> _0x553e=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x259219,_0x241445</span>)</span>{_0x259219=_0x259219<span class="hljs-number">-0x0</span>;<span class="hljs-keyword">var</span> _0x56d72d=_0x151c[_0x259219];<span class="hljs-keyword">return</span> _0x56d72d;};<span class="hljs-keyword">let</span> x=<span class="hljs-string">'1'</span>+<span class="hljs-number">0x1</span>;<span class="hljs-built_in">console</span>[_0x553e(<span class="hljs-string">'0x0'</span>)](<span class="hljs-string">'x'</span>,x);
</code></pre>
<p>可以看到单行显示的时候，对变量名进行了进一步的混淆和控制流扁平化操作。</p>
<h5>变量名混淆</h5>
<p>变量名混淆可以通过配置 identifierNamesGenerator 参数实现，我们通过这个参数可以控制变量名混淆的方式，如 hexadecimal 则会替换为 16 进制形式的字符串，在这里我们可以设定如下值：</p>
<ul>
<li>hexadecimal：将变量名替换为 16 进制形式的字符串，如 0xabc123。</li>
<li>mangled：将变量名替换为普通的简写字符，如 a、b、c 等。</li>
</ul>
<p>该参数默认为 hexadecimal。</p>
<p>我们将该参数修改为 mangled 来试一下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">const</span> code = <span class="hljs-string">`
let hello = '1' + 1
console.log('hello', hello)
`</span>
<span class="hljs-keyword">const</span> options = {
  <span class="hljs-attr">compact</span>: <span class="hljs-literal">true</span>,
  <span class="hljs-attr">identifierNamesGenerator</span>: <span class="hljs-string">'mangled'</span>
}
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> a=[<span class="hljs-string">'hello'</span>];(<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">c,d</span>)</span>{<span class="hljs-keyword">var</span> e=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">f</span>)</span>{<span class="hljs-keyword">while</span>(--f){c[<span class="hljs-string">'push'</span>](c[<span class="hljs-string">'shift'</span>]());}};e(++d);}(a,<span class="hljs-number">0x9b</span>));<span class="hljs-keyword">var</span> b=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">c,d</span>)</span>{c=c<span class="hljs-number">-0x0</span>;<span class="hljs-keyword">var</span> e=a[c];<span class="hljs-keyword">return</span> e;};<span class="hljs-keyword">let</span> hello=<span class="hljs-string">'1'</span>+<span class="hljs-number">0x1</span>;<span class="hljs-built_in">console</span>[<span class="hljs-string">'log'</span>](b(<span class="hljs-string">'0x0'</span>),hello);
</code></pre>
<p>可以看到这里的变量命名都变成了 a、b 等形式。</p>
<p>如果我们将 identifierNamesGenerator 修改为 hexadecimal 或者不设置，运行结果如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> _0x4e98=[<span class="hljs-string">'log'</span>,<span class="hljs-string">'hello'</span>];(<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x4464de,_0x39de6c</span>)</span>{<span class="hljs-keyword">var</span> _0xdffdda=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x6a95d5</span>)</span>{<span class="hljs-keyword">while</span>(--_0x6a95d5){_0x4464de[<span class="hljs-string">'push'</span>](_0x4464de[<span class="hljs-string">'shift'</span>]());}};_0xdffdda(++_0x39de6c);}(_0x4e98,<span class="hljs-number">0xc8</span>));<span class="hljs-keyword">var</span> _0x53cb=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x393bda,_0x8504e7</span>)</span>{_0x393bda=_0x393bda<span class="hljs-number">-0x0</span>;<span class="hljs-keyword">var</span> _0x46ab80=_0x4e98[_0x393bda];<span class="hljs-keyword">return</span> _0x46ab80;};<span class="hljs-keyword">let</span> hello=<span class="hljs-string">'1'</span>+<span class="hljs-number">0x1</span>;<span class="hljs-built_in">console</span>[_0x53cb(<span class="hljs-string">'0x0'</span>)](_0x53cb(<span class="hljs-string">'0x1'</span>),hello);
</code></pre>
<p>可以看到选用了 mangled，其代码体积会更小，但 hexadecimal 其可读性会更低。</p>
<p>另外我们还可以通过设置 identifiersPrefix 参数来控制混淆后的变量前缀，示例如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">const</span> code = <span class="hljs-string">`
let hello = '1' + 1
console.log('hello', hello)
`</span>
<span class="hljs-keyword">const</span> options = {
  <span class="hljs-attr">identifiersPrefix</span>: <span class="hljs-string">'germey'</span>
}
</code></pre>
<p>运行结果：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> germey_0x3dea=[<span class="hljs-string">'log'</span>,<span class="hljs-string">'hello'</span>];(<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x348ff3,_0x5330e8</span>)</span>{<span class="hljs-keyword">var</span> _0x1568b1=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x4740d8</span>)</span>{<span class="hljs-keyword">while</span>(--_0x4740d8){_0x348ff3[<span class="hljs-string">'push'</span>](_0x348ff3[<span class="hljs-string">'shift'</span>]());}};_0x1568b1(++_0x5330e8);}(germey_0x3dea,<span class="hljs-number">0x94</span>));<span class="hljs-keyword">var</span> germey_0x30e4=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x2e8f7c,_0x1066a8</span>)</span>{_0x2e8f7c=_0x2e8f7c<span class="hljs-number">-0x0</span>;<span class="hljs-keyword">var</span> _0x5166ba=germey_0x3dea[_0x2e8f7c];<span class="hljs-keyword">return</span> _0x5166ba;};<span class="hljs-keyword">let</span> hello=<span class="hljs-string">'1'</span>+<span class="hljs-number">0x1</span>;<span class="hljs-built_in">console</span>[germey_0x30e4(<span class="hljs-string">'0x0'</span>)](germey_0x30e4(<span class="hljs-string">'0x1'</span>),hello);
</code></pre>
<p>可以看到混淆后的变量前缀加上了我们自定义的字符串 germey。</p>
<p>另外 renameGlobals 这个参数还可以指定是否混淆全局变量和函数名称，默认为 false。示例如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">const</span> code = <span class="hljs-string">`
var $ = function(id) {
  return document.getElementById(id);
};
`</span>
<span class="hljs-keyword">const</span> options = {
  <span class="hljs-attr">renameGlobals</span>: <span class="hljs-literal">true</span>
}
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> _0x4864b0=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x5763be</span>)</span>{<span class="hljs-keyword">return</span> <span class="hljs-built_in">document</span>[<span class="hljs-string">'getElementById'</span>](_0x5763be);};
</code></pre>
<p>可以看到这里我们声明了一个全局变量 $，在 renameGlobals 设置为 true 之后，$ 这个变量也被替换了。如果后文用到了这个 $ 对象，可能就会有找不到定义的错误，因此这个参数可能导致代码执行不通。</p>
<p>如果我们不设置 renameGlobals 或者设置为 false，结果如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> _0x239a=[<span class="hljs-string">'getElementById'</span>];(<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x3f45a3,_0x583dfa</span>)</span>{<span class="hljs-keyword">var</span> _0x2cade2=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x28479a</span>)</span>{<span class="hljs-keyword">while</span>(--_0x28479a){_0x3f45a3[<span class="hljs-string">'push'</span>](_0x3f45a3[<span class="hljs-string">'shift'</span>]());}};_0x2cade2(++_0x583dfa);}(_0x239a,<span class="hljs-number">0xe1</span>));<span class="hljs-keyword">var</span> _0x3758=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x18659d,_0x50c21d</span>)</span>{_0x18659d=_0x18659d<span class="hljs-number">-0x0</span>;<span class="hljs-keyword">var</span> _0x531b8d=_0x239a[_0x18659d];<span class="hljs-keyword">return</span> _0x531b8d;};<span class="hljs-keyword">var</span> $=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x3d8723</span>)</span>{<span class="hljs-keyword">return</span> <span class="hljs-built_in">document</span>[_0x3758(<span class="hljs-string">'0x0'</span>)](_0x3d8723);};
</code></pre>
<p>可以看到，最后还是有 $ 的声明，其全局名称没有被改变。</p>
<h5>字符串混淆</h5>
<p>字符串混淆，即将一个字符串声明放到一个数组里面，使之无法被直接搜索到。我们可以通过控制 stringArray 参数来控制，默认为 true。</p>
<p>我们还可以通过 rotateStringArray 参数来控制数组化后结果的元素顺序，默认为 true。<br>
还可以通过 stringArrayEncoding 参数来控制数组的编码形式，默认不开启编码，如果设置为 true 或 base64，则会使用 Base64 编码，如果设置为 rc4，则使用 RC4 编码。<br>
还可以通过 stringArrayThreshold 来控制启用编码的概率，范围 0 到 1，默认 0.8。</p>
<p>示例如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">const</span> code = <span class="hljs-string">`
var a = 'hello world'
`</span>
<span class="hljs-keyword">const</span> options = {
  <span class="hljs-attr">stringArray</span>: <span class="hljs-literal">true</span>,
  <span class="hljs-attr">rotateStringArray</span>: <span class="hljs-literal">true</span>,
  <span class="hljs-attr">stringArrayEncoding</span>: <span class="hljs-literal">true</span>, <span class="hljs-comment">// 'base64' or 'rc4' or false</span>
  <span class="hljs-attr">stringArrayThreshold</span>: <span class="hljs-number">1</span>,
}
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> _0x4215=[<span class="hljs-string">'aGVsbG8gd29ybGQ='</span>];(<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x42bf17,_0x4c348f</span>)</span>{<span class="hljs-keyword">var</span> _0x328832=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x355be1</span>)</span>{<span class="hljs-keyword">while</span>(--_0x355be1){_0x42bf17[<span class="hljs-string">'push'</span>](_0x42bf17[<span class="hljs-string">'shift'</span>]());}};_0x328832(++_0x4c348f);}(_0x4215,<span class="hljs-number">0x1da</span>));<span class="hljs-keyword">var</span> _0x5191=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x3cf2ba,_0x1917d8</span>)</span>{_0x3cf2ba=_0x3cf2ba<span class="hljs-number">-0x0</span>;<span class="hljs-keyword">var</span> _0x1f93f0=_0x4215[_0x3cf2ba];<span class="hljs-keyword">if</span>(_0x5191[<span class="hljs-string">'LqbVDH'</span>]===<span class="hljs-literal">undefined</span>){(<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">var</span> _0x5096b2;<span class="hljs-keyword">try</span>{<span class="hljs-keyword">var</span> _0x282db1=<span class="hljs-built_in">Function</span>(<span class="hljs-string">'return\x20(function()\x20'</span>+<span class="hljs-string">'{}.constructor(\x22return\x20this\x22)(\x20)'</span>+<span class="hljs-string">');'</span>);_0x5096b2=_0x282db1();}<span class="hljs-keyword">catch</span>(_0x2acb9c){_0x5096b2=<span class="hljs-built_in">window</span>;}<span class="hljs-keyword">var</span> _0x388c14=<span class="hljs-string">'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/='</span>;_0x5096b2[<span class="hljs-string">'atob'</span>]||(_0x5096b2[<span class="hljs-string">'atob'</span>]=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x4cc27c</span>)</span>{<span class="hljs-keyword">var</span> _0x2af4ae=<span class="hljs-built_in">String</span>(_0x4cc27c)[<span class="hljs-string">'replace'</span>](<span class="hljs-regexp">/=+$/</span>,<span class="hljs-string">''</span>);<span class="hljs-keyword">for</span>(<span class="hljs-keyword">var</span> _0x21400b=<span class="hljs-number">0x0</span>,_0x3f4e2e,_0x5b193b,_0x233381=<span class="hljs-number">0x0</span>,_0x3dccf7=<span class="hljs-string">''</span>;_0x5b193b=_0x2af4ae[<span class="hljs-string">'charAt'</span>](_0x233381++);~_0x5b193b&amp;&amp;(_0x3f4e2e=_0x21400b%<span class="hljs-number">0x4</span>?_0x3f4e2e*<span class="hljs-number">0x40</span>+_0x5b193b:_0x5b193b,_0x21400b++%<span class="hljs-number">0x4</span>)?_0x3dccf7+=<span class="hljs-built_in">String</span>[<span class="hljs-string">'fromCharCode'</span>](<span class="hljs-number">0xff</span>&amp;_0x3f4e2e&gt;&gt;(<span class="hljs-number">-0x2</span>*_0x21400b&amp;<span class="hljs-number">0x6</span>)):<span class="hljs-number">0x0</span>){_0x5b193b=_0x388c14[<span class="hljs-string">'indexOf'</span>](_0x5b193b);}<span class="hljs-keyword">return</span> _0x3dccf7;});}());_0x5191[<span class="hljs-string">'DuIurT'</span>]=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x51888e</span>)</span>{<span class="hljs-keyword">var</span> _0x29801f=atob(_0x51888e);<span class="hljs-keyword">var</span> _0x561e62=[];<span class="hljs-keyword">for</span>(<span class="hljs-keyword">var</span> _0x5dd788=<span class="hljs-number">0x0</span>,_0x1a8b73=_0x29801f[<span class="hljs-string">'length'</span>];_0x5dd788&lt;_0x1a8b73;_0x5dd788++){_0x561e62+=<span class="hljs-string">'%'</span>+(<span class="hljs-string">'00'</span>+_0x29801f[<span class="hljs-string">'charCodeAt'</span>](_0x5dd788)[<span class="hljs-string">'toString'</span>](<span class="hljs-number">0x10</span>))[<span class="hljs-string">'slice'</span>](<span class="hljs-number">-0x2</span>);}<span class="hljs-keyword">return</span> <span class="hljs-built_in">decodeURIComponent</span>(_0x561e62);};_0x5191[<span class="hljs-string">'mgoBRd'</span>]={};_0x5191[<span class="hljs-string">'LqbVDH'</span>]=!![];}<span class="hljs-keyword">var</span> _0x1741f0=_0x5191[<span class="hljs-string">'mgoBRd'</span>][_0x3cf2ba];<span class="hljs-keyword">if</span>(_0x1741f0===<span class="hljs-literal">undefined</span>){_0x1f93f0=_0x5191[<span class="hljs-string">'DuIurT'</span>](_0x1f93f0);_0x5191[<span class="hljs-string">'mgoBRd'</span>][_0x3cf2ba]=_0x1f93f0;}<span class="hljs-keyword">else</span>{_0x1f93f0=_0x1741f0;}<span class="hljs-keyword">return</span> _0x1f93f0;};<span class="hljs-keyword">var</span> a=_0x5191(<span class="hljs-string">'0x0'</span>);
</code></pre>
<p>可以看到这里就把字符串进行了 Base64 编码，我们再也无法通过查找的方式找到字符串的位置了。</p>
<p>如果将 stringArray 设置为 false 的话，输出就是这样：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> a=<span class="hljs-string">'hello\x20world'</span>;
</code></pre>
<p>字符串就仍然是明文显示的，没有被编码。</p>
<p>另外我们还可以使用 unicodeEscapeSequence 这个参数对字符串进行 Unicode 转码，使之更加难以辨认，示例如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">const</span> code = <span class="hljs-string">`
var a = 'hello world'
`</span>
<span class="hljs-keyword">const</span> options = {
  <span class="hljs-attr">compact</span>: <span class="hljs-literal">false</span>,
  <span class="hljs-attr">unicodeEscapeSequence</span>: <span class="hljs-literal">true</span>
}
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> _0x5c0d = [<span class="hljs-string">'\x68\x65\x6c\x6c\x6f\x20\x77\x6f\x72\x6c\x64'</span>];
(<span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x54cc9c, _0x57a3b2</span>) </span>{
  <span class="hljs-keyword">var</span> _0xf833cf = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x3cd8c6</span>) </span>{
    <span class="hljs-keyword">while</span> (--_0x3cd8c6) {
      _0x54cc9c[<span class="hljs-string">'push'</span>](_0x54cc9c[<span class="hljs-string">'shift'</span>]());
    }
};
_0xf833cf(++_0x57a3b2);
}(_0x5c0d, <span class="hljs-number">0x17d</span>));
<span class="hljs-keyword">var</span> _0x28e8 = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x3fd645, _0x2cf5e7</span>) </span>{
  _0x3fd645 = _0x3fd645 - <span class="hljs-number">0x0</span>;
  <span class="hljs-keyword">var</span> _0x298a20 = _0x5c0d[_0x3fd645];
  <span class="hljs-keyword">return</span> _0x298a20;
};
<span class="hljs-keyword">var</span> a = _0x28e8(<span class="hljs-string">'0x0'</span>);
</code></pre>
<p>可以看到，这里字符串被数字化和 Unicode 化，非常难以辨认。</p>
<p>在很多 JavaScript 逆向的过程中，一些关键的字符串可能会作为切入点来查找加密入口。用了这种混淆之后，如果有人想通过全局搜索的方式搜索 hello 这样的字符串找加密入口，也没法搜到了。</p>
<h5>代码自我保护</h5>
<p>我们可以通过设置 selfDefending 参数来开启代码自我保护功能。开启之后，混淆后的 JavaScript 会强制以一行形式显示，如果我们将混淆后的代码进行格式化（美化）或者重命名，该段代码将无法执行。</p>
<p>例如：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">const</span> code = <span class="hljs-string">`
console.log('hello world')
`</span>
<span class="hljs-keyword">const</span> options = {
  <span class="hljs-attr">selfDefending</span>: <span class="hljs-literal">true</span>
}
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> _0x26da=[<span class="hljs-string">'log'</span>,<span class="hljs-string">'hello\x20world'</span>];(<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x190327,_0x57c2c0</span>)</span>{<span class="hljs-keyword">var</span> _0x577762=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0xc9dabb</span>)</span>{<span class="hljs-keyword">while</span>(--_0xc9dabb){_0x190327[<span class="hljs-string">'push'</span>](_0x190327[<span class="hljs-string">'shift'</span>]());}};<span class="hljs-keyword">var</span> _0x35976e=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">var</span> _0x16b3fe={<span class="hljs-string">'data'</span>:{<span class="hljs-string">'key'</span>:<span class="hljs-string">'cookie'</span>,<span class="hljs-string">'value'</span>:<span class="hljs-string">'timeout'</span>},<span class="hljs-string">'setCookie'</span>:<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x2d52d5,_0x16feda,_0x57cadf,_0x56056f</span>)</span>{_0x56056f=_0x56056f||{};<span class="hljs-keyword">var</span> _0x5b6dc3=_0x16feda+<span class="hljs-string">'='</span>+_0x57cadf;<span class="hljs-keyword">var</span> _0x333ced=<span class="hljs-number">0x0</span>;<span class="hljs-keyword">for</span>(<span class="hljs-keyword">var</span> _0x333ced=<span class="hljs-number">0x0</span>,_0x19ae36=_0x2d52d5[<span class="hljs-string">'length'</span>];_0x333ced&lt;_0x19ae36;_0x333ced++){<span class="hljs-keyword">var</span> _0x409587=_0x2d52d5[_0x333ced];_0x5b6dc3+=<span class="hljs-string">';\x20'</span>+_0x409587;<span class="hljs-keyword">var</span> _0x4aa006=_0x2d52d5[_0x409587];_0x2d52d5[<span class="hljs-string">'push'</span>](_0x4aa006);_0x19ae36=_0x2d52d5[<span class="hljs-string">'length'</span>];<span class="hljs-keyword">if</span>(_0x4aa006!==!![]){_0x5b6dc3+=<span class="hljs-string">'='</span>+_0x4aa006;}}_0x56056f[<span class="hljs-string">'cookie'</span>]=_0x5b6dc3;},<span class="hljs-string">'removeCookie'</span>:<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">return</span><span class="hljs-string">'dev'</span>;},<span class="hljs-string">'getCookie'</span>:<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x30c497,_0x51923d</span>)</span>{_0x30c497=_0x30c497||<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x4b7e18</span>)</span>{<span class="hljs-keyword">return</span> _0x4b7e18;};<span class="hljs-keyword">var</span> _0x557e06=_0x30c497(<span class="hljs-keyword">new</span> <span class="hljs-built_in">RegExp</span>(<span class="hljs-string">'(?:^|;\x20)'</span>+_0x51923d[<span class="hljs-string">'replace'</span>](<span class="hljs-regexp">/([.$?*|{}()[]\/+^])/g</span>,<span class="hljs-string">'$1'</span>)+<span class="hljs-string">'=([^;]*)'</span>));<span class="hljs-keyword">var</span> _0x817646=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0xf3fae7,_0x5d8208</span>)</span>{_0xf3fae7(++_0x5d8208);};_0x817646(_0x577762,_0x57c2c0);<span class="hljs-keyword">return</span> _0x557e06?<span class="hljs-built_in">decodeURIComponent</span>(_0x557e06[<span class="hljs-number">0x1</span>]):<span class="hljs-literal">undefined</span>;}};<span class="hljs-keyword">var</span> _0x4673cd=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">var</span> _0x4c6c5c=<span class="hljs-keyword">new</span> <span class="hljs-built_in">RegExp</span>(<span class="hljs-string">'\x5cw+\x20*\x5c(\x5c)\x20*{\x5cw+\x20*[\x27|\x22].+[\x27|\x22];?\x20*}'</span>);<span class="hljs-keyword">return</span> _0x4c6c5c[<span class="hljs-string">'test'</span>](_0x16b3fe[<span class="hljs-string">'removeCookie'</span>][<span class="hljs-string">'toString'</span>]());};_0x16b3fe[<span class="hljs-string">'updateCookie'</span>]=_0x4673cd;<span class="hljs-keyword">var</span> _0x5baa80=<span class="hljs-string">''</span>;<span class="hljs-keyword">var</span> _0x1faf19=_0x16b3fe[<span class="hljs-string">'updateCookie'</span>]();<span class="hljs-keyword">if</span>(!_0x1faf19){_0x16b3fe[<span class="hljs-string">'setCookie'</span>]([<span class="hljs-string">'*'</span>],<span class="hljs-string">'counter'</span>,<span class="hljs-number">0x1</span>);}<span class="hljs-keyword">else</span> <span class="hljs-keyword">if</span>(_0x1faf19){_0x5baa80=_0x16b3fe[<span class="hljs-string">'getCookie'</span>](<span class="hljs-literal">null</span>,<span class="hljs-string">'counter'</span>);}<span class="hljs-keyword">else</span>{_0x16b3fe[<span class="hljs-string">'removeCookie'</span>]();}};_0x35976e();}(_0x26da,<span class="hljs-number">0x140</span>));<span class="hljs-keyword">var</span> _0x4391=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x1b42d8,_0x57edc8</span>)</span>{_0x1b42d8=_0x1b42d8<span class="hljs-number">-0x0</span>;<span class="hljs-keyword">var</span> _0x2fbeca=_0x26da[_0x1b42d8];<span class="hljs-keyword">return</span> _0x2fbeca;};<span class="hljs-keyword">var</span> _0x197926=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">var</span> _0x10598f=!![];<span class="hljs-keyword">return</span> <span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0xffa3b3,_0x7a40f9</span>)</span>{<span class="hljs-keyword">var</span> _0x48e571=_0x10598f?<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">if</span>(_0x7a40f9){<span class="hljs-keyword">var</span> _0x2194b5=_0x7a40f9[<span class="hljs-string">'apply'</span>](_0xffa3b3,<span class="hljs-built_in">arguments</span>);_0x7a40f9=<span class="hljs-literal">null</span>;<span class="hljs-keyword">return</span> _0x2194b5;}}:<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{};_0x10598f=![];<span class="hljs-keyword">return</span> _0x48e571;};}();<span class="hljs-keyword">var</span> _0x2c6fd7=_0x197926(<span class="hljs-keyword">this</span>,<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">var</span> _0x4828bb=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">return</span><span class="hljs-string">'\x64\x65\x76'</span>;},_0x35c3bc=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">return</span><span class="hljs-string">'\x77\x69\x6e\x64\x6f\x77'</span>;};<span class="hljs-keyword">var</span> _0x456070=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">var</span> _0x4576a4=<span class="hljs-keyword">new</span> <span class="hljs-built_in">RegExp</span>(<span class="hljs-string">'\x5c\x77\x2b\x20\x2a\x5c\x28\x5c\x29\x20\x2a\x7b\x5c\x77\x2b\x20\x2a\x5b\x27\x7c\x22\x5d\x2e\x2b\x5b\x27\x7c\x22\x5d\x3b\x3f\x20\x2a\x7d'</span>);<span class="hljs-keyword">return</span>!_0x4576a4[<span class="hljs-string">'\x74\x65\x73\x74'</span>](_0x4828bb[<span class="hljs-string">'\x74\x6f\x53\x74\x72\x69\x6e\x67'</span>]());};<span class="hljs-keyword">var</span> _0x3fde69=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">var</span> _0xabb6f4=<span class="hljs-keyword">new</span> <span class="hljs-built_in">RegExp</span>(<span class="hljs-string">'\x28\x5c\x5c\x5b\x78\x7c\x75\x5d\x28\x5c\x77\x29\x7b\x32\x2c\x34\x7d\x29\x2b'</span>);<span class="hljs-keyword">return</span> _0xabb6f4[<span class="hljs-string">'\x74\x65\x73\x74'</span>](_0x35c3bc[<span class="hljs-string">'\x74\x6f\x53\x74\x72\x69\x6e\x67'</span>]());};<span class="hljs-keyword">var</span> _0x2d9a50=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x58fdb4</span>)</span>{<span class="hljs-keyword">var</span> _0x2a6361=~<span class="hljs-number">-0x1</span>&gt;&gt;<span class="hljs-number">0x1</span>+<span class="hljs-number">0xff</span>%<span class="hljs-number">0x0</span>;<span class="hljs-keyword">if</span>(_0x58fdb4[<span class="hljs-string">'\x69\x6e\x64\x65\x78\x4f\x66'</span>](<span class="hljs-string">'\x69'</span>===_0x2a6361)){_0xc388c5(_0x58fdb4);}};<span class="hljs-keyword">var</span> _0xc388c5=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x2073d6</span>)</span>{<span class="hljs-keyword">var</span> _0x6bb49f=~<span class="hljs-number">-0x4</span>&gt;&gt;<span class="hljs-number">0x1</span>+<span class="hljs-number">0xff</span>%<span class="hljs-number">0x0</span>;<span class="hljs-keyword">if</span>(_0x2073d6[<span class="hljs-string">'\x69\x6e\x64\x65\x78\x4f\x66'</span>]((!![]+<span class="hljs-string">''</span>)[<span class="hljs-number">0x3</span>])!==_0x6bb49f){_0x2d9a50(_0x2073d6);}};<span class="hljs-keyword">if</span>(!_0x456070()){<span class="hljs-keyword">if</span>(!_0x3fde69()){_0x2d9a50(<span class="hljs-string">'\x69\x6e\x64\u0435\x78\x4f\x66'</span>);}<span class="hljs-keyword">else</span>{_0x2d9a50(<span class="hljs-string">'\x69\x6e\x64\x65\x78\x4f\x66'</span>);}}<span class="hljs-keyword">else</span>{_0x2d9a50(<span class="hljs-string">'\x69\x6e\x64\u0435\x78\x4f\x66'</span>);}});_0x2c6fd7();<span class="hljs-built_in">console</span>[_0x4391(<span class="hljs-string">'0x0'</span>)](_0x4391(<span class="hljs-string">'0x1'</span>));
</code></pre>
<p>如果我们将上述代码放到控制台，它的执行结果和之前是一模一样的，没有任何问题。<br>
如果我们将其进行格式化，会变成如下内容：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> _0x26da = [<span class="hljs-string">'log'</span>, <span class="hljs-string">'hello\x20world'</span>];
(<span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x190327, _0x57c2c0</span>) </span>{
    <span class="hljs-keyword">var</span> _0x577762 = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0xc9dabb</span>) </span>{
        <span class="hljs-keyword">while</span> (--_0xc9dabb) {
            _0x190327[<span class="hljs-string">'push'</span>](_0x190327[<span class="hljs-string">'shift'</span>]());
        }
    };
    <span class="hljs-keyword">var</span> _0x35976e = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
        <span class="hljs-keyword">var</span> _0x16b3fe = {
            <span class="hljs-string">'data'</span>: {
                <span class="hljs-string">'key'</span>: <span class="hljs-string">'cookie'</span>,
                <span class="hljs-string">'value'</span>: <span class="hljs-string">'timeout'</span>
            },
            <span class="hljs-string">'setCookie'</span>: <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x2d52d5, _0x16feda, _0x57cadf, _0x56056f</span>) </span>{
                _0x56056f = _0x56056f || {};
                <span class="hljs-keyword">var</span> _0x5b6dc3 = _0x16feda + <span class="hljs-string">'='</span> + _0x57cadf;
                <span class="hljs-keyword">var</span> _0x333ced = <span class="hljs-number">0x0</span>;
                <span class="hljs-keyword">for</span> (<span class="hljs-keyword">var</span> _0x333ced = <span class="hljs-number">0x0</span>, _0x19ae36 = _0x2d52d5[<span class="hljs-string">'length'</span>]; _0x333ced &lt; _0x19ae36; _0x333ced++) {
                    <span class="hljs-keyword">var</span> _0x409587 = _0x2d52d5[_0x333ced];
                    _0x5b6dc3 += <span class="hljs-string">';\x20'</span> + _0x409587;
                    <span class="hljs-keyword">var</span> _0x4aa006 = _0x2d52d5[_0x409587];
                    _0x2d52d5[<span class="hljs-string">'push'</span>](_0x4aa006);
                    _0x19ae36 = _0x2d52d5[<span class="hljs-string">'length'</span>];
                    <span class="hljs-keyword">if</span> (_0x4aa006 !== !![]) {
                        _0x5b6dc3 += <span class="hljs-string">'='</span> + _0x4aa006;
                    }
                }
                _0x56056f[<span class="hljs-string">'cookie'</span>] = _0x5b6dc3;
            }, <span class="hljs-string">'removeCookie'</span>: <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
                <span class="hljs-keyword">return</span> <span class="hljs-string">'dev'</span>;
            }, <span class="hljs-string">'getCookie'</span>: <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x30c497, _0x51923d</span>) </span>{
                _0x30c497 = _0x30c497 || <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x4b7e18</span>) </span>{
                    <span class="hljs-keyword">return</span> _0x4b7e18;
                };
                <span class="hljs-keyword">var</span> _0x557e06 = _0x30c497(<span class="hljs-keyword">new</span> <span class="hljs-built_in">RegExp</span>(<span class="hljs-string">'(?:^|;\x20)'</span> + _0x51923d[<span class="hljs-string">'replace'</span>](<span class="hljs-regexp">/([.$?*|{}()[]\/+^])/g</span>, <span class="hljs-string">'$1'</span>) + <span class="hljs-string">'=([^;]*)'</span>));
                <span class="hljs-keyword">var</span> _0x817646 = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0xf3fae7, _0x5d8208</span>) </span>{
                    _0xf3fae7(++_0x5d8208);
                };
                _0x817646(_0x577762, _0x57c2c0);
                <span class="hljs-keyword">return</span> _0x557e06 ? <span class="hljs-built_in">decodeURIComponent</span>(_0x557e06[<span class="hljs-number">0x1</span>]) : <span class="hljs-literal">undefined</span>;
            }
        };
        <span class="hljs-keyword">var</span> _0x4673cd = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
            <span class="hljs-keyword">var</span> _0x4c6c5c = <span class="hljs-keyword">new</span> <span class="hljs-built_in">RegExp</span>(<span class="hljs-string">'\x5cw+\x20*\x5c(\x5c)\x20*{\x5cw+\x20*[\x27|\x22].+[\x27|\x22];?\x20*}'</span>);
            <span class="hljs-keyword">return</span> _0x4c6c5c[<span class="hljs-string">'test'</span>](_0x16b3fe[<span class="hljs-string">'removeCookie'</span>][<span class="hljs-string">'toString'</span>]());
        };
        _0x16b3fe[<span class="hljs-string">'updateCookie'</span>] = _0x4673cd;
        <span class="hljs-keyword">var</span> _0x5baa80 = <span class="hljs-string">''</span>;
        <span class="hljs-keyword">var</span> _0x1faf19 = _0x16b3fe[<span class="hljs-string">'updateCookie'</span>]();
        <span class="hljs-keyword">if</span> (!_0x1faf19) {
            _0x16b3fe[<span class="hljs-string">'setCookie'</span>]([<span class="hljs-string">'*'</span>], <span class="hljs-string">'counter'</span>, <span class="hljs-number">0x1</span>);
        } <span class="hljs-keyword">else</span> <span class="hljs-keyword">if</span> (_0x1faf19) {
            _0x5baa80 = _0x16b3fe[<span class="hljs-string">'getCookie'</span>](<span class="hljs-literal">null</span>, <span class="hljs-string">'counter'</span>);
        } <span class="hljs-keyword">else</span> {
            _0x16b3fe[<span class="hljs-string">'removeCookie'</span>]();
        }
    };
    _0x35976e();
}(_0x26da, <span class="hljs-number">0x140</span>));
<span class="hljs-keyword">var</span> _0x4391 = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x1b42d8, _0x57edc8</span>) </span>{
    _0x1b42d8 = _0x1b42d8 - <span class="hljs-number">0x0</span>;
    <span class="hljs-keyword">var</span> _0x2fbeca = _0x26da[_0x1b42d8];
    <span class="hljs-keyword">return</span> _0x2fbeca;
};
<span class="hljs-keyword">var</span> _0x197926 = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
    <span class="hljs-keyword">var</span> _0x10598f = !![];
    <span class="hljs-keyword">return</span> <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0xffa3b3, _0x7a40f9</span>) </span>{
        <span class="hljs-keyword">var</span> _0x48e571 = _0x10598f ? <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
            <span class="hljs-keyword">if</span> (_0x7a40f9) {
                <span class="hljs-keyword">var</span> _0x2194b5 = _0x7a40f9[<span class="hljs-string">'apply'</span>](_0xffa3b3, <span class="hljs-built_in">arguments</span>);
                _0x7a40f9 = <span class="hljs-literal">null</span>;
                <span class="hljs-keyword">return</span> _0x2194b5;
            }
        } : <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{};
        _0x10598f = ![];
        <span class="hljs-keyword">return</span> _0x48e571;
    };
}();
<span class="hljs-keyword">var</span> _0x2c6fd7 = _0x197926(<span class="hljs-keyword">this</span>, <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
    <span class="hljs-keyword">var</span> _0x4828bb = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
            <span class="hljs-keyword">return</span> <span class="hljs-string">'\x64\x65\x76'</span>;
        },
        _0x35c3bc = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
            <span class="hljs-keyword">return</span> <span class="hljs-string">'\x77\x69\x6e\x64\x6f\x77'</span>;
        };
    <span class="hljs-keyword">var</span> _0x456070 = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
        <span class="hljs-keyword">var</span> _0x4576a4 = <span class="hljs-keyword">new</span> <span class="hljs-built_in">RegExp</span>(<span class="hljs-string">'\x5c\x77\x2b\x20\x2a\x5c\x28\x5c\x29\x20\x2a\x7b\x5c\x77\x2b\x20\x2a\x5b\x27\x7c\x22\x5d\x2e\x2b\x5b\x27\x7c\x22\x5d\x3b\x3f\x20\x2a\x7d'</span>);
        <span class="hljs-keyword">return</span> !_0x4576a4[<span class="hljs-string">'\x74\x65\x73\x74'</span>](_0x4828bb[<span class="hljs-string">'\x74\x6f\x53\x74\x72\x69\x6e\x67'</span>]());
    };
    <span class="hljs-keyword">var</span> _0x3fde69 = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
        <span class="hljs-keyword">var</span> _0xabb6f4 = <span class="hljs-keyword">new</span> <span class="hljs-built_in">RegExp</span>(<span class="hljs-string">'\x28\x5c\x5c\x5b\x78\x7c\x75\x5d\x28\x5c\x77\x29\x7b\x32\x2c\x34\x7d\x29\x2b'</span>);
        <span class="hljs-keyword">return</span> _0xabb6f4[<span class="hljs-string">'\x74\x65\x73\x74'</span>](_0x35c3bc[<span class="hljs-string">'\x74\x6f\x53\x74\x72\x69\x6e\x67'</span>]());
    };
    <span class="hljs-keyword">var</span> _0x2d9a50 = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x58fdb4</span>) </span>{
        <span class="hljs-keyword">var</span> _0x2a6361 = ~<span class="hljs-number">-0x1</span> &gt;&gt; <span class="hljs-number">0x1</span> + <span class="hljs-number">0xff</span> % <span class="hljs-number">0x0</span>;
        <span class="hljs-keyword">if</span> (_0x58fdb4[<span class="hljs-string">'\x69\x6e\x64\x65\x78\x4f\x66'</span>](<span class="hljs-string">'\x69'</span> === _0x2a6361)) {
            _0xc388c5(_0x58fdb4);
        }
    };
    <span class="hljs-keyword">var</span> _0xc388c5 = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x2073d6</span>) </span>{
        <span class="hljs-keyword">var</span> _0x6bb49f = ~<span class="hljs-number">-0x4</span> &gt;&gt; <span class="hljs-number">0x1</span> + <span class="hljs-number">0xff</span> % <span class="hljs-number">0x0</span>;
        <span class="hljs-keyword">if</span> (_0x2073d6[<span class="hljs-string">'\x69\x6e\x64\x65\x78\x4f\x66'</span>]((!![] + <span class="hljs-string">''</span>)[<span class="hljs-number">0x3</span>]) !== _0x6bb49f) {
            _0x2d9a50(_0x2073d6);
        }
    };
    <span class="hljs-keyword">if</span> (!_0x456070()) {
        <span class="hljs-keyword">if</span> (!_0x3fde69()) {
            _0x2d9a50(<span class="hljs-string">'\x69\x6e\x64\u0435\x78\x4f\x66'</span>);
        } <span class="hljs-keyword">else</span> {
            _0x2d9a50(<span class="hljs-string">'\x69\x6e\x64\x65\x78\x4f\x66'</span>);
        }
    } <span class="hljs-keyword">else</span> {
        _0x2d9a50(<span class="hljs-string">'\x69\x6e\x64\u0435\x78\x4f\x66'</span>);
    }
});
_0x2c6fd7();
<span class="hljs-built_in">console</span>[_0x4391(<span class="hljs-string">'0x0'</span>)](_0x4391(<span class="hljs-string">'0x1'</span>));
</code></pre>
<p>如果把这段代码放到浏览器里面，浏览器会直接卡死无法运行。这样如果有人对代码进行了格式化，就无法正常对代码进行运行和调试，从而起到了保护作用。</p>
<h5>控制流平坦化</h5>
<p>控制流平坦化其实就是将代码的执行逻辑混淆，使其变得复杂难读。其基本思想是将一些逻辑处理块都统一加上一个前驱逻辑块，每个逻辑块都由前驱逻辑块进行条件判断和分发，构成一个个闭环逻辑，导致整个执行逻辑十分复杂难读。</p>
<p>我们通过 controlFlowFlattening 变量可以控制是否开启控制流平坦化，示例如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">const</span> code = <span class="hljs-string">`
(function(){
    function foo () {
        return function () {
            var sum = 1 + 2;
            console.log(1);
            console.log(2);
            console.log(3);
            console.log(4);
            console.log(5);
            console.log(6);
        }
    }

    foo()();
})();
`</span>
<span class="hljs-keyword">const</span> options = {
  <span class="hljs-attr">compact</span>: <span class="hljs-literal">false</span>,
  <span class="hljs-attr">controlFlowFlattening</span>: <span class="hljs-literal">true</span>
}
</code></pre>
<p>输出结果如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> _0xbaf1 = [
    <span class="hljs-string">'dZwUe'</span>,
    <span class="hljs-string">'log'</span>,
    <span class="hljs-string">'fXqMu'</span>,
    <span class="hljs-string">'0|1|3|4|6|5|2'</span>,
    <span class="hljs-string">'chYMl'</span>,
    <span class="hljs-string">'IZEsA'</span>,
    <span class="hljs-string">'split'</span>
];
(<span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x22d342, _0x4f6332</span>) </span>{
    <span class="hljs-keyword">var</span> _0x43ff59 = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x5ad417</span>) </span>{
        <span class="hljs-keyword">while</span> (--_0x5ad417) {
            _0x22d342[<span class="hljs-string">'push'</span>](_0x22d342[<span class="hljs-string">'shift'</span>]());
        }
    };
    _0x43ff59(++_0x4f6332);
}(_0xbaf1, <span class="hljs-number">0x192</span>));
<span class="hljs-keyword">var</span> _0x1a69 = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x8d64b1, _0x5e07b3</span>) </span>{
    _0x8d64b1 = _0x8d64b1 - <span class="hljs-number">0x0</span>;
    <span class="hljs-keyword">var</span> _0x300bab = _0xbaf1[_0x8d64b1];
    <span class="hljs-keyword">return</span> _0x300bab;
};
(<span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
    <span class="hljs-keyword">var</span> _0x19d8ce = {
        <span class="hljs-string">'chYMl'</span>: _0x1a69(<span class="hljs-string">'0x0'</span>),
        <span class="hljs-string">'IZEsA'</span>: <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x22e521, _0x298a22</span>) </span>{
            <span class="hljs-keyword">return</span> _0x22e521 + _0x298a22;
        },
        <span class="hljs-string">'fXqMu'</span>: <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x13124b</span>) </span>{
            <span class="hljs-keyword">return</span> _0x13124b();
        }
    };
    <span class="hljs-function"><span class="hljs-keyword">function</span> <span class="hljs-title">_0x4e2ee0</span>(<span class="hljs-params"></span>) </span>{
        <span class="hljs-keyword">var</span> _0x118a6a = {
            <span class="hljs-string">'LZAQV'</span>: _0x19d8ce[_0x1a69(<span class="hljs-string">'0x1'</span>)],
            <span class="hljs-string">'dZwUe'</span>: <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x362ef3, _0x352709</span>) </span>{
                <span class="hljs-keyword">return</span> _0x19d8ce[_0x1a69(<span class="hljs-string">'0x2'</span>)](_0x362ef3, _0x352709);
            }
        };
        <span class="hljs-keyword">return</span> <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
            <span class="hljs-keyword">var</span> _0x4c336d = _0x118a6a[<span class="hljs-string">'LZAQV'</span>][_0x1a69(<span class="hljs-string">'0x3'</span>)](<span class="hljs-string">'|'</span>), _0x2b6466 = <span class="hljs-number">0x0</span>;
            <span class="hljs-keyword">while</span> (!![]) {
                <span class="hljs-keyword">switch</span> (_0x4c336d[_0x2b6466++]) {
                <span class="hljs-keyword">case</span> <span class="hljs-string">'0'</span>:
                    <span class="hljs-keyword">var</span> _0xbfa3fd = _0x118a6a[_0x1a69(<span class="hljs-string">'0x4'</span>)](<span class="hljs-number">0x1</span>, <span class="hljs-number">0x2</span>);
                    <span class="hljs-keyword">continue</span>;
                <span class="hljs-keyword">case</span> <span class="hljs-string">'1'</span>:
                    <span class="hljs-built_in">console</span>[<span class="hljs-string">'log'</span>](<span class="hljs-number">0x1</span>);
                    <span class="hljs-keyword">continue</span>;
                <span class="hljs-keyword">case</span> <span class="hljs-string">'2'</span>:
                    <span class="hljs-built_in">console</span>[_0x1a69(<span class="hljs-string">'0x5'</span>)](<span class="hljs-number">0x6</span>);
                    <span class="hljs-keyword">continue</span>;
                <span class="hljs-keyword">case</span> <span class="hljs-string">'3'</span>:
                    <span class="hljs-built_in">console</span>[_0x1a69(<span class="hljs-string">'0x5'</span>)](<span class="hljs-number">0x2</span>);
                    <span class="hljs-keyword">continue</span>;
                <span class="hljs-keyword">case</span> <span class="hljs-string">'4'</span>:
                    <span class="hljs-built_in">console</span>[_0x1a69(<span class="hljs-string">'0x5'</span>)](<span class="hljs-number">0x3</span>);
                    <span class="hljs-keyword">continue</span>;
                <span class="hljs-keyword">case</span> <span class="hljs-string">'5'</span>:
                    <span class="hljs-built_in">console</span>[_0x1a69(<span class="hljs-string">'0x5'</span>)](<span class="hljs-number">0x5</span>);
                    <span class="hljs-keyword">continue</span>;
                <span class="hljs-keyword">case</span> <span class="hljs-string">'6'</span>:
                    <span class="hljs-built_in">console</span>[_0x1a69(<span class="hljs-string">'0x5'</span>)](<span class="hljs-number">0x4</span>);
                    <span class="hljs-keyword">continue</span>;
                }
                <span class="hljs-keyword">break</span>;
            }
        };
    }
    _0x19d8ce[_0x1a69(<span class="hljs-string">'0x6'</span>)](_0x4e2ee0)();
}());
</code></pre>
<p>可以看到，一些连续的执行逻辑被打破，代码被修改为一个 switch 语句，我们很难再一眼看出多条 console.log 语句的执行顺序了。</p>
<p>如果我们将 controlFlowFlattening 设置为 false 或者不设置，运行结果如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> _0x552c = [<span class="hljs-string">'log'</span>];
(<span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x4c4fa0, _0x59faa0</span>) </span>{
    <span class="hljs-keyword">var</span> _0xa01786 = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x409a37</span>) </span>{
        <span class="hljs-keyword">while</span> (--_0x409a37) {
            _0x4c4fa0[<span class="hljs-string">'push'</span>](_0x4c4fa0[<span class="hljs-string">'shift'</span>]());
        }
    };
    _0xa01786(++_0x59faa0);
}(_0x552c, <span class="hljs-number">0x9b</span>));
<span class="hljs-keyword">var</span> _0x4e63 = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x75ea1a, _0x50e176</span>) </span>{
    _0x75ea1a = _0x75ea1a - <span class="hljs-number">0x0</span>;
    <span class="hljs-keyword">var</span> _0x59dc94 = _0x552c[_0x75ea1a];
    <span class="hljs-keyword">return</span> _0x59dc94;
};
(<span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
    <span class="hljs-function"><span class="hljs-keyword">function</span> <span class="hljs-title">_0x507f38</span>(<span class="hljs-params"></span>) </span>{
        <span class="hljs-keyword">return</span> <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
            <span class="hljs-keyword">var</span> _0x17fb7e = <span class="hljs-number">0x1</span> + <span class="hljs-number">0x2</span>;
            <span class="hljs-built_in">console</span>[_0x4e63(<span class="hljs-string">'0x0'</span>)](<span class="hljs-number">0x1</span>);
            <span class="hljs-built_in">console</span>[<span class="hljs-string">'log'</span>](<span class="hljs-number">0x2</span>);
            <span class="hljs-built_in">console</span>[<span class="hljs-string">'log'</span>](<span class="hljs-number">0x3</span>);
            <span class="hljs-built_in">console</span>[_0x4e63(<span class="hljs-string">'0x0'</span>)](<span class="hljs-number">0x4</span>);
            <span class="hljs-built_in">console</span>[_0x4e63(<span class="hljs-string">'0x0'</span>)](<span class="hljs-number">0x5</span>);
            <span class="hljs-built_in">console</span>[_0x4e63(<span class="hljs-string">'0x0'</span>)](<span class="hljs-number">0x6</span>);
        };
    }
    _0x507f38()();
}());
</code></pre>
<p>可以看到，这里仍然保留了原始的 console.log 执行逻辑。</p>
<p>因此，使用控制流扁平化可以使得执行逻辑更加复杂难读，目前非常多的前端混淆都会加上这个选项。</p>
<p>但启用控制流扁平化之后，代码的执行时间会变长，最长达 1.5 倍之多。</p>
<p>另外我们还能使用 controlFlowFlatteningThreshold 这个参数来控制比例，取值范围是 0 到 1，默认 0.75，如果设置为 0，那相当于 controlFlowFlattening 设置为 false，即不开启控制流扁平化 。</p>
<h5>僵尸代码注入</h5>
<p>僵尸代码即不会被执行的代码或对上下文没有任何影响的代码，注入之后可以对现有的 JavaScript 代码的阅读形成干扰。我们可以使用 deadCodeInjection 参数开启这个选项，默认为 false。<br>
示例如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">const</span> code = <span class="hljs-string">`
(function(){
    if (true) {
        var foo = function () {
            console.log('abc');
            console.log('cde');
            console.log('efg');
            console.log('hij');
        };

        var bar = function () {
            console.log('klm');
            console.log('nop');
            console.log('qrs');
        };

        var baz = function () {
            console.log('tuv');
            console.log('wxy');
            console.log('z');
        };

        foo();
        bar();
        baz();
    }
})();
`</span>
<span class="hljs-keyword">const</span> options = {
  <span class="hljs-attr">compact</span>: <span class="hljs-literal">false</span>,
  <span class="hljs-attr">deadCodeInjection</span>: <span class="hljs-literal">true</span>
}
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> _0x5024 = [
    <span class="hljs-string">'zaU'</span>,
    <span class="hljs-string">'log'</span>,
    <span class="hljs-string">'tuv'</span>,
    <span class="hljs-string">'wxy'</span>,
    <span class="hljs-string">'abc'</span>,
    <span class="hljs-string">'cde'</span>,
    <span class="hljs-string">'efg'</span>,
    <span class="hljs-string">'hij'</span>,
    <span class="hljs-string">'QhG'</span>,
    <span class="hljs-string">'TeI'</span>,
    <span class="hljs-string">'klm'</span>,
    <span class="hljs-string">'nop'</span>,
    <span class="hljs-string">'qrs'</span>,
    <span class="hljs-string">'bZd'</span>,
    <span class="hljs-string">'HMx'</span>
];
<span class="hljs-keyword">var</span> _0x4502 = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x1254b1, _0x583689</span>) </span>{
    _0x1254b1 = _0x1254b1 - <span class="hljs-number">0x0</span>;
    <span class="hljs-keyword">var</span> _0x529b49 = _0x5024[_0x1254b1];
    <span class="hljs-keyword">return</span> _0x529b49;
};
(<span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
    <span class="hljs-keyword">if</span> (!![]) {
        <span class="hljs-keyword">var</span> _0x16c18d = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
            <span class="hljs-keyword">if</span> (_0x4502(<span class="hljs-string">'0x0'</span>) !== _0x4502(<span class="hljs-string">'0x0'</span>)) {
                <span class="hljs-built_in">console</span>[_0x4502(<span class="hljs-string">'0x1'</span>)](_0x4502(<span class="hljs-string">'0x2'</span>));
                <span class="hljs-built_in">console</span>[_0x4502(<span class="hljs-string">'0x1'</span>)](_0x4502(<span class="hljs-string">'0x3'</span>));
                <span class="hljs-built_in">console</span>[_0x4502(<span class="hljs-string">'0x1'</span>)](<span class="hljs-string">'z'</span>);
            } <span class="hljs-keyword">else</span> {
                <span class="hljs-built_in">console</span>[_0x4502(<span class="hljs-string">'0x1'</span>)](_0x4502(<span class="hljs-string">'0x4'</span>));
                <span class="hljs-built_in">console</span>[_0x4502(<span class="hljs-string">'0x1'</span>)](_0x4502(<span class="hljs-string">'0x5'</span>));
                <span class="hljs-built_in">console</span>[_0x4502(<span class="hljs-string">'0x1'</span>)](_0x4502(<span class="hljs-string">'0x6'</span>));
                <span class="hljs-built_in">console</span>[_0x4502(<span class="hljs-string">'0x1'</span>)](_0x4502(<span class="hljs-string">'0x7'</span>));
            }
        };
        <span class="hljs-keyword">var</span> _0x1f7292 = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
            <span class="hljs-keyword">if</span> (_0x4502(<span class="hljs-string">'0x8'</span>) === _0x4502(<span class="hljs-string">'0x9'</span>)) {
                <span class="hljs-built_in">console</span>[_0x4502(<span class="hljs-string">'0x1'</span>)](_0x4502(<span class="hljs-string">'0xa'</span>));
                <span class="hljs-built_in">console</span>[_0x4502(<span class="hljs-string">'0x1'</span>)](_0x4502(<span class="hljs-string">'0xb'</span>));
                <span class="hljs-built_in">console</span>[_0x4502(<span class="hljs-string">'0x1'</span>)](_0x4502(<span class="hljs-string">'0xc'</span>));
            } <span class="hljs-keyword">else</span> {
                <span class="hljs-built_in">console</span>[_0x4502(<span class="hljs-string">'0x1'</span>)](_0x4502(<span class="hljs-string">'0xa'</span>));
                <span class="hljs-built_in">console</span>[_0x4502(<span class="hljs-string">'0x1'</span>)](_0x4502(<span class="hljs-string">'0xb'</span>));
                <span class="hljs-built_in">console</span>[_0x4502(<span class="hljs-string">'0x1'</span>)](_0x4502(<span class="hljs-string">'0xc'</span>));
            }
        };
        <span class="hljs-keyword">var</span> _0x33b212 = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
            <span class="hljs-keyword">if</span> (_0x4502(<span class="hljs-string">'0xd'</span>) !== _0x4502(<span class="hljs-string">'0xe'</span>)) {
                <span class="hljs-built_in">console</span>[_0x4502(<span class="hljs-string">'0x1'</span>)](_0x4502(<span class="hljs-string">'0x2'</span>));
                <span class="hljs-built_in">console</span>[_0x4502(<span class="hljs-string">'0x1'</span>)](_0x4502(<span class="hljs-string">'0x3'</span>));
                <span class="hljs-built_in">console</span>[_0x4502(<span class="hljs-string">'0x1'</span>)](<span class="hljs-string">'z'</span>);
            } <span class="hljs-keyword">else</span> {
                <span class="hljs-built_in">console</span>[_0x4502(<span class="hljs-string">'0x1'</span>)](_0x4502(<span class="hljs-string">'0x4'</span>));
                <span class="hljs-built_in">console</span>[_0x4502(<span class="hljs-string">'0x1'</span>)](_0x4502(<span class="hljs-string">'0x5'</span>));
                <span class="hljs-built_in">console</span>[_0x4502(<span class="hljs-string">'0x1'</span>)](_0x4502(<span class="hljs-string">'0x6'</span>));
                <span class="hljs-built_in">console</span>[_0x4502(<span class="hljs-string">'0x1'</span>)](_0x4502(<span class="hljs-string">'0x7'</span>));
            }
        };
        _0x16c18d();
        _0x1f7292();
        _0x33b212();
    }
}());
</code></pre>
<p>可见这里增加了一些不会执行到的逻辑区块内容。</p>
<p>如果将 deadCodeInjection 设置为 false 或者不设置，运行结果如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> _0x402a = [
    <span class="hljs-string">'qrs'</span>,
    <span class="hljs-string">'wxy'</span>,
    <span class="hljs-string">'log'</span>,
    <span class="hljs-string">'abc'</span>,
    <span class="hljs-string">'cde'</span>,
    <span class="hljs-string">'efg'</span>,
    <span class="hljs-string">'hij'</span>,
    <span class="hljs-string">'nop'</span>
];
(<span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x57239e, _0x4747e8</span>) </span>{
    <span class="hljs-keyword">var</span> _0x3998cd = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x34a502</span>) </span>{
        <span class="hljs-keyword">while</span> (--_0x34a502) {
            _0x57239e[<span class="hljs-string">'push'</span>](_0x57239e[<span class="hljs-string">'shift'</span>]());
        }
    };
    _0x3998cd(++_0x4747e8);
}(_0x402a, <span class="hljs-number">0x162</span>));
<span class="hljs-keyword">var</span> _0x5356 = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x2f2c10, _0x2878a6</span>) </span>{
    _0x2f2c10 = _0x2f2c10 - <span class="hljs-number">0x0</span>;
    <span class="hljs-keyword">var</span> _0x4cfe02 = _0x402a[_0x2f2c10];
    <span class="hljs-keyword">return</span> _0x4cfe02;
};
(<span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
    <span class="hljs-keyword">if</span> (!![]) {
        <span class="hljs-keyword">var</span> _0x60edc1 = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
            <span class="hljs-built_in">console</span>[_0x5356(<span class="hljs-string">'0x0'</span>)](_0x5356(<span class="hljs-string">'0x1'</span>));
            <span class="hljs-built_in">console</span>[_0x5356(<span class="hljs-string">'0x0'</span>)](_0x5356(<span class="hljs-string">'0x2'</span>));
            <span class="hljs-built_in">console</span>[_0x5356(<span class="hljs-string">'0x0'</span>)](_0x5356(<span class="hljs-string">'0x3'</span>));
            <span class="hljs-built_in">console</span>[<span class="hljs-string">'log'</span>](_0x5356(<span class="hljs-string">'0x4'</span>));
        };
        <span class="hljs-keyword">var</span> _0x56405f = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
            <span class="hljs-built_in">console</span>[_0x5356(<span class="hljs-string">'0x0'</span>)](<span class="hljs-string">'klm'</span>);
            <span class="hljs-built_in">console</span>[<span class="hljs-string">'log'</span>](_0x5356(<span class="hljs-string">'0x5'</span>));
            <span class="hljs-built_in">console</span>[<span class="hljs-string">'log'</span>](_0x5356(<span class="hljs-string">'0x6'</span>));
        };
        <span class="hljs-keyword">var</span> _0x332d12 = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
            <span class="hljs-built_in">console</span>[_0x5356(<span class="hljs-string">'0x0'</span>)](<span class="hljs-string">'tuv'</span>);
            <span class="hljs-built_in">console</span>[_0x5356(<span class="hljs-string">'0x0'</span>)](_0x5356(<span class="hljs-string">'0x7'</span>));
            <span class="hljs-built_in">console</span>[<span class="hljs-string">'log'</span>](<span class="hljs-string">'z'</span>);
        };
        _0x60edc1();
        _0x56405f();
        _0x332d12();
    }
}());
</code></pre>
<p>另外我们还可以通过设置 deadCodeInjectionThreshold 参数来控制僵尸代码注入的比例，取值 0 到 1，默认是 0.4。</p>
<p>僵尸代码可以起到一定的干扰作用，所以在有必要的时候也可以注入。</p>
<h5>对象键名替换</h5>
<p>如果是一个对象，可以使用 transformObjectKeys 来对对象的键值进行替换，示例如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">const</span> code = <span class="hljs-string">`
(function(){
    var object = {
        foo: 'test1',
        bar: {
            baz: 'test2'
        }
    };
})(); 
`</span>
<span class="hljs-keyword">const</span> options = {
  <span class="hljs-attr">compact</span>: <span class="hljs-literal">false</span>,
  <span class="hljs-attr">transformObjectKeys</span>: <span class="hljs-literal">true</span>
}
</code></pre>
<p>输出结果如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> _0x7a5d = [
    <span class="hljs-string">'bar'</span>,
    <span class="hljs-string">'test2'</span>,
    <span class="hljs-string">'test1'</span>
];
(<span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x59fec5, _0x2e4fac</span>) </span>{
    <span class="hljs-keyword">var</span> _0x231e7a = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x46f33e</span>) </span>{
        <span class="hljs-keyword">while</span> (--_0x46f33e) {
            _0x59fec5[<span class="hljs-string">'push'</span>](_0x59fec5[<span class="hljs-string">'shift'</span>]());
        }
    };
    _0x231e7a(++_0x2e4fac);
}(_0x7a5d, <span class="hljs-number">0x167</span>));
<span class="hljs-keyword">var</span> _0x3bc4 = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params">_0x309ad3, _0x22d5ac</span>) </span>{
    _0x309ad3 = _0x309ad3 - <span class="hljs-number">0x0</span>;
    <span class="hljs-keyword">var</span> _0x3a034e = _0x7a5d[_0x309ad3];
    <span class="hljs-keyword">return</span> _0x3a034e;
};
(<span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
    <span class="hljs-keyword">var</span> _0x9f1fd1 = {};
    _0x9f1fd1[<span class="hljs-string">'foo'</span>] = _0x3bc4(<span class="hljs-string">'0x0'</span>);
    _0x9f1fd1[_0x3bc4(<span class="hljs-string">'0x1'</span>)] = {};
    _0x9f1fd1[_0x3bc4(<span class="hljs-string">'0x1'</span>)][<span class="hljs-string">'baz'</span>] = _0x3bc4(<span class="hljs-string">'0x2'</span>);
}());
</code></pre>
<p>可以看到，Object 的变量名被替换为了特殊的变量，这也可以起到一定的防护作用。</p>
<h5>禁用控制台输出</h5>
<p>可以使用 disableConsoleOutput 来禁用掉 console.log 输出功能，加大调试难度，示例如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">const</span> code = <span class="hljs-string">`
console.log('hello world')
`</span>
<span class="hljs-keyword">const</span> options = {
    <span class="hljs-attr">disableConsoleOutput</span>: <span class="hljs-literal">true</span>
}
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> _0x3a39=[<span class="hljs-string">'debug'</span>,<span class="hljs-string">'info'</span>,<span class="hljs-string">'error'</span>,<span class="hljs-string">'exception'</span>,<span class="hljs-string">'trace'</span>,<span class="hljs-string">'hello\x20world'</span>,<span class="hljs-string">'apply'</span>,<span class="hljs-string">'{}.constructor(\x22return\x20this\x22)(\x20)'</span>,<span class="hljs-string">'console'</span>,<span class="hljs-string">'log'</span>,<span class="hljs-string">'warn'</span>];(<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x2a157a,_0x5d9d3b</span>)</span>{<span class="hljs-keyword">var</span> _0x488e2c=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x5bcb73</span>)</span>{<span class="hljs-keyword">while</span>(--_0x5bcb73){_0x2a157a[<span class="hljs-string">'push'</span>](_0x2a157a[<span class="hljs-string">'shift'</span>]());}};_0x488e2c(++_0x5d9d3b);}(_0x3a39,<span class="hljs-number">0x10e</span>));<span class="hljs-keyword">var</span> _0x5bff=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x43bdfc,_0x52e4c6</span>)</span>{_0x43bdfc=_0x43bdfc<span class="hljs-number">-0x0</span>;<span class="hljs-keyword">var</span> _0xb67384=_0x3a39[_0x43bdfc];<span class="hljs-keyword">return</span> _0xb67384;};<span class="hljs-keyword">var</span> _0x349b01=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">var</span> _0x1f484b=!![];<span class="hljs-keyword">return</span> <span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x5efe0d,_0x33db62</span>)</span>{<span class="hljs-keyword">var</span> _0x20bcd2=_0x1f484b?<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">if</span>(_0x33db62){<span class="hljs-keyword">var</span> _0x77054c=_0x33db62[_0x5bff(<span class="hljs-string">'0x0'</span>)](_0x5efe0d,<span class="hljs-built_in">arguments</span>);_0x33db62=<span class="hljs-literal">null</span>;<span class="hljs-keyword">return</span> _0x77054c;}}:<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{};_0x1f484b=![];<span class="hljs-keyword">return</span> _0x20bcd2;};}();<span class="hljs-keyword">var</span> _0x19f538=_0x349b01(<span class="hljs-keyword">this</span>,<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">var</span> _0x7ab6e4=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{};<span class="hljs-keyword">var</span> _0x157bff;<span class="hljs-keyword">try</span>{<span class="hljs-keyword">var</span> _0x5e672c=<span class="hljs-built_in">Function</span>(<span class="hljs-string">'return\x20(function()\x20'</span>+_0x5bff(<span class="hljs-string">'0x1'</span>)+<span class="hljs-string">');'</span>);_0x157bff=_0x5e672c();}<span class="hljs-keyword">catch</span>(_0x11028d){_0x157bff=<span class="hljs-built_in">window</span>;}<span class="hljs-keyword">if</span>(!_0x157bff[_0x5bff(<span class="hljs-string">'0x2'</span>)]){_0x157bff[_0x5bff(<span class="hljs-string">'0x2'</span>)]=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x7ab6e4</span>)</span>{<span class="hljs-keyword">var</span> _0x5a8d9e={};_0x5a8d9e[_0x5bff(<span class="hljs-string">'0x3'</span>)]=_0x7ab6e4;_0x5a8d9e[_0x5bff(<span class="hljs-string">'0x4'</span>)]=_0x7ab6e4;_0x5a8d9e[_0x5bff(<span class="hljs-string">'0x5'</span>)]=_0x7ab6e4;_0x5a8d9e[_0x5bff(<span class="hljs-string">'0x6'</span>)]=_0x7ab6e4;_0x5a8d9e[_0x5bff(<span class="hljs-string">'0x7'</span>)]=_0x7ab6e4;_0x5a8d9e[_0x5bff(<span class="hljs-string">'0x8'</span>)]=_0x7ab6e4;_0x5a8d9e[_0x5bff(<span class="hljs-string">'0x9'</span>)]=_0x7ab6e4;<span class="hljs-keyword">return</span> _0x5a8d9e;}(_0x7ab6e4);}<span class="hljs-keyword">else</span>{_0x157bff[_0x5bff(<span class="hljs-string">'0x2'</span>)][_0x5bff(<span class="hljs-string">'0x3'</span>)]=_0x7ab6e4;_0x157bff[_0x5bff(<span class="hljs-string">'0x2'</span>)][_0x5bff(<span class="hljs-string">'0x4'</span>)]=_0x7ab6e4;_0x157bff[_0x5bff(<span class="hljs-string">'0x2'</span>)][<span class="hljs-string">'debug'</span>]=_0x7ab6e4;_0x157bff[_0x5bff(<span class="hljs-string">'0x2'</span>)][_0x5bff(<span class="hljs-string">'0x6'</span>)]=_0x7ab6e4;_0x157bff[_0x5bff(<span class="hljs-string">'0x2'</span>)][_0x5bff(<span class="hljs-string">'0x7'</span>)]=_0x7ab6e4;_0x157bff[_0x5bff(<span class="hljs-string">'0x2'</span>)][_0x5bff(<span class="hljs-string">'0x8'</span>)]=_0x7ab6e4;_0x157bff[_0x5bff(<span class="hljs-string">'0x2'</span>)][_0x5bff(<span class="hljs-string">'0x9'</span>)]=_0x7ab6e4;}});_0x19f538();<span class="hljs-built_in">console</span>[_0x5bff(<span class="hljs-string">'0x3'</span>)](_0x5bff(<span class="hljs-string">'0xa'</span>));
</code></pre>
<p>此时，我们如果执行这段代码，发现是没有任何输出的，这里实际上就是将 console 的一些功能禁用了，加大了调试难度。</p>
<h5>调试保护</h5>
<p>我们可以使用 debugProtection 来禁用调试模式，进入无限 Debug 模式。另外我们还可以使用 debugProtectionInterval 来启用无限 Debug 的间隔，使得代码在调试过程中会不断进入断点模式，无法顺畅执行。<br>
示例如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">const</span> code = <span class="hljs-string">`
for (let i = 0; i &lt; 5; i ++) {
    console.log('i', i)
}
`</span>
<span class="hljs-keyword">const</span> options = {
    <span class="hljs-attr">debugProtection</span>: <span class="hljs-literal">true</span>
}
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> _0x41d0=[<span class="hljs-string">'action'</span>,<span class="hljs-string">'debu'</span>,<span class="hljs-string">'stateObject'</span>,<span class="hljs-string">'function\x20*\x5c(\x20*\x5c)'</span>,<span class="hljs-string">'\x5c+\x5c+\x20*(?:_0x(?:[a-f0-9]){4,6}|(?:\x5cb|\x5cd)[a-z0-9]{1,4}(?:\x5cb|\x5cd))'</span>,<span class="hljs-string">'init'</span>,<span class="hljs-string">'test'</span>,<span class="hljs-string">'chain'</span>,<span class="hljs-string">'input'</span>,<span class="hljs-string">'log'</span>,<span class="hljs-string">'string'</span>,<span class="hljs-string">'constructor'</span>,<span class="hljs-string">'while\x20(true)\x20{}'</span>,<span class="hljs-string">'apply'</span>,<span class="hljs-string">'gger'</span>,<span class="hljs-string">'call'</span>];(<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x69147e,_0x180e03</span>)</span>{<span class="hljs-keyword">var</span> _0x2cc589=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x18d18c</span>)</span>{<span class="hljs-keyword">while</span>(--_0x18d18c){_0x69147e[<span class="hljs-string">'push'</span>](_0x69147e[<span class="hljs-string">'shift'</span>]());}};_0x2cc589(++_0x180e03);}(_0x41d0,<span class="hljs-number">0x153</span>));<span class="hljs-keyword">var</span> _0x16d2=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x3d813e,_0x59f7b2</span>)</span>{_0x3d813e=_0x3d813e<span class="hljs-number">-0x0</span>;<span class="hljs-keyword">var</span> _0x228f98=_0x41d0[_0x3d813e];<span class="hljs-keyword">return</span> _0x228f98;};<span class="hljs-keyword">var</span> _0x241eee=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">var</span> _0xeb17=!![];<span class="hljs-keyword">return</span> <span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x5caffe,_0x2bb267</span>)</span>{<span class="hljs-keyword">var</span> _0x16e1bf=_0xeb17?<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">if</span>(_0x2bb267){<span class="hljs-keyword">var</span> _0x573619=_0x2bb267[<span class="hljs-string">'apply'</span>](_0x5caffe,<span class="hljs-built_in">arguments</span>);_0x2bb267=<span class="hljs-literal">null</span>;<span class="hljs-keyword">return</span> _0x573619;}}:<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{};_0xeb17=![];<span class="hljs-keyword">return</span> _0x16e1bf;};}();(<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{_0x241eee(<span class="hljs-keyword">this</span>,<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">var</span> _0x5de4a4=<span class="hljs-keyword">new</span> <span class="hljs-built_in">RegExp</span>(_0x16d2(<span class="hljs-string">'0x0'</span>));<span class="hljs-keyword">var</span> _0x4a170e=<span class="hljs-keyword">new</span> <span class="hljs-built_in">RegExp</span>(_0x16d2(<span class="hljs-string">'0x1'</span>),<span class="hljs-string">'i'</span>);<span class="hljs-keyword">var</span> _0x5351d7=_0x227210(_0x16d2(<span class="hljs-string">'0x2'</span>));<span class="hljs-keyword">if</span>(!_0x5de4a4[_0x16d2(<span class="hljs-string">'0x3'</span>)](_0x5351d7+_0x16d2(<span class="hljs-string">'0x4'</span>))||!_0x4a170e[_0x16d2(<span class="hljs-string">'0x3'</span>)](_0x5351d7+_0x16d2(<span class="hljs-string">'0x5'</span>))){_0x5351d7(<span class="hljs-string">'0'</span>);}<span class="hljs-keyword">else</span>{_0x227210();}})();}());<span class="hljs-keyword">for</span>(<span class="hljs-keyword">let</span> i=<span class="hljs-number">0x0</span>;i&lt;<span class="hljs-number">0x5</span>;i++){<span class="hljs-built_in">console</span>[_0x16d2(<span class="hljs-string">'0x6'</span>)](<span class="hljs-string">'i'</span>,i);}<span class="hljs-function"><span class="hljs-keyword">function</span> <span class="hljs-title">_0x227210</span>(<span class="hljs-params">_0x30bc32</span>)</span>{<span class="hljs-function"><span class="hljs-keyword">function</span> <span class="hljs-title">_0x1971c7</span>(<span class="hljs-params">_0x19628c</span>)</span>{<span class="hljs-keyword">if</span>(<span class="hljs-keyword">typeof</span> _0x19628c===_0x16d2(<span class="hljs-string">'0x7'</span>)){<span class="hljs-keyword">return</span> <span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x3718f7</span>)</span>{}[_0x16d2(<span class="hljs-string">'0x8'</span>)](_0x16d2(<span class="hljs-string">'0x9'</span>))[_0x16d2(<span class="hljs-string">'0xa'</span>)](<span class="hljs-string">'counter'</span>);}<span class="hljs-keyword">else</span>{<span class="hljs-keyword">if</span>((<span class="hljs-string">''</span>+_0x19628c/_0x19628c)[<span class="hljs-string">'length'</span>]!==<span class="hljs-number">0x1</span>||_0x19628c%<span class="hljs-number">0x14</span>===<span class="hljs-number">0x0</span>){(<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">return</span>!![];}[_0x16d2(<span class="hljs-string">'0x8'</span>)](<span class="hljs-string">'debu'</span>+_0x16d2(<span class="hljs-string">'0xb'</span>))[_0x16d2(<span class="hljs-string">'0xc'</span>)](_0x16d2(<span class="hljs-string">'0xd'</span>)));}<span class="hljs-keyword">else</span>{(<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">return</span>![];}[_0x16d2(<span class="hljs-string">'0x8'</span>)](_0x16d2(<span class="hljs-string">'0xe'</span>)+_0x16d2(<span class="hljs-string">'0xb'</span>))[_0x16d2(<span class="hljs-string">'0xa'</span>)](_0x16d2(<span class="hljs-string">'0xf'</span>)));}}_0x1971c7(++_0x19628c);}<span class="hljs-keyword">try</span>{<span class="hljs-keyword">if</span>(_0x30bc32){<span class="hljs-keyword">return</span> _0x1971c7;}<span class="hljs-keyword">else</span>{_0x1971c7(<span class="hljs-number">0x0</span>);}}<span class="hljs-keyword">catch</span>(_0x58d434){}}
</code></pre>
<p>如果我们将代码粘贴到控制台，其会不断跳到 debugger 代码的位置，无法顺畅执行。</p>
<h5>域名锁定</h5>
<p>我们可以通过控制 domainLock 来控制 JavaScript 代码只能在特定域名下运行，这样就可以降低被模拟的风险。</p>
<p>示例如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">const</span> code = <span class="hljs-string">`
console.log('hello world')
`</span>
<span class="hljs-keyword">const</span> options = {
    <span class="hljs-attr">domainLock</span>: [<span class="hljs-string">'cuiqingcai.com'</span>]
}
</code></pre>
<p>运行结果如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> _0x3203=[<span class="hljs-string">'apply'</span>,<span class="hljs-string">'return\x20(function()\x20'</span>,<span class="hljs-string">'{}.constructor(\x22return\x20this\x22)(\x20)'</span>,<span class="hljs-string">'item'</span>,<span class="hljs-string">'attribute'</span>,<span class="hljs-string">'value'</span>,<span class="hljs-string">'replace'</span>,<span class="hljs-string">'length'</span>,<span class="hljs-string">'charCodeAt'</span>,<span class="hljs-string">'log'</span>,<span class="hljs-string">'hello\x20world'</span>];(<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x2ed22c,_0x3ad370</span>)</span>{<span class="hljs-keyword">var</span> _0x49dc54=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0x53a786</span>)</span>{<span class="hljs-keyword">while</span>(--_0x53a786){_0x2ed22c[<span class="hljs-string">'push'</span>](_0x2ed22c[<span class="hljs-string">'shift'</span>]());}};_0x49dc54(++_0x3ad370);}(_0x3203,<span class="hljs-number">0x155</span>));<span class="hljs-keyword">var</span> _0x5b38=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0xd7780b,_0x19c0f2</span>)</span>{_0xd7780b=_0xd7780b<span class="hljs-number">-0x0</span>;<span class="hljs-keyword">var</span> _0x2d2f44=_0x3203[_0xd7780b];<span class="hljs-keyword">return</span> _0x2d2f44;};<span class="hljs-keyword">var</span> _0x485919=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">var</span> _0x5cf798=!![];<span class="hljs-keyword">return</span> <span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params">_0xd1fa29,_0x2ed646</span>)</span>{<span class="hljs-keyword">var</span> _0x56abf=_0x5cf798?<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">if</span>(_0x2ed646){<span class="hljs-keyword">var</span> _0x33af63=_0x2ed646[_0x5b38(<span class="hljs-string">'0x0'</span>)](_0xd1fa29,<span class="hljs-built_in">arguments</span>);_0x2ed646=<span class="hljs-literal">null</span>;<span class="hljs-keyword">return</span> _0x33af63;}}:<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{};_0x5cf798=![];<span class="hljs-keyword">return</span> _0x56abf;};}();<span class="hljs-keyword">var</span> _0x67dcc8=_0x485919(<span class="hljs-keyword">this</span>,<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">var</span> _0x276a31;<span class="hljs-keyword">try</span>{<span class="hljs-keyword">var</span> _0x5c8be2=<span class="hljs-built_in">Function</span>(_0x5b38(<span class="hljs-string">'0x1'</span>)+_0x5b38(<span class="hljs-string">'0x2'</span>)+<span class="hljs-string">');'</span>);_0x276a31=_0x5c8be2();}<span class="hljs-keyword">catch</span>(_0x5f1c00){_0x276a31=<span class="hljs-built_in">window</span>;}<span class="hljs-keyword">var</span> _0x254a0d=<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">return</span>{<span class="hljs-string">'key'</span>:_0x5b38(<span class="hljs-string">'0x3'</span>),<span class="hljs-string">'value'</span>:_0x5b38(<span class="hljs-string">'0x4'</span>),<span class="hljs-string">'getAttribute'</span>:<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>)</span>{<span class="hljs-keyword">for</span>(<span class="hljs-keyword">var</span> _0x5cc3c7=<span class="hljs-number">0x0</span>;_0x5cc3c7&lt;<span class="hljs-number">0x3e8</span>;_0x5cc3c7--){<span class="hljs-keyword">var</span> _0x35b30b=_0x5cc3c7&gt;<span class="hljs-number">0x0</span>;<span class="hljs-keyword">switch</span>(_0x35b30b){<span class="hljs-keyword">case</span>!![]:<span class="hljs-keyword">return</span> <span class="hljs-keyword">this</span>[_0x5b38(<span class="hljs-string">'0x3'</span>)]+<span class="hljs-string">'_'</span>+<span class="hljs-keyword">this</span>[_0x5b38(<span class="hljs-string">'0x5'</span>)]+<span class="hljs-string">'_'</span>+_0x5cc3c7;<span class="hljs-keyword">default</span>:<span class="hljs-keyword">this</span>[_0x5b38(<span class="hljs-string">'0x3'</span>)]+<span class="hljs-string">'_'</span>+<span class="hljs-keyword">this</span>[_0x5b38(<span class="hljs-string">'0x5'</span>)];}}}()};};<span class="hljs-keyword">var</span> _0x3b375a=<span class="hljs-keyword">new</span> <span class="hljs-built_in">RegExp</span>(<span class="hljs-string">'[QLCIKYkCFzdWpzRAXMhxJOYpTpYWJHPll]'</span>,<span class="hljs-string">'g'</span>);<span class="hljs-keyword">var</span> _0x5a94d2=<span class="hljs-string">'cuQLiqiCInKYkgCFzdWcpzRAaXMi.hcoxmJOYpTpYWJHPll'</span>[_0x5b38(<span class="hljs-string">'0x6'</span>)](_0x3b375a,<span class="hljs-string">''</span>)[<span class="hljs-string">'split'</span>](<span class="hljs-string">';'</span>);<span class="hljs-keyword">var</span> _0x5c0da2;<span class="hljs-keyword">var</span> _0x19ad5d;<span class="hljs-keyword">var</span> _0x5992ca;<span class="hljs-keyword">var</span> _0x40bd39;<span class="hljs-keyword">for</span>(<span class="hljs-keyword">var</span> _0x5cad1 <span class="hljs-keyword">in</span> _0x276a31){<span class="hljs-keyword">if</span>(_0x5cad1[_0x5b38(<span class="hljs-string">'0x7'</span>)]==<span class="hljs-number">0x8</span>&amp;&amp;_0x5cad1[_0x5b38(<span class="hljs-string">'0x8'</span>)](<span class="hljs-number">0x7</span>)==<span class="hljs-number">0x74</span>&amp;&amp;_0x5cad1[_0x5b38(<span class="hljs-string">'0x8'</span>)](<span class="hljs-number">0x5</span>)==<span class="hljs-number">0x65</span>&amp;&amp;_0x5cad1[_0x5b38(<span class="hljs-string">'0x8'</span>)](<span class="hljs-number">0x3</span>)==<span class="hljs-number">0x75</span>&amp;&amp;_0x5cad1[_0x5b38(<span class="hljs-string">'0x8'</span>)](<span class="hljs-number">0x0</span>)==<span class="hljs-number">0x64</span>){_0x5c0da2=_0x5cad1;<span class="hljs-keyword">break</span>;}}<span class="hljs-keyword">for</span>(<span class="hljs-keyword">var</span> _0x29551 <span class="hljs-keyword">in</span> _0x276a31[_0x5c0da2]){<span class="hljs-keyword">if</span>(_0x29551[_0x5b38(<span class="hljs-string">'0x7'</span>)]==<span class="hljs-number">0x6</span>&amp;&amp;_0x29551[_0x5b38(<span class="hljs-string">'0x8'</span>)](<span class="hljs-number">0x5</span>)==<span class="hljs-number">0x6e</span>&amp;&amp;_0x29551[_0x5b38(<span class="hljs-string">'0x8'</span>)](<span class="hljs-number">0x0</span>)==<span class="hljs-number">0x64</span>){_0x19ad5d=_0x29551;<span class="hljs-keyword">break</span>;}}<span class="hljs-keyword">if</span>(!(<span class="hljs-string">'~'</span>&gt;_0x19ad5d)){<span class="hljs-keyword">for</span>(<span class="hljs-keyword">var</span> _0x2b71bd <span class="hljs-keyword">in</span> _0x276a31[_0x5c0da2]){<span class="hljs-keyword">if</span>(_0x2b71bd[_0x5b38(<span class="hljs-string">'0x7'</span>)]==<span class="hljs-number">0x8</span>&amp;&amp;_0x2b71bd[_0x5b38(<span class="hljs-string">'0x8'</span>)](<span class="hljs-number">0x7</span>)==<span class="hljs-number">0x6e</span>&amp;&amp;_0x2b71bd[_0x5b38(<span class="hljs-string">'0x8'</span>)](<span class="hljs-number">0x0</span>)==<span class="hljs-number">0x6c</span>){_0x5992ca=_0x2b71bd;<span class="hljs-keyword">break</span>;}}<span class="hljs-keyword">for</span>(<span class="hljs-keyword">var</span> _0x397f55 <span class="hljs-keyword">in</span> _0x276a31[_0x5c0da2][_0x5992ca]){<span class="hljs-keyword">if</span>(_0x397f55[<span class="hljs-string">'length'</span>]==<span class="hljs-number">0x8</span>&amp;&amp;_0x397f55[_0x5b38(<span class="hljs-string">'0x8'</span>)](<span class="hljs-number">0x7</span>)==<span class="hljs-number">0x65</span>&amp;&amp;_0x397f55[_0x5b38(<span class="hljs-string">'0x8'</span>)](<span class="hljs-number">0x0</span>)==<span class="hljs-number">0x68</span>){_0x40bd39=_0x397f55;<span class="hljs-keyword">break</span>;}}}<span class="hljs-keyword">if</span>(!_0x5c0da2||!_0x276a31[_0x5c0da2]){<span class="hljs-keyword">return</span>;}<span class="hljs-keyword">var</span> _0x5f19be=_0x276a31[_0x5c0da2][_0x19ad5d];<span class="hljs-keyword">var</span> _0x674f76=!!_0x276a31[_0x5c0da2][_0x5992ca]&amp;&amp;_0x276a31[_0x5c0da2][_0x5992ca][_0x40bd39];<span class="hljs-keyword">var</span> _0x5e1b34=_0x5f19be||_0x674f76;<span class="hljs-keyword">if</span>(!_0x5e1b34){<span class="hljs-keyword">return</span>;}<span class="hljs-keyword">var</span> _0x593394=![];<span class="hljs-keyword">for</span>(<span class="hljs-keyword">var</span> _0x479239=<span class="hljs-number">0x0</span>;_0x479239&lt;_0x5a94d2[<span class="hljs-string">'length'</span>];_0x479239++){<span class="hljs-keyword">var</span> _0x19ad5d=_0x5a94d2[_0x479239];<span class="hljs-keyword">var</span> _0x112c24=_0x5e1b34[<span class="hljs-string">'length'</span>]-_0x19ad5d[<span class="hljs-string">'length'</span>];<span class="hljs-keyword">var</span> _0x51731c=_0x5e1b34[<span class="hljs-string">'indexOf'</span>](_0x19ad5d,_0x112c24);<span class="hljs-keyword">var</span> _0x173191=_0x51731c!==<span class="hljs-number">-0x1</span>&amp;&amp;_0x51731c===_0x112c24;<span class="hljs-keyword">if</span>(_0x173191){<span class="hljs-keyword">if</span>(_0x5e1b34[<span class="hljs-string">'length'</span>]==_0x19ad5d[_0x5b38(<span class="hljs-string">'0x7'</span>)]||_0x19ad5d[<span class="hljs-string">'indexOf'</span>](<span class="hljs-string">'.'</span>)===<span class="hljs-number">0x0</span>){_0x593394=!![];}}}<span class="hljs-keyword">if</span>(!_0x593394){data;}<span class="hljs-keyword">else</span>{<span class="hljs-keyword">return</span>;}_0x254a0d();});_0x67dcc8();<span class="hljs-built_in">console</span>[_0x5b38(<span class="hljs-string">'0x9'</span>)](_0x5b38(<span class="hljs-string">'0xa'</span>));
</code></pre>
<p>这段代码就只能在指定域名 cuiqingcai.com 下运行，不能在其他网站运行，不信你可以试试。</p>
<h5>特殊编码</h5>
<p>另外还有一些特殊的工具包，如使用 aaencode、jjencode、jsfuck 等工具对代码进行混淆和编码。</p>
<p>示例如下：</p>
<pre><code data-language="js" class="lang-js"><span class="hljs-keyword">var</span> a = <span class="hljs-number">1</span>
</code></pre>
<p>jsfuck 的结果：</p>
<pre><code data-language="js" class="lang-js">[][(![]+[])[!+[]+!![]+!![]]+([]+{})[+!![]]+(!![]+[])[+!![]]+(!![]+[])[+[]]][([]+{})[!+[]+!![]+!![]+!![]+!![]]+([]+{})[+!![]]+([][[]]+[])[+!![]]+(![]+[])[!+[]+!![]+!![]]+(!![]+[])[+[]]+(!![]+[])[+!![]]+([][[]]+[])[+[]]+([]+{})[!+[]+!![]+!![]+!![]+!![]]+(!![]+[])[+[]]+([]+{})[+!![]]+(!![]+[])[+!![]]]([][(![]+[])[!+[]+!![]+!![]]+([]+{})[+!![]]+(!![]+[])[+!![]]+(!![]+[])[+[]]][([]+{})[!+[]+!![]+!![]+!![]+!![]]+([]+{})[+!![]]+([][[]]+[])[+!![]]+(![]+[])[!+[]+!![]+!![]]+(!![]+[])[+[]]+(!![]+[])[+!![]]+([][[]]+[])[+[]]+([]+{})[!+[]+!![]+!![]+!![]+!![]]+(!![]+[])[+[]]+([]+{})[+!![]]+(!![]+[])[+!![]]]((!![]+[])[+!![]]+([][[]]+[])[!+[]+!![]+!![]]+(!![]+[])[+[]]+([][[]]+[])[+[]]+(!![]+[])[+!![]]+([][[]]+[])[+!![]]+([]+{})[!+[]+!![]+!![]+!![]+!![]+!![]+!![]]+([][[]]+[])[+[]]+([][[]]+[])[+!![]]+([][[]]+[])[!+[]+!![]+!![]]+(![]+[])[!+[]+!![]+!![]]+([]+{})[!+[]+!![]+!![]+!![]+!![]]+(+{}+[])[+!![]]+([]+[][(![]+[])[!+[]+!![]+!![]]+([]+{})[+!![]]+(!![]+[])[+!![]]+(!![]+[])[+[]]][([]+{})[!+[]+!![]+!![]+!![]+!![]]+([]+{})[+!![]]+([][[]]+[])[+!![]]+(![]+[])[!+[]+!![]+!![]]+(!![]+[])[+[]]+(!![]+[])[+!![]]+([][[]]+[])[+[]]+([]+{})[!+[]+!![]+!![]+!![]+!![]]+(!![]+[])[+[]]+([]+{})[+!![]]+(!![]+[])[+!![]]]((!![]+[])[+!![]]+([][[]]+[])[!+[]+!![]+!![]]+(!![]+[])[+[]]+([][[]]+[])[+[]]+(!![]+[])[+!![]]+([][[]]+[])[+!![]]+([]+{})[!+[]+!![]+!![]+!![]+!![]+!![]+!![]]+(![]+[])[!+[]+!![]]+([]+{})[+!![]]+([]+{})[!+[]+!![]+!![]+!![]+!![]]+(+{}+[])[+!![]]+(!![]+[])[+[]]+([][[]]+[])[!+[]+!![]+!![]+!![]+!![]]+([]+{})[+!![]]+([][[]]+[])[+!![]])(!+[]+!![]+!![]+!![]+!![]))[!+[]+!![]+!![]]+([][[]]+[])[!+[]+!![]+!![]])(!+[]+!![]+!![]+!![])([][(![]+[])[!+[]+!![]+!![]]+([]+{})[+!![]]+(!![]+[])[+!![]]+(!![]+[])[+[]]][([]+{})[!+[]+!![]+!![]+!![]+!![]]+([]+{})[+!![]]+([][[]]+[])[+!![]]+(![]+[])[!+[]+!![]+!![]]+(!![]+[])[+[]]+(!![]+[])[+!![]]+([][[]]+[])[+[]]+([]+{})[!+[]+!![]+!![]+!![]+!![]]+(!![]+[])[+[]]+([]+{})[+!![]]+(!![]+[])[+!![]]]((!![]+[])[+!![]]+([][[]]+[])[!+[]+!![]+!![]]+(!![]+[])[+[]]+([][[]]+[])[+[]]+(!![]+[])[+!![]]+([][[]]+[])[+!![]]+([]+{})[!+[]+!![]+!![]+!![]+!![]+!![]+!![]]+([][[]]+[])[!+[]+!![]+!![]]+(![]+[])[!+[]+!![]+!![]]+([]+{})[!+[]+!![]+!![]+!![]+!![]]+(+{}+[])[+!![]]+([]+[][(![]+[])[!+[]+!![]+!![]]+([]+{})[+!![]]+(!![]+[])[+!![]]+(!![]+[])[+[]]][([]+{})[!+[]+!![]+!![]+!![]+!![]]+([]+{})[+!![]]+([][[]]+[])[+!![]]+(![]+[])[!+[]+!![]+!![]]+(!![]+[])[+[]]+(!![]+[])[+!![]]+([][[]]+[])[+[]]+([]+{})[!+[]+!![]+!![]+!![]+!![]]+(!![]+[])[+[]]+([]+{})[+!![]]+(!![]+[])[+!![]]]((!![]+[])[+!![]]+([][[]]+[])[!+[]+!![]+!![]]+(!![]+[])[+[]]+([][[]]+[])[+[]]+(!![]+[])[+!![]]+([][[]]+[])[+!![]]+([]+{})[!+[]+!![]+!![]+!![]+!![]+!![]+!![]]+(![]+[])[!+[]+!![]]+([]+{})[+!![]]+([]+{})[!+[]+!![]+!![]+!![]+!![]]+(+{}+[])[+!![]]+(!![]+[])[+[]]+([][[]]+[])[!+[]+!![]+!![]+!![]+!![]]+([]+{})[+!![]]+([][[]]+[])[+!![]])(!+[]+!![]+!![]+!![]+!![]))[!+[]+!![]+!![]]+([][[]]+[])[!+[]+!![]+!![]])(!+[]+!![]+!![]+!![]+!![])(([]+{})[+[]])[+[]]+(!+[]+!![]+!![]+!![]+!![]+!![]+!![]+[])+(!+[]+!![]+!![]+!![]+!![]+!![]+[]))+(+{}+[])[+!![]]+(!![]+[])[+!![]]+([]+{})[!+[]+!![]+!![]+!![]+!![]+!![]+!![]]+(+{}+[])[+!![]]+([]+{})[!+[]+!![]+!![]+!![]+!![]+!![]+!![]]+[][(![]+[])[!+[]+!![]+!![]]+([]+{})[+!![]]+(!![]+[])[+!![]]+(!![]+[])[+[]]][([]+{})[!+[]+!![]+!![]+!![]+!![]]+([]+{})[+!![]]+([][[]]+[])[+!![]]+(![]+[])[!+[]+!![]+!![]]+(!![]+[])[+[]]+(!![]+[])[+!![]]+([][[]]+[])[+[]]+([]+{})[!+[]+!![]+!![]+!![]+!![]]+(!![]+[])[+[]]+([]+{})[+!![]]+(!![]+[])[+!![]]]((!![]+[])[+!![]]+([][[]]+[])[!+[]+!![]+!![]]+(!![]+[])[+[]]+([][[]]+[])[+[]]+(!![]+[])[+!![]]+([][[]]+[])[+!![]]+([]+{})[!+[]+!![]+!![]+!![]+!![]+!![]+!![]]+([][[]]+[])[+[]]+([][[]]+[])[+!![]]+([][[]]+[])[!+[]+!![]+!![]]+(![]+[])[!+[]+!![]+!![]]+([]+{})[!+[]+!![]+!![]+!![]+!![]]+(+{}+[])[+!![]]+([]+[][(![]+[])[!+[]+!![]+!![]]+([]+{})[+!![]]+(!![]+[])[+!![]]+(!![]+[])[+[]]][([]+{})[!+[]+!![]+!![]+!![]+!![]]+([]+{})[+!![]]+([][[]]+[])[+!![]]+(![]+[])[!+[]+!![]+!![]]+(!![]+[])[+[]]+(!![]+[])[+!![]]+([][[]]+[])[+[]]+([]+{})[!+[]+!![]+!![]+!![]+!![]]+(!![]+[])[+[]]+([]+{})[+!![]]+(!![]+[])[+!![]]]((!![]+[])[+!![]]+([][[]]+[])[!+[]+!![]+!![]]+(!![]+[])[+[]]+([][[]]+[])[+[]]+(!![]+[])[+!![]]+([][[]]+[])[+!![]]+([]+{})[!+[]+!![]+!![]+!![]+!![]+!![]+!![]]+(![]+[])[!+[]+!![]]+([]+{})[+!![]]+([]+{})[!+[]+!![]+!![]+!![]+!![]]+(+{}+[])[+!![]]+(!![]+[])[+[]]+([][[]]+[])[!+[]+!![]+!![]+!![]+!![]]+([]+{})[+!![]]+([][[]]+[])[+!![]])(!+[]+!![]+!![]+!![]+!![]))[!+[]+!![]+!![]]+([][[]]+[])[!+[]+!![]+!![]])(!+[]+!![]+!![]+!![])([][(![]+[])[!+[]+!![]+!![]]+([]+{})[+!![]]+(!![]+[])[+!![]]+(!![]+[])[+[]]][([]+{})[!+[]+!![]+!![]+!![]+!![]]+([]+{})[+!![]]+([][[]]+[])[+!![]]+(![]+[])[!+[]+!![]+!![]]+(!![]+[])[+[]]+(!![]+[])[+!![]]+([][[]]+[])[+[]]+([]+{})[!+[]+!![]+!![]+!![]+!![]]+(!![]+[])[+[]]+([]+{})[+!![]]+(!![]+[])[+!![]]]((!![]+[])[+!![]]+([][[]]+[])[!+[]+!![]+!![]]+(!![]+[])[+[]]+([][[]]+[])[+[]]+(!![]+[])[+!![]]+([][[]]+[])[+!![]]+([]+{})[!+[]+!![]+!![]+!![]+!![]+!![]+!![]]+([][[]]+[])[!+[]+!![]+!![]]+(![]+[])[!+[]+!![]+!![]]+([]+{})[!+[]+!![]+!![]+!![]+!![]]+(+{}+[])[+!![]]+([]+[][(![]+[])[!+[]+!![]+!![]]+([]+{})[+!![]]+(!![]+[])[+!![]]+(!![]+[])[+[]]][([]+{})[!+[]+!![]+!![]+!![]+!![]]+([]+{})[+!![]]+([][[]]+[])[+!![]]+(![]+[])[!+[]+!![]+!![]]+(!![]+[])[+[]]+(!![]+[])[+!![]]+([][[]]+[])[+[]]+([]+{})[!+[]+!![]+!![]+!![]+!![]]+(!![]+[])[+[]]+([]+{})[+!![]]+(!![]+[])[+!![]]]((!![]+[])[+!![]]+([][[]]+[])[!+[]+!![]+!![]]+(!![]+[])[+[]]+([][[]]+[])[+[]]+(!![]+[])[+!![]]+([][[]]+[])[+!![]]+([]+{})[!+[]+!![]+!![]+!![]+!![]+!![]+!![]]+(![]+[])[!+[]+!![]]+([]+{})[+!![]]+([]+{})[!+[]+!![]+!![]+!![]+!![]]+(+{}+[])[+!![]]+(!![]+[])[+[]]+([][[]]+[])[!+[]+!![]+!![]+!![]+!![]]+([]+{})[+!![]]+([][[]]+[])[+!![]])(!+[]+!![]+!![]+!![]+!![]))[!+[]+!![]+!![]]+([][[]]+[])[!+[]+!![]+!![]])(!+[]+!![]+!![]+!![]+!![])(([]+{})[+[]])[+[]]+(!+[]+!![]+!![]+[])+([][[]]+[])[!+[]+!![]])+([]+{})[!+[]+!![]+!![]+!![]+!![]+!![]+!![]]+(+!![]+[]))(!+[]+!![]+!![]+!![]+!![]+!![]+!![]+!![])
</code></pre>
<p>aaencode 的结果：</p>
<pre><code data-language="js" class="lang-js">ﾟωﾟﾉ= <span class="hljs-regexp">/｀ｍ´）ﾉ ~┻━┻ /</span> [<span class="hljs-string">'_'</span>]; o=(ﾟｰﾟ) =_=<span class="hljs-number">3</span>; c=(ﾟΘﾟ) =(ﾟｰﾟ)-(ﾟｰﾟ); (ﾟДﾟ) =(ﾟΘﾟ)= (o^_^o)/ (o^_^o);(ﾟДﾟ)={ﾟΘﾟ: <span class="hljs-string">'_'</span> ,ﾟωﾟﾉ : ((ﾟωﾟﾉ==<span class="hljs-number">3</span>) +<span class="hljs-string">'_'</span>) [ﾟΘﾟ] ,ﾟｰﾟﾉ :(ﾟωﾟﾉ+ <span class="hljs-string">'_'</span>)[o^_^o -(ﾟΘﾟ)] ,ﾟДﾟﾉ:((ﾟｰﾟ==<span class="hljs-number">3</span>) +<span class="hljs-string">'_'</span>)[ﾟｰﾟ] }; (ﾟДﾟ) [ﾟΘﾟ] =((ﾟωﾟﾉ==<span class="hljs-number">3</span>) +<span class="hljs-string">'_'</span>) [c^_^o];(ﾟДﾟ) [<span class="hljs-string">'c'</span>] = ((ﾟДﾟ)+<span class="hljs-string">'_'</span>) [ (ﾟｰﾟ)+(ﾟｰﾟ)-(ﾟΘﾟ) ];(ﾟДﾟ) [<span class="hljs-string">'o'</span>] = ((ﾟДﾟ)+<span class="hljs-string">'_'</span>) [ﾟΘﾟ];(ﾟoﾟ)=(ﾟДﾟ) [<span class="hljs-string">'c'</span>]+(ﾟДﾟ) [<span class="hljs-string">'o'</span>]+(ﾟωﾟﾉ +<span class="hljs-string">'_'</span>)[ﾟΘﾟ]+ ((ﾟωﾟﾉ==<span class="hljs-number">3</span>) +<span class="hljs-string">'_'</span>) [ﾟｰﾟ] + ((ﾟДﾟ) +<span class="hljs-string">'_'</span>) [(ﾟｰﾟ)+(ﾟｰﾟ)]+ ((ﾟｰﾟ==<span class="hljs-number">3</span>) +<span class="hljs-string">'_'</span>) [ﾟΘﾟ]+((ﾟｰﾟ==<span class="hljs-number">3</span>) +<span class="hljs-string">'_'</span>) [(ﾟｰﾟ) - (ﾟΘﾟ)]+(ﾟДﾟ) [<span class="hljs-string">'c'</span>]+((ﾟДﾟ)+<span class="hljs-string">'_'</span>) [(ﾟｰﾟ)+(ﾟｰﾟ)]+ (ﾟДﾟ) [<span class="hljs-string">'o'</span>]+((ﾟｰﾟ==<span class="hljs-number">3</span>) +<span class="hljs-string">'_'</span>) [ﾟΘﾟ];(ﾟДﾟ) [<span class="hljs-string">'_'</span>] =(o^_^o) [ﾟoﾟ] [ﾟoﾟ];(ﾟεﾟ)=((ﾟｰﾟ==<span class="hljs-number">3</span>) +<span class="hljs-string">'_'</span>) [ﾟΘﾟ]+ (ﾟДﾟ) .ﾟДﾟﾉ+((ﾟДﾟ)+<span class="hljs-string">'_'</span>) [(ﾟｰﾟ) + (ﾟｰﾟ)]+((ﾟｰﾟ==<span class="hljs-number">3</span>) +<span class="hljs-string">'_'</span>) [o^_^o -ﾟΘﾟ]+((ﾟｰﾟ==<span class="hljs-number">3</span>) +<span class="hljs-string">'_'</span>) [ﾟΘﾟ]+ (ﾟωﾟﾉ +<span class="hljs-string">'_'</span>) [ﾟΘﾟ]; (ﾟｰﾟ)+=(ﾟΘﾟ); (ﾟДﾟ)[ﾟεﾟ]=<span class="hljs-string">'\\'</span>; (ﾟДﾟ).ﾟΘﾟﾉ=(ﾟДﾟ+ ﾟｰﾟ)[o^_^o -(ﾟΘﾟ)];(oﾟｰﾟo)=(ﾟωﾟﾉ +<span class="hljs-string">'_'</span>)[c^_^o];(ﾟДﾟ) [ﾟoﾟ]=<span class="hljs-string">'\"'</span>;(ﾟДﾟ) [<span class="hljs-string">'_'</span>] ( (ﾟДﾟ) [<span class="hljs-string">'_'</span>] (ﾟεﾟ+(ﾟДﾟ)[ﾟoﾟ]+ (ﾟДﾟ)[ﾟεﾟ]+(ﾟΘﾟ)+ ((o^_^o) +(o^_^o))+ ((o^_^o) +(o^_^o))+ (ﾟДﾟ)[ﾟεﾟ]+(ﾟΘﾟ)+ (ﾟｰﾟ)+ (ﾟΘﾟ)+ (ﾟДﾟ)[ﾟεﾟ]+(ﾟΘﾟ)+ ((o^_^o) +(o^_^o))+ ((o^_^o) - (ﾟΘﾟ))+ (ﾟДﾟ)[ﾟεﾟ]+(ﾟｰﾟ)+ (c^_^o)+ (ﾟДﾟ)[ﾟεﾟ]+(ﾟΘﾟ)+ (ﾟｰﾟ)+ (ﾟΘﾟ)+ (ﾟДﾟ)[ﾟεﾟ]+(ﾟｰﾟ)+ (c^_^o)+ (ﾟДﾟ)[ﾟεﾟ]+((ﾟｰﾟ) + (o^_^o))+ ((ﾟｰﾟ) + (ﾟΘﾟ))+ (ﾟДﾟ)[ﾟεﾟ]+(ﾟｰﾟ)+ (c^_^o)+ (ﾟДﾟ)[ﾟεﾟ]+((o^_^o) +(o^_^o))+ (ﾟΘﾟ)+ (ﾟДﾟ)[ﾟoﾟ])(ﾟΘﾟ))((ﾟΘﾟ)+(ﾟДﾟ)[ﾟεﾟ]+((ﾟｰﾟ)+(ﾟΘﾟ))+(ﾟΘﾟ)+(ﾟДﾟ)[ﾟoﾟ]);
</code></pre>
<p>jjencode 的结果：</p>
<pre><code data-language="js" class="lang-js">$=~[];$={<span class="hljs-attr">___</span>:++$,<span class="hljs-attr">$$$$</span>:(![]+<span class="hljs-string">""</span>)[$],<span class="hljs-attr">__$</span>:++$,<span class="hljs-attr">$_$_</span>:(![]+<span class="hljs-string">""</span>)[$],<span class="hljs-attr">_$_</span>:++$,<span class="hljs-attr">$_$$</span>:({}+<span class="hljs-string">""</span>)[$],<span class="hljs-attr">$$_$</span>:($[$]+<span class="hljs-string">""</span>)[$],<span class="hljs-attr">_$$</span>:++$,<span class="hljs-attr">$$$_</span>:(!<span class="hljs-string">""</span>+<span class="hljs-string">""</span>)[$],<span class="hljs-attr">$__</span>:++$,<span class="hljs-attr">$_$</span>:++$,<span class="hljs-attr">$$__</span>:({}+<span class="hljs-string">""</span>)[$],<span class="hljs-attr">$$_</span>:++$,<span class="hljs-attr">$$$</span>:++$,<span class="hljs-attr">$___</span>:++$,<span class="hljs-attr">$__$</span>:++$};$.$_=($.$_=$+<span class="hljs-string">""</span>)[$.$_$]+($._$=$.$_[$.__$])+($.$$=($.$+<span class="hljs-string">""</span>)[$.__$])+((!$)+<span class="hljs-string">""</span>)[$._$$]+($.__=$.$_[$.$$_])+($.$=(!<span class="hljs-string">""</span>+<span class="hljs-string">""</span>)[$.__$])+($._=(!<span class="hljs-string">""</span>+<span class="hljs-string">""</span>)[$._$_])+$.$_[$.$_$]+$.__+$._$+$.$;$.$$=$.$+(!<span class="hljs-string">""</span>+<span class="hljs-string">""</span>)[$._$$]+$.__+$._+$.$+$.$$;$.$=($.___)[$.$_][$.$_];$.$($.$($.$$+<span class="hljs-string">"\""</span>+<span class="hljs-string">"\\"</span>+$.__$+$.$$_+$.$$_+$.$_$_+<span class="hljs-string">"\\"</span>+$.__$+$.$$_+$._$_+<span class="hljs-string">"\\"</span>+$.$__+$.___+$.$_$_+<span class="hljs-string">"\\"</span>+$.$__+$.___+<span class="hljs-string">"=\\"</span>+$.$__+$.___+$.__$+<span class="hljs-string">"\""</span>)())();
</code></pre>
<p>这些混淆方式比较另类，但只需要输入到控制台即可执行，其没有真正达到强力混淆的效果。</p>
<p>以上便是对 JavaScript 混淆方式的介绍和总结。总的来说，经过混淆的 JavaScript 代码其可读性大大降低，同时防护效果也大大增强。</p>
<h4>JavaScript 加密</h4>
<p>不同于 JavaScript 混淆技术，JavaScript 加密技术可以说是对 JavaScript 混淆技术防护的进一步升级，其基本思路是将一些核心逻辑使用诸如 C/C++ 语言来编写，并通过 JavaScript 调用执行，从而起到二进制级别的防护作用。</p>
<p>其加密的方式现在有 Emscripten 和 WebAssembly 等，其中后者越来越成为主流。<br>
下面我们分别来介绍下。</p>
<h5>Emscripten</h5>
<p>现在，许多 3D 游戏都是用 C/C++ 语言写的，如果能将 C / C++ 语言编译成 JavaScript 代码，它们不就能在浏览器里运行了吗？众所周知，JavaScript 的基本语法与 C 语言高度相似。于是，有人开始研究怎么才能实现这个目标，为此专门做了一个编译器项目 Emscripten。这个编译器可以将 C / C++ 代码编译成 JavaScript 代码，但不是普通的 JavaScript，而是一种叫作 asm.js 的 JavaScript 变体。</p>
<p>因此说，某些 JavaScript 的核心功能可以使用 C/C++ 语言实现，然后通过 Emscripten 编译成 asm.js，再由 JavaScript 调用执行，这可以算是一种前端加密技术。</p>
<h5>WebAssembly</h5>
<p>如果你对 JavaScript 比较了解，可能知道还有一种叫作 WebAssembly 的技术，也能将 C/C++ 转成 JavaScript 引擎可以运行的代码。那么它与 asm.js 有何区别呢？</p>
<p>其实两者的功能基本一致，就是转出来的代码不一样：asm.js 是文本，WebAssembly 是二进制字节码，因此运行速度更快、体积更小。从长远来看，WebAssembly 的前景更光明。</p>
<p>WebAssembly 是经过编译器编译之后的字节码，可以从 C/C++ 编译而来，得到的字节码具有和 JavaScript 相同的功能，但它体积更小，而且在语法上完全脱离 JavaScript，同时具有沙盒化的执行环境。</p>
<p>利用 WebAssembly 技术，我们可以将一些核心的功能利用 C/C++ 语言实现，形成浏览器字节码的形式。然后在 JavaScript 中通过类似如下的方式调用：</p>
<pre><code data-language="js" class="lang-js">WebAssembly.compile(<span class="hljs-keyword">new</span> <span class="hljs-built_in">Uint8Array</span>(<span class="hljs-string">`
  00 61 73 6d  01 00 00 00  01 0c 02 60  02 7f 7f 01
  7f 60 01 7f  01 7f 03 03  02 00 01 07  10 02 03 61
  64 64 00 00  06 73 71 75  61 72 65 00  01 0a 13 02
  08 00 20 00  20 01 6a 0f  0b 08 00 20  00 20 00 6c
  0f 0b`</span>.trim().split(<span class="hljs-regexp">/[\s\r\n]+/g</span>).map(<span class="hljs-function"><span class="hljs-params">str</span> =&gt;</span> <span class="hljs-built_in">parseInt</span>(str, <span class="hljs-number">16</span>))
)).then(<span class="hljs-function"><span class="hljs-params">module</span> =&gt;</span> {
  <span class="hljs-keyword">const</span> instance = <span class="hljs-keyword">new</span> WebAssembly.Instance(<span class="hljs-built_in">module</span>)
  <span class="hljs-keyword">const</span> { add, square } = instance.exports
  <span class="hljs-built_in">console</span>.log(<span class="hljs-string">'2 + 4 ='</span>, add(<span class="hljs-number">2</span>, <span class="hljs-number">4</span>))
  <span class="hljs-built_in">console</span>.log(<span class="hljs-string">'3^2 ='</span>, square(<span class="hljs-number">3</span>))
  <span class="hljs-built_in">console</span>.log(<span class="hljs-string">'(2 + 5)^2 ='</span>, square(add(<span class="hljs-number">2</span> + <span class="hljs-number">5</span>)))
})
</code></pre>
<p>这种加密方式更加安全，因为作为二进制编码它能起到的防护效果无疑是更好的。如果想要逆向或破解那得需要逆向 WebAssembly，难度也是很大的。</p>
<h3>总结</h3>
<p>以上，我们就介绍了接口加密技术和 JavaScript 的压缩、混淆和加密技术，知己知彼方能百战不殆，了解了原理，我们才能更好地去实现 JavaScript 的逆向。<br>
本节代码：<a href="https://github.com/Python3WebSpider/JavaScriptObfuscate">https://github.com/Python3WebSpider/JavaScriptObfuscate</a></p>
<h3>参考文献</h3>
<ul>
<li><a href="https://www.ruanyifeng.com/blog/2017/09/asmjs_emscripten.html">https://www.ruanyifeng.com/blog/2017/09/asmjs_emscripten.html</a></li>
<li><a href="https://juejin.im/post/5cfcb9d25188257e853fa71c#heading-23">https://juejin.im/post/5cfcb9d25188257e853fa71c#heading-23</a></li>
<li><a href="https://www.jianshu.com/p/326594cbd4fa">https://www.jianshu.com/p/326594cbd4fa</a></li>
<li><a href="https://github.com/javascript-obfuscator/javascript-obfuscator">https://github.com/javascript-obfuscator/javascript-obfuscator</a></li>
<li><a href="https://obfuscator.io/">https://obfuscator.io/</a></li>
<li><a href="https://www.sojson.com/jjencode.html">https://www.sojson.com/jjencode.html</a></li>
<li><a href="http://dean.edwards.name/packer/">http://dean.edwards.name/packer/</a></li>
</ul>


# JavaScript逆向爬取实战（上）
<p data-nodeid="25901" class="">上个课时我们介绍了网页防护技术，包括接口加密和 JavaScript 压缩、加密和混淆。这就引出了一个问题，如果我们碰到了这样的网站，那该怎么去分析和爬取呢？</p>
<p data-nodeid="25902">本课时我们就通过一个案例来介绍一下这种网站的爬取思路，本课时介绍的这个案例网站不仅在 API 接口层有加密，而且前端 JavaScript 也带有压缩和混淆，其前端压缩打包工具使用了现在流行的 Webpack，混淆工具是使用了 javascript-obfuscator，这二者结合起来，前端的代码会变得难以阅读和分析。</p>
<p data-nodeid="25903">如果我们不使用 Selenium 或 Pyppeteer 等工具来模拟浏览器的形式爬取的话，要想直接从接口层面上获取数据，基本上需要一点点调试分析 JavaScript 的调用逻辑、堆栈调用关系来弄清楚整个网站加密的实现方法，我们可以称这个过程叫 JavaScript 逆向。这些接口的加密参数往往都是一些加密算法或编码的组合，完全搞明白其中的逻辑之后，我们就能把这个算法用 Python 模拟出来，从而实现接口的请求了。</p>
<h3 data-nodeid="25904">案例介绍</h3>
<p data-nodeid="27858" class="">案例的地址为：<a href="https://dynamic6.scrape.center/" data-nodeid="27862">https://dynamic6.scrape.center/</a>，页面如图所示。</p>


<p data-nodeid="25906"><img src="https://s0.lgstatic.com/i/image/M00/00/DC/Ciqc1F6qkZqANciBAA8meafvsHM491.png" alt="image.png" data-nodeid="26054"></p>
<p data-nodeid="25907">初看之下并没有什么特殊的，但仔细观察可以发现其 Ajax 请求接口和每部电影的 URL 都包含了加密参数。</p>
<p data-nodeid="25908">比如我们点击任意一部电影，观察一下 URL 的变化，如图所示。</p>
<p data-nodeid="25909"><img src="https://s0.lgstatic.com/i/image/M00/00/DC/CgqCHl6qkaOAH4oXABChHxWKtLo031.png" alt="image (1).png" data-nodeid="26059"></p>
<p data-nodeid="25910">这里我们可以看到详情页的 URL 包含了一个长字符串，看似是一个 Base64 编码的内容。</p>
<p data-nodeid="25911">那么接下来直接看看 Ajax 的请求，我们从列表页的第 1 页到第 10 页依次点一下，观察一下 Ajax 请求是怎样的，如图所示。</p>
<p data-nodeid="25912"><img src="https://s0.lgstatic.com/i/image/M00/00/DD/CgqCHl6qkbiASEZbAAoPAmLuiLs518.png" alt="image (2).png" data-nodeid="26064"></p>
<p data-nodeid="25913">可以看到 Ajax 接口的 URL 里面多了一个 token，而且不同的页码 token 是不一样的，这个 token 同样看似是一个 Base64 编码的字符串。</p>
<p data-nodeid="25914">另外更困难的是，这个接口还是有时效性的，如果我们把 Ajax 接口 URL 直接复制下来，短期内是可以访问的，但是过段时间之后就无法访问了，会直接返回 401 状态码。</p>
<p data-nodeid="25915">接下来我们再看下列表页的返回结果，比如我们打开第一个请求，看看第一部电影数据的返回结果，如图所示。</p>
<p data-nodeid="25916"><img src="https://s0.lgstatic.com/i/image/M00/00/DD/CgqCHl6qkcOAQupqAAMYHAP-Uvk656.png" alt="image (3).png" data-nodeid="26070"></p>
<p data-nodeid="29434" class="">这里我们把看似是第一部电影的返回结果全展开了，但是刚才我们观察到第一部电影的 URL 的链接却为 <a href="https://dynamic6.scrape.center/detail/ZWYzNCN0ZXVxMGJ0dWEjKC01N3cxcTVvNS0takA5OHh5Z2ltbHlmeHMqLSFpLTAtbWIx" data-nodeid="29438">https://dynamic6.scrape.center/detail/ZWYzNCN0ZXVxMGJ0dWEjKC01N3cxcTVvNS0takA5OHh5Z2ltbHlmeHMqLSFpLTAtbWIx</a>，看起来是 Base64 编码，我们解码一下，结果为 ef34#teuq0btua#(-57w1q5o5--j@98xygimlyfxs*-!i-0-mb1，但是看起来似乎还是毫无规律，这个解码后的结果又是怎么来的呢？返回结果里面也并不包含这个字符串，那这又是怎么构造的呢？</p>


<p data-nodeid="25918">再然后，这仅仅是某一个详情页页面的 URL，其真实数据是通过 Ajax 加载的，那么 Ajax 请求又是怎样的呢，我们再观察下，如图所示。</p>
<p data-nodeid="25919"><img src="https://s0.lgstatic.com/i/image/M00/00/DD/CgqCHl6qkeKAfDyaAAk629L2zsE019.png" alt="image (4).png" data-nodeid="26083"></p>
<p data-nodeid="25920">好，这里我们发现其 Ajax 接口除了包含刚才所说的 URL 中携带的字符串，又多了一个 token，同样也是类似 Base64 编码的内容。</p>
<p data-nodeid="25921">那么总结下来这个网站就有如下特点：</p>
<ul data-nodeid="25922">
<li data-nodeid="25923">
<p data-nodeid="25924">列表页的 Ajax 接口参数带有加密的 token；</p>
</li>
<li data-nodeid="25925">
<p data-nodeid="25926">详情页的 URL 带有加密 id；</p>
</li>
<li data-nodeid="25927">
<p data-nodeid="25928">详情页的 Ajax 接口参数带有加密 id 和加密 token。</p>
</li>
</ul>
<p data-nodeid="25929">那如果我们要想通过接口的形式来爬取，必须要把这些加密 id 和 token 构造出来才行，而且必须要一步步来，首先我们要构造出列表页 Ajax 接口的 token 参数，然后才能获取每部电影的数据信息，然后根据数据信息构造出加密 id 和 token。</p>
<p data-nodeid="25930">OK，到现在为止我们就知道了这个网站接口的加密情况了，我们下一步就是去找这个加密实现逻辑了。</p>
<p data-nodeid="25931">由于是网页，所以其加密逻辑一定藏在前端代码中，但前面我们也说了，前端为了保护其接口加密逻辑不被轻易分析出来，会采取压缩、混淆的方式来加大分析的难度。</p>
<p data-nodeid="25932">接下来，我们就来看看这个网站的源代码和 JavaScript 文件是怎样的吧。</p>
<p data-nodeid="25933">首先看看网站源代码，我们在网站上点击右键，弹出选项菜单，然后点击“查看源代码”，可以看到结果如图所示。</p>
<p data-nodeid="25934"><img src="https://s0.lgstatic.com/i/image/M00/00/DD/CgqCHl6qkgGANjrqAASTjV0jn7Q608.png" alt="image (5).png" data-nodeid="26096"></p>
<p data-nodeid="25935">内容如下：</p>
<pre class="lang-html" data-nodeid="25936"><code data-language="html"><span class="hljs-meta">&lt;!DOCTYPE <span class="hljs-meta-keyword">html</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">html</span> <span class="hljs-attr">lang</span>=<span class="hljs-string">en</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">head</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">meta</span> <span class="hljs-attr">charset</span>=<span class="hljs-string">utf-8</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">meta</span> <span class="hljs-attr">http-equiv</span>=<span class="hljs-string">X-UA-Compatible</span> <span class="hljs-attr">content</span>=<span class="hljs-string">"IE=edge"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">meta</span> <span class="hljs-attr">name</span>=<span class="hljs-string">viewport</span> <span class="hljs-attr">content</span>=<span class="hljs-string">"width=device-width,initial-scale=1"</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">link</span> <span class="hljs-attr">rel</span>=<span class="hljs-string">icon</span> <span class="hljs-attr">href</span>=<span class="hljs-string">/favicon.ico</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">title</span>&gt;</span>Scrape | Movie<span class="hljs-tag">&lt;/<span class="hljs-name">title</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">link</span> <span class="hljs-attr">href</span>=<span class="hljs-string">/css/chunk-19c920f8.2a6496e0.css</span> <span class="hljs-attr">rel</span>=<span class="hljs-string">prefetch</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">link</span> <span class="hljs-attr">href</span>=<span class="hljs-string">/css/chunk-2f73b8f3.5b462e16.css</span> <span class="hljs-attr">rel</span>=<span class="hljs-string">prefetch</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">link</span> <span class="hljs-attr">href</span>=<span class="hljs-string">/js/chunk-19c920f8.c3a1129d.js</span> <span class="hljs-attr">rel</span>=<span class="hljs-string">prefetch</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">link</span> <span class="hljs-attr">href</span>=<span class="hljs-string">/js/chunk-2f73b8f3.8f2fc3cd.js</span> <span class="hljs-attr">rel</span>=<span class="hljs-string">prefetch</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">link</span> <span class="hljs-attr">href</span>=<span class="hljs-string">/js/chunk-4dec7ef0.e4c2b130.js</span> <span class="hljs-attr">rel</span>=<span class="hljs-string">prefetch</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">link</span> <span class="hljs-attr">href</span>=<span class="hljs-string">/css/app.ea9d802a.css</span> <span class="hljs-attr">rel</span>=<span class="hljs-string">preload</span> <span class="hljs-attr">as</span>=<span class="hljs-string">style</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">link</span> <span class="hljs-attr">href</span>=<span class="hljs-string">/js/app.5ef0d454.js</span> <span class="hljs-attr">rel</span>=<span class="hljs-string">preload</span> <span class="hljs-attr">as</span>=<span class="hljs-string">script</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">link</span> <span class="hljs-attr">href</span>=<span class="hljs-string">/js/chunk-vendors.77daf991.js</span> <span class="hljs-attr">rel</span>=<span class="hljs-string">preload</span> <span class="hljs-attr">as</span>=<span class="hljs-string">script</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">link</span> <span class="hljs-attr">href</span>=<span class="hljs-string">/css/app.ea9d802a.css</span> <span class="hljs-attr">rel</span>=<span class="hljs-string">stylesheet</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">head</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">body</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">noscript</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">strong</span>&gt;</span>We're sorry but portal doesn't work properly without JavaScript enabled. Please enable it to continue.<span class="hljs-tag">&lt;/<span class="hljs-name">strong</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">noscript</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">div</span> <span class="hljs-attr">id</span>=<span class="hljs-string">app</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">div</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">script</span> <span class="hljs-attr">src</span>=<span class="hljs-string">/js/chunk-vendors.77daf991.js</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">script</span>&gt;</span><span class="hljs-tag">&lt;<span class="hljs-name">script</span> <span class="hljs-attr">src</span>=<span class="hljs-string">/js/app.5ef0d454.js</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">script</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">body</span>&gt;</span><span class="hljs-tag">&lt;/<span class="hljs-name">html</span>&gt;</span>
</code></pre>
<p data-nodeid="25937">这是一个典型的 SPA（单页 Web 应用）的页面， 其 JavaScript 文件名带有编码字符、chunk、vendors 等关键字，整体就是经过 Webpack 打包压缩后的源代码，目前主流的前端开发，如 Vue.js、React.js 的输出结果都是类似这样的结果。</p>
<p data-nodeid="25938">好，那么我们再看下其 JavaScript 代码是什么样子的，我们在开发者工具中打开 Sources 选项卡下的 Page 选项卡，然后打开 js 文件夹，这里我们就能看到 JavaScript 的源代码，如图所示。</p>
<p data-nodeid="25939"><img src="https://s0.lgstatic.com/i/image/M00/00/DD/Ciqc1F6qkjGAGyKRAAYIIGMFEZw194.png" alt="image (6).png" data-nodeid="26102"></p>
<p data-nodeid="25940">我们随便复制一些出来，看看是什么样子的，结果如下：</p>
<pre class="lang-js" data-nodeid="25941"><code data-language="js">\(<span class="hljs-built_in">window</span>\[<span class="hljs-string">'webpackJsonp'</span>\]=<span class="hljs-built_in">window</span>\[<span class="hljs-string">'webpackJsonp'</span>\]\|\|\[\]\)\[<span class="hljs-string">'push'</span>\]\(\[\[<span class="hljs-string">'chunk\-19c920f8'</span>\]\,\{<span class="hljs-string">'5a19'</span>:<span class="hljs-function"><span class="hljs-keyword">function</span>\(<span class="hljs-params">\_0x3cb7c3\,\_0x5cb6ab\,\_0x5f5010\</span>)\</span>{\}\,<span class="hljs-string">'c6bf'</span>:<span class="hljs-function"><span class="hljs-keyword">function</span>\(<span class="hljs-params">\_0x1846fe\,\_0x459c04\,\_0x1ff8e3\</span>)\</span>{\}\,<span class="hljs-string">'ca9c'</span>:<span class="hljs-function"><span class="hljs-keyword">function</span>\(<span class="hljs-params">\_0x195201\,\_0xc41ead\,\_0x1b389c\</span>)\</span>{<span class="hljs-string">'use strict'</span>;<span class="hljs-keyword">var</span> \_0x468b4e=\_0x1b389c\(<span class="hljs-string">'5a19'</span>\)\,\_0x232454=\_0x1b389c[<span class="hljs-string">'n'</span>](_0x468b4e);\_0x232454[<span class="hljs-string">'a'</span>];},<span class="hljs-string">'d504'</span>:...,[\_0xd670a1[<span class="hljs-string">'\_v'</span>](_0xd670a1%<span class="hljs-number">5</span>B<span class="hljs-string">'_s'</span>%<span class="hljs-number">5</span>D(_0x2227b6)+<span class="hljs-string">'%5Cx0a%5Cx20%5Cx20%5Cx20%5Cx20%5Cx20%5Cx20%5Cx20%5Cx20%5Cx20%5Cx20%5Cx20%5Cx20%5Cx20%5Cx20'</span>)]);}),<span class="hljs-number">0x1</span>),\_0x4ef533(<span class="hljs-string">'div'</span>,{<span class="hljs-string">'staticClass'</span>:<span class="hljs-string">'m-v-sm\x20info'</span>},[\_0x4ef533(<span class="hljs-string">'span'</span>,[\_0xd670a1[<span class="hljs-string">'\_v'</span>](_0xd670a1%<span class="hljs-number">5</span>B<span class="hljs-string">'_s'</span>%<span class="hljs-number">5</span>D(_0x1cc7eb%<span class="hljs-number">5</span>B<span class="hljs-string">'regions'</span>%<span class="hljs-number">5</span>D%<span class="hljs-number">5</span>B<span class="hljs-string">'join'</span>%<span class="hljs-number">5</span>D(<span class="hljs-string">'%E3%80%81'</span>)))]),\_0x4ef533(<span class="hljs-string">'span'</span>,[\_0xd670a1[<span class="hljs-string">'\_v'</span>](<span class="hljs-string">'%5Cx20/%5Cx20'</span>)]),\_0x4ef533(<span class="hljs-string">'span'</span>,[\_0xd670a1[<span class="hljs-string">'\_v'</span>](_0xd670a1%<span class="hljs-number">5</span>B<span class="hljs-string">'_s'</span>%<span class="hljs-number">5</span>D(_0x1cc7eb%<span class="hljs-number">5</span>B<span class="hljs-string">'minute'</span>%<span class="hljs-number">5</span>D)+<span class="hljs-string">'%5Cx20%E5%88%86%E9%92%9F'</span>)])]),\_0x4ef533(<span class="hljs-string">'div'</span>,...,\_0x4ef533(<span class="hljs-string">'el-col'</span>,{<span class="hljs-string">'attrs'</span>:{<span class="hljs-string">'xs'</span>:<span class="hljs-number">0x5</span>,<span class="hljs-string">'sm'</span>:<span class="hljs-number">0x5</span>,<span class="hljs-string">'md'</span>:<span class="hljs-number">0x4</span>}},[\_0x4ef533(<span class="hljs-string">'p'</span>,{<span class="hljs-string">'staticClass'</span>:<span class="hljs-string">'score\x20m-t-md\x20m-b-n-sm'</span>},[\_0xd670a1[<span class="hljs-string">'\_v'</span>](_0xd670a1%<span class="hljs-number">5</span>B<span class="hljs-string">'_s'</span>%<span class="hljs-number">5</span>D(_0x1cc7eb%<span class="hljs-number">5</span>B<span class="hljs-string">'score'</span>%<span class="hljs-number">5</span>D%<span class="hljs-number">5</span>B<span class="hljs-string">'toFixed'</span>%<span class="hljs-number">5</span>D(<span class="hljs-number">0x1</span>)))\]\)\,\_0x4ef533\(<span class="hljs-string">'p'</span>\,\[\_0x4ef533\(<span class="hljs-string">'el\-rate'</span>\,\{<span class="hljs-string">'attrs'</span>:\{<span class="hljs-string">'value'</span>:\_0x1cc7eb\[<span class="hljs-string">'score'</span>\]/<span class="hljs-number">0x2</span>\,<span class="hljs-string">'disabled'</span>:<span class="hljs-string">''</span>\,<span class="hljs-string">'max'</span>:<span class="hljs-number">0x5</span>\,<span class="hljs-string">'text\-color'</span>:<span class="hljs-string">'\#ff9900'</span>\}\}\)\]\,<span class="hljs-number">0x1</span>\)\]\)\]\,<span class="hljs-number">0x1</span>\)\]\,<span class="hljs-number">0x1</span>\);\}\)\,<span class="hljs-number">0x1</span>\)\]\,<span class="hljs-number">0x1</span>\)\,\_0x4ef533\(<span class="hljs-string">'el\-row'</span>\,\[\_0x4ef533\(<span class="hljs-string">'el\-col'</span>\,\{<span class="hljs-string">'attrs'</span>:\{<span class="hljs-string">'span'</span>:<span class="hljs-number">0xa</span>\,<span class="hljs-string">'offset'</span>:<span class="hljs-number">0xb</span>\}\}\,\[\_0x4ef533\(<span class="hljs-string">'div'</span>\,\{<span class="hljs-string">'staticClass'</span>:<span class="hljs-string">'pagination\x20m\-v\-lg'</span>\}\,\[\_0x4ef533\(<span class="hljs-string">'el\-pagination'</span>\,\.\.\.:<span class="hljs-function"><span class="hljs-keyword">function</span>\(<span class="hljs-params">\_0x347c29\</span>)\</span>{\_0xd670a1\[<span class="hljs-string">'page'</span>\]=\_0x347c29;\}\,<span class="hljs-string">'update:current\-page'</span>:<span class="hljs-function"><span class="hljs-keyword">function</span>\(<span class="hljs-params">\_0x79754e\</span>)\</span>{\_0xd670a1\[<span class="hljs-string">'page'</span>\]=\_0x79754e;\}\}\}\)\]\,<span class="hljs-number">0x1</span>\)\]\)\]\,<span class="hljs-number">0x1</span>\)\]\,<span class="hljs-number">0x1</span>\);\}\,\_0x357ebc=\[\]\,\_0x18b11a=\_0x1a3e60\(<span class="hljs-string">'7d92'</span>\)\,\_0x4369=\_0x1a3e60\(<span class="hljs-string">'3e22'</span>\)\,\.\.\.;<span class="hljs-keyword">var</span> \_0x498df8=\.\.\.\[<span class="hljs-string">'then'</span>\]\(<span class="hljs-function"><span class="hljs-keyword">function</span>\(<span class="hljs-params">\_0x59d600\</span>)\</span>{<span class="hljs-keyword">var</span> \_0x1249bc=\_0x59d600\[<span class="hljs-string">'data'</span>\]\,\_0x10e324=\_0x1249bc\[<span class="hljs-string">'results'</span>\]\,\_0x47d41b=\_0x1249bc\[<span class="hljs-string">'count'</span>\];\_0x531b38\[<span class="hljs-string">'loading'</span>\]=\!<span class="hljs-number">0x1</span>\,\_0x531b38\[<span class="hljs-string">'movies'</span>\]=\_0x10e324\,\_0x531b38\[<span class="hljs-string">'total'</span>\]=\_0x47d41b;\}\);\}\}\}\,\_0x28192a=\_0x5f39bd\,\_0x5f5978=\(\_0x1a3e60\(<span class="hljs-string">'ca9c'</span>\)\,\_0x1a3e60\(<span class="hljs-string">'eb45'</span>\)\,\_0x1a3e60\(<span class="hljs-string">'2877'</span>\)\)\,\_0x3fae81=<span class="hljs-built_in">Object</span>\(\_0x5f5978\[<span class="hljs-string">'a'</span>\]\)\(\_0x28192a\,\_0x443d6e\,\_0x357ebc\,\!<span class="hljs-number">0x1</span>\,<span class="hljs-literal">null</span>\,<span class="hljs-string">'724ecf3b'</span>\,<span class="hljs-literal">null</span>\);\_0x6f764c\[<span class="hljs-string">'default'</span>\]=\_0x3fae81\[<span class="hljs-string">'exports'</span>\];\}\,<span class="hljs-string">'eb45'</span>:<span class="hljs-function"><span class="hljs-keyword">function</span>\(<span class="hljs-params">\_0x1d3c3c\,\_0x52e11c\,\_0x3f1276\</span>)\</span>{<span class="hljs-string">'use strict'</span>;<span class="hljs-keyword">var</span> \_0x79046c=\_0x3f1276\(<span class="hljs-string">'c6bf'</span>\)\,\_0x219366=\_0x3f1276[<span class="hljs-string">'n'</span>](_0x79046c);\_0x219366[<span class="hljs-string">'a'</span>];}}]);
</code></pre>
<p data-nodeid="25942">就是这种感觉，可以看到一些变量都是一些十六进制字符串，而且代码全被压缩了。</p>
<p data-nodeid="25943">没错，我们就是要从这里面找出 token 和 id 的构造逻辑，看起来是不是很崩溃？</p>
<p data-nodeid="25944">要完全分析出整个网站的加密逻辑还是有一定难度的，不过不用担心，我们本课时会一步步地讲解逆向的思路、方法和技巧，如果你能跟着这个过程学习完，相信还是能学会一定的 JavaScript 逆向技巧的。</p>
<blockquote data-nodeid="25945">
<p data-nodeid="25946">为了适当降低难度，本课时案例的 JavaScript 混淆其实并没有设置的特别复杂，并没有开启字符串编码、控制流扁平化等混淆方式。</p>
</blockquote>
<h3 data-nodeid="25947">列表页 Ajax 入口寻找</h3>
<p data-nodeid="25948">接下来，我们就开始第一步入口的寻找吧，这里简单介绍两种寻找入口的方式：</p>
<ul data-nodeid="25949">
<li data-nodeid="25950">
<p data-nodeid="25951">全局搜索标志字符串；</p>
</li>
<li data-nodeid="25952">
<p data-nodeid="25953">设置 Ajax 断点。</p>
</li>
</ul>
<h4 data-nodeid="25954">全局搜索标志字符串</h4>
<p data-nodeid="25955">一些关键的字符串通常会作为找寻 JavaScript 混淆入口的依据，我们可以通过全局搜索的方式来查找，然后根据搜索到的结果大体观察是否是我们想找的入口。</p>
<p data-nodeid="25956">然后，我们重新打开列表页的 Ajax 接口，看下请求的 Ajax 接口，如图所示。</p>
<p data-nodeid="25957" class="te-preview-highlight"><img src="https://s0.lgstatic.com/i/image/M00/00/DD/CgqCHl6qknSAaJi9AAZGLdfg3f4342.png" alt="image (7).png" data-nodeid="26117"></p>
<p data-nodeid="31014" class="">这里的 Ajax 接口的 URL 为 <a href="https://dynamic6.scrape.center/api/movie/?limit=10&amp;offset=0&amp;token=NTRhYWJhNzAyYTZiMTc0ZThkZTExNzBiNTMyMDJkN2UxZWYyMmNiZCwxNTg4MTc4NTYz" data-nodeid="31022">https://dynamic6.scrape.center/api/movie/?limit=10&amp;offset=0&amp;token=NTRhYWJhNzAyYTZiMTc0ZThkZTExNzBiNTMyMDJkN2UxZWYyMmNiZCwxNTg4MTc4NTYz</a>，可以看到带有 offset、limit、token 三个参数，入口寻找关键就是找 token，我们全局搜索下 token 是否存在，可以点击开发者工具右上角的下拉选项卡，然后点击 Search，如图所示。</p>


<p data-nodeid="25959"><img src="https://s0.lgstatic.com/i/image/M00/00/DD/Ciqc1F6qkoqANZ5EAAXALwrapv4539.png" alt="image (8).png" data-nodeid="26129"></p>
<p data-nodeid="25960">这样我们就能进入到一个全局搜索模式，我们搜索 token，可以看到的确搜索到了几个结果，如图所示。</p>
<p data-nodeid="25961"><img src="https://s0.lgstatic.com/i/image/M00/00/DD/CgqCHl6qkpWAbJRjAAdFuzcwiio327.png" alt="image (9).png" data-nodeid="26133"></p>
<p data-nodeid="25962">观察一下，下面的两个结果可能是我们想要的，我们点击进入第一个看下，定位到了一个 JavaScript 文件，如图所示。</p>
<p data-nodeid="25963"><img src="https://s0.lgstatic.com/i/image/M00/00/DD/Ciqc1F6qkpyAJI-uAAWL2y8otZc090.png" alt="image (10).png" data-nodeid="26137"></p>
<p data-nodeid="25964">这时候可以看到整个代码都是压缩过的，只有一行，不好看，我们可以点击左下角的 {} 按钮，美化一下 JavaScript 代码，如图所示。</p>
<p data-nodeid="25965"><img src="https://s0.lgstatic.com/i/image/M00/00/DD/CgqCHl6qkqSAUtJ4AAWPHUlveFY575.png" alt="image (11).png" data-nodeid="26141"></p>
<p data-nodeid="25966">美化后的结果就是这样子了，如图所示。</p>
<p data-nodeid="25967"><img src="https://s0.lgstatic.com/i/image/M00/00/DD/Ciqc1F6qkq2AEhDXAAgZeF5_6Xk163.png" alt="image (12).png" data-nodeid="26145"></p>
<p data-nodeid="25968">这时可以看到这里弹出来了一个新的选项卡，其名称是 JavaScript 文件名加上了 :formatted，代表格式化后代码结果，在这里我们再次定位到 token 观察一下。</p>
<p data-nodeid="25969">可以看到这里有 limit、offset、token，然后观察下其他的逻辑，基本上能够确定这就是构造 Ajax 请求的地方了，如果不是的话可以继续搜索其他的文件观察下。</p>
<p data-nodeid="25970">那现在，混淆的入口点我们就成功找到了，这是一个首选的找入口的方法。</p>
<h4 data-nodeid="25971">XHR 断点</h4>
<p data-nodeid="25972">由于这里的 token 字符串并没有被混淆，所以上面的这个方法是奏效的。之前我们也讲过，这种字符串由于非常容易成为找寻入口点的依据，所以这样的字符串也会被混淆成类似 Unicode、Base64、RC4 的一些编码形式，这样我们就没法轻松搜索到了。</p>
<p data-nodeid="25973">那如果遇到这种情况，我们该怎么办呢？这里再介绍一种通过打 XHR 断点的方式来寻找入口。</p>
<p data-nodeid="25974">XHR 断点，顾名思义，就是在发起 XHR 的时候进入断点调试模式，JavaScript 会在发起 Ajax 请求的时候停住，这时候我们可以通过当前的调用栈的逻辑顺着找到入口。怎么设置呢？我们可以在 Sources 选项卡的右侧，XHR/fetch Breakpoints 处添加一个断点选项。</p>
<p data-nodeid="25975">首先点击 + 号，然后输入匹配的 URL 内容，由于 Ajax 接口的形式是 /api/movie/?limit=10... 这样的格式，所这里我们就截取一段填进去就好了，这里填的就是 /api/movie，如图所示。</p>
<p data-nodeid="25976"><img src="https://s0.lgstatic.com/i/image/M00/00/DE/Ciqc1F6qkxqAK4pMAAhc_XTSt_Y367.png" alt="image (13).png" data-nodeid="26156"></p>
<p data-nodeid="25977">添加完毕之后重新刷新页面，可以发现进入了断点模式，如图所示。</p>
<p data-nodeid="25978"><img src="https://s0.lgstatic.com/i/image/M00/00/DE/CgqCHl6qkyKAVRkUAAUv1oHhpbk473.png" alt="image (14).png" data-nodeid="26160"></p>
<p data-nodeid="25979">好，接下来我们重新点下 {} 格式化代码，看看断点是在哪里，如图所示。</p>
<p data-nodeid="25980"><img src="https://s0.lgstatic.com/i/image/M00/00/DE/Ciqc1F6qkzyAQZcQAAguLqHyraA259.png" alt="image (15).png" data-nodeid="26164"></p>
<p data-nodeid="25981">那这里看到有个 send 的字符，我们可以初步猜测这就是相当于发送 Ajax 请求的一瞬间。</p>
<p data-nodeid="25982">到了这里感觉 Ajax 马上就要发出去了，是不是有点太晚了，我们想找的是构造 Ajax 的时刻来分析 Ajax 参数啊！不用担心，这里我们通过调用栈就可以找回去。我们点击右侧的 Call Stack，这里记录了 JavaScript 的方法逐层调用过程，如图所示。</p>
<p data-nodeid="25983"><img src="https://s0.lgstatic.com/i/image/M00/00/DE/Ciqc1F6qk0qAJ04jAAiOvyuo0wU233.png" alt="image (16).png" data-nodeid="26169"></p>
<p data-nodeid="25984">这里当前指向的是一个名字为 anonymouns，也就是匿名的调用，在它的下方就显示了调用这个 anonymouns 的方法，名字叫作 _0x594ca1，然后再下一层就又显示了调用 _0x594a1 这个方法的方法，依次类推。</p>
<p data-nodeid="25985">这里我们可以逐个往下查找，然后通过一些观察看看有没有 token 这样的信息，就能找到对应的位置了，最后我们就可以找到 onFetchData 这个方法里面实现了这个 token 的构造逻辑，这样我们也成功找到 token 的参数构造的位置了，如图所示。</p>
<p data-nodeid="25986"><img src="https://s0.lgstatic.com/i/image/M00/00/DE/CgqCHl6qk1qASl3iAAfB8DQP-C0830.png" alt="image (17).png" data-nodeid="26178"></p>
<p data-nodeid="25987">好，到现在为止我们就通过两个方法找到入口点了。</p>
<p data-nodeid="25988">其实还有其他的寻找入口的方式，比如 Hook 关键函数的方式，稍后的课程里我们会讲到，这里就暂时不讲了。</p>
<h3 data-nodeid="25989">列表页加密逻辑寻找</h3>
<p data-nodeid="25990">接下来我们已经找到 token 的位置了，可以观察一下这个 token 对应的变量叫作 _0xa70fc9，所以我们的关键就是要找这个变量是哪里来的了。</p>
<p data-nodeid="25991">怎么找呢？我们打个断点看下这个变量是在哪里生成的就好了，我们在对应的行打一个断点，如果打了刚才的 XHR 断点的话可以先取消掉，如图所示。</p>
<p data-nodeid="25992"><img src="https://s0.lgstatic.com/i/image/M00/00/DE/CgqCHl6qk2iAXuw4AAfT4EPsbZI161.png" alt="image (18).png" data-nodeid="26188"></p>
<p data-nodeid="25993">这时候我们就设置了一个新的断点了。由于只有一个断点，可以重新刷新下网页，这时候我们会发现网页停在了新的断点上面。</p>
<p data-nodeid="25994"><img src="https://s0.lgstatic.com/i/image/M00/00/DE/Ciqc1F6qk5KAbR6NAAeJXqFY-QQ969.png" alt="image (19).png" data-nodeid="26192"></p>
<p data-nodeid="25995">这里我们就可以观察下运行的一些变量了，比如我们把鼠标放在各个变量上面去，可以看到变量的一些值和类型，比如我们看 _0x18b11a 这个变量，会有一个浮窗显示，如图所示。</p>
<p data-nodeid="25996"><img src="https://s0.lgstatic.com/i/image/M00/00/DE/Ciqc1F6qk5-AfIb1AAdTDoapgWc715.png" alt="image (20).png" data-nodeid="26198"></p>
<p data-nodeid="25997">另外我们还可以通过在右侧的 Watch 面板添加想要查看的变量名称，如这行代码的内容为：</p>
<pre class="lang-js" data-nodeid="25998"><code data-language="js">, _0xa70fc9 = <span class="hljs-built_in">Object</span>(_0x18b11a[<span class="hljs-string">'a'</span>])(<span class="hljs-keyword">this</span>[<span class="hljs-string">'$store'</span>][<span class="hljs-string">'state'</span>][<span class="hljs-string">'url'</span>][<span class="hljs-string">'index'</span>]);
</code></pre>
<p data-nodeid="25999">我们比较感兴趣的可能就是 _0x18b11a 还有 this 里面的这个值了，我们可以展开 Watch 面板，然后点击 + 号，把想看的变量添加到 Watch 面板里面，如图所示。</p>
<p data-nodeid="26000"><img src="https://s0.lgstatic.com/i/image/M00/00/DE/CgqCHl6qk9eAGFwlAAedDDa3zXI764.png" alt="image (21).png" data-nodeid="26205"></p>
<p data-nodeid="26001">观察下可以发现 _0x18b11a 是一个 Object，它有个 a 属性，其值是一个 function，然后 this['$store']['state']['url']['index'] 的值其实就是 /api/movie，就是 Ajax 请求 URL 的 Path。_0xa70fc9 就是调用了前者这个 function 然后传入了 /api/movie 得到的。</p>
<p data-nodeid="26002">那么下一步就是去寻找这个 function 在哪里了，我们可以把 Watch 面板的 _0x18b11a 展开，这里会显示一个 FunctionLocation，就是这个 function 的代码位置，如图所示。</p>
<p data-nodeid="26003"><img src="https://s0.lgstatic.com/i/image/M00/00/DE/CgqCHl6qk-eAetpKAAfZ8WF5vF8742.png" alt="image (22).png" data-nodeid="26237"></p>
<p data-nodeid="26004">点击进入之后发现其仍然是未格式化的代码，再次点击 {} 格式化代码。</p>
<p data-nodeid="26005">这时候我们就进入了一个新的名字为 _0xc9e475 的方法里面，这个方法里面应该就是 token 的生成逻辑了，我们再打上断点，然后执行面板右上角蓝色箭头状的 Resume 按钮，如图所示。</p>
<p data-nodeid="26006"><img src="https://s0.lgstatic.com/i/image/M00/00/DE/CgqCHl6qk_KAUBCyAAhWMyGe2CQ797.png" alt="image (23).png" data-nodeid="26244"></p>
<p data-nodeid="26007">这时候发现我们已经单步执行到这个位置了。</p>
<p data-nodeid="26008">接下来我们不断进行单步调试，观察这里面的执行逻辑和每一步调试过程中结果都有什么变化，如图所示。</p>
<p data-nodeid="26009"><img src="https://s0.lgstatic.com/i/image/M00/00/DE/CgqCHl6qk_2AIAG8AActuiud1jI582.png" alt="image (24).png" data-nodeid="26249"></p>
<p data-nodeid="26010">在每步的执行过程中，我们可以发现一些运行值会被打到代码的右侧并带有高亮表示，同时在 watch 面板还能看到每步的变量的具体结果。</p>
<p data-nodeid="26011">最后我们总结出这个 token 的构造逻辑如下：</p>
<ul data-nodeid="26012">
<li data-nodeid="26013">
<p data-nodeid="26014">传入的 /api/movie 会构造一个初始化列表，变量命名为 _0x3dde76。</p>
</li>
<li data-nodeid="26015">
<p data-nodeid="26016">获取当前的时间戳，命名为 _0x4c50b4，push 到 _0x3dde76 这个变量里面。</p>
</li>
<li data-nodeid="26017">
<p data-nodeid="26018">将 _0x3dde76 变量用“,”拼接，然后进行 SHA1 编码，命名为 _0x46ba68。</p>
</li>
<li data-nodeid="26019">
<p data-nodeid="26020">将 _0x46ba68 （SHA1 编码的结果）和 _0x4c50b4 （时间戳）用逗号拼接，命名为 _0x495a44。</p>
</li>
<li data-nodeid="26021">
<p data-nodeid="26022">将 _0x495a44 进行 Base64 编码，命名为 _0x2a93f2，得到最后的 token。</p>
</li>
</ul>
<p data-nodeid="26023">以上的一些逻辑经过反复的观察就可以比较轻松地总结出来了，其中有些变量可以实时查看，同时也可以自己输入到控制台上进行反复验证，相信总结出这个结果并不难。</p>
<p data-nodeid="26024">好，那现在加密逻辑我们就分析出来啦，基本的思路就是：</p>
<ul data-nodeid="26025">
<li data-nodeid="26026">
<p data-nodeid="26027">先将 /api/movie 放到一个列表里面；</p>
</li>
<li data-nodeid="26028">
<p data-nodeid="26029">列表中加入当前时间戳；</p>
</li>
<li data-nodeid="26030">
<p data-nodeid="26031">将列表内容用逗号拼接；</p>
</li>
<li data-nodeid="26032">
<p data-nodeid="26033">将拼接的结果进行 SHA1 编码；</p>
</li>
<li data-nodeid="26034">
<p data-nodeid="26035">将编码的结果和时间戳再次拼接；</p>
</li>
<li data-nodeid="26036">
<p data-nodeid="26037">将拼接后的结果进行 Base64 编码。</p>
</li>
</ul>
<p data-nodeid="26038">验证下逻辑没问题的话，我们就可以用 Python 来实现出来啦。</p>
<h3 data-nodeid="26039">Python 实现列表页的爬取</h3>
<p data-nodeid="26040">要用 Python 实现这个逻辑，我们需要借助于两个库，一个是 hashlib，它提供了 sha1 方法；另外一个是 base64 库，它提供了 b64encode 方法对结果进行 Base64 编码。<br>
代码实现如下：</p>
<pre class="lang-js" data-nodeid="26291"><code data-language="js"><span class="hljs-keyword">import</span> hashlib
<span class="hljs-keyword">import</span> time
<span class="hljs-keyword">import</span> base64
<span class="hljs-keyword">from</span> typing <span class="hljs-keyword">import</span> List, Any
<span class="hljs-keyword">import</span> requests

INDEX\_URL = <span class="hljs-string">'https://dynamic6.scrape.center/api/movie?limit={limit}&amp;offset={offset}&amp;token={token}'</span>
LIMIT = <span class="hljs-number">10</span>
OFFSET = <span class="hljs-number">0</span>

def get\_token(args: List[Any]):
timestamp = str(int(time.time()))
args.append(timestamp)
sign = hashlib.sha1(<span class="hljs-string">','</span>.join(args).encode(<span class="hljs-string">'utf-8'</span>)).hexdigest()
<span class="hljs-keyword">return</span> base64.b64encode(<span class="hljs-string">','</span>.join([sign, timestamp]).encode(<span class="hljs-string">'utf-8'</span>)).decode(<span class="hljs-string">'utf-8'</span>)

args = [<span class="hljs-string">'/api/movie'</span>]
token = get\_token(args=args)
index\_url = INDEX\_URL.format(limit=LIMIT, offset=OFFSET, token=token)
response = requests.get(index\_url)
print(<span class="hljs-string">'response'</span>, response.json())
</code></pre>

<p data-nodeid="26042" class="">这里我们就根据上面的逻辑把加密流程实现出来了，这里我们先模拟爬取了第一页的内容，最后运行一下就可以得到最终的输出结果了。</p>


# JavaScript逆向爬取实战（下）
<h3 data-nodeid="33112" class="">详情页加密 id 入口的寻找</h3>
<p data-nodeid="33113">好，我们接着上一课时的内容往下讲，我们观察下上一步的输出结果，我们把结果格式化一下，看看部分结果：</p>
<pre class="lang-js" data-nodeid="33114"><code data-language="js">{
 <span class="hljs-string">'count'</span>: <span class="hljs-number">100</span>,
 <span class="hljs-string">'results'</span>: [
  {
     <span class="hljs-string">'id'</span>: <span class="hljs-number">1</span>,
     <span class="hljs-string">'name'</span>: <span class="hljs-string">'霸王别姬'</span>,
     <span class="hljs-string">'alias'</span>: <span class="hljs-string">'Farewell My Concubine'</span>,
     <span class="hljs-string">'cover'</span>: <span class="hljs-string">'https://p0.meituan.net/movie/ce4da3e03e655b5b88ed31b5cd7896cf62472.jpg@464w_644h_1e_1c'</span>,
     <span class="hljs-string">'categories'</span>: [
       <span class="hljs-string">'剧情'</span>,
       <span class="hljs-string">'爱情'</span>
    ],
     <span class="hljs-string">'published_at'</span>: <span class="hljs-string">'1993-07-26'</span>,
     <span class="hljs-string">'minute'</span>: <span class="hljs-number">171</span>,
     <span class="hljs-string">'score'</span>: <span class="hljs-number">9.5</span>,
     <span class="hljs-string">'regions'</span>: [
       <span class="hljs-string">'中国大陆'</span>,
       <span class="hljs-string">'中国香港'</span>
    ]
  },
   ...
]
}
</code></pre>
<p data-nodeid="33115">这里我们看到有个 id 是 1，另外还有一些其他的字段如电影名称、封面、类别，等等，那么这里面一定有什么信息是用来唯一区分某个电影的。</p>
<p data-nodeid="34099" class="">但是呢，这里我们点击下第一个部电影的信息，可以看到它跳转到了 URL 为 <a href="https://dynamic6.scrape.center/detail/ZWYzNCN0ZXVxMGJ0dWEjKC01N3cxcTVvNS0takA5OHh5Z2ltbHlmeHMqLSFpLTAtbWIx" data-nodeid="34103">https://dynamic6.scrape.center/detail/ZWYzNCN0ZXVxMGJ0dWEjKC01N3cxcTVvNS0takA5OHh5Z2ltbHlmeHMqLSFpLTAtbWIx</a> 的页面，可以看到这里 URL 里面有一个加密 id 为 ZWYzNCN0ZXVxMGJ0dWEjKC01N3cxcTVvNS0takA5OHh5Z2ltbHlmeHMqLSFpLTAtbWIx，那么这个和电影的这些信息有什么关系呢？</p>


<p data-nodeid="33117">这里，如果你仔细观察其实是可以比较容易地找出规律来的，但是这总归是观察出来的，如果遇到一些观察不出规律的那就不好处理了。所以还是需要靠技巧去找到它真正加密的位置。</p>
<p data-nodeid="33118">这时候我们该怎么办呢？首先为我们分析一下，这个加密 id 到底是什么生成的。</p>
<p data-nodeid="33119">我们在点击详情页的时候就看到它访问的 URL 里面就带上了 ZWYzNCN0ZXVxMGJ0dWEjKC01N3cxcTVvNS0takA5OHh5Z2ltbHlmeHMqLSFpLTAtbWIx 这个加密 id 了，而且不同的详情页的加密 id 是不同的，这说明这个加密 id 的构造依赖于列表页 Ajax 的返回结果，所以可以确定这个加密 id 的生成是发生在 Ajax 请求完成后或者点击详情页的一瞬间。</p>
<p data-nodeid="33120">为了进一步确定是发生在何时，我们看看页面源码，可以看到在没有点击之前，详情页链接的 href 里面就已经带有加密 id 了，如图所示。</p>
<p data-nodeid="33121"><img src="https://s0.lgstatic.com/i/image/M00/00/DF/CgqCHl6qlJCAdQd6AAk_ZX1vrAI092.png" alt="image (25).png" data-nodeid="33263"></p>
<p data-nodeid="33122">由此我们可以确定，这个加密 id 是在 Ajax 请求完成之后生成的，而且肯定也是由 JavaScript 生成的了。</p>
<p data-nodeid="33123">那怎么再去查找 Ajax 完成之后的事件呢？是否应该去找 Ajax 完成之后的事件呢？<br>
可以是可以，你可以试试，我们可以看到在 Sources 面板的右侧，有一个 Event Listener Breakpoints，这里有一个 XHR 的监听，包括发起时、成功后、发生错误时的一些监听，这里我们勾选上 readystatechange 事件，代表 Ajax 得到响应时的事件，其他的断点可以都删除了，然后刷新下页面看下，如图所示。</p>
<p data-nodeid="33124"><img src="https://s0.lgstatic.com/i/image/M00/00/DF/Ciqc1F6qlKSAQ_UFAAh49gbNW9s441.png" alt="image (26).png" data-nodeid="33270"></p>
<p data-nodeid="33125">这里我们可以看到就停在了 Ajax 得到响应时的位置了。</p>
<p data-nodeid="33126">那我们怎么才能弄清楚这个 id 是怎么加密的呢？可以选择一个断点一个断点地找下去，但估计找的过程会崩溃掉，因为这里可能会逐渐调用到页面 UI 渲染的一些底层实现，甚至可能即使找到了也不知道具体找到哪里去了。</p>
<p data-nodeid="33127">那怎么办呢？这里我们再介绍一种定位的方法，那就是 Hook。</p>
<p data-nodeid="33128">Hook 技术中文又叫作钩子技术，它就是在程序运行的过程中，对其中的某个方法进行重写，在原有的方法前后加入我们自定义的代码。相当于在系统没有调用该函数之前，钩子程序就先捕获该消息，可以先得到控制权，这时钩子函数便可以加工处理（改变）该函数的执行行为。</p>
<p data-nodeid="33129">通俗点来说呢，比如我要 Hook 一个方法 a，可以先临时用一个变量存一下，把它存成 _a，然后呢，我再重新声明一个方法 a，里面添加自己的逻辑，比如加点调试语句、输出语句等等，然后再调用 _a，这里调用的 _a 就是之前的 a。</p>
<p data-nodeid="33130">这样就相当于新的方法 a 里面混入了我们自己定义的逻辑，同时又把原来的方法 a 也执行了一遍。所以这不会影响原有的执行逻辑和运行效果，但是我们通过这种改写便可以顺利在原来的 a 方法前后加上了我们自己的逻辑，这就是 Hook。</p>
<p data-nodeid="33131">那么，我们这里怎么用 Hook 的方式来找到加密 id 的加密入口点呢？</p>
<p data-nodeid="33132">想一下，这个加密 id 是一个 Base64 编码的字符串，那么生成过程中想必就调用了 JavaScript 的 Base64 编码的方法，这个方法名叫作 btoa，这个 btoa 方法可以将参数转化成 Base64 编码。当然 Base64 也有其他的实现方式，比如利用 crypto-js 这个库实现的，这个可能底层调用的就不是 btoa 方法了。</p>
<p data-nodeid="33133">所以，其实现在并不确定是不是调用的 btoa 方法实现的 Base64 编码，那就先试试吧。要实现 Hook，其实关键在于将原来的方法改写，这里我们其实就是 Hook btoa 这个方法了，btoa 这个方法属于 window 对象，我们将 window 对象的 btoa 方法进行改写即可。</p>
<p data-nodeid="33134">改写的逻辑如下：</p>
<pre class="lang-java" data-nodeid="33135"><code data-language="java">(function () {
   <span class="hljs-string">'use strict'</span>
   <span class="hljs-function">function <span class="hljs-title">hook</span><span class="hljs-params">(object, attr)</span> </span>{
       <span class="hljs-keyword">var</span> func = object[attr]
       object[attr] = function () {
           console.log(<span class="hljs-string">'hooked'</span>, object, attr, arguments)
           <span class="hljs-keyword">var</span> ret = func.apply(object, arguments)
           debugger
           console.log(<span class="hljs-string">'result'</span>, ret)
           <span class="hljs-keyword">return</span> ret
      }
  }
   hook(window, <span class="hljs-string">'btoa'</span>)
})()
</code></pre>
<p data-nodeid="33136">我们定义了一个 hook 方法，传入 object 和 attr 参数，意思就是 Hook object 对象的 attr 参数。例如我们如果想 Hook 一个 alert 方法，那就把 object 设置为 window，把 attr 设置为 alert 字符串。这里我们想要 Hook Base64 的编码方法，那么这里就只需要 Hook window 对象的 btoa 方法就好了。</p>
<p data-nodeid="33137">我们来看下，首先是 var func = object[attr]，相当于先把它赋值为一个变量，我们调用 func 方法就可以实现和原来相同的功能。接着，我们再直接改写这个方法的定义，直接改写 object[attr]，将其改写成一个新的方法，在新的方法中，通过 func.apply 方法又重新调用了原来的方法。</p>
<p data-nodeid="33138">这样我们就可以保证，前后方法的执行效果是不受什么影响的，之前这个方法该干啥就还是干啥的。但是和之前不同的是，我们自定义方法之后，现在可以在 func 方法执行的前后，再加入自己的代码，如 console.log 将信息输出到控制台，如 debugger 进入断点等等。</p>
<p data-nodeid="33139">这个过程中，我们先临时保存下来了 func 方法，然后定义一个新的方法，接管程序控制权，在其中自定义我们想要的实现，同时在新的方法里面再重新调回 func 方法，保证前后结果是不受影响的。所以，我们达到了在不影响原有方法效果的前提下，可以实现在方法的前后实现自定义的功能，就是 Hook 的完整实现过程。</p>
<p data-nodeid="33140">最后，我们调用 hook 方法，传入 window 对象和 btoa 字符串即可。</p>
<p data-nodeid="33141">那这样，怎么去注入这个代码呢？这里我们介绍三种注入方法。</p>
<ul data-nodeid="33142">
<li data-nodeid="33143">
<p data-nodeid="33144">直接控制台注入；</p>
</li>
<li data-nodeid="33145">
<p data-nodeid="33146">复写 JavaScript 代码；</p>
</li>
<li data-nodeid="33147">
<p data-nodeid="33148">Tampermonkey 注入。</p>
</li>
</ul>
<h4 data-nodeid="33149">控制台注入</h4>
<p data-nodeid="33150">对于我们这个场景，控制台注入其实就够了，我们先来介绍这个方法。</p>
<p data-nodeid="33151">其实控制台注入很简单，就是直接在控制台输入这行代码运行，如图所示。</p>
<p data-nodeid="33152"><img src="https://s0.lgstatic.com/i/image/M00/00/DF/Ciqc1F6qlPqALm_6AAY-ro6hjwQ572.png" alt="image (27).png" data-nodeid="33309"></p>
<p data-nodeid="33153">执行完这段代码之后，相当于我们就已经把 window 的 btoa 方法改写了，可以控制台调用下 btoa 方法试试，如：</p>
<pre class="lang-js" data-nodeid="33154"><code data-language="js">btoa(<span class="hljs-string">'germey'</span>)
</code></pre>
<p data-nodeid="33155">回车之后就可以看到它进入了我们自定义的 debugger 的位置停下了，如图所示。</p>
<p data-nodeid="33156"><img src="https://s0.lgstatic.com/i/image/M00/00/DF/Ciqc1F6qlQiALo85AAgcGmKilPk416.png" alt="image (28).png" data-nodeid="33314"></p>
<p data-nodeid="33157">我们把断点向下执行，点击 Resume 按钮，然后看看控制台的输出，可以看到也输出了一些对应的结果，如被 Hook 的对象，Hook 的属性，调用的参数，调用后的结果等，如图所示。</p>
<p data-nodeid="33158"><img src="https://s0.lgstatic.com/i/image/M00/00/DF/Ciqc1F6qlRSAdfGaAAacrG5N8q4214.png" alt="image (29).png" data-nodeid="33318"></p>
<p data-nodeid="33159">这里我们就可以看到，我们通过 Hook 的方式改写了 btoa 方法，使其每次在调用的时候都能停到一个断点，同时还能输出对应的结果。</p>
<p data-nodeid="33160">接下来我们看下怎么用 Hook 找到对应的加密 id 的加密入口？</p>
<p data-nodeid="33161">由于此时我们是在控制台直接输入的 Hook 代码，所以页面一旦刷新就无效了，但由于我们这个网站是 SPA 式的页面，所以在点击详情页的时候页面是不会整个刷新的，所以这段代码依然还会生效。但是如果不是 SPA 式的页面，即每次访问都需要刷新页面的网站，这种注入方式就不生效了。</p>
<p data-nodeid="33162">好，那我们的目的是为了 Hook 列表页 Ajax 加载完成后的加密 id 的 Base64 编码的过程，那怎么在不刷新页面的情况下再次复现这个操作呢？很简单，点下一页就好了。</p>
<p data-nodeid="33163">这时候我们可以点击第 2 页的按钮，可以看到它确实再次停到了 Hook 方法的 debugger 处，由于列表页的 Ajax 和加密 id 都会带有 Base64 编码的操作，因此它每一个都能 Hook 到，通过观察对应的 Arguments 或当前网站的行为或者观察栈信息，我们就能大体知道现在走到了哪个位置了，从而进一步通过栈的调用信息找到调用 Base64 编码的位置。</p>
<p data-nodeid="33164">我们可以根据调用栈的信息来观察这些变量是在哪一层发生变化的，比如最后的这一层，我们可以很明显看到它执行了 Base64 编码，编码前的结果是：</p>
<pre class="lang-java" data-nodeid="33165"><code data-language="java">ef34#teuq0btua#(-57w1q5o5--j@98xygimlyfxs\*-!i-0-mb1
</code></pre>
<p data-nodeid="33166">编码后的结果是：</p>
<pre class="lang-java" data-nodeid="33167"><code data-language="java">ZWYzNCN0ZXVxMGJ0dWEjKC01N3cxcTVvNS0takA5OHh5Z2ltbHlmeHMqLSFpLTAtbWIx
</code></pre>
<p data-nodeid="33168">如图所示。</p>
<p data-nodeid="33169"><img src="https://s0.lgstatic.com/i/image/M00/00/DF/Ciqc1F6qlTOAXTlNAAbm2aEXYuA791.png" alt="image (30).png" data-nodeid="33329"></p>
<p data-nodeid="33170">这里很明显。</p>
<p data-nodeid="33171">那么核心问题就来了，编码前的结果 ef34#teuq0btua#(-57w1q5o5--j@98xygimlyfxs*-!i-0-mb1又是怎么来的呢？我们展开栈的调用信息，一层层看看这个字符串的变化情况。如果不变那就看下一层，如果变了那就停下来仔细看看。</p>
<p data-nodeid="33172">最后我们可以在第五层找到它的变化过程，如图所示。</p>
<p data-nodeid="33173"><img src="https://s0.lgstatic.com/i/image/M00/00/DF/Ciqc1F6qlVGAO62xAAmWfTuXTnI776.png" alt="image (31).png" data-nodeid="33339"></p>
<p data-nodeid="33174">那这里我们就一目了然了，看到了 _0x135c4d 是一个写死的字符串 ef34#teuq0btua#(-57w1q5o5--j@98xygimlyfxs*-!i-0-mb，然后和传入的这个 _0x565f18 拼接起来就形成了最后的字符串。</p>
<p data-nodeid="33175">那这个 _0x565f18 又是怎么来的呢？再往下追一层，那就一目了然了，其实就是 Ajax 返回结果的单个电影信息的 id。</p>
<p data-nodeid="33176">所以，这个加密逻辑的就清楚了，其实非常简单，就是 ef34#teuq0btua#(-57w1q5o5--j@98xygimlyfxs*-!i-0-mb1 加上电影 id，然后 Base64 编码即可。</p>
<p data-nodeid="33177">到此，我们就成功用 Hook 的方式找到加密的 id 生成逻辑了。</p>
<p data-nodeid="33178">但是想想有什么不太科学的地方吗？刚才其实也说了，我们的 Hook 代码是在控制台手动输入的，一旦刷新页面就不生效了，这的确是个问题。而且它必须是在页面加载完了才注入的，所以它并不能在一开始就生效。</p>
<p data-nodeid="33179">下面我们再介绍几种 Hook 注入方式</p>
<h4 data-nodeid="33180">重写 JavaScript</h4>
<p data-nodeid="33181">我们可以借助于 Chrome 浏览器的 Overrides 功能实现某些 JavaScript 文件的重写和保存，它会在本地生成一个 JavaScript 文件副本，以后每次刷新的时候会使用副本的内容。</p>
<p data-nodeid="33182">这里我们需要切换到 Sources 选项卡的 Overrides 选项卡，然后选择一个文件夹，比如这里我自定了一个文件夹名字叫作 modify，如图所示。</p>
<p data-nodeid="33183"><img src="https://s0.lgstatic.com/i/image/M00/00/DF/Ciqc1F6qlX6ADDfeAAZ2FWeRVsQ406.png" alt="image (32).png" data-nodeid="33365"></p>
<p data-nodeid="33184">然后我们随便选一个 JavaScript 脚本，后面贴上这段注入脚本，如图所示。</p>
<p data-nodeid="33185"><img src="https://s0.lgstatic.com/i/image/M00/00/DF/CgqCHl6qlYmAK5yrAAh8B0rKWe4441.png" alt="image (33).png" data-nodeid="33369"></p>
<p data-nodeid="33186">保存文件。此时可能提示页面崩溃，但是不用担心，重新刷新页面就好了，这时候我们就发现现在浏览器加载的 JavaScript 文件就是我们修改过后的了，文件的下方会有一个标识符，如图所示。</p>
<p data-nodeid="33187"><img src="https://s0.lgstatic.com/i/image/M00/00/DF/CgqCHl6qlZSADgU4AAZ9p5grU3A458.png" alt="image (34).png" data-nodeid="33373"></p>
<p data-nodeid="33188">同时我们还注意到这时候它就直接进入了断点模式，成功 Hook 到了 btoa 这个方法了。其实 Overrides 的这个功能非常有用，有了它我们可以持久化保存我们任意修改的 JavaScript 代码，所以我们想在哪里改都可以了，甚至可以直接修改 JavaScript 的原始执行逻辑也都是可以的。</p>
<h4 data-nodeid="33189">Tampermonkey 注入</h4>
<p data-nodeid="33190">如果我们不想用 Overrides 的方式改写 JavaScript 的方式注入的话，还可以借助于浏览器插件来实现注入，这里推荐的浏览器插件叫作 Tampermonkey，中文叫作“油猴”。它是一款浏览器插件，支持 Chrome。利用它我们可以在浏览器加载页面时自动执行某些 JavaScript 脚本。由于执行的是 JavaScript，所以我们几乎可以在网页中完成任何我们想实现的效果，如自动爬虫、自动修改页面、自动响应事件等等。</p>
<p data-nodeid="33191">首先我们需要安装 Tampermonkey，这里我们使用的浏览器是 Chrome。直接在 Chrome 应用商店或者在 Tampermonkey 的官网 <a href="https://www.tampermonkey.net/" data-nodeid="33380">https://www.tampermonkey.net/</a> 下载安装即可。</p>
<p data-nodeid="33192">安装完成之后，在 Chrome 浏览器的右上角会出现 Tampermonkey 的图标，这就代表安装成功了。</p>
<p data-nodeid="33193"><img src="https://s0.lgstatic.com/i/image/M00/00/E0/Ciqc1F6qlciAUhX0AAARZPlxmoI615.png" alt="image (35).png" data-nodeid="33385"></p>
<p data-nodeid="33194">我们也可以自己编写脚本来实现想要的功能。编写脚本难不难呢？其实就是写 JavaScript 代码，只要懂一些 JavaScript 的语法就好了。另外除了懂 JavaScript 语法，我们还需要遵循脚本的一些写作规范，这其中就包括一些参数的设置。</p>
<p data-nodeid="33195">下面我们就简单实现一个小的脚本，实现某个功能。</p>
<p data-nodeid="33196">首先我们可以点击 Tampermonkey 插件图标，点击“管理面板”按钮，打开脚本管理页面。</p>
<p data-nodeid="33197"><img src="https://s0.lgstatic.com/i/image/M00/00/E0/CgqCHl6qldmABWZBAABrf5xHC2s424.png" alt="image (36).png" data-nodeid="33391"></p>
<p data-nodeid="33198">界面类似显示如下图所示。</p>
<p data-nodeid="33199"><img src="https://s0.lgstatic.com/i/image/M00/00/E0/Ciqc1F6qleOAMwp5AAC3g20XxJI627.png" alt="image (37).png" data-nodeid="33395"></p>
<p data-nodeid="33200">在这里显示了我们已经有的一些 Tampermonkey 脚本，包括我们自行创建的，也包括从第三方网站下载安装的。</p>
<p data-nodeid="33201">另外这里也提供了编辑、调试、删除等管理功能，我们可以方便地对脚本进行管理。接下来我们来创建一个新的脚本来试试，点击左侧的“+”号，会显示如图所示的页面。</p>
<p data-nodeid="33202"><img src="https://s0.lgstatic.com/i/image/M00/00/E0/CgqCHl6qlfeAR-XCAAFU2XQaKcs123.png" alt="image (38).png" data-nodeid="33400"></p>
<p data-nodeid="33203">初始化的代码如下：</p>
<pre class="lang-js" data-nodeid="33204"><code data-language="js"><span class="hljs-comment">// ==UserScript==</span>
<span class="hljs-comment">// @name         New Userscript</span>
<span class="hljs-comment">// @namespace   http://tampermonkey.net/</span>
<span class="hljs-comment">// @version     0.1</span>
<span class="hljs-comment">// @description try to take over the world!</span>
<span class="hljs-comment">// @author       You</span>
<span class="hljs-comment">// @match       https://www.tampermonkey.net/documentation.php?ext=dhdg</span>
<span class="hljs-comment">// @grant       none</span>
<span class="hljs-comment">// ==/UserScript==</span>

(<span class="hljs-function"><span class="hljs-keyword">function</span>(<span class="hljs-params"></span>) </span>{
   <span class="hljs-string">'use strict'</span>;

   <span class="hljs-comment">// Your code here...</span>
})();
</code></pre>
<p data-nodeid="33205">这里最上面是一些注释，但这些注释是非常有用的，这部分内容叫作 UserScript Header ，我们可以在里面配置一些脚本的信息，如名称、版本、描述、生效站点等等。</p>
<p data-nodeid="33206">在 UserScript Header 下方是 JavaScript 函数和调用的代码，其中 use strict 标明代码使用 JavaScript 的严格模式，在严格模式下可以消除 Javascript 语法的一些不合理、不严谨之处，减少一些怪异行为，如不能直接使用未声明的变量，这样可以保证代码的运行安全，同时提高编译器的效率，提高运行速度。在下方 // Your code here... 这里我们就可以编写自己的代码了。</p>
<p data-nodeid="33207">我们可以将脚本改写为如下内容：</p>
<pre class="lang-js" data-nodeid="35414"><code data-language="js"><span class="hljs-comment">// ==UserScript==</span>
<span class="hljs-comment">// @name         HookBase64</span>
<span class="hljs-comment">// @namespace   https://scrape.center/</span>
<span class="hljs-comment">// @version     0.1</span>
<span class="hljs-comment">// @description Hook Base64 encode function</span>
<span class="hljs-comment">// @author       Germey</span>
<span class="hljs-comment">// @match       https://dynamic6.scrape.center/</span>
<span class="hljs-comment">// @grant       none</span>
<span class="hljs-comment">// @run-at     document-start</span>
<span class="hljs-comment">// ==/UserScript==</span>
(<span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
   <span class="hljs-string">'use strict'</span>
   <span class="hljs-function"><span class="hljs-keyword">function</span> <span class="hljs-title">hook</span>(<span class="hljs-params">object, attr</span>) </span>{
       <span class="hljs-keyword">var</span> func = object[attr]
       <span class="hljs-built_in">console</span>.log(<span class="hljs-string">'func'</span>, func)
       object[attr] = <span class="hljs-function"><span class="hljs-keyword">function</span> (<span class="hljs-params"></span>) </span>{
           <span class="hljs-built_in">console</span>.log(<span class="hljs-string">'hooked'</span>, object, attr)
           <span class="hljs-keyword">var</span> ret = func.apply(object, <span class="hljs-built_in">arguments</span>)
           <span class="hljs-keyword">debugger</span>
           <span class="hljs-keyword">return</span> ret
      }
  }
   hook(<span class="hljs-built_in">window</span>, <span class="hljs-string">'btoa'</span>)
})()
</code></pre>


<p data-nodeid="33209">这时候启动脚本，重新刷新页面，可以发现也可以成功 Hook 住 btoa 方法，如图所示。</p>
<p data-nodeid="33210"><img src="https://s0.lgstatic.com/i/image/M00/00/E0/Ciqc1F6qlkWAXng9AAXpKOp2pIg630.png" alt="image (39).png" data-nodeid="33408"></p>
<p data-nodeid="33211">然后我们再顺着找调用逻辑就好啦。</p>
<p data-nodeid="33212">以上，我们就成功通过 Hook 的方式找到加密 id 的实现了。</p>
<h3 data-nodeid="33213">详情页 Ajax 的 token 寻找</h3>
<p data-nodeid="33214">现在我们已经找到详情页的加密 id 了，但是还差一步，其 Ajax 请求也有一个 token，如图所示。</p>
<p data-nodeid="33215"><img src="https://s0.lgstatic.com/i/image/M00/00/E0/CgqCHl6qllaAX0VEAAcFtYFfiyc075.png" alt="image (40).png" data-nodeid="33415"></p>
<p data-nodeid="33216">其实这个 token 和详情页的 token 构造逻辑是一样的了。</p>
<p data-nodeid="33217">这里就不再展开说了，可以运用上文的几种找入口的方法来找到对应的加密逻辑。</p>
<h3 data-nodeid="33218">Python 实现详情页爬取</h3>
<p data-nodeid="33219">现在我们已经成功把详情页的加密 id 和 Ajax 请求的 token 找出来了，下一步就能使用 Python 完成爬取了，这里我就只实现第一页的爬取了，代码示例如下：</p>
<pre class="lang-js te-preview-highlight" data-nodeid="36724"><code data-language="js"><span class="hljs-keyword">import</span> hashlib
<span class="hljs-keyword">import</span> time
<span class="hljs-keyword">import</span> base64
<span class="hljs-keyword">from</span> typing <span class="hljs-keyword">import</span> List, Any
<span class="hljs-keyword">import</span> requests

INDEX_URL = <span class="hljs-string">'https://dynamic6.scrape.center/api/movie?limit={limit}&amp;offset={offset}&amp;token={token}'</span>
DETAIL_URL = <span class="hljs-string">'https://dynamic6.scrape.center/api/movie/{id}?token={token}'</span>
LIMIT = <span class="hljs-number">10</span>
OFFSET = <span class="hljs-number">0</span>
SECRET = <span class="hljs-string">'ef34#teuq0btua#(-57w1q5o5--j@98xygimlyfxs*-!i-0-mb'</span>

def get_token(args: List[Any]):
   timestamp = str(int(time.time()))
   args.append(timestamp)
   sign = hashlib.sha1(<span class="hljs-string">','</span>.join(args).encode(<span class="hljs-string">'utf-8'</span>)).hexdigest()
   <span class="hljs-keyword">return</span> base64.b64encode(<span class="hljs-string">','</span>.join([sign, timestamp]).encode(<span class="hljs-string">'utf-8'</span>)).decode(<span class="hljs-string">'utf-8'</span>) 

args = [<span class="hljs-string">'/api/movie'</span>]
token = get_token(args=args)
index_url = INDEX_URL.format(limit=LIMIT, offset=OFFSET, token=token)
response = requests.get(index_url)
print(<span class="hljs-string">'response'</span>, response.json())

result = response.json()
<span class="hljs-keyword">for</span> item <span class="hljs-keyword">in</span> result[<span class="hljs-string">'results'</span>]:
   id = item[<span class="hljs-string">'id'</span>]
   encrypt_id = base64.b64encode((SECRET + str(id)).encode(<span class="hljs-string">'utf-8'</span>)).decode(<span class="hljs-string">'utf-8'</span>)
   args = [f<span class="hljs-string">'/api/movie/{encrypt_id}'</span>]
   token = get_token(args=args)
   detail_url = DETAIL_URL.format(id=encrypt_id, token=token)
   response = requests.get(detail_url)
   print(<span class="hljs-string">'response'</span>, response.json())
</code></pre>


<p data-nodeid="33221">这里模拟了详情页的加密 id 和 token 的构造过程，然后请求了详情页的 Ajax 接口，这样我们就可以爬取到详情页的内容了。</p>
<h3 data-nodeid="33222">总结</h3>
<p data-nodeid="33223">本课时内容很多，一步步介绍了整个网站的 JavaScript 逆向过程，其中的技巧有：</p>
<ul data-nodeid="33224">
<li data-nodeid="33225">
<p data-nodeid="33226">全局搜索查找入口</p>
</li>
<li data-nodeid="33227">
<p data-nodeid="33228">代码格式化</p>
</li>
<li data-nodeid="33229">
<p data-nodeid="33230">XHR 断点</p>
</li>
<li data-nodeid="33231">
<p data-nodeid="33232">变量监听</p>
</li>
<li data-nodeid="33233">
<p data-nodeid="33234">断点设置和跳过</p>
</li>
<li data-nodeid="33235">
<p data-nodeid="33236">栈查看</p>
</li>
<li data-nodeid="33237">
<p data-nodeid="33238">Hook 原理</p>
</li>
<li data-nodeid="33239">
<p data-nodeid="33240">Hook 注入</p>
</li>
<li data-nodeid="33241">
<p data-nodeid="33242">Overrides 功能</p>
</li>
<li data-nodeid="33243">
<p data-nodeid="33244">Tampermonkey 插件</p>
</li>
<li data-nodeid="33245">
<p data-nodeid="33246">Python 模拟实现</p>
</li>
</ul>
<p data-nodeid="33247">掌握了这些技巧我们就能更加得心应手地实现 JavaScript 逆向分析。</p>
<p data-nodeid="33248" class="">本节代码：<a href="https://github.com/Python3WebSpider/ScrapeDynamic6" data-nodeid="33438">https://github.com/Python3WebSpider/ScrapeDynamic6</a></p>

