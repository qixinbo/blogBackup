---
title: 用Python开发web应用——Streamlit框架解析：1 -- 上手
tags: [Streamlit]
categories: digitalization 
date: 2022-3-8
---

# 简介
[Streamlit](https://streamlit.io/)是一个开源Python库，其旨在为机器学习和数据科学领域简单快速地创建和分享美观的、定制化的web应用。使用Streamlit，用户仅仅几分钟即可创建和部署强大的数据应用。
截几张范例的界面图：
![gallery1](https://user-images.githubusercontent.com/6218739/157160066-1781000e-b957-4fac-b0df-01d23b3c7f67.png)
![g2](https://user-images.githubusercontent.com/6218739/157160339-5e34e7fc-8886-4bd9-bc18-3e94fff749ce.png)
![g3](https://user-images.githubusercontent.com/6218739/157160476-7d64aa1a-4c92-4f80-8dd3-5e449fcc56be.png)
# 安装
使用pip安装：
```python
pip install streamlit
```

测试一下：
```python
streamlit hello
```
此时浏览器会打开`http://localhost:8501/`，然后出现streamlit关于动画、动态绘图、地图、pandas绘图的四个demo。

# 核心概念
## 运行方式
```python
streamlit run your_script.py [-- script args]
```

另外一种运行方式是通过Python模块运行（这对于使用IDE如pycharm有用）：
```python
# Running
$ python -m streamlit your_script.py

# is equivalent to:
$ streamlit run your_script.py
```

## 开启开发模式
在开发阶段，最好是开启“开发模式”，这样只要保存代码后，Streamlit就能重新运行app。这会极大地提高开发效率。
开启方式是在右上角选择“Always rerun”。

## 展示数据
### 使用“魔法”
魔法magic和`st.write()`可以用来展示很多数据类型，比如text、data、matplotlib图表、Altair图表。直接将这些数据传给`st.write()`或者magic即可，Streamlit可以自动识别。
这里魔法magic的意思是不用在代码里调用Streamlit的任何方法就可以直接展示数据，原因是当Streamlit看到在一行中只有一个变量名时，就会自动在这里加上`st.write()`。
比如下面代码：
```python
"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df
```
它跟下面的代码效果是一样的：
```python
import streamlit as st
import pandas as pd

st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))
```

### 展示data frame
前面说了，`st.write()`或magic几乎能展示所有数据，但也有其他的与具体数据相关的函数，如`st.dataframe()`和`st.table()`等。
这里可能有一个问题：“为什么我不能全用`st.write()`呢”，原因如下：
（1）`st.write()`或magic能自动渲染数据，但有时你可能想用另外一种方式渲染。比如，如果你不想将dataframe数据渲染成一种可交互的表格，此时就需要使用`st.table(df)`将它渲染成静态表格；
（2）其他方法返回的对象可以被使用和修改，比如在上面增加数据或替换数据；
（3）对于其他方法，可以传递更多的参数来定制化行为。
比如下面的例子使用Pandas的styler来高亮化某些元素：
```python
import streamlit as st
import numpy as np
import pandas as pd

dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))

st.dataframe(dataframe.style.highlight_max(axis=0))
```
以及静态图表：
```python
import streamlit as st
import numpy as np
import pandas as pd

dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))
st.table(dataframe)
```

### 展示charts和maps
Streamlit支持很多流行的绘图库，如Matplotlib、Altair、deck.gl、plotly等等。一些demo见[这里](https://docs.streamlit.io/library/api-reference#chart-elements)。
折线图举例：
```python
import streamlit as st
import numpy as np
import pandas as pd

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)
```
地图举例：
```python
import streamlit as st
import numpy as np
import pandas as pd

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)
```

## 部件
当想要探索得到的数据或模型时，可以使用部件进行调节，比如滑块`st.slider()`、按钮`st.button`、下拉列表`st.selectbox`。
使用方法也很简单，就像将这些部件视作变量。
### 滑块
常用来调节数值：
```python
import streamlit as st
x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)
```
### 复选框
常用来显示或关闭数据。
```python
import streamlit as st
import numpy as np
import pandas as pd

if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    chart_data
```

### 下拉列表
常用来选择数据
```python
import streamlit as st
import pandas as pd

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
    })

option = st.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected: ', option
```

### 部件的key
如果对某一部件附加了一个独特的key，那么，部件的值可以通过key来获取，比如：
```python
import streamlit as st
st.text_input("Your name", key="name")

# You can access the value at any point with:
st.session_state.name
```
有key的部件会被自动添加到Session State中，从而可以在部件间传递数据。
详情查看[这里](https://docs.streamlit.io/library/api-reference/session-state)。

### 进度条
当一个app需要运行很长时间时，可以添加进度条部件`st.progress()`来显示进度。进度条不能添加key。
```python
import streamlit as st
import time

'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'
```

## 布局
### sidebar
Streamlit提供了一个左侧侧边栏`st.sidebar`来组织上面的部件。每一个传给该侧边栏的元素都被“钉”到左边，这样用户就能专注于自己的app内容上。
比如使用`st.sidebar.slider`替代`st.slider`：
```python
import streamlit as st

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)
```

### columns和expander
除了侧边栏，Streamlit还提供了其他控制布局的方式，如`st.columns`可以一列一列地排放部件，`st.expander`可以将大片的内容隐藏或展开。
```python
import streamlit as st

left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")
```

## 主题
Streamlit默认支持Light和Dark两种主题。可以通过Settings进行设置。也可以自定义主题。

## 缓存
当从web上加载数据、操作大型数据集以及进行大规模计算时，对状态的缓存就非常重要，Streamlit缓存使得这些情况下app仍然快速执行。
Streamlit提供了一些缓存方案，如`@st.cache`、`@st.experimental_memo`、`@st.experimental_singleton`。
具体的用法可以参考[这里](https://docs.streamlit.io/library/advanced-features/experimental-cache-primitives)。
以`@st.cache`为例，当指定需要使用缓存时，就用这个装饰器包装一下函数：
```python
import streamlit as st

@st.cache  # 👈 This function will be cached
def my_slow_function(arg1, arg2):
    # Do something really slow in here!
    return the_output
```
这个装饰器告诉Streamlit，当该函数被调用时，它需要检查如下东西：
（1）该函数的输入参数；
（2）在函数内用到的任意外部变量；
（3）函数体；
（4）在该函数体内用到的其他函数体。
如果是Streamlit看到是这四个部分都是第一次以这些数值及其组合顺序运行，那么它就运行函数，然后将结果存储在局部缓存中。然后，当该缓存的函数下一次被调用时，如果上述四部分都没有改变，那么Streamlit就会跳过执行，而将上一次缓存的结果返回。

## 运行机理
知道了上面的零碎的知识，总结一下整体的运行机理：
（1）Streamlit的apps是从上到下执行的Python脚本；
（2）每次当一个用户打开浏览器，访问你的app后，上述脚本就会重新执行；
（3）当脚本执行时，Streamlit在浏览器渲染它的输出；
（4）脚本使用Streamlit缓存来避免重复执行昂贵的运算，所以结果更新会非常快；
（5）每次当用户与部件进行交互时，脚本就会重新运行，部件的返回值也会更新为最新状态。

# 上手总结
以上就是最基本的Streamlit用法，总体来看，确实极大地降低了开发web app的难度，可以使用原生python语法来做这件事是太香了。
