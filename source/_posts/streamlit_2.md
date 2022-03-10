---
title: 用Python开发web应用——Streamlit框架解析：2 -- 会话状态
tags: [Streamlit]
categories: digitalization 
date: 2022-3-10
---

# 简介
在Streamlit中，与一个部件widget的交互会触发“重新运行”rerun，这样一来每次运行后代码中的变量都会被重新初始化。这可能会带来很麻烦的问题，比如实现一个简单的“累加计数按钮”，每次点击后都会清零，无法实现累加功能。
为了解决类似问题，Streamlit引入了一种新的特性：会话状态Session State，它是一种可以在rerun之间保存变量状态、配合使用回调函数处理部件上的事件、动态改变部件状态等等的功能（注意，这些功能发生在一个session中，一个session可以简单理解为用户通过浏览器的一个标签页来访问Streamlit）。其可以用在如下场景中：
（1）数据或图像标注[code](https://github.com/streamlit/release-demos/blob/0.84/0.84/demos/labelling.py)；
（2）创建分页[code](https://github.com/streamlit/release-demos/blob/0.84/0.84/demos/pagination.py)；
（3）基于其他部件来添加部件；
（4）创建简单的基于状态的小游戏，如井字棋[code](https://github.com/streamlit/release-demos/blob/0.84/0.84/demos/tic_tac_toe.py)；
（5）待办事项清单[code](https://github.com/streamlit/release-demos/blob/0.84/0.84/demos/todo_list.py)。

# 累加计数例子
比如最开头提到的“累加计数按钮”的实现，如下：
```python
import streamlit as st

st.title('Counter Example')

# Streamlit runs from top to bottom on every iteraction so
# we check if `count` has already been initialized in st.session_state.

# If no, then initialize count to 0
# If count is already initialized, don't do anything
if 'count' not in st.session_state:
    st.session_state.count = 0

# Create a button which will increment the counter
increment = st.button('Increment')
if increment:
    st.session_state.count += 1

# A button to decrement the counter
decrement = st.button('Decrement')
if decrement:
    st.session_state.count -= 1

st.write('Count = ', st.session_state.count)
```

# 配合回调函数更新会话状态
首先看一下什么是回调函数，援引维基百科上的一张图：
![Callback-notitle](https://user-images.githubusercontent.com/6218739/157572549-24b73b56-bd8d-4f5b-8825-c66eb25e0449.svg)
这张图说明了几个事情：
（1）底层有一个库函数，它被其他程序（这里是Main program这个程序）所调用
（2）这个库函数有脾气，它不能被简单调用，需要给它提前传一个函数（即回调函数），这样才能在合适的时候执行该函数
（3）这个回调函数与main program处于同一层级，是由main program来指定的。
可以这样记忆回调函数：从库函数的视角，你先给我这个函数，我回头再调用你。
[回调函数（callback）是什么？ - no.body的回答 - 知乎](https://www.zhihu.com/question/19801131/answer/27459821)挺好。

## 使用回调函数更新会话状态
```python
import streamlit as st

st.title('Counter Example using Callbacks')
if 'count' not in st.session_state:
    st.session_state.count = 0

def increment_counter():
    st.session_state.count += 1

st.button('Increment', on_click=increment_counter)

st.write('Count = ', st.session_state.count)
```
即可以在输入部件（比如按钮、滑块、文本框等）的`on_change`或`on_click`的事件上绑定回调函数。

## 在回调函数中使用args和kwargs
可以在回调函数中传入参数：
```python
import streamlit as st

st.title('Counter Example using Callbacks with args')
if 'count' not in st.session_state:
    st.session_state.count = 0

increment_value = st.number_input('Enter a value', value=0, step=1)

def increment_counter(increment_value):
    st.session_state.count += increment_value

increment = st.button('Increment', on_click=increment_counter,
    args=(increment_value, ))

st.write('Count = ', st.session_state.count)
```
也可以传入字典类型的命名参数：
```python
import streamlit as st

st.title('Counter Example using Callbacks with kwargs')
if 'count' not in st.session_state:
    st.session_state.count = 0

def increment_counter(increment_value=0):
    st.session_state.count += increment_value

def decrement_counter(decrement_value=0):
    st.session_state.count -= decrement_value

st.button('Increment', on_click=increment_counter,
    kwargs=dict(increment_value=5))

st.button('Decrement', on_click=decrement_counter,
    kwargs=dict(decrement_value=1))

st.write('Count = ', st.session_state.count)
```

## 在表单上绑定回调函数
```python
import streamlit as st
import datetime

st.title('Counter Example')
if 'count' not in st.session_state:
    st.session_state.count = 0
    st.session_state.last_updated = datetime.time(0,0)

def update_counter():
    st.session_state.count += st.session_state.increment_value
    st.session_state.last_updated = st.session_state.update_time

with st.form(key='my_form'):
    st.time_input(label='Enter the time', value=datetime.datetime.now().time(), key='update_time')
    st.number_input('Enter a value', value=0, step=1, key='increment_value')
    submit = st.form_submit_button(label='Update', on_click=update_counter)

st.write('Current Count = ', st.session_state.count)
st.write('Last Updated = ', st.session_state.last_updated)
```
在form中仅有`st.form_submit_button`可以设置回调函数，其他在form中的部件不允许有回调函数。

# 变量状态和部件状态的关联
Session State存储了变量的值，而部件widgets的状态也可以存储在Session State中，变量的状态与部件的状态就可以实现梦幻联动了，方法就是将变量名设置为部件的key值。
比如：
```python
import streamlit as st

if "celsius" not in st.session_state:
    # set the initial default value of the slider widget
    st.session_state.celsius = 50.0

st.slider(
    "Temperature in Celsius",
    min_value=-100.0,
    max_value=100.0,
    key="celsius"
)

# This will get the value of the slider widget
st.write(st.session_state.celsius)
```
但是，有两个例外，不能通过Session State的API来改变`st.button`、`st.download_button`和`st.file_uploader`部件的状态。
还需要注意的是Session State变量和部件初始化的顺序，如果先初始化了部件，再通过Session State的API来更改它的状态，此时就会报错，抛出`StreamlitAPIException`的错误。
所以，一定注意先在Session State中定义好变量。

# API
## 初始化状态
```python
# Initialization
if 'key' not in st.session_state:
    st.session_state['key'] = 'value'

# Session State also supports attribute based syntax
if 'key' not in st.session_state:
    st.session_state.key = 'value'
```

## 读取和更新
读取Session State：
```python
# 读取某一个状态
st.write(st.session_state.key)

# 读取所有状态
st.write(st.session_state)

# With magic:
st.session_state
```
更新状态（有两种方式）：
```python
st.session_state.key = 'value2'     # Attribute API
st.session_state['key'] = 'value2'  # Dictionary like API
```

## 删除
```python
# Delete a single key-value pair
del st.session_state[key]

# Delete all the items in Session state
for key in st.session_state.keys():
    del st.session_state[key]
```
也可以通过在Settings中`Clear Cache`来删除，并rerun整个app。


# 注意点
使用Session State时需要注意以下几点：
（1）Session State的生命周期存在于浏览器的标签页打开且连接到server期间。一旦关闭标签页后，Session State中存储的东西都会丢失。
（2）Session State也不能持久化，一旦server关闭，其存储的东西也会被擦除。

# 参考资料
[Session State for Streamlit](https://blog.streamlit.io/session-state-for-streamlit/)
[Add statefulness to apps](https://docs.streamlit.io/library/advanced-features/session-state)
[Session State](https://docs.streamlit.io/library/api-reference/session-state)
