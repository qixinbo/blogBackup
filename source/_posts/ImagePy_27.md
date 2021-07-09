---
title: ImagePy解析： 27 -- 工作流workflow组件
tags: [ImagePy]
categories: computer vision 
date: 2021-7-10
---

ImagePy的工作流worflow功能能够以可视化的方式逐步执行已定义的一系列图像处理动作，即有机地将复杂的图像处理步骤串联起来，也提供了可视化便捷的交互方式，可以认为是更人性化的“宏命令”。
![image](https://user-images.githubusercontent.com/6218739/125029058-a4018000-e0bb-11eb-841c-58f2cffa19c8.png)
本文就是解析一下这个组件的底层原理。
# 文本解析
如下parse函数是读取描述workflow的文件，然后根据每行的标识对其进行解析，比如如果是两个井号开头，则这一行代表是chapter，以及在某个chapter下面还有若干section及其提示信息hint。在底层来说，就是将这些文件信息存储为有层级的python字典。
```python
def parse(cont):
	ls = cont.split('\n')
	workflow = {'title':ls[0], 'chapter':[]}
	for line in ls[2:]:
		line = line.strip()
		if line == '':continue
		if line.startswith('## '):
			chapter = {'title':line[3:], 'section':[]}
			workflow['chapter'].append(chapter)
		elif line[1:3] == '. ':
			section = {'title':line[3:]}
		else:
			section['hint'] = line
			chapter['section'].append(section)
	return workflow
```

# 界面实现
先来看整体的界面布局图：
![Untitled](https://user-images.githubusercontent.com/6218739/125015230-aa82fe00-e0a1-11eb-9bdb-623c5516413f.png)
可以看出，整个界面由三部分构成：
（1）微调按钮
它使用的组件是wxPython的SpinButton：
```python
self.spn_scroll = wx.SpinButton( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SP_HORIZONTAL )
```
它是用来切换后面的工作流中包含的各个Chapter控件的显示，具体看一下它绑定的事件：
```python
self.spn_scroll.Bind( wx.EVT_SPIN, self.on_spn )

def on_spn(self, event):
	v = self.spn_scroll.GetValue()
	self.scr_workflow.Scroll(v, 0)
	self.spn_scroll.SetValue(self.scr_workflow.GetViewStart()[0])
```

（2）工作流组件显示
这个部分是核心，是用来显示工作流中包含的各个图像处理功能组件，并赋予相应的功能。
因为事先不知道一个工作流中具体包含多少个图像处理功能，因此，需要使用可以滚动显示的方式来承载未知个数的组件，具体应用的是wxPython的ScrolledCanvas这种画布：
```python
self.scr_workflow = wx.ScrolledCanvas( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize)
```
然后再将之前解析的工作流一个个添加到该Canvas中。
第一层级是以chapter为单位，多个chapter用水平排列的方式添加到canvas中；
第二层级是在每个chapter中，以垂直排列的方式依次添加chapter的标题、包含的Sections（即具体图像处理功能）及下面的Snap、load等等（目前这两个没有实际功能）。
添加的Section要与具体的图像处理操作绑定，所以要给它添加鼠标事件：
```python
for section in chapter['section']:
	btn = wx.Button( self.pan_chapter, wx.ID_ANY, section['title'], wx.DefaultPosition, wx.DefaultSize, wx.BU_EXACTFIT )
	sizer_section.Add( btn, 0, wx.ALL, 3 )
	btn.Bind(wx.EVT_BUTTON, lambda e, x=section['title']: self.f(x))
	btn.Bind( wx.EVT_ENTER_WINDOW, lambda e, info=section['hint']: self.info(info))
```
可以看出，有两个事件绑定，一个是鼠标单击事件，与一个匿名函数进行了绑定，该函数所做的是将section的title传入self.f函数中，并执行它（默认的f函数就是print）。另一个事件是当鼠标进入该button时，会在右侧的info窗口显示hint内容。
这个地方需要深究一下鼠标单击事件，即这个button是怎样执行具体的图像处理功能的：
首先，刚才已提到，该button与self.f是绑定的，即点击button时，会将title传入f函数来执行，那么就看一下f函数是啥。
```python
def Bind(self, event, f=print): self.f = f
```
从这个Bind函数可知，可以从外部传入一个f函数，然后赋值给该workflow组件的f函数。
那进一步探究外部是怎样传入f函数的。
具体看一下imagepy这个app中的实现：
```python
    def _show_workflow(self, cont, title='ImagePy'):
        pan = WorkFlowPanel(self)
        pan.SetValue(cont)
        pan.Bind(None, lambda x:self.run_macros(['%s>None'%x]))
```
在imagepy这个app中，f函数就是一个匿名函数：
```python
lambda x:self.run_macros(['%s>None'%x])
```
它执行了imagepy的宏执行命令，关键就是在这个地方了，它巧妙地将工作流中地命令映射到了执行宏命令上。
同时需要注意地是，这里的宏命令中的参数那一项是None，即默认不传入参数，因此就会跳出GUI窗口来让用户输入自己的参数，这正是宏命令与工作流的区别：底层都是宏命令，但一个是带参数的，一个是不带参数的。
（3）消息窗口
最右边就是消息提示窗口：
```python
self.txt_info = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_AUTO_URL|wx.TE_MULTILINE|wx.TE_READONLY )
```
前面已经说了，当鼠标进入某个button时，会在这里显示该button的提示消息：
```python
btn.Bind( wx.EVT_ENTER_WINDOW, lambda e, info=section['hint']: self.info(info))

def info(self, info):
	self.txt_info.SetValue(info)
```
