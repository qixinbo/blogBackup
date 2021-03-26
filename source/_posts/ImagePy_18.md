---
title: ImagePy解析：18 -- 参数对话框ParaDialog详解
tags: [ImagePy]
categories: computer vision 
date: 2020-3-24
---

本文对sciwx的独立组件——参数对话框ParaDialog进行解析。

# demo全景
首先还是直接给出sciwx库中可运行的demo：
```python
from sciwx.widgets import ParaDialog

if __name__ == '__main__':
    para = {'name':'yxdragon', 'age':10, 'h':1.72, 'w':70, 'sport':True, 'sys':'Mac', 'lan':['C/C++', 'Python'], 'c':(255,0,0), 'path': ' '}

    view = [('lab', 'lab', 'This is a questionnaire'),
            (str, 'name', 'name', 'please'),
            (int, 'age', (0,150), 0, 'age', 'years old'),
            (float, 'h', (0.3, 2.5), 2, 'height', 'm'),
            ('slide', 'w', (1, 150), 0, 'weight','kg'),
            (bool, 'sport', 'do you like sport'),
            (list, 'sys', ['Windows','Mac','Linux'], str, 'favourite', 'system'),
            ('chos', 'lan', ['C/C++','Java','Python'], 'lanuage you like(multi)'),
            ('color', 'c', 'which', 'you like'),
            ('path', 'path', 'Select the image', ['jpg', 'jpeg', 'png'])]

    app = wx.App()
    pd = ParaDialog(None, 'Test')
    pd.init_view(view, para, preview=True, modal=False)
    pd.pack()
    pd.ShowModal()
    print(para)
    app.MainLoop()
```

运行结果如图：
![paradiglog](https://user-images.githubusercontent.com/6218739/77433470-5efc0080-6e1a-11ea-85e1-83e3d54cad10.png)

可以看出，该对话框中包含了输入框（包含文本输入、数值输入）、滑动条（集成了微调器）、单选框、复选框、下拉列表等效果，完全满足日常人机交互的需求。

下面再分步细看具体设置。

# 参数字典para
变量para是一个参数字典，用来设置该对话框所提供的想要用户进行交互调节的参数。
该字典中的key都是字符串，而value则视参数类型的不同，要设置好默认值，比如数值输入框的默认值是数值，颜色输入框的默认值是(255, 0, 0)这样的RGB元组等。

# 视图列表view
变量view是一个列表，用来控制所形成的对话框的图形界面。
可以看出，view列表中又有若干元组，这些元组就是对话框中的各个组件。还有一点非常重要的是，注意到，不同组件所需的参数量不同，因此编写view时需要将这些参数搞明白。
编写view时，要将元组中的参数分成三类进行考虑（下文会对原因有详细说明），第一类是该元组的第一个元素，第二类是元组的第二个元素，第三类是后面所有元素。
第一个元素就是ImagePy/sciwx的内部组件类型，第二个元素就是para中的各个键值key（lab类型除外），第三个元素就是该组件需要的参数。因为第二个元素是para变量所定义的，所以下面只对其他元素进行说明：
（1）标签：内部类型'lab'，参数为title，view中写法为：('lab', 'lab', title)
（2）文本输入框：内部类型str，参数为(title, unit)，即名称（或称前缀）和单位（或称后缀）， view写法为：(str, key, title, unit)
（3）数值输入框：内部类型int或float，即使用int或float皆可，这里只是语义上的差别，两者实际调用的是同一个组件，使用int时后面的精度设为0，使用float时精度设为大于0的整数，即保留多少位小数；参数为(rang, accury, title, unit)，即范围、精度、名称和单位，view写法为：(int, key, (lim1, lim2), accu, title, unit)
（4）滑动条（集成了微调器）：内部类型'slide'（注意别忘了单引号），参数为(rang, accury, title, unit='')，即范围、精度、名称和单位（单位有默认参数，可以不显式指定），view写法为('slide', key, rang, accury, title, unit)
（5）单选框：内部类型bool，参数为title，view写法为(bool, key, title)
（6）下拉列表：内部类型list，参数为(choices, type, title, unit)，choices为字符选项，type为期望输出类型，比如str或int，view写法为(list, key, [choices], type, title, unit)
（7）复选框：内部类型'chos'，参数为(choices, title)，choices为字符选项，与上面的list不同的是，其支持多选，view写法为('chos', key, [choices], title)
（8）颜色框：内部类型'color'，参数为(title, unit)，即前缀和后缀，view写法为：('color', key, title, unit)
（9）路径选择框：内部类型'path'，参数为(title, filter)，filter是所指定的文件扩展名，view写法为：('path', key, title, filter)

# 初始化类
```python
pd = ParaDialog(None, 'Test')
```
这一步是初始化ParaDialog类。
该类派生自wx.Dialog类，从该类的初始化函数可以看出，其需要传入parent和title两个参数，所以该对话框的标题就是Test。

# 初始化视图
```python
pd.init_view(view, para, preview=True, modal=False)
```
这一步是调用了ParaDialog类的init_view函数，传入的就是上面的view和para，preview是指定是否显示preview单选框，如果勾选了该框，那么当参数发生变化时就会在后台打印参数值（因为这里的self.handle=print），。
该步是关键一步，在后台做了很多事情，如果想理清ParaDialog的生成机理，需要深入研究一下。
先看一下该函数的全貌：
```python
    def init_view(self, items, para, preview=False, modal = True):
        self.para = para
        for item in items:
            self.add_ctrl_(widgets[item[0]], item[1], item[2:])

        if preview:self.add_ctrl_(Check, 'preview', ('preview',))
        self.reset(para)
        self.add_confirm(modal)
        self.pack()
        self.Bind(wx.EVT_WINDOW_DESTROY, self.OnDestroy)
        print('bind close')
```
## 参数字典传递
```python
        self.para = para
```
这一步是将外部传过来的para赋值给内部属性self.para。
## 添加组件
```python
        for item in items:
            self.add_ctrl_(widgets[item[0]], item[1], item[2:])
```
这一步是将view视图中的组件添加进来。
因为view是个列表，所以这里先用循环逐个提取。然后在添加时将item拆分成了三部分，将其第0个元素、第1个元素、(第2个元素及后面所有元素)分别传入不同的位置，下面具体看为什么这么做。

### view的第0个元素
```python
widgets[item[0]]
```
这个元素是传入了widgets，而widgets也是一个字典：
```python
widgets = { 'ctrl':None, 'slide':FloatSlider, int:NumCtrl, 'path':PathCtrl,
            float:NumCtrl, 'lab':Label, bool:Check, str:TextCtrl, list:Choice,
            'color':ColorCtrl, 'any':AnyType, 'chos':Choices, 'hist':HistPanel}
```
可以看出，view的第0个元素是作为widgets的key，这样就能对应提取widgets中的value。而这些value就是sciwx独有的基于wxPython各种组件二次开发的组件。以NumCtrl为例：
```python
class NumCtrl(wx.Panel):
    """NumCtrl: diverid from wx.core.TextCtrl """
    def __init__(self, parent, rang, accury, title, unit):
        wx.Panel.__init__(self, parent)
        sizer = wx.BoxSizer( wx.HORIZONTAL )
        self.prefix = lab_title = wx.StaticText( self, wx.ID_ANY, title,
                                  wx.DefaultPosition, wx.DefaultSize)

        lab_title.Wrap( -1 )
        sizer.Add( lab_title, 0, wx.ALIGN_CENTER|wx.ALL, 5 )
        self.ctrl = wx.TextCtrl(self, wx.TE_RIGHT)
        self.ctrl.Bind(wx.EVT_KEY_UP, lambda x : self.para_changed(key))
        sizer.Add( self.ctrl, 2, wx.ALL, 5 )

        self.postfix = lab_unit = wx.StaticText( self, wx.ID_ANY, unit,
                                  wx.DefaultPosition, wx.DefaultSize)

        lab_unit.Wrap( -1 )
        sizer.Add( lab_unit, 0, wx.ALIGN_CENTER|wx.ALL, 5 )
        self.SetSizer(sizer)

        self.min, self.max = rang
        self.accury = accury
        self.ctrl.Bind(wx.EVT_KEY_UP, self.ontext)

    def Bind(self, z, f):self.f = f

    def ontext(self, event):
        self.f(self)
        if self.GetValue()==None:
            self.ctrl.SetBackgroundColour((255,255,0))
        else:
            self.ctrl.SetBackgroundColour((255,255,255))
        self.Refresh()

    def SetValue(self, n):
        self.ctrl.SetValue(str(round(n,self.accury) if self.accury>0 else int(n)))

    def GetValue(self):
        sval = self.ctrl.GetValue()
        try:
            num = float(sval) if self.accury>0 else int(sval)
        except ValueError:
            return None
        if num<self.min or num>self.max:
            return None
        if abs(round(num, self.accury) - num) > 1E-5:
            return None
        return num
```
可以看出NumCtrl就是由两个wx.StaticText静态文本框和一个wx.TextCtrl输入框组合而成，表现出来就是一个前缀说明和一个后缀说明及中间的输入框。
同时可以可以看出NumCtrl有数值检查这一功能，当超出所设定的范围后，就会报警。

### view的第1个元素
view的第1个元素就是para中的各个key值

### view的第2个及后面所有元素
因为不同组件需要的参数量不同，因为这里将第2个及后面所有元素统一打包然后传入。

### 添加组件 
以上三组元素都传入了下面方法，分别作为它的Ctrl、key和p参数：
```python
    def add_ctrl_(self, Ctrl, key, p):
        ctrl = Ctrl(self, *p)
        if not p[0] is None:
            self.ctrl_dic[key] = ctrl
        if hasattr(ctrl, 'Bind'):
            ctrl.Bind(None, self.para_changed)
        pre = ctrl.prefix if hasattr(ctrl, 'prefix') else None
        post = ctrl.postfix if hasattr(ctrl, 'postfix') else None
        self.tus.append((pre, post))
        self.lst.Add( ctrl, 0, wx.EXPAND, 0 )
```
这里面挺有创意的一点是通过判断组件类里是否有某个特定的属性来进行下一步操作，比如判断是否有Bind属性，若有则Bind对话框的para_changed方法，还有prefix、postfix属性等，具体参见代码。

# 显示
```python
    pd.ShowModal()
```
通过调用父类wx.Dialog的ShowModal()方法来将对话框显示出来。

# 隐式输出
该对话框提供了图形界面供用户来调节para变量中的参数，当调节完成后，隐含地就对para变量中的key值所对应的value进行了调节。原理就是在para_changed方法中的GetValue()，即调用各个组件的GetValue()来获取输入值。
