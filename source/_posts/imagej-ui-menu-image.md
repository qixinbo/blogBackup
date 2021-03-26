---
title: ImageJ 用户指南 -- 7. 菜单栏之Image
tags: [ImageJ]
categories: computer vision
date: 2018-9-6
---

# 本章说明
这里详解Image菜单的功能。
# Image
## Type
显示当前活动图像的类型（子菜单打钩的即是当前类型）以及将其转化成另一种类型。
- 8-bit：转化为8-bit灰度图。ImageJ将16-bit和32-bit的图像通过线性地将"min-max"缩放到"0-255"来转换成8-bit图像，其中min和max可以通过Image-Adjust-Brightness/Contrast来查看。注意，如果Edit-Options-Conversions中的If Scale When Converting 没有勾选，那么就不会缩放。
- 16-bit：转为unsigned 16-bit灰度图
- 32-bit：转为signed 32-bit浮点型灰度图
- 8-bit color：转为8-bit indexed 彩色图。当前图片必须是RGB图。
- RGB Color：转为32-bit RGB图
- RGB Stack：转为一个3-slice(RGB)的stack。
- HSB Stack：转为一个3-slice(HSV)的stack。

## Adjust
该菜单用来调节亮度/对比度、阈值和尺寸等。
### Brightness/Contrast
使用该工具来交互地调节图片的亮度和对比度。对于8-bit图片，亮度和对比度是通过修改查询表LUT来改变，所以像素值是不变的。对于16-bit和32-bit图片，是通过改变从像素值到8-bit显示值的映射来改变的，所以像素值也不会改变。对于RGB图，亮度和对比度是通过修改像素值来改变的。
- 直方图：显示像素值是怎样映射到8-bit（0-255）范围的。直方图下方的两个数是要显示的最小和最大像素值，这两个值可以在下面调节。如果实际像素值小于最小值，则显示为黑色；如果实际像素值大于最大值，则显示为白色。
- Minium and Maximum sliders：控制要显示的最小和最大像素值。按住Shift可以同时调节一个复合图片的所有通道。
- Brightness slider：通过移动显示范围（即同时增大或减小最小和最大像素值）来增加或降低图片亮度。
- Contrast slider：通过改变显示范围的宽度（即扩大或减小显示范围）来调节图片的对比度。
- Auto：ImageJ将会基于图片直方图的分析来自动优化亮度和对比度。创建一个选区后，整个图片也将会根据该选区的分析来自动优化。优化的方式是使得一小部分的像素值是饱和的，即显示为白色或黑色。每次额外的点击都会增加饱和像素的数目。
- Reset：恢复原来的亮度和对比度设置，将会显示整个范围的像素值。
- Set：允许用户输入最小和最大显示范围。
- Apply：应用当前设置。如果是选区，那么仅仅选区内的像素被修改。这是唯一的修改非RGB图片像素值的方式。

### Window/Level
该命令和Brightness/Contrast是重复的，它更多地适用于医学图片的处理。
### Color Balance
该命令改变一个标准RGB图片的每个通道的亮度和对比度。使用下拉菜单选择应用于哪个通道。
这里面的滑块和按钮的应用跟上面的B/C相同。
注意：当在色彩通道之间改变时，如果不点击Apply，那么之前的修改会丢失。
### Threshold
该命令自动或交互地设置阈值的上界和下界，从而将灰度图片分割成感兴趣的区域和背景。红框内框住的范围即是选择区域。
- Upper slider：调节阈值的下界。按住Shift能够在保持固定宽度阈值窗口的条件下调节下界。
- Lower slider：调节阈值的上界。
- Method：有16种不同的自动阈值算法可供选择。默认算法是改进IsoData算法。
- Display：有三种选择方式：（1）Red：用红色显示阈值以内的数值；（2）B&W：用黑色显示特征，白色作为北京。（3）Over/Under：在阈值下界以下的像素用蓝色显示，阈值范围以内的用灰色显示，阈值上界以上的用绿色显示。
- Dark background：但特征要比背景浅的话，就会被勾选。
- Stack histogram：勾选后，ImageJ将首先计算整个stack的直方图，然后基于此再计算阈值。如果不勾选，则每个slice的阈值会单独计算。
- Auto：使用当前选择的method以及当前图片或选区的的直方图来计算阈值。
- Apply：将阈值范围内的像素设为黑色，其他设为白色。对于32-bit图片，Apply也会运行Process-Math-NaN Background。
- Reset：去掉阈值，然后更新直方图。
- Set：手动输入阈值的上下界。

### Color Threshold
上面的Threshold是对灰度图像进行操作。这里的Color Threshold是对24-bit的RGB图像基于HSB、RGB或YUV等进行阈值设定。
- Pass：勾选后，范围以内的值被选定和显示，否则，范围以外的值被选定和显示。
- Thresholding Method：选择16种不同的自动阈值算法中的某一个
- Threshold Color：选择阈值显示颜色
- Color Space：选择色彩空间，有HSB、RGB、CIE Lab或YUV
- Dark Background：当特征比背景要浅时，就勾选
- Orginal：恢复原先的图片
- Filtered：显示滤波后的图片，最终图片类型是RGB，不是8-bit灰度图
- Select：基于当前设定创建一个ROI选区，选区是根据Process-Binary-Options对话框中的设定定义的。
- Sample：基于用户自定义的ROI中的像素值设定滤波的范围
- Stack：使用当前的设定处理剩下的slices（如果有的话）
- Macro：基于当前设定创建一个宏
- Help：打开内置的help对话框

### Size
将当前图片或选区缩放到一个特定的以像素为单位的Width和Height。
缩放时可以设定是否保持长宽比，以及是否插值。
### Canvas Size
改变画布尺寸，而不缩放真实图片。如果画布尺寸增加了，边界用当前的背景色填充，如果选择了Zero Fill，那么边界用数值为0的像素填充。也可以指定原图片在新画布中的位置。
### Line Width
设置线宽，更简单的方式是双击Line Selection Tools的图标。

## Show Info
显示图片信息

## Properties
使用该命令显示和设置当前图片或stack的属性。
Channels、Slices、Frames的数目都可以更改，只要三者的成绩等于stack中图片的数目。
Unit of Length是一个字符串，用来表明下方的Pixel Width、Pixel Height和Voxel Depth的测量单位。这三个维度可以自动转换，如果单位在ImageJ已知的单位之间转换，这些单位有：$nm$、$\mu m$（或写成$um$和$micron$）、$mm$、$cm$、$meter$、$km$和$inch$等。
对于时间序列的stack，可以设定Frame Interval，即frame rate的倒数。如果单位是sec，这个设置也会同时设定Animation Options中所用的frame rate。
Origin是图片坐标系的参考点，该参考点的x和y坐标永远是像素为单位。
如果勾选Global，这里的设置将会施加到当前session打开的所有图像。

## Color
该菜单是处理彩色图片。
### Split Channels
将一张RGB图分割成三个8-bit的灰度图，分别是红绿蓝的三个通道。如果是复合图片或hyperstacks，该命令将分割这个stack成不同的channels。
### Merge Channels
把2-7张图片合并成RGB图片或多通道的复合图片。
如果勾选了Create composite，那么就会创建一个多通道的复合图片，如果不勾选，那么就会创建一个RGB图片。当创建复合图片时，原始的LUT和显示范围都会保留，除非勾选了下面的Ignore source LUTs。创建RGB时总是忽略原始的LUTs。
如果勾选了Keep source Images，源图片不会被清除。
### Channels Tool
等同于Image-Hyperstacks-Channels Tool。
### Stack to RGB
将一个含2个或3个slices的stack转化成RGB，假定slices是按R、G、B的顺序排列的。stack必须是8-bit或16-bit的灰度图。也可以将一个复合图片转成RGB。
### Make Composite
将RGB图、stack等转成复合图片。
### Show LUT
显示当前图片的LUT。
### Edit LUT
打开ImageJ的LUT编辑器。
### Color Picker
设定前景色和背景色。当前调色板是基于HSB，双击某个颜色可以设置RGB值。

## Stacks
该菜单包含与Stacks相关的命令。
### Add Slice
在当前slice之后插入一个空白slice，按住Alt则在当前slice之前插入。
### Delete Slice
删除当前slice。
### Next Slice
显示下一个slice
### Previous Slice
显示上一个的slice
### Set Slice
显示一个特定的slice
### Images to Stack
从当前在不同窗口显示的图片创建一个新的stack。
如果图片尺寸不同，那么可以选择转换的Method。Copy(center)和Copy(top-left)：将最宽的照片的宽度设为stack的宽度，将最高的照片的高度设为stack的高度。较小的图片将会复制到slice的中间center或左上角top-left。边界用数值为0的像素填充。Scale(smallest)和Scale(largest)：Stack将会选择最小或最大的图片的尺寸，其他的图片会被缩放到新的尺寸，如果勾选了Bicubic interpolation，就会使用双三次插件。
Title Contains：输入一个字符串，然后ImageJ将会仅仅转换包含该string的图片。
### Stack to Images
将当前stack的slices转成分开的图片窗口。
### Make Montage
创建拼贴集。
### Reslice
通过当前stack或hyperstack的图片体重新切片。
- Output spacing：输出间距，决定了重构的蒸饺的图片的数目，spacing越大，输出的stack的size越小
- Start at：决定图片的边缘，即重构从哪个地方开始
- Flip vertically：勾选后，输出的每个slice都是垂直翻转
- Rotate 90 degree：勾选后，每个slice都旋转90度
- Avoid interpolation：勾选后，不做插值

### Orthogonal Views
提供当前stack的正交视图，即如果原stack是XY视图，则该命令提供YZ和XZ视图。
### Z Project
将stack沿着垂直于图片的轴，即Z轴，进行投影。
### 3D Project
可以很自由地对stack进行各个方向的投影。
### Plot Z-Axis Profile
将ROI选区的平均灰度值对slice进行作图。该命令需要一个点选区或线选区。
### Label
对stack添加一系列数字（比如时间戳）和/或标签。数字和标签使用当前前景色绘制。
标签的初始X和Y坐标及字体尺寸等基于当前的矩形选区（如果有的话）。
- Format：指定标签的结构。0：普通序列；0000：用前导的0填充数字；00:00：将标签转为minutes:seconds这样的时间戳；00:00:00：将标签转为hours:minutes:seconds这样的时间戳；Text：仅包含下面的Text输入框中的内容；Label：显示slice的标签。
- Starting value and Interval：指定第一个数值和间隔。注意，对于时间戳，必须使用公制时间间隔，比如Interval为3600时将创建1 hour的间隔
- Text：字符串
- Use overlay：勾选后，创建的标签就作为无损的Overlay，之前添加的overlay将会被删除。
- Use text tool font：勾选后，标签将使用Fonts部件中指定的风格

### Tools
- Combine：将两个stack组合，创建一个新的stack
- Concatenate：将多个图片或stack连接起来，类型和尺寸不符的图片将被忽略。
- Reduce：按照指定的Reduction Factor减少stack的尺寸。
- Reverse：与Image-Transfrom-Flip Z命令相同
- Insert：在指定的位置在目标图片上插入一张源图片。目标图片和源图片可以是单一图片或stacks，但必须相同类型，且目标图片一旦被插入后就被永久修改。如果源图片是单一图片，一种更简单的组合两个图片的方法是：通过Edit-Selection-Image to Selection创建图片ROI，然后Image-Overlay-Add Image
- Montage to Stack：将一个拼贴集转为一个stack，这与上面的创建拼贴集是相反操作
- Make Substack：从当前stack中提取一些图片成为新的stack。
- Grouped Z Project：创建Z轴投影的多个结果
- Remove Slice Labels：从stack中去除slice标签

### Animation
- Start Animation：重复按次序显示该stack的slices。
- Stop Animation：停止动画播放
- Animation Options：设置每秒多少帧，即动画速率。

## Hyperstacks
这个菜单针对于Hyperstacks，即4D或5D的图片。
### New Hyperstack
创建一个新的hyperstack，属性主要有Width(w)、Height(h)、Channels(c)、Slices(z)、Frames(t)。
### Stack to Hyperstack
将stack转化为hyperstack。RGB的stack将转为3个通道的hyperstack。Order就是channels、slices和frames的次序。ImageJ的hyperstack总是czt次序，不是czt顺序的stack将被重新排序为czt。
### Hyperstack to Stack
将hyperstack转为stack。
### Reduce Dimensionality
该命令通过创建一个新的hyperstack而将原hyperstack降维，比如抽取给定z坐标的所有的channels和时间点，或者抽取在当前channel和时间点的所有的z的slices。
不勾选channels将会删除所有的channels、但保留当前channel，不勾选Slices将仅保留当前的slice，不勾选Frames仅保留当前时间点。
### Channels Tool
打开Channels部件。

## Crop
基于当前的矩形选区来裁剪图片或stack。

## Duplicate
创建一个新的窗口，包含当前图片或矩形选区的副本。对于stack和hyperstack，可以指定channels、slices和Frames的复制范围。

## Rename
重命名当前图片。

## Scale
通过对话框中的缩放因子来调整当前图片或选区的大小，可以选择两种重采样方法：双线性或双三次插值。
为了更好的显示效果，对于图片和文字，使用整数缩放因子，如果该因子小于1，则勾选Average when downsizing。
如果勾选了Create New window，则缩放的图片或选区可以复制到一个新的图片；如果缩放一个选区，且不复制到新图片，则勾选Fill with Background Color将提供背景色，而不是填充0。勾选Process entire stack后将缩放整个stack。

## Transform
该菜单包含常用的几何图形变换的命令。
### Flip Horizontally
水平翻转
### Flip Vertically
垂直翻转
### Flip Z
将stack中的slice的顺序翻转
### Rotate 90 Degrees Right
顺时针90度旋转
### Rotate 90 Degrees Left
逆时针90度旋转
### Rotate
旋转特定角度。
- Grid Lines：可以用预览模式在图片上加上网格线
- Interpolation：可选择双线性或双三次的重采样方法
- Fill with Background Color：对于8-bit或RGB图片，勾选此项后会填充当前背景色，而不是0
- Enlarge to Fit Result：勾选后，图片将会被避免裁剪

### Translate
平移特定的像素值。对于stacks，可以平移当前图片或所有图片。勾选Preview可以预览效果。图片边缘的背景将被设为0。
### Bin
通过指定X、Y、Z方向的收缩因子，来减小图片的尺寸。最终的像素可以通过Average、Median、Maximum或Minimum等方法计算。Undo撤销操作仅对二维图片有效，即对stack无效。
Z方向的操作与Image-Stacks-Tools-Grouped Z Project效果相同。然而，有两个主要的不同点：Bin替代了当前图片，Grouped Z Project则创建了一个新的substack；Bin中的Z shrink factor可以填入任意值，而Group size必须能够stack尺寸所整除。
### Image to Results
将当前选区打印到Resutls Table中，同时清除之前的结果。如果没有ROI，则处理整个图片。表格中详细显示了XY坐标及其像素值。
对于RGB图片，每个像素通过平均或加权平均算法转化为灰度值。
### Results to Image
是上面操作的逆操作，将Results Table中的表格数据转化为32-bit图片。

## Zoom
该菜单控制怎样显示图片。对于下面的In和Out命令，更提倡使用+、-或上下箭头。如果有选区时，使用上下箭头时需要按住Shift或者Ctrl。
### In
有21种可能的放大级别。放大时，如果箭头在画布中，那么将会围绕箭头放大，如果箭头不在画布中，将会围绕图片的中心扩大。左上角的Zoom Indicator表明了当前显示的是图片的哪一部分。当放大到一定级别后，默认就会显示像素的格点，除非勾选Edit-Options-Appearance中的Interpolate zoomed images。当需要滚动放大的图片时，在拖拽鼠标的同时按住空格键。
默认Overlays和选区是按一个像素的宽度来显示，如果想要在较高放大级别下加粗ROI边缘，将Edit-Selection-Properties中的Stroke width设为非零。
### Out
缩小放大层级。
### Original Scale
显示最初打开时的尺寸。快捷键是双击“放大镜”工具的图标
### View 100%
使用100%放大，即1个图片像素等于1个屏幕像素。将Edit-Options-Appearance中的Enable Open Images at 100%勾选后，即可设置图片在打开时就是100%显示。
### To Selection
基于当前的选区进行缩放。如果没有选区的话，就会使得图片缩放到fit to screen级别。
### Set
手动设定精确值供缩放，也可以同时设定缩放的中心点的坐标。

## Overlay
该菜单用于设置对图片无损的Overlay。Overlay包含一个或多个选区：箭头、线段、点、各种形状和文本等，也可以包含图片选区，即ImageROI。
### Add Selection
该命令用于将选区立即加入当前的Overlay，快捷键是B。按住Alt+B将会显示一个对话框供设置Stroke Color、Width和Fill color。除了文本选区，Stroke color和width这两个与Fill color是不共存的。
如果勾选了New overlay，那么之前添加的Overlay将被删除。
如果在Analyze-Set Measurements中勾选了Add to overlay，那么要测量的选区(Analyze-Measure)将会自动添加到Overlay。
### Add Image
通过将一张图片添加到另一张图片的overlay而实现组合图片的效果。要组合的图片可以是任意类型，但不能比主图大。组合时可以设置透明度，初始的XY坐标是基于当前矩形选区。
默认情形下，创建的新图片不能随意在画布上移动，即不是一个图片选区ImageROI，它存在TIFF的header中。如果想得到一个图片选区，可以通过Edit-Selection-Image to Selection或者Image-Overlay-To ROI Manager。
### Hide Overlay
隐藏Overlay
### Show Overlay
显示Overlay
### From ROI Manager
从ROI管理器中的选区创建一个overlay，注意之前添加的overlay将被删除。
### To ROI Manager
把当前Overlay中的选区复制到ROI管理器，这样就可以对其进行编辑。注意，ROI管理器中的之前项目会被删除。
### Remove Overlay
永久清除overlay，使其不可被恢复
### Flatten
创建一个新的RGB图片，其中的overlay被渲染成图片数据，该RGB图片与原图片的尺寸相同，这跟Plugins-Utilities-Capture Image不同，后者是创建一个“所见即所得”的与当前窗口尺寸相同的图片。
### Labels
定义怎样对overlay打标签。比如定义颜色、标签字体、标题、背景等。
### Overlay Options
定义默认的overlay的Stroke color、width和Fill Color。将Stroke width设为0，则选区的边缘的宽度就是1个像素，不管放大多少倍。

## LookUp Table
该菜单包含选择哪种色彩查询表用来将灰度图创建成伪彩色图。
### Invert LUT
反转当前的LUT。对于8-bit图片，表中的每一个值v都被255-v所替代。与Edit-Invert不同的是，像素值没有被改变，只是在屏幕上显示的方式改变了。
### Apply LUT
将当前的LUT施加到图片或选区的像素值上。该命令等价于Image-Adjust-Brightness/Contrast的Apply操作。对于阈值处理过的图片，等价于Image-Adjust-Threshold的Apply操作。
