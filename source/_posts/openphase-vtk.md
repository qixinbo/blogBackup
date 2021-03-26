---
title: OpenPhase实例学习系列：VTK文件格式详解
date: 2016-2-22
tags: [openphase]
categories: simulation 
---
# VTK文件格式概况
VTK，全称为Visualization Toolkit，即“可视化工具箱”，制订了一个统一的文件输入输出格式，这样就可在不同的软件间方便通信。
VTK文件格式包括五个基本部分：
(1)第一部分是文件版本说明：
\# vtk DataFile Version 3.0 
(2)第二部分是文件头，是一个由"\n"结尾的字符串，最大为256个字符，如：
PhaseField                 
(3)第三部分说明文件的格式，是ASCII或BINARY，两者必填其一。
(4)第四部分是数据集的结构。
该部分起始是关键词DATASET加上数据集的类型，数据集的类型，即几何/拓扑的类型，包括()：STRUCTURED_POINTS, STURCTURED_GRID, UNSTRUCTURED_GRID, POLYDATA, RECTILINEAR_GRID, 然后基于该类型，再确定具体性质。
(5)第五部分说明数据的属性，即数据值或"场"。
以关键词POINT_DATA或CELL_DATA开头，加上一个整型数值确定点或胞的数目。然后是具体的属性值，如标量、矢量、张量等。

# 一些注意事项：
(1)所有的关键词项都是用ASCII书写的。对于二进制数据，只有定义点的坐标、数值时才用二进制形式。
(2)指标是从0开始的，因此第一个点的id是0。
(3)如果数据的属性和几何/拓扑部分都存在，那么两者需要精确吻合。
(4)胞cell的类型和指标是INT型。
(5)二进制数据必须紧随前面的ASCII字符后的"\n"换行符后。
(6)几何/拓扑部分必须在属性之前设定。
(7)数据类型dataType包括bit, unsigned_char, char, unsigned_short, short, unsigned_int, int, unsigned_long, long, float和double。

# 具体的数据集格式举例： 
## Structured Points： 
DATASET STRUCTURED_POINTS 
DIMENSIONS nx ny nz       
ORIGIN x y z              
SPACING sx sy sz          
## Structured Grid:    
DATASET STRUCTURED_GRID   
DIMENSIONS nx ny nz       
POINTS n dataType         
p0x p0y p0z               
p1x p1y p1z               
...                       
p(n-1)x p(n-1)y p(n-1)z   
## Rectilinear Grid    
DATASET RECTILINEAR_GRID  
DIMENSIONS nx ny nz       
X_COORDINATES nx dataType 
x0 x1 ... x(nx-1)         
Y_COORDINATES ny dataType 
y0 y1 ... y(ny-1)         
Z_COORDINATES nz dataType 
z0 z1 ... z(nz-1)         

# 具体的属性/场格式举例：
VTK可以存储标量、矢量、张量等，每个属性有一个名称dataName，用来提取该属性。因此，在同一套数据集上，可以有多个属性，如压力、温度等。
## Scalars
SCALARS dataName dataType numComp 
LOOKUP_TABLE tableName            
s0                                
s1                               
...                               
s(n-1)                            
## Vectors 
VECTORS dataName dataType         
v0x v0y v0z                      
v1x v1y v1z                     
...                               
v(n-1)x v(n-1)y v(n-1)z          
## Normals 
NORMALS dataName dataType         
n0x n0y n0z                       
n1x n1y n1z                       
...                               
n(n-1)x n(n-1)y n(n-1)z           
