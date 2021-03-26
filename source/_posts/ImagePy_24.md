---
title: ImagePy解析：24 -- 骨架图转图论sknw解析
tags: [ImagePy]
categories: computer vision 
date: 2020-11-20
---

sknw是一个从骨架图中创建图网络的库，代码在[这里](https://github.com/Image-Py/sknw)。

它不仅可以实现将单线转变成图graph的效果，而且里面的trace函数还可以实现像素追踪，将图像中的单线的坐标序列依次提取出来，从而将图像转变为矢量图。（sknw可以对闭合曲线进行坐标提取，对于闭合曲线，也可以使用find_contour来提取这些坐标序列）

# 输入图像
输入图像必须是一个二值的骨架图。
比如，这里的示例图像矩阵为：
```python
    img = np.array([
        [0,0,0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0,1,0],
        [0,0,0,1,0,0,0,0,0],
        [1,1,1,1,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0],
        [0,1,0,0,0,1,0,0,0],
        [1,0,1,0,0,1,1,1,1],
        [0,1,0,0,1,0,0,0,0],
        [0,0,0,1,0,0,0,0,0]])
```

# 标识节点
```python
node_img = mark_node(img)

def mark_node(ske):
    buf = np.pad(ske, (1,1), mode='constant')
    nbs = neighbors(buf.shape)
    acc = np.cumprod((1,)+buf.shape[::-1][:-1])[::-1]
    mark(buf, nbs)
    return buf

def mark(img, nbs): # mark the array use (0, 1, 2)
    img = img.ravel()
    for p in range(len(img)):
        if img[p]==0:continue
        s = 0
        for dp in nbs:
            if img[p+dp]!=0:s+=1
        if s==2:img[p]=1
        else:img[p]=2
```
这一步是将上面的骨架图中的特有节点标识出来（注意：新形成的图是在原图周围附加了一圈0作为缓冲）：
（1）像素值原来为0的地方，仍然为0；
（2）如果某非0像素，其八邻域有2个非0像素，那么标识该像素为1；这种像素代表了骨架图中的中间段的像素（其中有一部分1代表了环形闭合结构，在后面会特殊处理）；
（3）如果某非0像素，其八邻域中的非0像素个数不是2（比如是1、3等），那么标识该像素为2，这种像素代表了骨架图中端点和交点部分的像素。

经过标识后，得到的图像矩阵为：
```python
[[0 0 0 0 0 0 0 0 0 0 0]
[0 0 0 0 2 0 0 0 0 0 0]
[0 0 0 0 1 0 0 0 2 0 0]
[0 0 0 0 2 0 0 0 0 0 0]
[0 2 1 2 2 0 0 0 0 0 0]
[0 0 0 0 0 1 0 0 0 0 0]
[0 0 1 0 0 0 2 0 0 0 0]
[0 1 0 1 0 0 2 2 1 2 0]
[0 0 1 0 0 1 0 0 0 0 0]
[0 0 0 0 2 0 0 0 0 0 0]
[0 0 0 0 0 0 0 0 0 0 0]]
```

# 泛洪填充
```python
    img = img.ravel()
    buf = np.zeros(131072, dtype=np.int64)
    num = 10
    nodes = []
    for p in range(len(img)):
        if img[p] == 2:
            isiso, nds = fill(img, p, num, nbs, acc, buf)
            if isiso and not iso: continue
            num += 1
            nodes.append(nds)
```
依次观察值为2的像素，对其进行等值填充（主要函数就是fill函数），并记录这些节点在原图中的坐标位置。
fill函数为：
```python
def fill(img, p, num, nbs, acc, buf):
    img[p] = num
    buf[0] = p
    cur = 0; s = 1; iso = True;

    while True:
        p = buf[cur]
        for dp in nbs:
            cp = p+dp
            if img[cp]==2:
                img[cp] = num
                buf[s] = cp
                s+=1
            if img[cp]==1: iso=False
        cur += 1
        if cur==s:break
    return iso, idx2rc(buf[:s], acc)
```
原理就是探究这些值为2的像素的邻居是否仍然是2，若是，则将其亦纳入探究范围，这样就标识出了这些节点。
经过上述代码后，得到的nodes数值为：
```python
[array([[0, 3]], dtype=int16), array([[1, 7]], dtype=int16), array([[2, 3],
       [3, 2],
       [3, 3]], dtype=int16), array([[3, 0]], dtype=int16), array([[5, 5],
       [6, 5],
       [6, 6]], dtype=int16), array([[6, 8]], dtype=int16), array([[8, 3]], dtype=int16)]
```
如何理解呢？很好理解，比如第一个坐标[0,3]就是第一个值为2的像素在原图中的位置，而坐标组合([[2, 3], [3, 2], [3, 3]])代表那三个相邻的值为2的像素。
同时img中的像素值也发生了变化，比如第一个值为2的像素，它的值由2变成了10（如上代码硬编码），而第一个值为2的像素则变成了11，同理，那三个相邻的值为2的像素变成了12，依次类推，最后一个值为2的像素变成了16。

# 像素追溯
```python
    edges = []
    for p in range(len(img)):
        if img[p] <10: continue
        for dp in nbs:
            if img[p+dp]==1:
                edge = trace(img, p+dp, nbs, acc, buf)
                edges.append(edge)
```
像素追溯部分的观察点就变成了与上述节点相邻且值为1的那些像素，即骨架图中的中间部分的像素（原理就是通过是否大于10而选择过滤出上面那些节点，然而通过其邻居是否是1来过滤得到这些值为1的像素），然后通过trace函数寻找其两端所连接的具体节点标识。

trace函数为：
```python
def trace(img, p, nbs, acc, buf):
    c1 = 0; c2 = 0;
    newp = 0
    cur = 1
    while True:
        buf[cur] = p
        img[p] = 0
        cur += 1
        for dp in nbs:
            cp = p + dp
            if img[cp] >= 10:
                if c1==0:
                    c1 = img[cp]
                    buf[0] = cp
                else:
                    c2 = img[cp]
                    buf[cur] = cp
            if img[cp] == 1:
                newp = cp
        p = newp
        if c2!=0:break
    return (c1-10, c2-10, idx2rc(buf[:cur+1], acc))
```
trace的原理就是：观察这些值为1的像素的邻居，若为大于10，即找到了那些节点nodes，分别通过c1和c2来存储两侧的节点；若为1，则也将其设为进一步的观察点。
经过上述代码后，得到的edges的数值为：
```python
[(0, 2, array([[0, 3],
       [1, 3],
       [2, 3]], dtype=int16)), (3, 2, array([[3, 0],
       [3, 1],
       [3, 2]], dtype=int16)), (2, 4, array([[3, 3],
       [4, 4],
       [5, 5]], dtype=int16)), (4, 6, array([[6, 5],
       [7, 4],
       [8, 3]], dtype=int16)), (4, 5, array([[6, 6],
       [6, 7],
       [6, 8]], dtype=int16))]
```
如何理解呢？也很好理解。比如第一个元组((0, 2, array([[0, 3], [1, 3], [2, 3]])，前两个元素0和2分别是img中的值为10和12的像素减去10所得，第三个元素就是第一个值为1的元素的坐标，以及它所连接的两个值为2的节点的坐标。其他元组也是这个意思。通过这样的元组表示，就很自然地为后面的图graph中的首尾节点及中间连接做了准备。

同时img中的像素值又发生了变化：与上面值为2的节点相连接的值为1的像素的值都变为了0（见trace函数中的硬编码），这是为了接下来的闭合曲线的处理。

## 闭合曲线的处理
上面的代码可以处理非闭合的曲线，因为很容易根据八邻域中的节点数目来获得交点部分所在。而对于闭合曲线，比如原图中左下方的四个1所形成的闭合曲线，其中无法找到值为2的这种像素，且无法对应到图graph中的节点的概念，因此需要特殊处理一下（这里是否处理这种闭合曲线，是用ring这个参数来指定）：
```python
    for p in range(len(img)):
        if img[p]!=1: continue
        img[p] = num; num += 1
        nodes.append(idx2rc([p], acc))
        for dp in nbs:
            if img[p+dp]==1:
                edge = trace(img, p+dp, nbs, acc, buf)
                edges.append(edge)
```
注意，因为上面已经将值为1且与2相连的像素置为0，所以这里寻找的是剩下的值为1的像素。若为1，然后会将它继续添加到节点nodes中。
然后再对其邻居点进行追溯trace，不断地将邻居的为1的像素加入到edge中，最终得到的edge结果为：
```python
(7, 7, array([[5, 1],
       [6, 0],
       [7, 1],
       [6, 2],
       [5, 1]], dtype=int16))
```
代表由7号节点到7号节点的一个循环。

# 创建图graph
经过上述节点标识和像素追溯，可以得到节点及其边为：
```python
nodes =
[array([[0, 3]], dtype=int16), array([[1, 7]], dtype=int16), array([[2, 3],
       [3, 2],
       [3, 3]], dtype=int16), array([[3, 0]], dtype=int16), array([[5, 5],
       [6, 5],
       [6, 6]], dtype=int16), array([[6, 8]], dtype=int16), array([[8, 3]], dtype=int16), array([[5, 1]], dtype=int16)]
 
edges =
[(0, 2, array([[0, 3],
       [1, 3],
       [2, 3]], dtype=int16)), (3, 2, array([[3, 0],
       [3, 1],
       [3, 2]], dtype=int16)), (2, 4, array([[3, 3],
       [4, 4],
       [5, 5]], dtype=int16)), (4, 6, array([[6, 5],
       [7, 4],
       [8, 3]], dtype=int16)), (4, 5, array([[6, 6],
       [6, 7],
       [6, 8]], dtype=int16)), (7, 7, array([[5, 1],
       [6, 0],
       [7, 1],
       [6, 2],
       [5, 1]], dtype=int16))]
```
使用networkx库来构建graph：
```python
import networkx as nx
def build_graph(nodes, edges, multi=False):
    graph = nx.MultiGraph() if multi else nx.Graph()
    for i in range(len(nodes)):
        graph.add_node(i, pts=nodes[i], o=nodes[i].mean(axis=0))
    for s,e,pts in edges:
        l = np.linalg.norm(pts[1:]-pts[:-1], axis=1).sum()
        graph.add_edge(s,e, pts=pts, weight=l)
    return graph
```
