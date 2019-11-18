---
title: ImagePy解析：14 -- 寻找局部极值（Find Maximum和Find Minimum）
tags: [ImagePy]
categories: computational material science 
date: 2019-11-17
---

源码在：[这里](https://github.com/Image-Py/imagepy/blob/master/imagepy/ipyalg/hydrology/findmax.py)
还有一篇参考文献：[局部极值提取算法](https://www.cnblogs.com/wjy-lulu/p/7416135.html)

# 图像准备
首先创建一张10 pixels乘以10 pixels的背底黑色、中间白色的图像，如下图所示（下图仅是为了显示，实际图像是100个像素的面积大小，这样是为了后面显示像素矩阵时更方便）：
![](https://user-images.githubusercontent.com/6218739/69029452-e91cfe00-0a0f-11ea-9e7f-b13dec002fc2.png)
然后上面这张图像不能直接传入findmax脚本中，实际用到的是它的距离变换图（注意将这张图的显示范围调整为0-3，因为面积很小，所以距离很近，如果正常0-255显示则就是一片黑色）：
![](https://user-images.githubusercontent.com/6218739/69029482-04880900-0a10-11ea-80a9-496a538fabd5.png)
这张图像的像素矩阵就是：
```python
[[0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 1 1 1 0 0 0]
 [0 0 0 1 1 2 1 1 0 0]
 [0 0 1 1 2 3 2 1 1 0]
 [0 0 1 1 2 3 2 1 1 0]
 [0 0 0 1 1 2 1 1 0 0]
 [0 0 0 0 1 1 1 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]]
```

然后基于这张图像创建一个掩膜和缓冲区：
创建掩膜：
```python
msk = np.zeros_like(img, dtype=np.uint8)
msk[tuple([slice(1,-1)]*img.ndim)] = 1
```
该掩膜的矩阵为：
```python
[[0 0 0 0 0 0 0 0 0 0]
 [0 1 1 1 1 1 1 1 1 0]
 [0 1 1 1 1 1 1 1 1 0]
 [0 1 1 1 1 1 1 1 1 0]
 [0 1 1 1 1 1 1 1 1 0]
 [0 1 1 1 1 1 1 1 1 0]
 [0 1 1 1 1 1 1 1 1 0]
 [0 1 1 1 1 1 1 1 1 0]
 [0 1 1 1 1 1 1 1 1 0]
 [0 0 0 0 0 0 0 0 0 0]]
```
可以看出来，实际就是将边界一圈都置为0，内部区域置为1。
创建缓冲区：
```python
buf = np.zeros(img.size//3, dtype=np.int64)
```
该缓冲区为：
```python
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
```
可以看出，是33个0，即对图像尺寸对3进行向下取整的除法。这里的缓冲区是为了存放泛洪填充时的中心像素的位置，因此，很有可能泛洪漫延时超过33个位置，不过也没关系，程序中已经做了，如果超过33个，就会及时地将后面的数据挪到前面，保证不溢出。如这两部分处的代码：
```python
if s==len(buf):
    buf[:s-cur] = buf[cur:]
    s-=cur; cur=0;
```
和：
```python
if s==msk.size//3:
    cut = cur//2
    msk[bur[:cut]] = 2
    bur[:s-cut] = bur[cut:]
    cur -= cut
    s -= cut
```

# 获得邻居像素的位置距离
neighbors()方法是为了在以某个像素为中心时，获得它周围的8个像素相对它的位置距离。
源码为：
```python
def neighbors(shape):
    dim = len(shape)
    block = np.ones([3]*dim)
    block[tuple([1]*dim)] = 0
    idx = np.where(block>0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx-[1]*dim)
    acc = np.cumprod((1,)+shape[::-1][:-1])
    return np.dot(idx, acc[::-1])
```
比如，这里的示例图像是10乘以10的大小，那么经过neighbour方法后，得到的邻居像素的位置距离就是：
```python
nbs = neighbors(img.shape)
```
得到：
```python
nbs = array([-11, -10,  -9,  -1,   1,   9,  10,  11])
```
后面对img进行Ravel()的扁平化处理后，就可以根据相对距离得到中心像素的8个邻居。

# 以某个像素点为起点进行等值填充
fill()方法是实现针对于某一像素点，如果其周围的八个邻居与其像素值相等，则填充为相同的背景。
```python
def fill(img, msk, p, nbs, buf):
    buf[0] = p
    back = img[p]
    cur = 0; s = 1;

    while cur<s:
        p = buf[cur]
        for dp in nbs:
            cp = p+dp
            if img[cp]==back and msk[cp]==1:
                msk[cp] = 3
                buf[s] = cp
                s+=1
                if s==len(buf):
                    buf[:s-cur] = buf[cur:]
                    s-=cur; cur=0;
        cur += 1
```
注意理解该算法所接收的参数：
img是原图像，msk是掩膜（初始为边界一圈为0，内部为1），p是填充时所选的起始像素（一切都是该像素引发的，后面的idx中存储的就是这种引起填充行为的像素的位置），nbs就是上面所说的邻居像素的相对距离，buf存储了填充时的中心像素位置（再说一遍，起始像素引起了这次填充，它周围的像素则可以作为继续填充时的中心像素，以此类推）。
具体算法为：
（1）将起始像素点的位置记入buffer的第一个元素；
（2）将起始像素点的像素值另存为back变量，后面的像素都是与它做比较；
（3）创建一个游标cur和不断变化的可以作为填充的中心像素的计数变量s；
（4）对于起始像素，判断它周围的8个像素，如果满足两个条件：一是邻居像素的像素值与它的像素值相等，二是邻居像素不在边界上及未被填充过（这里再说明一下，边界点的掩膜值为0，填充过的掩膜值为3，未填充的掩膜值为1，后面还会有一个值为2，是代表该像素是否被某一局部极值占有），那么就把该邻居像素的掩膜值设为3，同时把它的位置存入buf中，同时把可作为填充的中心像素数目加1；
（5）后面再以邻居像素为中心继续填充，直到不满足上面的两个条件。

前面也说了，如果buf的长度不够，会及时把它后面的数据挪到前面来。

# 对整张图像进行标记
mark()方法就是对整张图像进行填充标记：
```python
def mark(img, msk, buf, mode):
    nbs = neighbors(img.shape)
    idx = np.zeros(msk.size//3, dtype=np.int64)
    img = img.ravel()
    msk = msk.ravel()
    s = 0
    for p in range(len(img)):
        if msk[p]!=1:continue
        sta = 0
        for dp in nbs:
            if img[p+dp]==img[p]:sta+=1
            if mode and img[p+dp]>img[p]:
                sta = 100
                break
            elif not mode and img[p+dp]<img[p]:
                sta = 100
                break
        if sta==100:continue
        msk[p] = 3
        if sta>0:
            fill(img, msk, p, nbs, buf)

        idx[s] = p
        s += 1
        if s==len(idx):break
    return idx[:s].copy()
```
需要明确的是该方法所接收的mode，如果为True，则是寻找局部极大值，如果为False，则是寻找局部极小值。
它首先是要对图像中的所有像素点进行循环：
（1）如果该点在边界上（掩膜值为0）或之前已经被标记过（掩膜值为3），那么就会跳过；
（2）如果没有被标记（掩膜值为1），那么就会对它周围的八个邻居像素进行循环：
（2.1）如果邻居像素的像素值与该中心像素的像素值相同，则将状态加1；
（2.2）如果是想寻找局部极大值（mode为True），且有个邻居像素大于该中心像素的像素值，则直接将状态置为100，并跳出循环；
（2.3）如果是想寻找局部极小值（mode为False），且有个邻居像素小于该中心像素的像素值，则直接将状态置为100，并跳出循环；
然后就判断状态值，如果为100，那么就不以该像素为起点进行填充，反之，则将该像素的掩膜值置为3，然后如果状态>0，那么就以它为起点进行填充，同时记录下该起点像素的位置，填入idx中进行保存。

举一个例子，因为边界上的像素都不会被标记，因为第一个被标记的像素是编号为11的像素（即除了边界外的左上角的那个像素，此时的编号是基于img已经被Ravel()压平了），以它为起始点进行填充后，得到的掩膜矩阵为：
```python
[[0 0 0 0 0 0 0 0 0 0]
 [0 3 3 3 3 3 3 3 3 0]
 [0 3 3 3 1 1 1 3 3 0]
 [0 3 3 1 1 1 1 1 3 0]
 [0 3 1 1 1 1 1 1 1 0]
 [0 3 1 1 1 1 1 1 1 0]
 [0 3 3 1 1 1 1 1 3 0]
 [0 3 3 3 1 1 1 3 3 0]
 [0 3 3 3 3 3 3 3 3 0]
 [0 0 0 0 0 0 0 0 0 0]]
```
然后，第二个可作为填充的起始像素是编号为42的像素，经过再一次填充后，掩膜变为：
```python
[[0 0 0 0 0 0 0 0 0 0]
 [0 3 3 3 3 3 3 3 3 0]
 [0 3 3 3 3 3 3 3 3 0]
 [0 3 3 3 3 1 3 3 3 0]
 [0 3 3 3 1 1 1 3 3 0]
 [0 3 3 3 1 1 1 3 3 0]
 [0 3 3 3 3 1 3 3 3 0]
 [0 3 3 3 3 3 3 3 3 0]
 [0 3 3 3 3 3 3 3 3 0]
 [0 0 0 0 0 0 0 0 0 0]]
```
第三次可作为填充的起始像素的编号为45，经过填充后，掩膜变为：
```python
[[0 0 0 0 0 0 0 0 0 0]
 [0 3 3 3 3 3 3 3 3 0]
 [0 3 3 3 3 3 3 3 3 0]
 [0 3 3 3 3 1 3 3 3 0]
 [0 3 3 3 1 3 1 3 3 0]
 [0 3 3 3 1 3 1 3 3 0]
 [0 3 3 3 3 1 3 3 3 0]
 [0 3 3 3 3 3 3 3 3 0]
 [0 3 3 3 3 3 3 3 3 0]
 [0 0 0 0 0 0 0 0 0 0]]
```
可以看出，像素值为2的区域没有被填充，就是因为以这些像素进行填充时，会碰到像素值为3的像素，导致sta被置为100，无法执行填充操作。

经过后面对idx中的元素截取，得到mark()方法后的返回值为：
```python
[11, 42, 45]
```

# 寻找局部极值
filter()方法就是用来实现局部极值的寻找：
```python
def filter(img, msk, idx, bur, tor, mode):
    nbs = neighbors(img.shape)
    acc = np.cumprod((1,)+img.shape[::-1][:-1])[::-1]
    img = img.ravel()
    msk = msk.ravel()

    arg = np.argsort(img[idx])[::-1 if mode else 1]
   
    for i in arg:
        if msk[idx[i]]!=3:
            idx[i] = 0
            continue
        cur = 0; s = 1;
        bur[0] = idx[i]
        while cur<s:
            p = bur[cur]
            if msk[p] == 2:
                idx[i]=0
                break

            for dp in nbs:
                cp = p+dp
                if msk[cp]==0 or cp==idx[i] or msk[cp] == 4: continue
                if mode and img[cp] < img[idx[i]]-tor: continue
                if not mode and img[cp] > img[idx[i]]+tor: continue
                bur[s] = cp
                s += 1
                if s==msk.size//3:
                    cut = cur//2
                    msk[bur[:cut]] = 2
                    bur[:s-cut] = bur[cut:]
                    cur -= cut
                    s -= cut

                if msk[cp]!=2:msk[cp] = 4
            cur += 1
        msk[bur[:s]] = 2
    return idx2rc(idx[idx>0], acc)
```
主要步骤是：
（1）首先对找到的极值按数值大小进行排序：
```python
arg = np.argsort(img[idx])[::-1 if mode else 1]
```
如果是寻找极大值，那么就是按从大到小来排序。比如在这里，idx中存储的位置对应的像素的数值分别是0、1和3，所以，arg就是：
```python
arg = Array([2, 1, 0], dtype=int64)
```
（2）按顺序依次寻找极值的领地
（2.1）首先还是以idx存储的像素为起始点，对它的8个邻居像素进行遍历，如果这些邻居像素在边界上（掩膜值为0）或被临时占有（掩膜值为4），那么就跳过；
（2.2）如果是寻找极大值（mode为True）以及邻居像素要小于该idx存储的起点像素的像素值减去容忍度tolerance，那么也跳过（那么这里，容忍度就是非常重要的一个参数，它是由用户输入，代表这个极值的“势力范围”或“领地”，如果容忍度为0，那么就严格地认为只有拥有该数值的像素才属于该极值，直观理解看，就是容忍度决定了势力范围，如果tolerance较大，说明划定的势力范围较大，那么两个局部极值就可以碰到，导致合并成一个极值，所以tolerance越大，极值点越少）；
（2.3）同理，如果是寻找极小值，邻居像素如果大于该起点像素的像素值加上容忍度，那么也跳过；
（2.4）如果邻居像素属于该极值的势力范围（即数值满足起点像素值和tolerance），那么就将它的位置存入buf中，后续还要以它为中心进行循环，同时如果它的掩膜值不为2（即之前没有被占有过），则将它的掩膜值置为4，即临时占有；
（2.5）在对这些中心像素都遍历一遍后，就将它们的掩膜值置为2，即它们被该idx存储的像素所永久占有。
如果将tolerance设为1，下面是位置编号为45的像素所占有的领地（掩膜矩阵）：
```python
[[0 0 0 0 0 0 0 0 0 0]
 [0 3 3 3 3 3 3 3 3 0]
 [0 3 3 3 3 3 3 3 3 0]
 [0 3 3 3 3 2 3 3 3 0]
 [0 3 3 3 2 2 2 3 3 0]
 [0 3 3 3 2 2 2 3 3 0]
 [0 3 3 3 3 2 3 3 3 0]
 [0 3 3 3 3 3 3 3 3 0]
 [0 3 3 3 3 3 3 3 3 0]
 [0 0 0 0 0 0 0 0 0 0]]
```
那么在对下一个idx存储的像素为起点进行探索时，也会执行上面的步骤，但此时可以在一开始或中途触发某个掩膜值为2的邻居像素，那么此时就会终止该次循环，并且将idx中存储的这个像素的位置置为0，即下面的代码：
```python
while cur<s:
    p = bur[cur]
    if msk[p] == 2:
        idx[i]=0
        break
```
这个地方非常重要，这代表着两个极值所涉及的领地是否有冲突，如果冲突了，那么说明这两个极值实际是接触的，因为前面的循环是基于更大的极值（以寻找极大值来说），那么这个第二次的极值就会被刚才的极值占领，它的位置也就可以在idx中被抹去。
比如位置编号为42的像素开始占有后：
```python
[[0 0 0 0 0 0 0 0 0 0]
 [0 2 2 2 2 2 3 3 3 0]
 [0 2 2 2 2 2 3 3 3 0]
 [0 2 2 2 2 2 3 3 3 0]
 [0 2 2 2 2 2 2 3 3 0]
 [0 2 2 2 2 2 2 3 3 0]
 [0 2 2 2 2 2 3 3 3 0]
 [0 3 3 3 3 3 3 3 3 0]
 [0 3 3 3 3 3 3 3 3 0]
 [0 0 0 0 0 0 0 0 0 0]]
```
就是因为在不断地寻找过程中，某个邻居像素碰到了之前已经被标记为2的像素，导致提前终止，使得idx变成了：
```python
idx = array([11,  0, 45], dtype=int64)
```
而接着在对idx中的位置为11的像素进行探索时，发现它一上来掩膜值就已经是2，所以直接就被终止，idx变成：
```python
idx = array([0,  0, 45], dtype=int64)
```
那么下面一步就是挑选不为0的位置编号，并且将它转换为行号和列号，即：
```python
idx2rc(idx[idx>0], acc)
 
def idx2rc(idx, acc):
    rst = np.zeros((len(idx), len(acc)), dtype=np.int16)
    for i in range(len(idx)):
        for j in range(len(acc)):
            rst[i,j] = idx[i]//acc[j]
            idx[i] -= rst[i,j]*acc[j]
    return rst
```
这里找到的极值点就是在行号为4、列号为5的像素点，即第5行、第6列。
