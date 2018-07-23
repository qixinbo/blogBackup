---
title: 多相材料的相场模型
tags: [phasefield]
categories: computational material science
date: 2016-2-27
---
# 传统Ginzburg-Landau自由能泛函形式
$$ F_{CH}=\int\gamma[\frac{\epsilon}{2}|\nabla\phi|^2+\frac{1}{4\epsilon}(\phi^2-1)^2]dx $$
其中$\gamma$是传统明锐界面模型中的表面张力，$\epsilon$是界面宽度。

# 多相体系的混合能形式
$$ 
\begin{equation}
\begin{split}
F&=\int\_\Omega W(\phi,\nabla\phi,\psi,\nabla\psi)dx \\\
 &=\int\_\Omega[\gamma\_1(\frac{\psi-1}{2})^2(\frac{\epsilon\_1}{2}|\nabla\phi|^2+\frac{1}{4\epsilon\_1}(\phi^2-1)^2)+\gamma\_2(\frac{\epsilon\_2}{2}|\nabla\psi|^2+\frac{1}{4\epsilon\_2}(\psi^2-1)^2]dx
\end{split}
\end{equation}
$$
其中$(\frac{\psi-1}{2})^2$这一项是为了保证两个不同相(标记为$\phi=1,\psi=-1$和$\phi=-1,\psi=-1$)之间的相互作用不直接影响第三相(标记为$\psi=1$)。

# 多相运动体系的总能量
通过在系统中加入流体方程，得到整个流体动力学体系的总能量，其是动能和混合能的加权之和：
$$ E=\int\_\Omega(\frac{1}{2}\rho|u|^2+\lambda W(\phi,\nabla\phi,\psi,\nabla\psi))dx $$ 
这里$\lambda$表示两种能量之间的竞争。

# 应力张量
通过虚功原理求得Ginzburg-Landau能中的应力张量:
弹性应力张量:
$$ \sigma^e=-\gamma\_1\epsilon\_1(\frac{\psi-1}{2})^2\nabla\phi\otimes\nabla\phi
-\gamma\_2\epsilon\_2\nabla\psi\otimes\nabla\psi $$
粘性应力张量:
$$ \sigma^v=\frac{1}{2}(\nabla u+(\nabla u)^T) $$

# 控制方程组
$$
\begin{equation}
\begin{split}
\rho(u\_t+u\nabla u)+\nabla p&=\lambda\nabla(\sigma^e+\sigma^v) \\\
\phi\_t+u\nabla\phi&=-M\_1\frac{\delta F}{\delta\phi}           \\\
\psi\_t+u\nabla\psi&=-M\_2\frac{\delta F}{\delta\psi}           \\\
\end{split}
\end{equation}
$$
其中：
$$
\begin{equation}
\begin{split}
\frac{\delta F}{\delta\phi}&=-\gamma\_1[\epsilon\_1\nabla[(\frac{\phi-1}{2}^2\nabla\phi]
-\frac{1}{\epsilon\_1}(\frac{\psi-1}{2})^2(\phi^2-1)\phi]       \\\
\frac{\delta F}{\delta\psi}&=-\gamma\_2[\epsilon\_2\Delta\psi-\frac{1}{\epsilon\_2}(\psi^2-1)\psi]+\gamma\_1\epsilon\_1\frac{\psi-1}{2}[\frac{1}{2}|\nabla\phi|^2+\frac{1}{4\epsilon\_1}(\phi^2-1)^2]           \\\
\end{split}
\end{equation}
$$

# 参考文献
J. Brannick, C. Liu, T. Qian, H. Sun. Diffuse Interface Methods for Multiple Phase Materials: An Energetic Variational Approach, Numerical Mathematics: Theory, Methods and Applications 8 (2015) 220-236.

