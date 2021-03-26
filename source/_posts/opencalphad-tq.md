---
title: OpenCalphad的TQ库编译及链接
tags: [opencalphad]
categories: simulation
date: 2016-1-13
---
TQ库的基本库文件有boceq.a、liboceqplus.mod、liboctq.o、liboctq.mod。

1、通过make编译OC主程序得到liboceq.a和liboceqplus.mod文件;

2、将以上两个文件复制到TQlib路径下，并编译liboctq.F90文件：gfortran -c liboctq.F90
得到liboctq.o和liboctq.mod两个文件。
若想调用oc，则将上述四个文件复制到当前程序路径下。

如果是编译Fortran程序，执行：
gfortran -o app Example.F90 liboctq.o liboceq.a
注意liboctq.o和liboceq.a的链接顺序，否则会出错。

如果是编译c/c++程序，需要编译liboctqc.F90或liboctqisoc.F90：
gfortran -c liboctqisoc.F90

然后编译c源文件：
gcc -o app Example.cpp liboctqisoc.o liboctq.o liboceq.a -lgfortran -lm
别忘了后面的-lgfortran。

如果编译c++源文件：
g++ -o app -lstdc++ Example.cpp liboctqisoc.o liboctq.o liboceq.a -lgfortran -lm

如果是编译并行的C++程序（这里使用openmp并行）：
g++ -o app -fopenmp -lstdc++ Example.cpp liboctqisoc.o liboctq.o liboceq.a -lgfortran -lm
