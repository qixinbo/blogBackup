---
title: C++默认构造函数——深入理解 (转载)
tags: [c++]
categories: coding 
date: 2017-7-11
---
本文转载自[这里](http://blog.csdn.net/hankai1024/article/details/7947989)，修正了一些格式和文字错误。

# 引子
错误认识1：若程序员没有自己定义无参数的构造函数，那么编译器会自动生成默认构造函数，来进行对成员函数的初始化。
错误认识2：编译器合成出来的default constructor会明确设定“class内每一个data member的默认值”。但这两种种认识是有误的，不全面的。
正确认识：
默认的构造函数分为有用的和无用的，所谓无用的默认构造函数就是一个空函数、什么操作也不做，而有用的默认构造函数是可以初始化成员的函数。
对构造函数的需求也是分为两类：一类是编辑器需求，一类是程序的需求。
程序的需求：若程序需求构造函数时，就是要程序员自定义构造函数来显示的初始化类的数据成员。
编辑器的需求：编辑器的需求也分为两类：一类是无用的空的构造函数(trivial)，一类是编辑器自己合成的有用的构造函数(non-trivival)。
在用户没有自定义构造函数的情况下：
一、由于编辑器的需求，编辑器会调用空的无用的默认构造函数。如：类中没有显式定义任何构造函数。
二、但在某些情况下,编辑器就一定会合成有用的默认构造函数。

# 无关紧要（trivial）的默认构造函数【无用构造函数】
用户并没有显式地定义默认构造函数，编译器会为它自动生成一个无关紧要（trivial）的默认构造函数，生成的默认构造函数什么也不做，既不会将其成员变量置零，也不会做其他的任何事情，只是为了保证程序能够正确运行而已，这就是所谓的“需要”，如果还需要给初始化成员变量，这件事情还是交给程序员做吧！
```cpp
#include<iostream>  
using namespace std;  
class Foo  
{   
public:   
  int val;   
  Foo *pnext;  
};  
void foo_fo(){  
 Foo fo;   
if (fo.val || fo.pnext)   
   {   
      cout << fo.val << endl;   
     cout << fo.pnext << endl;   
   }  
}  
int main()  
{   
foo_fo();   
}  
```
输出为：
```cpp
4196608
0x400750
```
这两个值是无意义的垃圾值。

如果基类和继承类都没有自定义的构造函数。那么，都会生成trival的默认构造函数。例：
```cpp
#include<iostream>   
using namespace std;   
class Foo   
{   
  private:   
  int val;   
};   
class Bar:public Foo   
{   
public:   
  char *str;   
  int i;   
};   
void foo_bar()   
{   
   Bar bar;   
   cout<<bar.i<<endl;   
if (bar.str )   
{   
   cout << "Print the content !" << endl; }  
}   
int main()   
{   
  foo_bar();   
  return 0;   
}
```
输出为：
```cpp
4196656
```
# 非平凡（non-trivival）默认构造函数【有用构造函数】
C++标准描述了哪些情况下这样的隐式默认构造函数是无关紧要的。一个非平凡（non-trivival）的默认构造函数是ARM中所说的被实现所“需要”，并在必要的时候被编译器自动生成。下面来看看默认构造函数是非平凡的四种情况：
## 含有包含默认构造函数的成员类对象
如果该类包含一个成员类对象，它有默认的构造函数，那么这个类的隐式构造函数是非平凡的，并且编译器需要为包含的这个成员类对象生成一个默认构造函数。然后，这个编译器生成的默认构造函数只有在实际上被调用时才会被真正的生成。如下例中编译器为Bar类生成一个默认构造函数。
如果一个class中含有成员对象，而且这个对象有default constructor（如Foo foo,Foo类中有default constructor）, 那么编译器就会给这个class（Bar）合成一个default constructor, 但是这个合成动作只有在调用需要时才会产生。也就是说，在需要时才会合成。
```cpp
#include<iostream>   
using namespace std;   
class Foo   
{   
public:   
  Foo();   
  Foo( int );  
private:    
  int val;   
};   
Foo::Foo()   
{   
  cout << "Call Foo::Foo() Constructor !"<< endl;   
  val = 0;   
}   
Foo::Foo(int i)   
{   
  cout << "Call Foo::Foo(int i) Constructor !"<< endl;   
  val = i;   
}   
class Bar   
{   
public:  
  Foo foo;   
  char *str;   
};  
void foo_bar()   
{   
  Bar bar; // Bar::foo must be initialized here 
  if (bar.str )   
  { cout << "Print the content !" << endl; }   
}   
int main()   
{   
  foo_bar();   
  return 0;   
}  
```
输出为：
```cpp
Call Foo::Foo() Constructor !
Print the content !
```
在这个程序片段中Bar的成员foo含有默认构造函数，它初始化自己的类成员val为0而Bar本身并没有定义默认的构造函数，这个构造函数的目的是为了初始化它的成员变量foo，实际上就是调用Bar::foo的默认构造函数，但它并不会做一丁点关于另外一个变量str的初始化和赋值工作，初始化Bar::foo是编译器的责任，而初始化str是程序员的责任。
我们可以用以下代码来大致描述一下编译器的工作：
```cpp
inline Bar::Bar() 
{ // Pseudo C++ Code 
  foo.Foo::Foo(); 
}  
```
结论：如果class中内含一个以上的含有default constructor的object,那在为class合成的default constructor中，会按照object的声明次序调用object 的 default constructor。
如果Bar中用户自定义的默认构造函数（用户有定义，则编译器不会再自动生成）：
```cpp
#include<iostream>   
using namespace std;  
class Foo   
{   
public:   
  Foo();   
  Foo( int );   
private:   
  int val;   
};  
Foo::Foo()   
{   
  cout << "Call Foo::Foo() Constructor !"<< endl;   
  val = 0;   
}   
Foo::Foo(int i)   
{   
  cout << "Call Foo::Foo(int i) Constructor !"<< endl;   
  val = i;   
}   
class Bar   
{   
public:   
  Foo foo;   
  char *str;   
  Bar(){}//默认构造函数   
};   
void foo_bar()   
{   
  Bar bar;   
if (bar.str ) {  
  cout << "Print the content !" << endl;   
}   
}   
int main()   
{   
  foo_bar();   
  return 0;   
} 
```
结果一样。

对比，以下代码不符合上述要求，会报错：
```cpp
#include<iostream>   
using namespace std;   
class Foo   
{   
public:   
  Foo(); //去掉默认构造函数会报错   
  Foo( int );   
private:   
  int val;  
};   
Foo::Foo()   
{   
  cout << "Call Foo::Foo() Constructor !"<< endl;   
  val = 0;   
}   
Foo::Foo(int i)   
{   
  cout << "Call Foo::Foo(int i) Constructor !"<< endl;   
  val = i;   
}   
class Bar   
{   
public:   
  Foo foo(1); //不是默认构造函数产生的成员，也会报错。expected `;' before '(' token char *str;   
};   
void foo_bar()   
{   
  Bar bar; // Bar::foo must be initialized here   
if (bar.str )   
{   
  cout << "Print the content !" << endl; }   
}   
int main()   
{   
  foo_bar();   
  return 0;   
} 
```  

## 一个类的父类自定义的无参构造函数(有non-trival的默认构造函数)。
父类有自定义的默认构造函数，子类无任何自定义构造函数。
```cpp
#include<iostream>   
using namespace std;   
class Foo   
{   
public:   
  Foo()   
  {   
    cout << "Call Foo::Foo() Constructor !"<< endl;   
    val = 0;   
  }   
  Foo(int i)   
  {   
    cout << "Call Foo::Foo(int i) Constructor !"<< endl;   
    val = i;   
   }   
private:   
  int val;   
};   
class Bar:public Foo   
{   
public: 
//Foo foo;   
char *str;   
};   
void foo_bar()   
{   
  Bar bar;   
  if (bar.str )   
  {   
  cout << "Print the content !" << endl;   
  }   
}   
int main()   
{   
  foo_bar();   
  return 0;   
}  
```
输出为：
```cpp
Call Foo::Foo() Constructor !
Print the content !
```
如果一个没有任何constructor的派生类继承自一个带有default constructor(用户定义的)的base class, 那么这个派生类的default constructor被认为是nontrivial，而对于nontrivial的default constructor, 编译器会为他合成出来。在合成出的default constructor中调用base class的default constuctor。
如果设计者提供了多个constructor,但未提供default constuctor,那编译器不会合成新的default constructor,而是会扩展所有的现有的constructor，安插进去default constructor所必须的代码。如果此类中仍存在第一种情况，也就是说存在有member object, 而且object含有default constructor, 那这些default constructor 也会被调用，在base class的default constructor被调用后。
如果把//Foo foo这句也加上，结果为：
```cpp
Call Foo::Foo() Constructor !
Call Foo::Foo() Constructor !
Print the content !
```
因为在创建子类对象时需要调用父类的构造函数，同时，由于子类包含父类成员对象，初始化成员对象时也需要调用父类构造函数。

## 一个类里隐式的含有Virtual tabel(Vtbl)或者pointer member(vptr)，并且其基类无任何构造函数或者有用户自定义的默认构造函数。
vtbl或vptr需要编辑器隐式地合成出来，那么编辑器就把合成动作放在了默认构造函数里，所以编辑器必需自己产生一个构造函数来完成这些动作。所以你的类里只要含有virtual function，那么编辑器就会生成默认的构造函数。
无论一个class是声明（或继承）了一个virtual function, 还是派生自一个继承串联，其中有一个或多个virtual base class.不管上述哪种情况，由于缺乏由user声明的constructor, 编译器会详细记录合成一个default constructor的详细信息。
 在编译期间，会做以下的扩张工作：
(1) 一个virtual function table会被编译器产生出来，内含virtual functions的地址。
(2) 编译器会合成一个vptr, 插入每一个object中。
而合成出来的default constructor，当然会为每一个object 设定vptr的初值。但不会为类的成员变量初始化。
例，基类有虚函数，并且无构造函数或者有用户自定义的默认构造函数：
```cpp
#include<iostream>   
using namespace std;   
class Base
{ 
  public: 
    int x; 
    // Base(){} 有没有这句效果都是一样的 
    virtual void set() 
    { cout<<"Base set"<<endl; cout<<x<<endl; } 
};
class Derived:public Base
{ 
  public: 
    int y; 
    void set() 
    { cout<<"Derived set"<<endl; cout<<y<<endl; } 
};
void func(Base &b)
{ 
  b.set(); 
}
int main()
{ 
  Base b; 
  Derived d; 
  func(b); 
  func(d); 
}
```
输出为：
```cpp
Base set
0
Derived set
0
```
例：基类为抽象类，且无构造函数。
```cpp
#include<iostream>  
using namespace std;  
class Widget  
{  
public:   
    virtual void flip() = 0;  
};  
class Bell: public Widget  
{  
public:   
  void flip()  
  {   
    cout <<"Bell." << endl;   
   }  
};  
class Whistle: public Widget  
{  
public:   
  void flip()  
  {   
   cout <<"Whistle." << endl; }  
  };  
void flip(Widget &widget)  
{   
   widget.flip();  
}  
void foo()  
{   
  Bell b;   
  Whistle w;   
  flip(b);   
  flip(w);  
}  
int main()  
{  
 foo();   
}  
```
输出为：
```cpp
Bell.
Whistle.
```
## 如果一个类虚继承与其他类
理由和上一个一样，虚基类也需要vtbl和vptr管理，那么这些管理就需要合成构造函数来实现管理，则需要生成默认的构造函数。
例：
```cpp
#include<iostream>  
using namespace std;  
class Base  
{   
public:   
  int x;   
void set()   
{   
  cout<<"Base set"<<endl;   
  cout<<x<<endl;   
}   
};  
class Derived1:virtual public Base  
{   
public:   
  int y;   
void set()   
{   
  cout<<"Derived1 set"<<endl;   
  cout<<y<<endl;   
}   
};  
class Derived2:virtual public Base  
{   
public:   
int z;   
void set()   
{   
cout<<"Derived2 set"<<endl;   
cout<<z<<endl;   
}   
};  
int main()  
{   
Base b;   
b.set();   
Derived1 d1;   
d1.set();   
Derived2 d2;   
d2.set();   
}  
```
输出为：
```cpp
Base set
-1913988864
Derived1 set
4196432
Derived2 set
0
```
与如下代码效果一样:
```cpp
#include<iostream>  
using namespace std;  
class Base  
{   
public:   
int x;   
Base(){};//用户定义的默认构造函数   
Base(int i)//用用户定义的含参构造函数   
{ x=i; }   
void set()   
{   
cout<<"Base set"<<endl;   
cout<<x<<endl;   
}   
};  
class Derived1:virtual public Base  
{   
public:   
int y;  
void set()   
{   
cout<<"Derived1 set"<<endl;   
cout<<y<<endl;   
}   
};  
class Derived2:virtual public Base  
{   
public:   
int z;   
void set()   
{   
cout<<"Derived2 set"<<endl;   
cout<<z<<endl;   
}   
};  
int main()  
{   
Base b;   
b.set();   
Derived1 d1;   
d1.set();   
Derived2 d2;   
d2.set();   
}  
```
# 当用带有默认参数的构造函数进行初始化的时候，可以用类似默认参数初始化类的对象的方式来进行初始化。
特别注意：以下所有情况均为把有参构造函数用默认值初始化的特例(长得像，但并不是默认构造函数)，并非有默认构造函数。创建对象方式与默认构造函数相同，但意义不一样。以下是在声明时成成员初始化为0，则调用形式与默认构造函数相同。其
## 例1：父类无构造函数（有编译器自己创建的trival型的），子类已经有有参构造函数。生成对象的时候实际调用的是用参构造函数，只不过参数都是0，可以省略不写。
```cpp
#include<iostream>  
using namespace std;  
class Point  
{   
protected:   
int x0,y0;   
public:   
void set()//或者virtual void set()   
{ cout<<"Base"<<endl; }   
};   
class Derived:public Point   
{ protected:   
int x1,y1;   
public:   
Derived(int m=0,int n=0)   
{ x1=m; y1=n; }   
void set()  
{cout<<"Derived_set()"<<endl;}   
void draw(){cout<<"Derived_draw()"<<endl;}   
};  
void test(Point *b){ b->set(); }  
int main()  
{   
Derived *dr=new Derived;   
test(dr);   
Derived drr;   
test(&drr);  
 system("pause");   
}  
```
输出为：
```cpp
Base
Base
```
如果是
virtual void set()
则结果为：
```cpp
Derived_set()
Derived_set()
```

## 例2：基类含无参构造函数，子类已经有有参构造函数。
```cpp
#include<iostream>  
using namespace std;  
class Point  
{   
protected:   
int x0,y0;   
public:   
void set()   
{ cout<<"Base"<<endl;   
}   
Point(){//基类无参构造函数  
 }  
};   
class Derived:public Point   
{   
protected:   
int x1,y1;   
public:   
Derived(int m=0,int n=0)   
{ x1=m; y1=n; }   
void set(){cout<<"Derived_set()"<<endl;}   
void draw(){cout<<"Derived_draw()"<<endl;}   
};  
void test(Point *b)  
{   
b->set();   
}  
int main()  
{   
Derived *dr=new Derived;   
test(dr);   
Derived drr;   
test(&drr);   
system("pause");   
}  
```
结果：
```cpp
Base
Base
```
## 例3：基类含有参构造函数
```cpp
#include<iostream>  
using namespace std;  
class Point  
{   
protected:   
int x0,y0;   
public:   
void set()   
{ cout<<"Base"<<endl; }   
Point(int i,int j)  
{ x0=i; y0=j; }   
};   
class Derived:public Point   
{   
protected: int x1,y1;   
public:   
Derived(int m=0,int n=0,int i=0,int j=0):Point(i,j)   
{ x1=m; y1=n; }   
void set()  
{cout<<"Derived_set()"<<endl;}   
void draw()  
{cout<<"Derived_draw()"<<endl;}   
};  
void test(Point *b)  
{ b->set(); }  
int main()  
{   
Derived *dr=new Derived;   
test(dr);   
Derived drr;   
test(&drr);   
system("pause");   
}  
```
结果：
```cpp
Base
Base
```
## 例4：对于基类含有虚函数和抽象类也和上面所说的情况一致。
```cpp
#include<iostream>  
using namespace std;  
class Point  
{   
protected:   
int x0,y0;   
public:   
Point(int i,int j)   
{ x0=i; y0=j; }   
virtual void set()=0;   
virtual void draw()=0;   
};  
class Line:public Point  
{   
protected: int x1,y1;   
public: Line(int i=0,int j=0,int m=0,int n=0):Point(i,j)   
{ x1=m; y1=n; }   
void set()  
{cout<<"Line_set()"<<endl;}   
void draw(){cout<<"Line_draw()"<<endl;}   
};  
class Ellipse:public Point  
{   
protected:   
int x2,y2;   
public:   
Ellipse(int i=0,int j=0,int p=0,int q=0):Point(i,j)   
{ x2=p; y2=q; }   
void set()  
{cout<<"Ellipse_set()"<<endl;}   
void draw()  
{cout<<"Ellipse_draw()"<<endl;}   
};  
void drawobj(Point *p)  
{   
p->draw();   
}  
void setobj(Point *p)  
{ p->set(); }  
int main()  
{   
Line *li=new Line();//new Line;   
drawobj(li);   
setobj(li);   
cout<<endl;   
Ellipse *el=new Ellipse();//new Ellipse;   
drawobj(el);   
setobj(el);   
cout<<endl;   
Line *li2=new Line;   
drawobj(li2);   
setobj(li2);   
cout<<endl;   
Ellipse elp;   
drawobj(&elp);   
setobj(&elp);   
cout<<endl;   
}  
```
输出为：
```cpp
Line_draw()
Line_set()

Ellipse_draw()
Ellipse_set()

Line_draw()
Line_set()

Ellipse_draw()
Ellipse_set()
```
