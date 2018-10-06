---
title: 使用skopt贝叶斯搜索寻找scikit-learn中算法的最优超参数
tags: [Machine Learning]
categories: programming
date: 2018-9-18
---
# 介绍
机器学习的寻找最优超参数是个老大难问题，scikit-learn提供了网格搜索GridSearchCV和随机搜索RandomizedSearchCV这两个函数来帮助寻找这些超参数。网格搜索的本质就是对参数空间形成的所有参数组合进行一个个的尝试，然后选出得分最高的那个，可能会忽略这些组合以外的参数，同时随着参数的增多，计算量也会指数增加。随机搜索是对参数的随机搜索，但没有充分利用搜索空间的结构。
skopt是一个超参数优化库，包括随机搜索、贝叶斯搜索、决策森林和梯度提升树等，用于辅助寻找机器学习算法中的最优超参数。这里是利用skopt的贝叶斯搜索来替代scikit-learn中的默认搜索方法，从而更快更好地寻找到最优超参数。
本文是对这个[官方文档](https://scikit-optimize.github.io/notebooks/sklearn-gridsearchcv-replacement.html)的翻译。
# 上手实例
用skopt来优化sklearn的支持向量机分类器：
```python
from skopt import BayesSearchCV
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X, y = load_digits(10, True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

opt = BayesSearchCV(
    SVC(),
    {
        'C': (1e-6, 1e+6, 'log-uniform'),
        'gamma': (1e-6, 1e+1, 'log-uniform'),
        'degree': (1, 8), # 整型类型的空间
        'kernel': ['linear', 'poly', 'rbf'], # Categorical类型的空间
    },
    n_iter = 32
)

opt.fit(X_train, y_train)

print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test, y_test))
```
由上面定义的搜索空间可以看出，skopt只需给定参数的上下界即可，它会自动在此范围内寻找。注意那个log-uniform，这样指定的话，就会以通过改变x而改变exp(x)这样的形式变动参数。

# 进阶实例
这个例子给出了使用多个模型及其对应的搜索空间：
```python
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

X, y = load_digits(10, True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

pipe = Pipeline(
    [
        ('model', SVC())
    ])

# 隐式地指定参数类型
linsvc_search = {
    'model': [LinearSVC(max_iter=10000)],
    'model__C': (1e-6, 1e+6, 'log-uniform'),
}
# 显式地指定参数类型
svc_search = {
    'model': Categorical([SVC()]),
    'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
    'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    'model__degree': Integer(1,8),
    'model__kernel': Categorical(['linear', 'poly', 'rbf']),
}

opt = BayesSearchCV(
    pipe,
    [(svc_search, 20), (linsvc_search, 16)], # (parameter space, # of evaluations)
)

opt.fit(X_train, y_train)

print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test, y_test))
```

# 使用fit函数的callback参数实现过程监控
可以实现在每一步子空间探索过程中通过一个事件句柄来监控BayesSearchCV的进度。对于串行任务，每一步评估后调用，对于并行任务，当并行执行了n\_jobs后调用。
除此以外，如果callback返回了True，还可以停止进程。这可以用于当精度足够高时提前终止学习。
```python
from skopt import BayesSearchCV

from sklearn.datasets import load_iris
from sklearn.svm import SVC

X, y = load_iris(True)

searchcv = BayesSearchCV(
    SVC(),
    search_spaces={'C': (0.01, 100.0, 'log-uniform')},
    n_iter=10
)

def on_step(optim_result):
    score = searchcv.best_score_
    print("best score: %s" % score)
    if score >= 0.98:
        print('Interrupting!')
        return True

searchcv.fit(X, y, callback=on_step)
```

# 计算探索所有子空间所需的总迭代次数
```python
from skopt import BayesSearchCV

from sklearn.datasets import load_iris
from sklearn.svm import SVC

X, y = load_iris(True)

searchcv = BayesSearchCV(
    SVC(),
    search_spaces=[
        ({'C': (0.1, 1.0)}, 19),  # 19 iterations for this subspace
        {'gamma':(0.1, 1.0)}
    ],
    n_iter=23
)

print(searchcv.total_iterations)
```
