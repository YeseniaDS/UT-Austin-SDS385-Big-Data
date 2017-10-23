# Output of R code:
- http://rpubs.com/leexiner/bigdata-exercise05

The following code includes the computaion of Leave-one-out Cross Validation 'MOOSE' and compares the best lambda that has smallest 'MOOSE' with Mallows' Cp.
- http://rpubs.com/leexiner/bigdata-exercise05-updated

# Data source:
- https://github.com/Cindy-UTSDS/SDS385/tree/master/data

The culmination of the first four exercises was stochastic gradient descent, which is one of the core algorithms that powers modern data science. Over the next few sets of exercises, we will build up to two other such core algorithms: the proximal gradient method, and ADMM, which stands for the alternating direction method of multipliers. These algorithms are broadly useful for optimizing objective functions f(x) in statistics that have either or both of the following two features:

- f(x) is a sum of two terms, one of which measures fit to the data, and the other of which penalizes model complexity.
- f(x) is not everywhere smooth, so that we cannot assume derivatives exist.

Both features come up in problems where we wish to impose sparsity on a parameter in a statistical model (i.e. the lasso of the previous exercises).

In this set of exercises, we begin our study of scalable algorithms that can handle sparsity, with the proximal gradient method.