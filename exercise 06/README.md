# Output of R code:
- http://rpubs.com/leexiner/bigdata-exercise05

The following code includes the computaion of Leave-one-out Cross Validation 'MOOSE' and compares the best lambda that has smallest 'MOOSE' with Mallows' Cp.
- http://rpubs.com/leexiner/bigdata-exercise05-updated

# Data source:
- https://github.com/Cindy-UTSDS/SDS385/tree/master/data

# Sparisty
In many problems, we wish to impose sparsity on the parameters of a statistical model -- that is, the assumption that some parameters are zero. In this set of exercises, we will learn a few basic ideas that are important for thinking about sparse statistical models at scale.

After reading Chapter 3.4 of The Elements of Statistical Learning, we have the following key things:

- the lasso
- the idea of the lasso solution path (Figure 3.10)
- the degrees of freedom for the lasso (page 77)

Note that we will work the lasso objective in "Lagrangian form" (Equation 3.52). 
