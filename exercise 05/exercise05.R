# Big Data Exercise 5 Sparsity
# Yanxin Li
# Oct 12 2017

rm(list=ls())
setwd("D:/2017 UT Austin/Statistical Models for Big Data/R")

# Part (A)
# Penalized likelihood function without the argmin (objective function)
obj = function(y, theta, lambda){
  0.5 * (y-theta)^2 + lambda * theta
}

# Toy example with given lamda and y
lambda = 1
y = 5 

# Plot S(y) function
curve(obj(y,x,lambda),-10,10,xlab=expression(theta),ylab=expression(S[lambda](5)))
a = abs(y) - lambda
abline(v=ifelse(y<0,-1,1)*ifelse(a>0,a,0),col=2)

# Hard thresholding function 
hard.thresh = function(y,lambda){
  ifelse(abs(y)>=lambda,y,0)
}

# Soft thresholing function
soft.thresh = function(y,lambda){
  a = abs(y) - lambda
  results = ifelse(y<0,-1,1)*ifelse(a>0,a,0)
  return(results)
}

# Plot different thresholdings
curve(soft.thresh(x,1),-3,3,n=10000,xlab='y',ylab=expression(hat(theta)))
curve(hard.thresh(x,1),-3,3,n=10000,xlab='y',ylab=expression(hat(theta)),
      col=2,lty=3,add=T,lwd=3)
legend('topleft',legend = c('Hard','Soft'), col=1:2, lty=c(1,3), lwd=c(1,3))
title(expression(paste('Thresholding, ',lambda,'=1')))

#------------------------------------------------------------------
# Part (B)
# Toy-example to illustrate how soft -thresholding can be used to enforce sparsity
# Number of datapoints
n = 100
sparsity = 0.2

# Step 1: sparse theta vector
nonzero = round(n*sparsity)
theta = numeric(n)
theta[sample(1:n,nonzero)] <- 1:nonzero
sigma = numeric(n) + 1  # all sigma_i = 1

#  Step 2: generate data, Gaussian sequence model
z = rnorm(n = n, mean = theta, sd = sigma)

# Step 3: Soft thresholding for multiple lambda values
lambda = 0:5
plot(0,0,type='n',xlim = range(theta)*1.1, ylim = range(theta)*1.2, 
     xlab=expression(theta),ylab=expression(paste(hat(theta),'(y)',sep='')),
     main='Soft Thresholding')

for(i in 1:length(lambda)){
  points = soft.thresh(y = z, lambda = lambda[i]*sigma)
  points(theta, points, col = i, pch=19, cex = 0.5)
}
abline(0,1,lty=3)
text(22,23,expression(paste(hat(theta),'=',theta)))
legend('topleft',legend=lambda,title=expression(paste(lambda,' value')),col=1:6,pch=19)

# Step 4: Try out varying sparsity
n = 1000 # Number of datapoints
lambda = seq(0,3,length.out = 100) # A discrete grid of different lambda 
sparsity = c(0.01,0.05,0.10,0.25,0.50) # Varying sparsity

plot(0,0,type='n',xlim = range(lambda)*1.1, ylim = c(0,6), 
     xlab=expression(lambda),ylab='Scaled MSE')

# Loop over varying sparsity
for(i in 1:length(sparsity)){
  # Initialize vector to hold MSE for each lambda
  MSE = rep(0,length(lambda))
  
  # Sparse theta vector
  nonzero = round(n*sparsity[i])
  theta = numeric(n)
  theta[sample(1:n,nonzero)] = rpois(nonzero,5)
  
  # Vector of sigma's
  sigma = numeric(n) + 1
  
  # Simulate one data point
  z = rnorm(n = n, mean = theta, sd = sigma)
  
  # Loop over lambda's
  for(j in 1:length(lambda)){
    # Soft thresholding
    theta.hat = soft.thresh(y = z, lambda = lambda[j]*sigma)
    # Calculate MSE
    MSE[j] = mean((theta.hat-theta)^2)
  }
  
  # Find the minimum
  minimum = which.min(MSE)
  lines(lambda,MSE/MSE[minimum],col = i) # Scale by minimum
  # abline(v=lambda[minimum],col=i)
  points(lambda[minimum],1,col=i,pch=19,cex=1.5) # Dot the minimum 
}

# Add legend
legend('bottomright',legend=1-sparsity,title='Sparsity',
       col=1:length(sparsity),pch=19)


#---------------------------------------------------------------------
# The Lasso
# Part (A)
rm(list=ls())
library(glmnet)

# Load X into matrix
X = read.csv('diabetesX.csv')
X = as.matrix(X)

# Load Y into vector
Y = read.csv('diabetesY.csv',header=F)
Y = Y[,1]

# Choice of lambda's
m = 100 # Number of lambda's
lambda = exp(seq(4,-4,length.out=m))

# Fit Lasso model across a range of lambda values (which glmnet does automatically)
myLasso = glmnet(X,Y,alpha=1,lambda=lambda)

# Plot both coefficient sizes and MSE change
# Plot beta's by lambda - R has a default
plot(myLasso) # the L1 norm is the regularization term for Lasso
par(mfrow=c(1,2))
plot(0,0,type='n',
     ylim=range(myLasso$beta), ylab=expression(hat(beta)[lambda]),
     xlim=log(range(myLasso$lambda)), xlab = expression(paste('log(',lambda,')')),
     main = 'Shrinkage of Coefficients'
)
for(i in 1:nrow(myLasso$beta)){
  lines(log(lambda),myLasso$beta[i,],col=i)
}

# Plot in-sample MSE by lambda
MSE = myLasso$lambda * 0 # Initialize

for(i in 1:ncol(myLasso$beta)){
  MSE[i] = mean((Y - myLasso$a0[i] - X%*%myLasso$beta[,i])^2)
}
plot(log(myLasso$lambda), MSE, type = 'l', col='blue', main = 'In-sample MSE', 
     ylab= "MSE", xlab = expression(paste('log(',lambda,')')))
abline(v=log(myLasso$lambda[which.min(MSE)]),col='red')

# -------------------------------------------------------------------
# Part (B): Cross validation
# 10-fold Cross-validation to choose the best tuning parameter lambda
set.seed(1)
ratio = 0.5 # Split data
index = sample(1:nrow(X), floor(ratio*nrow(X))) # Make equal partition

# Training data set and test data set
x.train = X[index, ]
x.test = X[-index,]
y.train = Y[index]
y.test = Y[-index]

# 10-fold cross-validation
set.seed(1)
cv.out = cv.glmnet(x.train,y.train,alpha =1)
par(mfrow=c(1,1))
plot(cv.out)
bestlam = cv.out$lambda.min
print(bestlam) # The value of lambda that results in the 
               # smallest cross-validation error is 0.9351282

# Compute cross-validation MSE for Lasso model with best lambda
cv.pred.lasso = predict(myLasso,s=bestlam,newx=x.test)
MOOSE.best = mean((cv.pred.lasso -y.test)^2)
print(MOOSE.best) # 2808.36
MOOSE = cv.out$cvm # All Mean out-of-sample squared errors
length(cv.out$lambda) # number of the values of lambda used in the fit

# How many parameters at best lambda?
Lasso.mod  = glmnet(X,Y,alpha=1,lambda=lambda)
lasso.coef <- predict(Lasso.mod,type ="coefficients",s=bestlam)[1:ncol(X),]
print(lasso.coef)
sum(lasso.coef != 0) # number of nonzero betas for minimum MOOSE

###################
k = 10
set.seed(1) # Reproducibility
folds = sample(1:k, nrow(x.train), replace=TRUE)
cv.errors <- matrix(NA,k,m,dimnames=list(NULL, paste(1:m)))
for(j in 1:k){
  model = glmnet(x.train[folds!=j, ], y.train[folds!=j],alpha=1, lambda=lambda)
  for(i in 1:m){
    pred <- predict(model, s=lambda[i], newx=x.train[folds==j,])
    cv.errors[j,i] = mean((y.train[folds==j] - pred)^2)
    }
}
# Get Mean OOS square error (MOOSE)
MOOSE = apply(cv.errors,2,mean)
best.lam = lambda[which.min(MOOSE)]
print(best.lam)

# How many parameters at best lambda?
lasso.cv = glmnet(X,Y,alpha=1,lambda=lambda)
coef.cv <- predict(lasso.cv,type ="coefficients",s=best.lam)[1:ncol(X),]
print(coef.cv)
sum(coef.cv != 0) # number of nonzero betas for minimum MOOSE

#----------------------------------------------------------------
# Part (C): Collom Mallow's CP stat
n = length(Y)
p = ncol(X)
# Setup sequence of lambdas
lambda = exp(seq(4,-4,length.out=m))

# initialize CP vector
Cp = lambda * 0

# Run model with several lambda's
model = glmnet(X,Y,alpha=1,lambda=lambda)

# calculate MSE for each labmda
for(i in 1:ncol(model$beta)){
  errors = Y - model$a0[i] - X%*%model$beta[,i]
  Cp[i] = mean((errors)^2) + 2*model$df[i]*var(errors)/n
}
lambda[which.min(Cp)]
lasso.cp = glmnet(X,Y,alpha=1,lambda=lambda)
coef.cp <- predict(lasso.cp,type ="coefficients",s=lambda[which.min(Cp)])[1:ncol(X),]
print(coef.cp)
sum(coef.cp != 0) # number of nonzero betas for minimum Cp

# Plot log(lambda) v.s. MSE, MOOSE and Cp
plot(log(lambda),MSE,type='l',xlab=expression(paste("log(",lambda,")")), ylab='Error')
lines(log(lambda),MOOSE,col=2)
lines(log(lambda),Cp,col=3)
abline(v=log(lambda[which.min(MSE)]))
abline(v=log(lambda[which.min(MOOSE)]),col="red")
abline(v=log(lambda[which.min(Cp)]),col="green")
legend('topleft',legend=c('IS MSE','MOOSE','Cp'),col=1:3,lty=1)

# How many parameters at best lambda?
sum(model$beta[,which.min(Cp)] != 0)

