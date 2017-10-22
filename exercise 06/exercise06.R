# Big Data Exercise06 The Proximal Gradient Method
# Yanxin Li
# Oct 20 2017

rm(list=ls())
setwd("D:/2017 UT Austin/Statistical Models for Big Data/R")

#---------------------------------------------------------------
# Negative log likelihood function
nll <- function(X, Y, beta) {
  A <- Y - X %*% beta
  loglike <- (0.5/nrow(X)) * crossprod(A)
  return(loglike)
}

#---------------------------------------------------------------
# Gradient of negative log likelihood function
gradient <- function(X, Y, beta) {
  A <- Y - X %*% beta
  grad <- -(1/nrow(X)) * crossprod(X, A)
  return(grad)
}

#---------------------------------------------------------------
# Proximal operator
prox.l1 <- function(u, lambda) {
  uhat <- abs(u) - lambda
  ubind <- cbind(rep(0, length(u)), uhat)
  prox <- sign(u) * apply(ubind, 1, max) # 1: row, 2: column
  return(prox)
}

#---------------------------------------------------------------
# Proximal gradient descent
proxGD <- function(X,Y,lambda=0.01,gamma=0.01,beta0=NA,iter=50000,conv=1e-10){
  # Set beta0 equal to a series of zeros
  if (is.na(beta0)) { beta0 <- rep(0, ncol(X)) }
  
  # Initialize coefficients matrix 
  beta <- matrix(rep(NA, ncol(X)*(iter+1)), nrow=ncol(X))
  beta[, 1] <- beta0
  
  # Initialize objective funtion
  obj <- rep(NA, iter+1)
  obj[1] <- nll(X, Y, beta0) + lambda * sum(abs(beta0))
  
  # Initialize convergence message in case convergence not reached
  message <- "Convergence not reached..."
  
  for (t in 1:iter) {
    # Update u, beta and obj
    u <- beta[,t] - gamma * gradient(X, Y, beta[,t])
    beta[,t+1] <- prox.l1(u, gamma * lambda)
    obj[t+1] <- nll(X, Y, beta[,t+1]) + lambda * sum(abs(beta[,t+1]))
    
    # Check convergence
    delta <- abs(obj[t+1]-obj[t]) /(abs(obj[t])+conv)
    if (delta < conv) {
      # Remove excess betas and obj
      beta <- beta[, -((t+2):ncol(beta))]
      obj <- obj[-((t+2):length(obj))]
      
      # Update convergence message
      message <- sprintf("Convergence reached after %i iterations", (t+1))
      break
    }
  }
  
  result <- list("beta.hat"=beta[,ncol(beta)],"beta"=beta,"objective"=obj,"conv"=message)
  return(result)
}

#-----------------------------------------------------------------
# Accelerated proximal gradient method
proxACCE <- function(X,Y,lambda=0.01,gamma=0.01,beta0=NA,iter=50000,conv=1e-10){
  # Set beta0 equal to a series of zeros
  if (is.na(beta0)) { beta0 <- rep(0, ncol(X)) }
  
  # Create s vector
  s <- rep(NA, iter+1)
  s[1] <- 1
  
  for (j in 2:length(s)) {
    s[j] <- (1+sqrt(1+4*s[j-1]^2))/2
  }
  
  # Initialize z matrix
  z <- matrix(0, nrow=ncol(X), ncol=iter+1)
  
  # Initialize coefficients matrix 
  beta <- matrix(rep(NA, ncol(X)*(iter+1)), nrow=ncol(X))
  beta[, 1] <- beta0
  
  # Initialize objective funtion
  obj <- rep(NA, iter+1)
  obj[1] <- nll(X, Y, beta0) + lambda * sum(abs(beta0))
  
  # Initialize convergence message in case convergence not reached
  message <- "Convergence not reached..."
  
  for (t in 1:iter) {
    # Update u, beta, z and obj
    u <- z[,t] - gamma * gradient(X, Y, z[,t])
    beta[,t+1] <- prox.l1(u, gamma * lambda)
    z[,t+1] <- beta[,t+1] + (s[t] - 1)/s[t+1] * (beta[,t+1] - beta[,t])
    obj[t+1] <- nll(X, Y, beta[,t+1]) + lambda * sum(abs(beta[,t+1]))
    
    # Check convergence 
    delta <- abs(obj[t+1]-obj[t]) /(abs(obj[t])+conv)
    if (delta < conv) {
      # Remove excess betas and nll
      beta <- beta[, -((t+2):ncol(beta))]
      obj <- obj[-((t+2):length(obj))]
      
      # Update convergence message
      message <- sprintf("Convergence reached after %i iterations", (t+1))
      break
    }
  }
  
  result <- list("beta.hat"=beta[,ncol(beta)],"beta"=beta,"objective"=obj,"conv"=message)
  return(result)
}

#------------------------------------------------------------------
library(MASS)
library(glmnet)
library(ggplot2)

# Read in data
X <- as.matrix(read.csv("diabetesX.csv", header = TRUE))
Y <- as.numeric(unlist(read.csv("diabetesY.csv", header = FALSE)))

# Scale data
X <- scale(X)
Y <- scale(Y)

# Setup the range of lambda
lambda <- exp(seq(0,-7,length.out=100)) # [0.00091, 1]

#-------------------------------------------------------------------
# Compare proximal gradient algorithm with glmnet
# Calculate coefficients using proximal gradient algorithm
betamat <- matrix(rep(NA, length(lambda)*ncol(X)), ncol=length(lambda))
for (i in 1:length(lambda)) {
  mylasso <- proxGD(X, Y, lambda=lambda[i])
  betamat[, i]  <- mylasso$beta.hat
}

# Fit Lasso model across a range of lambda values
myLasso <- glmnet(X,Y,alpha=1,lambda=lambda)

# Plot all beta's vs. lambda both from accelerated proximal gradient method and glmnet
par(mar=c(4,5,4,2))
par(mfrow=c(1,2))
plot(0,0,type='n', ylim=c(min(betamat), max(betamat)), cex.main=0.8,
     ylab=expression(hat(beta)[lambda]), xlim=log(range(lambda)),
     xlab=expression(paste('log(',lambda,')')), main='Coefficients of Proximal Gradient')

for (i in 1:nrow(betamat)) { lines(log(lambda), betamat[i, ], col=i) }

# plot(myLasso) # R default, the L1 norm is the regularization term for Lasso
plot(0,0,type='n', ylim=range(myLasso$beta), cex.main=0.8,
     ylab=expression(hat(beta)[lambda]), xlim=log(range(myLasso$lambda)),
     xlab=expression(paste('log(',lambda,')')), main='Coefficients of glmnet Package')

for(i in 1:nrow(myLasso$beta)){ lines(log(lambda),myLasso$beta[i,],col=i) }

#--------------------------------------------------------------------
# 10-fold Cross-validation to choose the best tuning parameter lambda for glmnet
set.seed(1)
ratio <- 0.5 # Split data
index <- sample(1:nrow(X), floor(ratio*nrow(X))) # Make equal partition

# 10-fold cross-validation
set.seed(1)
cv.out <- cv.glmnet(X[index, ],Y[index],alpha =1)
par(mfrow=c(1,1))
plot(cv.out)
bestlam <- cv.out$lambda.min
# The value of lambda that results in the smallest CV error is 0.01212987
print(bestlam)

#-------------------------------------------------------------------------
# Compare proximal gradient algorithm with accelerated proximal gradient algorithm
lassoGD <- proxGD(X, Y, lambda=0.01, gamma=0.01, conv=1e-10)
lassoACCE <- proxACCE(X, Y, lambda=0.01, gamma=0.01, conv=1e-10)

# Number of iterations when convergence met
n1 <- length(lassoGD$objective)
n2 <- length(lassoACCE$objective)
print(n1); print(n2)

# Plot objective values and compare number of iterations used when convergence met
par(mfrow=c(1,1))
plot(lassoGD$objective, type="l", col="black", log="x",
     xlab="iteration", ylab="Objective Function", cex.main=1,
     main=sprintf("Objective for Prox and Accelerated Prox Methods, conv = 1e%i", -10))
lines(lassoACCE$objective, col="red")
points(n1, lassoGD$objective[n1], pch=19, col="red")
points(n2, lassoACCE$objective[n2], pch=19, col="blue")
legend("topright", legend=c("Proximal Method", "Accelerated Proximal Method"),
       col=c("black", "red"), lty=c(1,1), cex=1)

# Compare beta.hat's of the two algorithms
plot(lassoGD$beta.hat, lassoACCE$beta.hat,
     main="Comparison of Coefficients from Two Methods",
     xlab="Coefficients from Prox Method",
     ylab="Coefficients from Accel Prox Method",pch=16,col="purple")
abline(0, 1)

# Eyeball beta.hat
cbind(lassoGD$beta.hat,lassoACCE$beta.hat)
# Number of nonzero coefficients by using proximal gradient method
sum(lassoGD$beta.hat != 0) 
# Number of nonzero coefficients by using accelerated proximal gradient
sum(lassoACCE$beta.hat != 0) 
