### SDS 385 - Exercises 02 - Part C & D
# This code implements stochastic gradient descent to estimate the 
# beta coefficients for binomial logistic regression.

#-------------------------------------------------------------
library(Matrix)
rm(list=ls())
set.seed(10) # Reproducibility

# Read in data
wdbc <- read.csv("https://raw.githubusercontent.com/jgscott/SDS385/master/data/wdbc.csv", header = FALSE)

X <- as.matrix(wdbc [,3:12]) # Select first 10 features to keep and scale features
X <- scale(X) # Normalize design matrix features
X <- cbind(rep (1, nrow(X)), X)

Y <- wdbc[, 2] 
y <- rep(0,length(Y))
y[Y=='M'] <- 1 # Convert y values to 1/0's, response vector

beta <- as.matrix(rep (0, ncol(X))) # Initial guess of beta
m <- 1 # Number of trials is 1

iter <- 1E5 # Maximum iterations allowed
alpha <- 0.01 # Stepsize
epsilon <- 1E-10 # Convergence threshold

#---------------------------------------------------------------
# Sigmoid function
sigmoid <- function(beta, X){
  w <- 1/(1+exp(-X %*% beta)) 
  w <- pmax(w, 1E-5); w <- pmin(w, 1-1E-5) # Avoid probability of 0 or 1
  return(w)
}

#---------------------------------------------------------------
# Negative log likelihood function
loglike <- function(beta, y, X, m){
  # Input: beta: regression parameter P x 1
  #        y: vector of response N x 1
  #        X: matrix of features N x P
  #        m: number of trial for the ith case
  # Output:
  #        Negative log likelihood of binomial distribution
  loglik <- -sum(dbinom(y, m, sigmoid(beta, X), log = TRUE))
  return(loglik)
}

#---------------------------------------------------------------
# Gradient of the negative log likelihood
gradient <- function(beta, y, X, m){
  grad <- as.numeric(y - m * sigmoid(beta, X)) * X
  return(-colSums(grad))
}

#---------------------------------------------------------------
# Stochastic Gradient Descent
SGD <- function(beta, y, X, m, iter, epsilon, alpha){
  # Input: y: vector of response N x 1
  #        X: matrix of features N x P
  #        beta: initial guees of beta P x 1
  #        m: number of trials for the ith case
  #        iter: maximum iterations allowed if it doesn't converge
  #        epsilon: minimum error allowed for convergency creteria
  #        alpha: step size
  # Output:
  #        negative log likelihood per iteration using stochastic gradient descent
  #        beta: regression parameter P x 1
  
  # Initial guess for beta and negative log likelihood
  betas <- array(NA, dim=c(iter, ncol(X)))
  betas[1,] <- beta
  nll <- array(NA, dim = iter)
  nll[1] <- loglike(betas[1,], y, X, m)
  
  # Iterations
  for (i in 2:iter){
    rs <- sample(nrow(X), 1) # Draw a random sample with replacement
    xvector <- matrix(X[rs, ],nrow = 1)
    yvector <- matrix(y[rs], nrow = 1)
    grad <- gradient(betas[i-1,], yvector, xvector, m)
    
    # Gradient Descent
    betas[i,] <- betas[i-1,] - alpha * grad 
    nll[i] <- loglike(betas[i,], y, X, m)
    
    # Check for convergence
    rate <- abs(nll[i] - nll[i-1])/(abs(nll[i-1]) + epsilon)
    if (rate < epsilon){
      cat('Stochastic Gradient Descent converged in iterations:', i, sep = "\n")
      nll <- nll[1:i]
      betas <- betas[1:i,]
      break;
    } else if (i == iter && rate >= epsilon){
      print('Stochastic Gradient Descent did not converge')
      break;
    }
  }
  return(list("NegLoglike" = nll, "beta" = betas[i,]))
}

#-------------------------------------------------------------------
# Plot the result
stepsize <-  c(0.001, 0.01, 0.05, 0.1, 0.5, 1)
for (k in 1:length(stepsize)){
  sgd <- SGD(beta, y, X, m, iter, epsilon, alpha = stepsize[k])
  nllSGD <- as.matrix(unlist(sgd[1], use.names = FALSE))
  betaSGD <- as.matrix(unlist(sgd[2], use.names = FALSE))
  
  #png(filename=paste('SGD',stepsize[k],'.png'),width=15,height=12,units="cm",res=200)
  plot(1:length(nllSGD), nllSGD, type = "l",col = "blue", 
       xlab = "Iterations (Log-Scale)", ylab = "Negative Log-Likelihood", 
       log = "x", xlim = c(1,iter))
  #dev.off()
}

#-------------------------------------------------------------------
# Stochastic Gradient Descent using Robbins-Monro Rule 
# Decaying Step Size
SGDRM <- function(beta, y, X, m, iter, epsilon, C, t0, alpha){
  # Input:  C: constant in the Robbins-Monro rule
  #         t: t0 prior number of steps
  #         alpha: learning rate
  # Output: Negative log likelihood per iteration using 
  #         stochastic gradient descent with decaying step size
  #         beta: regression parameter P x 1
  
  # Initial guess
  betas <- array(NA, dim=c(iter, ncol(X)))
  betas[1,] <- beta
  nll <- array(NA, dim = iter)
  nll[1] <- loglike(betas[1,], y, X, m)
  
  # Iterations
  for (i in 2:iter){
    rs <- sample(nrow(X), 1) # Draw a random sample with replacement
    xvector <- matrix(X[rs, ],nrow = 1)
    yvector <- matrix(y[rs], nrow = 1)
    grad <- gradient(betas[i-1,], yvector, xvector, m)
    
    # Gradient Descent
    step <- C*(i+t0)^(-alpha) # Robbins-Monro rule
    betas[i,] <- betas[i-1,] - step * grad 
    nll[i] <- loglike(betas[i,], y, X, m)
    
    # Checking for Convergence
    rate <- abs(nll[i] - nll[i-1])/(abs(nll[i-1]) + epsilon)
    if (rate < epsilon){
      cat('Stochastic Gradient Descent converged in iterations:',i,sep = "\n")
      nll <- nll[1:i]
      betas <- betas[1:i,]
      break;
    } else if (i == iter && rate >= epsilon){
      print('Stochastic Gradient Descent did not converge')
      break;
    }
  }
  return(list("NegLoglike" = nll, "beta" = betas[i,]))
}

#------------------------------------------------------------------
# Loop through varying C
t0 <- 1
alpha <- 0.75
C <- c(0.01,0.1,1,10,50,100)
for (k in 1:length(C)){
  sgdrm <- SGDRM(beta, y, X, m, iter, epsilon,C[k], t0, alpha)
  nllSGDRM <- as.matrix(unlist(sgdrm[1], use.names = FALSE))
  betaSGDRM <- as.matrix(unlist(sgdrm[2], use.names = FALSE))
  
  plot(1:length(nllSGDRM),nllSGDRM,type="l",col="blue", 
       xlab="Iterations (Log-Scale)", ylab="Negative Log-Likelihood", 
       log="x", xlim=c(1, iter))
}

# Loop through varying alpha
t0 <- 1
alpha <- c(0.5,0.6,0.7,0.8,0.9,1)
C <- 50
for (k in 1:length(alpha)){
  sgdrm <- SGDRM(beta, y, X, m, iter, epsilon, C, t0, alpha[k])
  nllSGDRM <- as.matrix(unlist(sgdrm[1], use.names = FALSE))
  betaSGDRM <- as.matrix(unlist(sgdrm[2], use.names = FALSE))
  
  plot(1:length(nllSGDRM),nllSGDRM,type="l",col="blue", 
       xlab="Iterations (Log-Scale)", ylab="Negative Log-Likelihood", 
       log="x", xlim=c(1, iter))
}