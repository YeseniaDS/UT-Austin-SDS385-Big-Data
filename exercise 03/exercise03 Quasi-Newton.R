### SDS 385 - Exercises 03 - Quasi-Newton
# Some of the code is based on the R code of Jennifer Starling
# Jennifer Starling: https://github.com/jestarling/bigdata/tree/master/Exercise-03

# This code implements Quasi-Newton algorithm to estimate the 
# beta coefficients for binomial logistic regression.

# Yanxin Li   Oct 2 2017

rm(list=ls())
library(Matrix)
set.seed(1) # Reproducibility

#-----------------------------------------------------------------
# Read in data
wdbc <- read.csv("https://raw.githubusercontent.com/jgscott/SDS385/master/data/wdbc.csv",
                 header = FALSE)
y <- wdbc[,2]

X <- as.matrix(wdbc [,3:12]) # Select first 10 features to keep and scale features
X <- scale(X) # Normalize design matrix features
X <- cbind(rep (1, nrow(X)), X)

Y <- wdbc[, 2] 
y <- rep(0,length(Y))
y[Y=='M'] <- 1 # Convert y values to 1/0's, response vector

beta <- as.matrix(rep (0, ncol(X))) # Initial guess of beta
m <- 1 # Number of trials is 1

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
  # Output:Negative log likelihood of binomial distribution
  
  loglik <- -sum(dbinom(y, m, sigmoid(beta, X), log = TRUE))
  return(loglik)
}

#---------------------------------------------------------------
# Gradient of the negative log likelihood
gradient <- function(beta, y, X, m){
  grad <- as.numeric(y - m * sigmoid(beta, X)) * X
  return(-colSums(grad))
}

#--------------------------------------------------------------
# Line Search Function: p is the direction vector
line.search <- function(beta,y,X,gr,p,m,maxalpha=1){
  c <- 0.1			# a constant, in (0,1)
  alpha <- maxalpha	# max step size, ie the starting step size
  rho <- 0.5				# the multiplier for the step size at each iteration
  
  while((loglike(beta+alpha*p,y,X,m)) > loglike(beta,y,X,m) + c*alpha*t(gr)%*%p){
    alpha <- rho*alpha
  }
  return(alpha)
}

#--------------------------------------------------------------
# Quasi-Newton with Backtracking Line Search Algorithm
# Inputs:  conv: tolerance level for evaluating convergence
#	         a: step size

# Outputs: beta_hat: a vector of estimated beta coefficients
#	         iter: The number of iterations until convergence
#       	 converged: 1/0, depending on whether algorithm converged
#	         loglik: negative log-likelihood

quasi_newton <- function(X,y,m,maxiter=50000,conv=1E-10){
  
  converged <- 0		# Indicator for whether convergence met
  
  # 1. Initialize matrix to hold beta vector for each iteration
  betas <- matrix(0,nrow=maxiter+1,ncol=ncol(X)) 
  betas[1,] <- rep(0,ncol(X))	# Initialize beta vector to 0
  
  # 2. Initialize values for log likelihood
  loglik <- rep(0,maxiter) 	# Initialize vector to hold log likelihood
  loglik[1] <- loglike(betas[1,],y,X,m)
  
  # 3. Initialize matrix to hold gradients for each iteration					
  grad <- matrix(0,nrow=maxiter,ncol=ncol(X)) 		
  grad[1,] <- gradient(betas[1,],y,X,m)
  
  # 4. Initialize list of approximations of Hessian inverse B 
  B <- list()
  B[[1]] <- diag(ncol(betas)) # use identity matrix as initial value
  
  # 5. Perform gradient descent
  for (i in 2:maxiter){
    
    # Compute direction and step size for beta update
    p <- -B[[i-1]] %*% grad[i-1,]
    alpha <- line.search(betas[i-1,],y,X,grad[i-1,],p,m,maxalpha=1)
    
    # Update beta values based on step/direction
    betas[i,] <- betas[i-1,] + alpha*p
    
    # Calculate loglikelihood for each iteration
    loglik[i] <- loglike(betas[i,],y,X,m)
    
    # Calculate gradient for new betas
    grad[i,] <- gradient(betas[i,],y,X,m)
    
    # Update values needed for BFGS Hessian inverse approximation
    s <- alpha*p
    z <- grad[i,] - grad[i-1,]
    rho <- as.vector(1/(t(z) %*% s))	 # as.vector to make rho a scalar
    tau <- rho * s %*% t(z)
    I <- diag(ncol(grad))
    
    # BFGS formula for updating approx of H inverse
    B[[i]] <- (I-tau) %*% B[[i-1]] %*% (I-t(tau)) + rho * s %*% t(s) 
    
    # print(i)
    
    # Check if convergence met: If yes, exit loop
    if (abs(loglik[i]-loglik[i-1])/(abs(loglik[i-1])+conv) < conv ){
      converged=1;
      break;
    }
    
  } # End gradient descent iterations
  
  return(list(betas=betas[1:i,],beta_hat=betas[i,], iter=i, 
              converged=converged, loglik=loglik[1:i]))
}

#------------------------------------------------------------------
# Run gradient descent and view results
# 1. Fit glm model for comparison. (No intercept: already added to X.)
glm <- glm(y~X-1, family='binomial') # fits model, obtains beta values
beta <- glm$coefficients

# 2. Call Quasi-Newton function to estimate
output <- quasi_newton(X,y,m,maxiter=10000,conv=1E-10)

# 3. Eyeball values for accuracy & display convergence
round(beta,2)				# glm estimated beta values
round(output$beta_hat,2)	# gradient descent estimated beta values

# Check whether the algorithm has converged, and the number of iterations
if(output$converged>0){cat('Algorithm converged in',output$iter, 'iterations.')}
if(output$converged<1){cat('Algorithm did not converge. Ran for max iterations.')}

# 4. Plot the convergence of the beta variables compared to glm
par(mfrow=c(4,3))
par(mar=c(4,4,1,3))
for (j in 1:length(output$beta_hat)){
  plot(1:nrow(output$betas),output$betas[,j],type='l',
       xlab='iterations',ylab=paste('beta',j))
  abline(h=beta[j],col='red')
}

# 5. Plot log-likelihood function for convergence
par(mar=c(5.1,4.1,4.1,2.1))
par(mfrow=c(1,1))
plot(1:length(output$loglik),output$loglik,type='l',xlab='iterations',ylab='loglike',
     col='blue',log='xy',main='Quasi-Newton Negative Log Likelihood Function')
