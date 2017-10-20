### SDS 385 - Exercises 03 - Backtracking Line Search
# Some of the code is based on the R code of Jennifer Starling
# Jennifer Starling: https://github.com/jestarling/bigdata/tree/master/Exercise-03

# This code implements gradient descent to estimate the 
# beta coefficients for binomial logistic regression.
# It uses backtracking line search to calculate the step size.

# Yanxin Li   Oct 2 2017

rm(list=ls())
library(Matrix)
set.seed(1) # Reproducibility

#-----------------------------------------------------------------
# Read in data
wdbc <- read.csv("https://raw.githubusercontent.com/jgscott/SDS385/master/data/wdbc.csv",
                 header = FALSE)
y = wdbc[,2]

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

#---------------------------------------------------------------
# Function for calculating Euclidean norm of a vector.
norm_vec <- function(x) sqrt(sum(x^2)) 

#---------------------------------------------------------------
# Line Search Function
# Input:  X: design matrix
#		      y: vector of 1/0 response values
#		      beta: vector of betas
# 		    gr: gradient for beta vector
#         m: sample size vector m
#  	      maxalpha: the maximum allowed step size
# Output: alpha: the multiple of the search direction
linesearch <- function(beta, y, X, gr, m, maxalpha=1){
  c <- 0.001			# a constant, in (0,1)
  alpha <- maxalpha	# the max step size, ie the starting step size
  p <- 0.5				# the multiplier for the step size at each iteration
  
  # Update alpha while line search condition holds
  while((loglike(beta - alpha*gr,y,X,m))>loglike(beta,y,X,m)-c*alpha*norm_vec(gr)^2){
    alpha <- p*alpha
  }
  return(alpha)
}

#------------------------------------------------------------------
# Gradient Descent Algorithm:
# Input:  X: n x p design matrix
#	        y: response vector length n
#         conv: Tolerance level for determining convergence
#	        a: Step size
# Output: beta_hat: A vector of estimated beta coefficients
#	        iter: The number of iterations until convergence
#	        converged: 1/0, depending on whether algorithm converged
#	        loglik: Log-likelihood function

gradient_descent <- function(X,y,m,maxiter=50000,conv=1E-10){
  
  # 1. Initialize matrix to hold beta vector for each iteration
  betas <- matrix(0,nrow=maxiter+1,ncol=ncol(X)) 
  betas[1,] <- rep(0, ncol(X))	# Initialize beta vector to 0 to start
  
  # 2. Initialize values for log-likelihood
  loglik <- rep(0,maxiter) 	#Initialize vector to hold loglikelihood function
  loglik[1] <- loglike(betas[1,], y, X, m)
  
  # 3. Initialize matrix to hold gradients for each iteration					
  grad <- matrix(0,nrow=maxiter,ncol=ncol(X)) 		
  grad[1,] <- gradient(betas[1,], y, X, m)
  
  converged <- 0		# Indicator for whether convergence met.
  iter <- 1			# Counter to track iterations for function output.
  
  # Perform gradient descent
  for (i in 2:maxiter){
    
    # Backtracking line search to calculate step size
    step <- linesearch(beta=betas[i-1,],y,X,gr=grad[i-1,],m,maxalpha=1)
    
    # Set new beta equal to beta - step*gradient(beta).
    betas[i,] <- betas[i-1,] - step * grad[i-1,]
    
    # Calculate loglikelihood for each iteration.
    loglik[i] <- loglike(betas[i,],y,X,m)
    
    # Calculate gradient for beta.
    grad[i,] <- gradient(betas[i,],y,X,m)
    
    iter <- i	# Track iterations
    
    # check if convergence met: If yes, exit loop
    if (abs(loglik[i]-loglik[i-1])/(abs(loglik[i-1])+conv) < conv ){
      converged=1;
      break;
    }
    
  } # End gradient descent iterations
  
  return(list(beta_hat=betas[i,],iter=iter,converged=converged,loglik=loglik[1:i]))
}

#------------------------------------------------------------------
# Run gradient descent and view results
# 1. Fit glm model for comparison (No intercept: already added to X)
glm1 <- glm(y ~ X-1, family='binomial') # Fits model, obtains beta values
beta <- glm1$coefficients

# 2. Call gradient descent function to estimate beta_hat
beta_hat <- gradient_descent(X,y,m,maxiter=50000,conv=1E-10)

# 3. Eyeball values for accuracy & display convergence.
round(beta,2)				      # glm estimated beta values
round(beta_hat$beta_hat,2)	# Gradient descent estimated beta values

print(c("Algorithm converged?",beta_hat$converged,"(1=converged,0=did not converge)"))
print(beta_hat$iter)

# 4. Plot log-likelihood function for convergence
plot(1:length(beta_hat$loglik),beta_hat$loglik,type='l',xlab='iterations',
     col='blue',log='xy',main='Gradient Descent w Bt Line Search Log Likelihood')
