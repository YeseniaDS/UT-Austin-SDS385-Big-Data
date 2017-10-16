### SDS385 Exercise 01 Generalized Linear Models: Part B
# This code implements gradient descent to estimate the
# beta coefficients(MLE) for binomial logistic regression.

#-------------------------------------------------------------
library(Matrix)
# Read in data
wdbc <- read.csv("D:/2017 UT Austin/Statistical Models for Big Data/R/wdbc.csv", header = FALSE)

X <- as.matrix (wdbc [, 3:12]) # Select first 10 features to keep and scale features
X <- scale (X) # Normalize design matrix features
X <- cbind (rep (1, nrow (X)), X)

y <- wdbc [, 2] 
Y <- rep(0,length(y)); Y[y=='M'] <- 1 # Convert y values to 1/0's, response vector

beta <- as.matrix (rep (0, ncol (X)))
m <- rep(1, nrow (X)) # Number of trials is 1

#---------------------------------------------------------------
# Compute Sigmoid function wi
sigmoid <- function(z){
  w <- 1 / (1 + exp(-z)) 
  return(w)
}

#---------------------------------------------------------------
# Function for computing likelihood, which handles the case that -X^T * beta is huge

loglik <- function(X, y, beta, m) {
  w <- sigmoid(X %*% beta)
  loglik <- - sum(y * log(w+1E-5) + (m-y) * log (1-w+1E-5)) 
  # Adding a constant to resolve issues with probabilities near 0 or 1.
  return(loglik)
}

#----------------------------------------------------------------
# Function for computing gradient for likelihood
   # Input: design matrix X, response vector Y
   #        Coefficient matrix beta, sample size (all 1's) vector m
   # Output: Returns value of gradient function for binomial logistic fuction

gradient <- function(X, y, beta, m){
  w <- sigmoid(X %*% beta) # Probabilities vector
  gradient <- array(NA, dim = length(beta)) # Initialize the gradient
  gradient <- apply(X * as.numeric(m * w - y), 2, sum)
  return(gradient)
}

#----------------------------------------------------------------
# Gradient Descent Algorithm
stepsize <- 0.01
n.steps <- 50000
epsi <- 1E-10 # Level for determining convergence
converged <- 0 # 1/0, depending on whether algorithm converged

# Initialize vector to hold loglikelihood function
log.lik <- rep(NULL, n.steps) 
# Initialize values for first iteration
log.lik[1] <- loglik(X, y, beta, m) 

# Initialize matrix to hold gradients for each iteration
grad <- matrix (0, nrow = n.steps, ncol = ncol (X))
# Initialize values for first iteration
grad[1,] <- gradient (X, y, beta, m)

for (step in 2:n.steps) {
  beta <- beta - stepsize * gradient(X, y, beta, m)
  
  # Calculate log liklihood for each iteration
  log.lik[step] <- loglik(X, y, beta, m)
  
  # Calculate gradient for beta
  grad[step, ] <- gradient(X, y, beta, m)
  
  # Check if convergence met: If yes, exit loop
  if (abs(log.lik[step] - log.lik[step-1]) / (abs(log.lik[step-1]) + 1E-3) < epsi){
    converged = 1;
    break ;
  }
}

#-------------------------------------------------------------
# view the result
beta

# Fit glm model for comparison (No intercept: already added to X)
fit <- glm(y ~ X[, c(-1)], family = "binomial") # Fits model, obtains beta values.
summary(fit)
beta.glm <- fit$coefficients

# Create trace plot of likelihood, check for convergence again
plot (1:length(log.lik), log.lik, type ="l", ylab = "log likelihood(beta)(log scale)",
      xlab ="iterations",log ="xy")

#----------------------------------------------------------------
#Newton's Methods
beta.nt <- as.matrix(rep(0, ncol(X)))
nsteps <- 10

# Initialize values
log_lik <- rep(NULL, nsteps)
log_lik[1] <- loglik(X, y, beta.nt, m)

for (step in 2:nsteps) {
  w <- as.numeric(sigmoid(X %*% beta.nt))
  Hessian <- t(X) %*% diag(w*(1-w)) %*% X # Compute Hessian matrix
  beta.nt <- beta.nt - solve(Hessian) %*% gradient(X, y, beta.nt, m)
  log_lik[step] <- loglik(X, y, beta.nt, m)
  
  # Check if convergence met: If yes, exit loop
  if (abs(log_lik[step] - log_lik[step-1]) / (abs(log_lik[step-1]) + 1E-3) < epsi){
    converged = 1;
    break ;
  }
}

# Show estimates from Newton's method
beta.nt
log_lik

# Create trace plot of likelihood, check for convergence
plot (1:length(log_lik), log_lik, type ="l", ylab = "log likelihood(beta) using Newton's Method",
      xlab ="iterations")

#END