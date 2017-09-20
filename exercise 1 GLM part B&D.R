### SDS385 Exercise 1 Generalized Linear Models: Part B
# This code implements gradient descent to estimate the
# beta coefficients for binomial logistic regression.

#-------------------------------------------------------------
library (Matrix)
rm(list =ls())

# Read in data
wdbc <- read.csv("D:/2017 UT Austin/Statistical Models for Big Data/R/wdbc.csv", header = FALSE)
y <- wdbc[ ,2]

# Convert y values to 1/0's
# Select first 10 features to keep and scale features
X <- as.matrix ( wdbc [, 3:12])
X <- scale (X) #Normalize design matrix features
X <- cbind (rep (1, nrow (X)), X)
y <- wdbc [, 2]
y <- y == "M"
beta <- as.matrix (rep (0, ncol (X)))
mi <- 1

#-------------------------------------------------------------
# Function for computing w.i
comp.wi <- function (X, beta ) {
  wi <- 1 / (1 + exp (-X %*% beta ))
  return (wi)
}

#-------------------------------------------------------------
# Binomial Negative Loglikelihood function
     # Inputs: Design matrix X, vector of 1/0 Y
     # Coefficient matrix beta, sample size vector m

     # Output: Return value of negative log????likelihood
     # function for binomial logistic regression.
# Function for computing likelihood, which handles the case that -X^T * beta is huge
loglik <- function (beta , y, X, mi) {
  XtBeta <- -X %*% beta
  if ( max ( XtBeta ) > 700) {
    loglik <- apply (( mi - y) * (X %*% beta )+ mi*XtBeta , 2, sum )
  }
  else {
    loglik <- apply (( mi - y) * (X %*% beta )+ mi*log (1 + exp ( XtBeta )), 2, sum )
  }
  return ( loglik )
}

#------------------------------------------------------------
# Gradient Function:
# Inputs: Design matrix X, vector of 1/0 vals Y,
# coefficient matrix beta, sample size vector m.
# Output: Returns value of gradient function for binomial
# logistic regression.
# Function for computing gradient for likelihood
grad.loglik <- function (beta , y, X, mi ){
  grad <- array (NA, dim = length ( beta ))
  wi <- comp.wi(X, beta )
  grad <- apply (X*as.numeric (mi * wi - y), 2, sum )
  return (grad)
  }

# GRADIENT DESCENT
stepfactor <- 0.025
n.steps <- 50000
log.lik <- rep (NULL , n.steps + 1)
log.lik [1] <- loglik (beta, y, X, mi)


for ( step in 1:n.steps ) {
  beta <- beta - stepfactor * grad.loglik (beta , y, X, mi)
  log.lik[ step + 1] <- loglik (beta , y, X, mi)
}

# Create trace plot of likelihood, check for convergence
plot (log.lik,
      main = " Trace Plot for log likelihood (beta) using gradient descent ",
      xlab = " Step (log scale)",
      ylab = "log likelihood (beta) (log scale)",
      log = "xy",
      pch = 20)

# Compare results to R's glm function
glm.model <- glm (y ~ X[, c( -1)] , family = "binomial")
summary (glm.model)
print ( beta )

# Newton's Method
beta.N <- as.matrix (rep (0, ncol (X)))
n.steps2 <- 10
log.lik2 <- rep (NULL , n.steps + 1)
log.lik2 [1] <- loglik ( beta.N, y, X, mi)
for (step in 1:n.steps2 ) {
  w.i <- as.numeric ( comp.wi(X, beta.N))
  Hessian <- t(X) %*% diag (w.i*(1-w.i)) %*% X
  beta.N <- beta.N - solve ( Hessian ) %*% grad.loglik ( beta.N, y, X, mi)
  log.lik2 [ step + 1] <- loglik ( beta.N, y, X, mi)
}

plot (log.lik2,
      main = "Trace plot for log likelihood ( beta ) using Newton 's method ",
      xlab = "Step",
      ylab = "log likelihood (beta)",
      pch = 20)

# Show estimates from Newton's method
print( beta.N)


