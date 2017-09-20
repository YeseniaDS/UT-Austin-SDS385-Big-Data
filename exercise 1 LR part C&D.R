# SDS385 Exercise 1 Linear Regression: Part C
# compare various matrix decomposition methods with inversion method
# and benchmarks peformance of the Cholesky and LU vesus inversion

library ( Matrix ) 
library ( microbenchmark ) # For benchmarking

#----------------------------------------------------------------
# Inversion method function
inv_method <- function (X,W,y){
  beta_hat_inv <- solve (t(X) %*% W %*% X) %*% t(X) %*% W %*% y
  return (beta_hat_inv)
}

#----------------------------------------------------------------
# Cholesky decomposition
chol_method <- function (X,W,y){
  A <- t(X) %*% (X*diag (W))   # Efficient way of A = t(X) %*% W %*% X as W is diagonal
  b <- t(X) %*% (y*diag (W))   # Avoid multiply by 0's
  
  # Cholesky decomposition of A
  U <- chol(A)  # Upper Cholesky decomposition of A and U'U = A
  
  # Replace Ax=b with U'Ux=b, solve U'z=b for z
  z <- solve(t(U))%*%b
  # Solve Ux = z for x (x=beta_hat)
  beta_hat_chol <- solve(U) %*% z
  
  return(beta_hat_chol)
}

#----------------------------------------------------------------
# LU method function
lu_method <- function (X,W,y){
  A <- t(X) %*% (X*diag (W))   # Efficient way of A = t(X) %*% W %*% X as W is diagonal
  b <- t(X) %*% (y*diag (W))   # Avoid multiply by 0's
  
  # LU decomposition of A
  decomp <- lu(A)
  L <- expand ( decomp )$L  # Upper triangular matrix
  U <- expand ( decomp )$U  # Lower triangular matrix
  
  # Replace Ax=b with LUx=b, solve Lz=b for z
  z <- solve (L) %*% b
  # Solve Ux=z for x (x=beta_hat)
  beta_hat_lu <- solve(U) %*% z
  
  return(beta_hat_lu)
}

#----------------------------------------------------------------
# Simulate data from the linear model for a range of values of N and P
# Assume weights w_i are all 1, data are Gaussian

N <- c (20, 100, 400, 1200)
P <- N/4 # N>P

res <- list () # Performance results
for (i in 1: length (N)){
  n <- N[i]
  p <- P[i]
  print (n)
  
  # Set up matrices of size N, P parameters: (dummy data)
  X <- matrix (rnorm (n*p),nrow =n, ncol =p)
  y <- rnorm (n)
  W <- diag (1, nrow =n)
  
  # Perform benchmarking:
  res[[i]] <- microbenchmark (
    inv_method (X,W,y),
    lu_method (X,W,y),
    chol_method (X,W,y), 
    unit ='ms'
  )
}
names(res) <- (c('N=20, P=5', 'N=100, P=25', 'N=400, P=100', 'N=1200, P=300'))
res # Display benchmarking results

#--------------------------------------------------------------
# SDS385 Exercise 1 Linear Regression: Part D
# Benchmark the inversion method, Cholesky method, LU method,
# and the sparse method across some different scenarios (including different 
# sparsity levels in X (0.01, 0.05, 0.25)

#--------------------------------------------------------------
# Sparse Cholesky factorization
sparse_method <- function(X,W,y) {
  
  X <- Matrix(X, sparse = T)
  A <- t(X) %*% (X*diag (W))   # Efficient way of A = t(X) %*% W %*% X as W is diagonal
  b <- t(X) %*% (y*diag (W))   # Avoid multiply by 0's
  
  # Cholesky decomposition of A
  U <- chol(A)  # Upper Cholesky decomposition of A and U'U = A
  
  # Replace Ax=b with U'Ux=b, solve U'z=b for z
  z <- forwardsolve(t(U), b)
  # Solve Ux = z for x (x=beta_hat_sparse)
  beta_hat_sparse <- backsolve(U, z)
  
  return(beta_hat_sparse)
}

#--------------------------------------------------------------
# Set different sparsity: 0.01, 0.05, 0.25
theta <- c(0.01, 0.05, 0.25)

results <- list () # Performance results
for (i in 1: length (theta)){
  
  N <- 2000
  P <- 1000
  
  X <- matrix(rnorm(N * P), nrow = N)
  mask <- matrix(rbinom(N * P, 1, theta), nrow = N)
  X <- mask * X
  y <- rnorm (N)
  W <- diag(rep(1, N))
  
  # Perform benchmarking:
  results[[i]] <- microbenchmark (
    inv_method (X,W,y),
    lu_method (X,W,y),
    chol_method (X,W,y),
    sparse_method (X,W,y),
    times=10)
}
names(results) <- (c('theta=0.01', 'theta=0.05', 'theta=0.25'))
results # Display benchmarking results

#END
