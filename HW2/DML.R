#File: Machine_Learning_HW2.R
#Project: Double Machine Learning
#Author: Renee Li
#Date: 2024-02-28

# Setup -------------------------------------------------------------------
### CLEAN UP MY CONSOLE/ENVIRONMENT
cat("\014")
rm(list = ls())
# My working directory
setwd('/Users/renee/Desktop/lab2_Bingxue_Li') 
# Installs packages
list.of.packages <- c("caret", "ggplot2", "glmnet", "clusterGeneration", "mvtnorm")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos = "http://cran.us.r-project.org")
invisible(lapply(list.of.packages, library, character.only = TRUE))

# Data Generating Process ------------------------------------------------------
generate_data <- function(N=1000, k=50, true_beta=1) {
  b=1/(1:k)
  # = Generate covariance matrix of x = #
  sigma=genPositiveDefMat(k,"unifcorrmat")$Sigma
  sigma=cov2cor(sigma)
  
  x=rmvnorm(N,sigma=sigma) # Generate x
  g=as.vector(cos(x%*%b)^2) # Generate the function g
  m=as.vector(sin(x%*%b)+cos(x%*%b)) # Generate the function m
  d=m+rnorm(N) # Generate d
  y=true_beta*d+g+rnorm(N) # Generate y
  
  dgp = list(y=y, d=d, x=x)
  
  return(dgp)
}

#################################
## Part1: Lab Session Revision ##
#################################
# Simulation with OLS ----------------------------------------------------------
#generate 100 different datasets and estimate the coefficient of D with OLS
set.seed(123)
beta = c()
for (i in 1:30) {
  dgp = generate_data()
  lm <- lm(dgp$y ~ dgp$d + unlist(dgp$x))
  beta[i] <- coef(summary(lm))[2]
}
beta_df <- data.frame(beta=beta)
beta_ols <-mean(beta_df$beta)

# Simulation with LASSO --------------------------------------------------------
set.seed(123)
beta = c()
for (i in 1:30) {
  dgp = generate_data()
  lasso <- train(y = dgp$y,
                 x = data.frame(d=dgp$d, x=dgp$x),
                 method = "glmnet",
                 metric = "RMSE",
                 trControl = trainControl(method = "cv",
                                          number = 5),
                 tuneGrid = expand.grid(alpha = 1,
                                        lambda = seq(0.02, 2, by = 0.002))
  )
  beta = c(beta, coef(lasso$finalModel, lasso$bestTune$lambda)[2])
}
beta_df <- data.frame(beta=beta)
beta_lasso <- mean(beta_df$beta)

# Simulation with Post-LASSO ---------------------------------------------------
# Function to extract non-zero variables from lasso
set.seed(123)
lasso_nzero_vars <- function(lasso = lasso) {
  vars <- data.frame(as.matrix(coef(lasso$finalModel, lasso$bestTune$lambda))[-1,])
  colnames(vars)[1] <- "varname"
  nzero_vars <- rownames(vars)[vars$varname!=0]
  return(nzero_vars)
}
# Run 30 post-lasso regressions and store the beta value
beta = c()
for (i in 1:30) {
  dgp = generate_data()
  lasso <- train(y = dgp$y,
                 x = data.frame(d=dgp$d, x=dgp$x),
                 method = "glmnet",
                 metric = "RMSE",
                 trControl = trainControl(method = "cv",
                                          number = 5),
                 tuneGrid = expand.grid(alpha = 1,
                                        lambda = seq(0.02, 2, by = 0.002))
  )
  #Extract the nonzero vars from the lasso selected outcomes and store as a new df
  nzero_vars <- lasso_nzero_vars(lasso)
  covariates = data.frame(data.frame(d=dgp$d, x=dgp$x)[,colnames(data.frame(d=dgp$d, x=dgp$x)) %in% nzero_vars])
  y = dgp$y
  data = cbind(y,covariates)
  #Run Post-lasso-OLS with the new dataset
  lm_screening_lasso <- lm(y ~ ., data = data)
  # might drop variable d by lasso selection 
  if (names(coef(lm_screening_lasso)[2])=="d") { 
    beta = c(beta, coef(lm_screening_lasso)[2])
  } else {
    beta = c(beta, 0)
  }
}
beta_df <- data.frame(beta=beta)
beta_plols1 <- mean(beta_df$beta)

# Forcing LASSO to keep the variable of interest
set.seed(123)
beta = c()
for (i in 1:30) {
  dgp = generate_data()
  lasso <- train(y = dgp$y,
                 x = data.frame(d=dgp$d, x=dgp$x),
                 method = "glmnet",
                 metric = "RMSE",
                 trControl = trainControl(method = "cv",
                                          number = 5),
                 tuneGrid = expand.grid(alpha = 1,
                                        lambda = seq(0.02, 2, by = 0.002))
  )
  
  nzero_vars <- lasso_nzero_vars(lasso)
  covariates = data.frame(cbind(d=dgp$d), data.frame(x=dgp$x)[,colnames(data.frame(x=dgp$x)) %in% nzero_vars])
  y = dgp$y
  data = cbind(y, covariates)
  lm_screening_lasso_forced <- lm(y ~ ., data = data)
  beta = c(beta, coef(lm_screening_lasso_forced)[2])
  
}
beta_df <- data.frame(beta=beta)
beta_plols2 <- mean(beta_df$beta)

# Simulation with Post-Double-Selection-LASSO ----------------------------------------------------
set.seed(123)
beta = c()
for (i in 1:30) {
  print(i)
  dgp = generate_data()
  
  lasso1 <- train(y = dgp$y,
                  x = data.frame(d=dgp$d, x=dgp$x),
                  method = "glmnet",
                  metric = "RMSE",
                  trControl = trainControl(method = "cv",
                                           number = 5),
                  tuneGrid = expand.grid(alpha = 1,
                                         lambda = seq(0.02, 2, by = 0.002))
  )
  
  nzero_vars <- lasso_nzero_vars(lasso1)
  nzero_vars1 <- nzero_vars[nzero_vars != "d"]
  #nzero_vars1 selected the xs that gives a consistent conditional mean prediction for y 
  lasso2 <- train(y = dgp$d,
                  x = data.frame(x=dgp$x),
                  method = "glmnet",
                  metric = "RMSE",
                  trControl = trainControl(method = "cv",
                                           number = 5),
                  tuneGrid = expand.grid(alpha = 1,
                                         lambda = seq(0.02, 2, by = 0.002))
  )
  nzero_vars2 <- lasso_nzero_vars(lasso2)
  #nzero_vars2 selected the xs that gives a consistent conditional mean prediction for d  
  covariates <- data.frame(cbind(d=dgp$d), data.frame(x=dgp$x)[,colnames(data.frame(x=dgp$x)) %in% c(nzero_vars1,nzero_vars2)])
  y = dgp$y
  data = cbind(y, covariates)
  #create a new df with the union of double lasso-selected x and d
  post_double_selection <- lm(y ~ ., data = data)
  #estimate the double selection post lasso
  beta = c(beta, coef(post_double_selection)[2])
}
beta_df <- data.frame(beta=beta)
beta_pdbslasso <- mean(beta_df$beta)

# Naive Double Machine Learning with FWL ---------------------------------------------------- 
set.seed(123)
library("randomForest")
# Define function to follow the FW intuition drawing on ML algorithms
fwML <- function(X, d, y) {
  # STEP 1: Estimate E[Y|X]
  modely = randomForest(X, y)
  y_hat = predict(modely,X)
  U_hat = y-y_hat
  # STEP 2: Estimate E[D|X]
  modeld = randomForest(X, d)
  d_hat = predict(modeld,X)
  V_hat = d-d_hat
  # Compute Naive ML
  fw_ML = lm(U_hat~V_hat)
  beta_ml = coef(fw_ML)[2]
  return(beta_ml)
}
beta = c()
for (i in 1:30) {
  dgp = generate_data()
  beta_ml = fwML(X=dgp$x, d=dgp$d, y=dgp$y)
  beta = c(beta, beta_ml)
}
beta_df <- data.frame(beta=beta)
beta_fwml <- mean(beta_df$beta)

# Double Machine Learning ---------------------------------------------------- 
set.seed(123)
doubleml <- function(X, d, y) {
  # STEP 1 : Split sample
  I=sort(sample(1:length(dgp$y),length(dgp$y)/2))
  IC=setdiff(1:length(dgp$y),I)
  # STEP 2 : Train models for E[Y|X] (and E[D|X]) on set 1 and predict Y (abd D) on set 2 using this model - and vice versa
  modely1 = randomForest(X[IC,], y[IC], maxnodes = 10)
  modely2 = randomForest(X[I,], y[I], maxnodes = 10)
  y_hat1 = predict(modely1,X[I,])
  y_hat2 = predict(modely2,X[IC,])
  modeld1 = randomForest(X[IC,], d[IC], maxnodes = 10)
  modeld2 = randomForest(X[I,], d[I], maxnodes = 10)
  d_hat1 = predict(modeld1,X[I,])
  d_hat2 = predict(modeld2,X[IC,])
  # STEP 3 : Get the residuals
  U_hat1 = y[I]-y_hat1
  U_hat2 = y[IC]-y_hat2
  V_hat1 = d[I]-d_hat1
  V_hat2 = d[IC]-d_hat2
  # STEP 4: Compute cross-fitted DML beta
  U_hat = c(U_hat1,U_hat2)
  V_hat = c(V_hat1,V_hat2)
  beta1 = mean(V_hat1*U_hat1)/mean(V_hat1*d[I])
  beta2 = mean(V_hat2*U_hat2)/mean(V_hat2*d[IC])
  beta = mean(c(beta1,beta2))
  # STEP 5: Compute standard errors. These are the usual OLS standard errors in the regression res_y = res_d*beta + eps. 
  psi_stack = c((U_hat1 - V_hat1*beta), (U_hat2 - V_hat2*beta))
  res_stack = c(V_hat1, V_hat2)
  se = sqrt(mean(res_stack^2)^(-2)*mean(res_stack^2*psi_stack^2))/sqrt(length(dgp$y))
  return(c(beta,se))
}
beta = c()
se = c()
for (i in 1:30) {
  tryCatch({
    print(i)
    dgp = generate_data()
    DML = doubleml(d=dgp$d, X=dgp$x, y=dgp$y)
    beta = c(beta, DML[1])
    se = c(se, DML[2])
  }, error = function(e) {
    print(paste("Error for", i))
  })
}
beta_df <- data.frame(beta=beta)
beta_dml <- mean(beta_df$beta)
se_df <- data.frame(se=se)
se_dml <- mean(se_df$se)
beta_ols
beta_lasso
beta_plols1
beta_pdbslasso
beta_fwml
beta_dml 

###################################
## Part2: Extension of Double ML ##
###################################
# Installs packages if not already installed, then loads packages 
list.of.packages <- c("caret", "ggplot2", "glmnet", "clusterGeneration", "mvtnorm")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos = "http://cran.us.r-project.org")

invisible(lapply(list.of.packages, library, character.only = TRUE))

source('Bingxue_Li_k2ml.R') #### Imports the function you wrote in Part 2

### generate_data() is the data generating process from class ### 
generate_data <- function(N=1000, k=50, true_beta=1) {
  
  b=1/(1:k)
  
  # = Generate covariance matrix of x = #
  sigma=genPositiveDefMat(k,"unifcorrmat")$Sigma
  sigma=cov2cor(sigma)
  
  x=rmvnorm(N,sigma=sigma) # Generate x
  g=as.vector(cos(x%*%b)^2) # Generate the function g
  m=as.vector(sin(x%*%b)+cos(x%*%b)) # Generate the function m
  d=m+rnorm(N) # Generate d
  y=true_beta*d+g+rnorm(N) # Generate y
  
  dgp = list(y=y, d=d, x=x)
  
  return(dgp)
}

# Specify for example random forests with random selection of 3 variables as your ML algorithm
results <- k2ml(X=dgp$x, d=dgp$d, y=dgp$y, K=5,
                method_y = "rf",
                method_d = "rf",
                tunegrid_y = expand.grid(.mtry = 3),
                tunegrid_d = expand.grid(.mtry = 3)
)

print(c('beta estimate is :', results[1])) 
print(c('SE estimate is :', results[2])) 

method_y = "rf"
method_d = "rf"
tgrid_y=expand.grid(.mtry = 3)
tgrid_d=expand.grid(.mtry = 3)
beta = c()
se = c()
for (i in 1:5) {
  tryCatch({
    print(i)
    dgp = generate_data()
    KML = k2ml(X=dgp$x,
               d=dgp$d, 
               y=dgp$y, 
               K=5, 
               method_y= method_y, 
               method_d= method_y, 
               tunegrid_y =tgrid_y, 
               tunegrid_d =tgrid_d)
    beta = c(beta, KML[1])
    se = c(se, KML[2])
  }, error = function(e) {
    print(paste("Error for", i))
  })
}
beta_df <- data.frame(beta=beta)
beta_k2ml <- mean(beta_df$beta)
se_df <- data.frame(se=se)
se_k2ml <- mean(se_df$se)

###########################
## Part3: An Application ##
###########################
set.seed(123)
st <- read.csv("social_turnout.csv")
#d-variable: treat_neighbors
#y-variable: outcome_voted
D <- st$treat_neighbors
Y <- st$outcome_voted
X <- st[, setdiff(names(st), c("treat_neighbors","outcome_voted"))]
library(caret)
library(glmnet)
source('Bingxue_Li_k2ml.R') #### Imports the function you wrote in Part 2
result_OLS <- lm(outcome_voted~treat_neighbors+. ,data = st)

result1 <- k2ml(X, as.factor(D), as.factor(Y), 
                k = 2,
                method_y = "glmnet",
                method_d = "glmnet",
                tunegrid_y = expand.grid(alpha = seq(0, 1, by = 0.3),#alpha=1: ridge
                                         lambda = seq(0.001, 0.1, by = 0.03)),
                tunegrid_d = expand.grid(alpha = seq(0, 1, by = 0.3),#alpha=1: ridge
                                         lambda = seq(0.001, 0.1, by = 0.03)) )

result2 <- k2ml(X,as.factor(D),as.factor(Y),
                k = 2,
                method_y = "rf",
                method_d = "rf",
                tunegrid_y = expand.grid(.mtry = 7),
                tunegrid_d = expand.grid(.mtry = 7)) 


result3 <- k2ml(X,D,Y,
                K = 2,
                method_y = "glmneț",
                method_d = "glmnet",
                tunegrid_y = expand.grid(alpha = seq(0, 1, by = 0.3),#alpha=1: ridge
                                         lambda = seq(0.001, 0.1, by = 0.03)),
                tunegrid_d = expand.grid(alpha = seq(0, 1, by = 0.3),#alpha=1: ridge
                                         lambda = seq(0.001, 0.1, by = 0.03)))

result4 <- k2ml(X,D,Y,
                k = 2,
                method_y = "rf",
                method_d = "rf",
                tunegrid_y = expand.grid(.mtry = 7),
                tunegrid_d = expand.grid(.mtry = 7)) 

result5 <- k2ml(X,D,Y,
                k = 4,
                method_y = "rf",
                method_d = "rf",
                tunegrid_y = expand.grid(.mtry = 7),
                tunegrid_d = expand.grid(.mtry = 7)) 
