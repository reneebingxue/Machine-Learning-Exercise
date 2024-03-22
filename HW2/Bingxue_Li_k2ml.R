library(caret)

k2ml <- function(X, d, y, K,
                 method_y, 
                 method_d, 
                 tunegrid_y, 
                 tunegrid_d) 
  {
  X <- as.data.frame(X)

  # STEP 1 : Split sample
  # Initialize lists to store the k-fold splits
  I_list <- list()
  IC_list <- list()
  # create k-fold splits
  I_list <- createFolds(1:length(y), K)
  for (i in 1:K) {
    IC_list[[i]] <- setdiff(1:length(y), I_list[[i]])
  }
  
  # Initialize lists to store models and predictions
  modely_list <- list()
  y_hat_list <- list()
  modeld_list <- list()
  d_hat_list <- list()
  
  # Initialize lists to store residuals
  U_hat_list <- list()
  V_hat_list <- list()
  
  # Initialize list to store beta values
  beta_list <- list()
  psi_stack <- c()
  res_stack <- c()
  
  # TrainContorl
  trainControl <- trainControl(method = "cv",           
                               number = 5,              
                               savePredictions = "final") # Only prediction from the last iteration are saved
  
  # Loop over each split
  for (i in 1:K) {

    # STEP 2 : Train models for E[Y|X] on set i and predict Y on the complementary set using this model
    modely <- train(y= y[IC_list[[i]]], 
                    x= X[IC_list[[i]], ],
                    method = method_y, 
                    tuneGrid = tunegrid_y,
                    trContrl = trainControl,
                    allowParallel = T)
    y_hat <- predict(modely, X[I_list[[i]], ])
    
    # STEP 2 : Train models for E[D|X] on set i and predict D on the complementary set using this model
    modeld <- train(y= d[IC_list[[i]]],
                    x= X[IC_list[[i]], ], 
                    method = method_d, 
                    tuneGrid = tunegrid_d,
                    trContrl = trainControl,
                    allowParallel = T)
    d_hat <- predict(modeld, X[I_list[[i]], ])
    
    # STEP 3: Get the residuals
    U_hat <- as.numeric(y[I_list[[i]]]) - as.numeric(y_hat)
    V_hat <- as.numeric(d[I_list[[i]]]) - as.numeric(d_hat)
    
    # STEP 4: Compute cross-fitted DML beta
    beta <- mean(V_hat * U_hat) / mean(V_hat * V_hat)
    
    # STEP 5: Store models, predictions, residuals, and beta values for this split
    modely_list[[i]] <- modely
    y_hat_list[[i]] <- y_hat
    modeld_list[[i]] <- modeld
    d_hat_list[[i]] <- d_hat
    U_hat_list[[i]] <- U_hat
    V_hat_list[[i]] <- V_hat
    beta_list[[i]] <- beta
  }
  beta <- mean(unlist(beta_list))

  # STEP 6: Compute standard errors. These are the usual OLS standard errors in the regression res_y = res_d*beta + eps.
  # Combine U_hat and V_hat lists across all folds
  for (i in 1:K) {
    psi_stack <- c(psi_stack, (U_hat_list[[i]] - V_hat_list[[i]] * beta))
    res_stack <- c(res_stack, V_hat_list[[i]])
  }
  se = sqrt(mean(res_stack^2)^(-2)*mean(res_stack^2*psi_stack^2))/sqrt(length(y))
  
  return(c(beta,se))
}
