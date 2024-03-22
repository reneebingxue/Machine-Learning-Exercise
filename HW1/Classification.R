#File: Machine_Learning_Lab1.R
#Project: Assignment 1 for Machine_Learning
#Author: Bingxue Li
#Date: 2024-02-06

#################
##### SETUP #####
#################
cat("\014")
rm(list = ls())

# My working directory
setwd('/Users/renee/Desktop/lab1_Bingxue_Li/Data') 
# Load the packages
list.of.packages <- c("tidyverse", "stopwords", "tidytext", "scales", "reshape2", "glmnet",
                      "nnls", "quadprog", "caret", "ggplot2", "rpart", "randomForest", "kknn")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos = "http://cran.us.r-project.org")
invisible(lapply(list.of.packages, library, character.only = TRUE))
select <- dplyr::select
library(caret)
#####################
##### READ DATA #####
#####################
train <- read_csv("train.csv") %>% mutate(id = row_number()) 
test <- read_csv("test.csv") %>% mutate(id = row_number())

#######################
##### CLEAN WORDS #####
#######################
## in this part we clean the words in the headlines and create a dataframe in long form (one line per headline id and per word)
### get rid of stopwords 
stwds = stopwords::stopwords("en", source = "stopwords-iso") 
stwds

replace_reg <- "http[s]?://[A-Za-z\\d/\\.]+|&amp;|&lt;|&gt;"
unnest_reg  <- "([^A-Za-z_\\d#@']|'(?![A-Za-z_\\d#@]))"

tidy_headlines_train <- train %>%
  mutate(headline = str_replace_all(headline, replace_reg, "")) %>%
  unnest_tokens(
    word, headline, token = "regex", pattern = unnest_reg) %>%
  filter(!word %in% stwds, str_detect(word, "[a-z]"))

tidy_headlines_test <- test %>%
  mutate(headline = str_replace_all(headline, replace_reg, "")) %>%
  unnest_tokens(
    word, headline, token = "regex", pattern = unnest_reg) %>%
  filter(!word %in% stwds, str_detect(word, "[a-z]"))

#############################
##### GET TOP 500 WORDS #####
#############################
## in this part we want to create a dataframe with a list of headline ids and dummy variables for the top 500 words
# list of words in train that are also in test
list_of_words = tidy_headlines_train$word[tidy_headlines_train$word %in% tidy_headlines_test$word]
# list of words dataframe: collapse and counting numbers
freq_words <- data.frame(word=list_of_words) %>%
  group_by(word) %>%
  mutate(n = n()) %>%
  unique() %>%
  ungroup() %>%
  arrange(-n) %>%
  .[1:500,] %>% # top 500 words
  mutate(word=as.character(word))
# words appearance in training data
tidy_headlines_topwords_train <- tidy_headlines_train %>% 
  mutate(topwords = ifelse(word %in% freq_words$word, 1,0)) %>%
  mutate(word = ifelse(topwords==1, word, "no_top_word")) %>%
  unique() %>%
  group_by(id) %>%
  mutate(notopwords = 1-max(topwords)) %>%
  ungroup() %>%
  filter(!(word=="no_top_word" & notopwords==0)) %>%
  select(-topwords, -notopwords) %>%
  unique() %>%
  dcast(id+is_sarcastic~word, function(x) 1, fill = 0) 
# save the dataset cleaned
#saveRDS(tidy_headlines_topwords_train, "train_top500_words.rds")
tidy_headlines_topwords_train <- readRDS("train_top500_words.rds")
head(tidy_headlines_topwords_train)
# repeat the same procedure for the test dataset
tidy_headlines_topwords_test <- tidy_headlines_test %>% 
  mutate(topwords = ifelse(word %in% freq_words$word, 1,0)) %>%
  mutate(word = ifelse(topwords==1, word, "no_top_word")) %>%
  unique() %>%
  group_by(id) %>%
  mutate(notopwords = 1-max(topwords)) %>%
  ungroup() %>%
  filter(!(word=="no_top_word" & notopwords==0)) %>%
  select(-topwords, -notopwords) %>%
  unique() %>%
  dcast(id~word, function(x) 1, fill = 0) 
saveRDS(tidy_headlines_topwords_test, "test_top500_words.rds")
head(tidy_headlines_topwords_test)

###############################
##### GET WORD EMBEDDINGS #####
###############################
## in this part we want to get the word embeddings in 300 dimensional space for all the words in the headlines
w2v <- read_lines("GoogleNews-vectors-negative300-SLIM.txt")
# from https://github.com/eyaler/word2vec-slim/tree/master
words_embeddings_w2v <- regmatches(w2v, regexpr(" ", w2v), invert = TRUE) %>% unlist(.)
words_w2v = words_embeddings_w2v[c(TRUE, FALSE)]
embeddings_w2v = words_embeddings_w2v[c(FALSE, TRUE)]
list_of_words_test_train = c(tidy_headlines_train$word, tidy_headlines_test$word) %>% .[!duplicated(.)]
w2v_mat = matrix(nrow = length(list_of_words_test_train), ncol = 300)
for (i in 1:length(list_of_words_test_train)){
  j = which(words_w2v %in% list_of_words_test_train[i])
  tryCatch({
    w2v_mat[i,] = as.numeric(unlist(strsplit(embeddings_w2v[j], " ")))
  }, error = function(e) {
    print(paste("Cannot find ", list_of_words_test_train[i]))
  })
}
word2vec_df = data.frame(w2v_mat)
word2vec_df = cbind(word= list_of_words_test_train, word2vec_df)
word2vec_df <- word2vec_df[complete.cases(word2vec_df),]
head(word2vec_df)
#saveRDS(word2vec_df, "word2vec_df.rds")
word2vec_df <- readRDS("word2vec_df.rds")

################################
##### TIDY WORD EMBEDDINGS #####
################################
## in this part we want to create a dataframe with a list of headline ids and word embeddings. We also add dummy variables for top words that are not embedded
## we take the average of the word embeddings for each headline
### train ### 
tidy_w2v_train_1 <- tidy_headlines_train %>% # gets top 500 words not in w2v
  filter(word %in% freq_words$word & !(word %in% word2vec_df$word)) %>%
  unique() %>%
  dcast(id+is_sarcastic~word, function(x) 1, fill = 0) %>%
  select(-is_sarcastic)

tidy_w2v_train_2 <- tidy_headlines_train %>% # gets word embeddings
  left_join(word2vec_df, by ="word") %>%
  filter(complete.cases(.)) %>%
  select(-word, -is_sarcastic) %>%
  group_by(id) %>%
  summarise_all(list(mean)) %>%
  ungroup()

tidy_w2v_train <- tidy_headlines_topwords_train %>% # combines previous two
  select(id, is_sarcastic) %>%
  left_join(tidy_w2v_train_2, by="id") %>%
  left_join(tidy_w2v_train_1, by="id") 

tidy_w2v_train[is.na(tidy_w2v_train)] = 0
#saveRDS(tidy_w2v_train, "tidy_w2v_train.rds")

### test ### 
tidy_w2v_test_1 <- tidy_headlines_test %>% # gets top 500 words not in w2v
  filter(word %in% freq_words$word & !(word %in% word2vec_df$word)) %>%
  unique() %>%
  dcast(id~word, function(x) 1, fill = 0) 

tidy_w2v_test_2 <- tidy_headlines_test %>% # gets word embeddings
  left_join(word2vec_df, by ="word") %>%
  filter(complete.cases(.)) %>%
  select(-word) %>%
  group_by(id) %>%
  summarise_all(list(mean)) %>%
  ungroup()

tidy_w2v_test <- tidy_headlines_topwords_test %>% # gets top 500 words not in w2v
  select(id) %>%
  left_join(tidy_w2v_test_2, by="id") %>%
  left_join(tidy_w2v_test_1, by="id") 

tidy_w2v_test[is.na(tidy_w2v_test)] = 0
#saveRDS(tidy_w2v_test, "tidy_w2v_test.rds")

#########################################
##### Alternative: Train\Test Split #####
#########################################
set.seed(123)
tidy_w2v_train <- readRDS("tidy_w2v_train.rds")
tidy_w2v_test <- readRDS("tidy_w2v_test.rds")
#Instead of using the full train, we further split it into train and validation sets. 
fulldf <-  readRDS("tidy_w2v_train.rds")
fulldf$is_sarcastic <- as.factor(fulldf$is_sarcastic)
trainIndex <- createDataPartition(fulldf$is_sarcastic, p = .8, 
                                  list = FALSE, 
                                  times = 1)
head(trainIndex)
dfTrain <- fulldf[ trainIndex,]
dfTest  <- fulldf[-trainIndex,]

###########################################
##### Machine Learning: Linear Models #####
###########################################
training_y <- as.factor(tidy_w2v_train$is_sarcastic)  # Outcome variables
training_x <- tidy_w2v_train[, -c(1:2), drop = FALSE] # Predictors
training_x <- data.frame(scale(training_x)) 

#Input Function Setup
metric <- "Accuracy"
trainControl <- trainControl(method = "cv",           # Resampling method: Cross-validation
                             number = 5,                # Number of folds for cross-validation: 5
                             savePredictions = "final") # Only prediction from the last iteration are saved

trainControlbs <- trainControl(method = "boot",         # Use bootstrap validation
                               number = 50,             # Number of bootstrap replicates
                               savePredictions = "final")

#LOGISTIC
glm <- train(y = training_y,
             x = training_x,
             method = "glm",
             metric = metric,
             trControl = trainControl)

#LINEAR DISCRIMINATE ANALYSIS
lda <- train(y = training_y,
             x = training_x,
             method = "lda",
             metric = metric,
             trControl = trainControl)

#Shrinkage Methods:
#LASSO
start.time <- Sys.time()
lasso <- train(y = training_y,
               x = training_x,
               method = "glmnet",
               metric = metric,
               trControl = trainControl,
               tuneGrid = expand.grid(alpha = 1,#alpha=1: lasso
                                      lambda = seq(0.001, 0.1, by = 0.001)))
end.time <- Sys.time()
time.takencv <- end.time - start.time
time.takencv

#lasso1 uses a different resampling method
start.time <- Sys.time()
lasso1 <- train(y = training_y,
               x = training_x,
               method = "glmnet",
               metric = metric,
               trControl = trainControlbs,
               tuneGrid = expand.grid(alpha = 1,#alpha=1: lasso
                                      lambda = seq(0.001, 0.1, by = 0.001)))
end.time <- Sys.time()
time.takenbs <- end.time - start.time
time.takenbs
#Ridge
ridge <- train(y = training_y,
               x = training_x,
               method = "glmnet",
               metric = metric,
               trControl = trainControl,
               tuneGrid = expand.grid(alpha = 0,#alpha=1: ridge
                                      lambda = seq(0.001, 0.1, by = 0.001)))
ridge1 <- train(y = training_y,
               x = training_x,
               method = "glmnet",
               metric = metric,
               trControl = trainControlbs,
               tuneGrid = expand.grid(alpha = 0,#alpha=1: ridge
                                      lambda = seq(0.001, 0.1, by = 0.001)))

#Elastic Net
lenet <- train(y = training_y,
               x = training_x,
               method = "glmnet",
               family = "binomial", 
               metric = metric,
               trControl = trainControl,
               tuneLength=5) #automatically selects 5 values of alpha and lambda to be evaluated 

## Plot comparison of Accuracy by choice of lambda
complambda_lasso_ridge <- data.frame(rbind(cbind(lasso$results$lambda, c("lasso"),lasso$results$Accuracy), 
                                           cbind(ridge$results$lambda,  c("ridge"), ridge$results$Accuracy)))
colnames(complambda_lasso_ridge) <- c("lambda", "Type", "Accuracy")
complambda_lasso_ridge$lambda <- as.numeric(complambda_lasso_ridge$lambda)
complambda_lasso_ridge$Accuracy <- as.numeric(complambda_lasso_ridge$Accuracy)
ggplot(data=complambda_lasso_ridge, aes(x=lambda, y=Accuracy, group=Type)) +
  xlab("lambda") +
  geom_line(aes(color=Type)) +
  geom_point(aes(color=Type)) +
  scale_color_discrete(name = "Model", labels = c("Lasso", "Ridge"))  +
  ggtitle("Penalization and prediction error") +
  theme_minimal()

# Plot and compare number of nzero variables by choice of lambda
compnzero_lasso_ridge <- data.frame(cbind( seq(0.001, 0.1, by = 0.001),
                                           colSums(coef(lasso$finalModel, seq(0.001, 0.1, by = 0.001)) != 0),
                                           colSums(coef(ridge$finalModel, seq(0.001, 0.1, by = 0.001)) != 0)))
colnames(compnzero_lasso_ridge) <- c("lambda","lasso","ridge")
compnzero_lasso_ridge_long <- pivot_longer(compnzero_lasso_ridge, -c(lambda), values_to = "nzero", names_to = "Type")
ggplot(data=compnzero_lasso_ridge_long, aes(x=lambda, y=nzero, group=Type)) +
  xlab("lambda") +
  geom_line(aes(color=Type)) +
  geom_point(aes(color=Type)) +
  scale_color_discrete(name = "Model", labels = c("Lasso", "Ridge")) +
  ggtitle("Penalization and non-zero variables") +
  theme_minimal()

## Plot comparison of Accuracy by choice of alpha/lambda
trellis.par.set(caretTheme())
plot(lenet, metric = "Accuracy")

## Store the Best Accuracy Choice of alpha/lambda
best_lasso_index <- which.max(lasso$results$Accuracy)
best_ridge_index <- which.max(ridge$results$Accuracy)
best_lasso_accuracy <- lasso$results$Accuracy[best_lasso_index]
best_ridge_accuracy <- ridge$results$Accuracy[best_ridge_index]
best_lasso_lambda <- lasso$results$lasso[best_lasso_index]
best_ridge_lambda <- ridge$results$lambda[best_ridge_index]
best_lasso_model <- lasso$results[best_lasso_index, ]
best_ridge_model <- ridge$results[best_ridge_index, ]

##############################################
##### Machine Learning: Nonlinear Models #####
##############################################
# KNN - k-Nearest Neighbors 
knn <- train(y = training_y,
             x = training_x, 
             method = "knn",
             metric = "Accuracy",
             trControl = trainControl(method = "cv",
                        number = 5,
                        savePredictions = "final"),
             tuneLength=5)
# Summary of the trained model
summary(knn)
# Resampling results
knn$results

# Tree-based Method: Decision Tree
library(rpart)       # performing regression trees
library(rpart.plot) # plotting regression trees
rpart <- train(y = training_y,
               x = training_x,
               method = "rpart",
               metric = "Accuracy",
               trControl = trainControl(method = "cv",
                                        number = 5,
                                        savePredictions = "final"),
               tuneLength = 5)
rpart.plot(rpart$finalModel)

# Support Vector Machine
training_x <- data.frame(scale(training_x))
tidy_w2v_train$is_sarcastic <- as.factor(tidy_w2v_train$is_sarcastic)
svmlinear <- train(y = tidy_w2v_train$is_sarcastic,
                   x = training_x,
                   method = "svmLinear",
                   metric = "AUC")

svmradial <- train(x = training_x, 
                   y = training_yn,
                   method = "svmRadial",
                   metric = metric)

###########################################
##### Model Training: Ensemble Method #####
###########################################
# Random Forest
training_yn <- make.names(tidy_w2v_train$is_sarcastic)
rf <- train(y = training_yn,
            x = training_x,
            method = "rf", 
            metric = metric,
            trControl = trainControl(method = "cv", 
                                     number = 5,
                                     classProbs = TRUE))

# Bagging 
treebag <- train(y = training_y,
                 x = training_x,
                 method = "treebag", 
                 metric = metric,
                 trControl = trainControl)

# Boosting: Stochastic Gradient Boosting
gbm <- train(y = training_y,
             x = training_x,
             method = "gbm",
             metric = metric,
             trControl = trainControl)

library(caretEnsemble)
tidy_w2v_train$is_sarcastic <- as.factor(tidy_w2v_train$is_sarcastic) # Ensure 'is_sarcastic' is a factor
tidy_w2v_train$is_sarcastic <- make.names(tidy_w2v_train$is_sarcastic)

myFolds <- createFolds(tidy_w2v_train$is_sarcastic, k = 5, list = TRUE, returnTrain = TRUE)
model_list_all <- caretList(
  y = tidy_w2v_train$is_sarcastic,
  x = training_x,
  trControl = trainControl(method = "cv",
                           number = 5,
                           classProbs = TRUE,
                           summaryFunction = prSummary,
                           savePredictions = "final",
                           index = myFolds),
  metric = "AUC",
  methodList = c("glm"),  # methodList = c("glm","glmnet")
  tuneList = list(
    lasso = caretModelSpec(method = "glmnet", tuneGrid = expand.grid(alpha = 1, lambda = lasso$bestTune$lambda)),
    ridge = caretModelSpec(method = "glmnet", tuneGrid = expand.grid(alpha = 0, lambda = ridge$bestTune$lambda))))
    #,rf1 = caretModelSpec(method = "rf", tuneGrid = data.frame(.mtry = 80)),  # mtry= 80  since Accuracy = 0.6414827 (highest among 5 folds)
    #svmLinear = caretModelSpec(method = "svmLinear", tuneGrid = data.frame(.C = 0.1)) # SVM with linear kernel

bwplot(resamples(model_list_all))
modelCor(resamples(model_list_all))
ensemble <- caretEnsemble(model_list_all,
                          metric="AUC",
                          trControl=trainControl(method = "cv",
                                                 number = 5,
                                                 classProbs = TRUE,
                                                 summaryFunction = prSummary,
                                                 savePredictions = "final",
                                                 index = myFolds))

summary(ensemble)
#############################################################
##### Model Training: Screening Method (Post-Lasso/PCA) #####
#############################################################
install.packages("MLmetrics")
library("MLmetrics")

# Screening with Lasso:
vars <- data.frame(as.matrix(coef(lasso$finalModel, lasso$bestTune$lambda))[-1,])
colnames(vars)[1] <- "varname"
nzero_vars <- rownames(vars)[vars$varname!=0]

#Lasso-Screening
#lasso to glm
glm_scrlasso <- train(y = training_y,
                     x = training_x[,colnames(training_x) %in% nzero_vars],
                     method = "glm",
                     metric = "Accuracy",
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              savePredictions = "final"))

#lasso to svm
svm_scrlasso <- train(y = training_yn,
                   x = training_x[,colnames(training_x) %in% nzero_vars],
                   method = "svmLinear", # Using linear SVM
                   metric = "AUC",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            classProbs = TRUE,
                                            summaryFunction = prSummary,
                                            savePredictions = "final")
                   # Note: No tuneGrid is specified here as svmLinear will automatically perform tuning
)

#lasso to random forest
rf_scrlasso <- train(y = training_yn,
                     x = training_x[,colnames(training_x) %in% nzero_vars],
             method = "rf",
             metric = "AUC",
             trControl = trainControl(method = "cv",
                                      number = 5,
                                      classProbs = TRUE,
                                      summaryFunction = prSummary,
                                      savePredictions = "final"),  # Cross-validation with 5 folds
             tuneLength = 5) # Number of models to evaluate

# PCA-Screening
#pca to glm
logisticP <- train(y= training_yn,
                   x = training_x,
                   method = "glm",
                   family = "binomial",
                   metric = "AUC",
                   trControl = trainControl(method = "cv",
                                            number = 5,
                                            classProbs = TRUE,
                                            summaryFunction = prSummary,
                                            savePredictions = "final"),
                   preProcess = "pca" # Specify PCA preprocessing
)

#pca to svb 
svbP <- train(y= training_yn,
              x = training_x,
              method = "svmLinear", 
              metric = "AUC",
              trControl = trainControl(method = "cv",
                                       number = 5,
                                       classProbs = TRUE,
                                       summaryFunction = prSummary,
                                       savePredictions = "final"),
              preProcess = "pca" # Specify PCA preprocessing
)

#pca to random forest
rfP <- train(y = training_yn,
             x = training_x,
             method = "rf",
              metric = "AUC",
              trControl = trainControl(method = "cv",
                                       number = 5,
                                       classProbs = TRUE,
                                       summaryFunction = prSummary,
                                       savePredictions = "final"),
             tuneLength = 5,
             preProcess = "pca" # Specify PCA preprocessing
)

#The following is a chunk to manually do pca with train and test subsets
if (FALSE){
library("FactoMineR")
# training set 
dfTrainPCA <- dfTrain[, -c(1:2), drop = FALSE] # Predictors
dfTestPCA <- dfTest[, -c(1:2), drop = FALSE] # Predictors
pcares <- PCA(dfTrainPCA,          # perform pca on training data only
              ncp = 30,             # number of principal component to keep
              graph = FALSE)
training_data_pca <- pcares$ind$coord # pcares: coordinates of the individuals (rows) in the principal component space. 
                                      # coord extracts these coordinates and store
pcapred <- predict.PCA(pcares,newdata = dfTestPCA) # predicts the coordinates of new data using the PCA trained
validation_data_pca <- pcapred$coord
# 1. Combine PCA coordinates with outcome variable
combined_train <- data.frame(cbind(training_data_pca,is_sarcastic = dfTrain$is_sarcastic))
combined_train$is_sarcastic <- as.factor(combined_train$is_sarcastic)
combined_test <- data.frame(cbind(validation_data_pca,is_sarcastic = dfTest$is_sarcastic))
combined_test$is_sarcastic <- as.factor(combined_test$is_sarcastic)
validation_data_pca <- as.data.frame(validation_data_pca)
# 2. Train a classification model
pcaglm <- glm(is_sarcastic ~ ., data = combined_train, family = binomial)
# 3. Predict outcomes for validation set
predictions <- predict(pcaglm, newdata = validation_data_pca, type = "response")
# 4. Evaluate model performance
conf_matrix <- table(predicted = ifelse(predictions > 0.5, 2, 2), actual = combined_test$is_sarcastic)}

#############################################
##### Comparing and Selecting the Model #####
#############################################
# Comparing the predicting power: the best one is glm-post-lasso

#Train/Test Split for Unbiased Prediction Error Estimate
#train with train subset again 
train_y <- as.factor(dfTrain$is_sarcastic)  # Outcome variables
train_x <- dfTrain[, -c(1:2), drop = FALSE] # Predictors
train_x <- data.frame(scale(train_x)) 
#Train\Test Split Approach
lasso2 <- train(as.factor(is_sarcastic) ~ .-id, 
                data = dfTrain,
                method = "glmnet",
                metric = metric,
                trControl = trainControl,
                tuneGrid = expand.grid(alpha = 1,#alpha=1: lasso
                                       lambda = seq(0.001, 0.1, by = 0.001)))
predict_lasso <- predict(lasso2, dfTest)
cf_lasso <- confusionMatrix(predict_lasso, as.factor(dfTest$is_sarcastic))
best_lasso2_index <- which.max(lasso2$results$Accuracy)
best_lasso2_lambda <-lasso2$results$lasso2[best_lasso2_index]
best_lasso2_model <- lasso2$results[best_lasso2_index, ]
vars <- data.frame(as.matrix(coef(lasso2$finalModel, lasso2$bestTune$lambda))[-1,])
colnames(vars)[1] <- "varname"
nzero_vars <- rownames(vars)[vars$varname!=0]
glm_scrlasso2 <- train(y = train_y,
                      x = train_x[,colnames(training_x) %in% nzero_vars],
                      method = "glm",
                      metric = "Accuracy",
                      trControl = trainControl(method = "cv",
                                               number = 5,
                                               savePredictions = "final"))
predict_lasso <- predict(glm_scrlasso2, dfTest)
# Confusion Matrix
cf_compare <- confusionMatrix(predict_lasso, dfTest$is_sarcastic)

#############################################
##### Perform Classification Prediction #####
#############################################
testing_x <- tidy_w2v_test[, -c(1), drop = FALSE]
testing_x <- data.frame(scale(testing_x))
#Predicting with algorithms selected
prelasso <- predict(glm_scrlasso, testing_x)
# Write predictions to a text file
writeLines(as.character(prelasso), "predictions.txt")
