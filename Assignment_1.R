#File: Machine_Learning_Lab1.R
#Project: Assignment 1 for Machine_Learning
#Author: B Li
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
saveRDS(tidy_headlines_topwords_train, "train_top500_words.rds")
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

###########################################
##### Machine Learning: Linear Models #####
###########################################
set.seed(123)
training_y <- as.numeric(tidy_w2v_train$is_sarcastic)
training_y <- factor(training_y, levels = c(0, 1))
training_x <- tidy_w2v_train[, -c(1:2), drop = FALSE]

#Simple Logistic Model
lgreg <- glm(is_sarcastic ~ ., data = tidy_w2v_train, family = "binomial")

#LOGISTIC
glm <- train(y = training_y,
             x = training_x,
             method="glm",
             metric="Accuracy",
             trControl = trainControl(method = "cv",
                                      number = 5,
                                      savePredictions = "final"))
#LINEAR DISCRIMINATE ANALYSIS
lda <- train(y = training_y,
             x = training_x,
             method="lda",
             metric="Accuracy",
             trControl = trainControl(method = "cv",
                                    number = 5,
                                    savePredictions = "final"))

#LASSO
lasso <- train(y = training_y,
               x = training_x,
               method = "glmnet",
               metric = "Accuracy",
               trControl = trainControl(method = "cv",
                                        number = 5,
                                        savePredictions = "final"),
               tuneGrid = expand.grid(alpha = 1,#alpha=1: lasso
                                      lambda = seq(0.001, 0.1, by = 0.001)))

#Ridge
ridge <- train(y = training_y,
               x = training_x,
               method = "glmnet",
               metric = "Accuracy",
               trControl = trainControl(method = "cv",
                                        number = 5,
                                        savePredictions = "final"),
               tuneGrid = expand.grid(alpha = 0,#alpha=1: ridge
                                      lambda = seq(0.001, 0.1, by = 0.001)))

#Elastic Net
lenet <- train(y = training_y,
               x = training_x,
               method = "glmnet",
               family = "binomial", 
               metric = "Accuracy",
               trControl = trainControl(method = "cv",
                                        number = 5,
                                        savePredictions = "final"),
               tuneLength=5) #automatically selects 5 values of alpha and lambda to be evaluated 

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
                        savePredictions = "final"))

# Tree-based Method: 
install.packages("rpart.plot") 
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
rpart
rpart.plot(rpart$finalModel)

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
## Plot comparison of Accuracy by choice of alpha/lambda
trellis.par.set(caretTheme())
plot(lenet, metric = "Accuracy")

############################
##### Screening Method #####
############################
# Screening with Lasso
vars <- data.frame(as.matrix(coef(lasso$finalModel, lasso$bestTune$lambda))[-1,])
colnames(vars)[1] <- "varname"
nzero_vars <- rownames(vars)[vars$varname!=0]
lm_scrlasso <- train(y = training_y,
                     x = training_x[,colnames(training_x) %in% nzero_vars],
                     method = "glm",
                     metric = "Accuracy",
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              savePredictions = "final"))
# Screening with Ridge
vars2 <- data.frame(as.matrix(coef(ridge$finalModel, ridge$bestTune$lambda))[-1,])
colnames(vars)[1] <- "varname"
nzero_vars2 <- rownames(vars)[vars$varname!=0]
lm_scrridge <- train(y = training_y,
                     x = training_x[,colnames(training_x) %in% nzero_vars2],
                     method = "glm",
                     metric="Accuracy",
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              savePredictions = "final"))
# Comparing the predicting power with Ridge
resamples <- resamples(list("Lasso" = lasso,
                            "Ridge" = ridge,
                            "slass"= lm_scrlasso,
                            "sridg"= lm_scrridge))
summary(resamples)

#############################################
##### Perform Classification Prediction #####
#############################################
testing_x <- tidy_w2v_test[, -c(1), drop = FALSE]
#Predicting with algorithms selected
prelasso <- predict(lasso, testing_x)
preridge <- predict(ridge, testing_x)
prelenet <- predict(lenet, testing_x)

