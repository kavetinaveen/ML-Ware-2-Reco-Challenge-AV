######################### ML WARE 1 - Recommendation Engine ######################

# Setting working directory
filepath <- c("/Users/nkaveti/Documents/Kaggle/ML WARE 2 - Recommendation Engine")
setwd(filepath)

# Loading required packages
library(readr)
library(dplyr)
library(h2o)
library(lsa)
library(xgboost)
library(Matrix)

# Reading files
train <- read_csv("train_MLWARE2.csv")
test <- read_csv("test_MLWARE2.csv")

# Converting files into R dataframe
train <- as.data.frame(train)
test <- as.data.frame(test)

train$ID <- NULL

# User-Item rating matrix
RatingMat <- reshape(train, idvar = c("userId"), timevar  = c("itemId"), direction = "wide")
colnames(RatingMat)[-1] <- unique(train$itemId)
row.names(RatingMat) <- RatingMat[,1]
RatingMat <- RatingMat[,-1]
RatingMat[is.na(RatingMat)] <- 0

# Cosine Similarity of items
item_sim <- cosine(as.matrix(RatingMat))
diag(item_sim) <- 0
mean_item_sim <- as.data.frame(colMeans(item_sim))
mean_item_sim$itemId <- as.integer(row.names(mean_item_sim))
row.names(mean_item_sim) <- NULL
colnames(mean_item_sim)[1] <- "item_sim_avg"
mean_item_sim <- mean_item_sim[, c("itemId", "item_sim_avg")]

# Feature Extraction
UserFeatures <- train %>% group_by(userId) %>% summarise(MeanRatUser = mean(rating), MdRatUser = median(rating), SdRatUser = sd(rating), Count_User = length(userId))

UserFeatures[is.na(UserFeatures)] <- 0

ItemFeatures <- train %>% group_by(itemId) %>% summarise(MeanRatItem = mean(rating), MdRatItem = median(rating), SdRatItem = sd(rating), Count_Items = length(itemId))

ItemFeatures[is.na(ItemFeatures)] <- 0

top5simitems <- do.call(rbind, lapply(colnames(item_sim), function(x){
  temp <- as.integer(names(sort(item_sim[, x], decreasing = TRUE))[1:5])
  ItemFeatures[ItemFeatures$itemId %in% temp, "MeanRatItem"]
}))

top5simitems <- data.frame(itemId = as.integer(colnames(item_sim)), top5simitems)

ItemFeatures <- ItemFeatures %>% left_join(top5simitems, by = "itemId")

UserFeatures <- as.data.frame(UserFeatures)
ItemFeatures <- as.data.frame(ItemFeatures)

# Merging features with train and test data
train <- train %>% left_join(UserFeatures, by = "userId")
train <- train %>% left_join(ItemFeatures, by = "itemId")
train <- train %>% left_join(mean_item_sim, by = "itemId")

test <- test %>% left_join(UserFeatures, by = "userId")
test <- test %>% left_join(ItemFeatures, by = "itemId")
test <- test %>% left_join(mean_item_sim, by = "itemId")

# Using h2o to build tree structure models
train$userId <- as.factor(train$userId)
test$userId <- as.factor(test$userId)
train$itemId <- as.factor(train$itemId)
test$itemId <- as.factor(test$itemId)

# [1] "userId"       "itemId"       "rating"       "MeanRatUser"  "MdRatUser"    "SdRatUser"    "MeanRatItem" 
# [8] "MdRatItem"    "SdRatItem"    "item_sim_avg"

predictors <- setdiff(colnames(train), c("ID", "userId", "MdRatUser", "MdRatItem", "SdRatItem", "rating"))
response <- c("rating")
gbm <- h2o.gbm(x = predictors, y = response, training_frame = as.h2o(train), nfolds = 4)

pred <- predict(gbm, as.h2o(test[, predictors]))
pred_train <- predict(gbm, as.h2o(train[, predictors]))
pred[pred < 0] <- 0
pred[pred > 10] <- 10
pred <- as.data.frame(pred)

pred_train[pred_train < 0] <- 0
pred_train[pred_train > 10] <- 10
pred_train <- as.data.frame(pred_train)

result <- data.frame(ID = test$ID, rating = pred$predict)
write.csv(result, paste0("GBM_Central_Dispersion2.csv"), row.names = FALSE)

train$pred_gbm <- pred_train$predict
test$pred_gbm <- pred$predict

train_sparse <- sparse.model.matrix(~., train[, predictors])
test_sparse <- sparse.model.matrix(~., test[, predictors])

dtrain <- xgb.DMatrix(train_sparse, label = train$rating)

param <- list(max_depth = 10, eta = 0.01, objective = "reg:linear", eval_metric = "rmse")

xgb <- xgboost(data = dtrain, 
               max_depth = 10,
               eta = 0.01,
               objective = "reg:linear",
               eval_metric = "rmse",
               nround = 500,
               subsample = 0.7,
               colsample_bytree = 0.4,
               nthread = 7
)

pred_xgb <- predict(xgb, test_sparse)
pred_xgb[pred_xgb < 0] <- 0
pred_xgb[pred_xgb > 10] <- 10

result_xgb <- data.frame(ID = test$ID, rating = (pred_xgb + pred)/2)
write.csv(result_xgb, file= "XGB.csv", row.names = FALSE)

