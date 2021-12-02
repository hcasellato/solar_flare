### Code Summary: ###############################################################################################
# This project's goal is to create an algorithm that predicts and classifies Solar Flares. 
#
### Basic packages: #############################################################################################
repo <- "http://cran.us.r-project.org"

# Required packages:
if(!require(randomForest)) install.packages("randomForest", repos = repo, dependencies = TRUE)
if(!require(data.table))   install.packages("data.table",   repos = repo, dependencies = TRUE)
if(!require(tidyverse))    install.packages("tidyverse",    repos = repo, dependencies = TRUE)
if(!require(Boruta))       install.packages("Boruta",       repos = repo, dependencies = TRUE)
if(!require(caret))        install.packages("caret",        repos = repo, dependencies = TRUE)
if(!require(dplyr))        install.packages("dplyr",        repos = repo, dependencies = TRUE)
if(!require(e1071))        install.packages("e1071",         repos = repo, dependencies = TRUE)
if(!require(pROC))         install.packages("pROC",         repos = repo, dependencies = TRUE)

library(randomForest)
library(data.table)
library(tidyverse)
library(Boruta)
library(caret)
library(dplyr)
library(e1071)
library(pROC)

### Data Preparation: ###########################################################################################
setwd(getwd())
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/solar-flare/"

## Download
download.file(paste(url,"flare.data1", sep = ""), destfile = "flare.data1")
download.file(paste(url,"flare.data2", sep = ""), destfile = "flare.data2")

## Reading
# Attribute Information:
#   1. Code for class (modified Zurich class)  (A,B,C,D,E,F,H)
#   2. Code for largest spot size              (X,R,S,A,H,K)
#   3. Code for spot distribution              (X,O,I,C)
#   4. Activity                                (1 = reduced, 2 = unchanged)
#   5. Evolution                               (1 = decay, 2 = no growth, 
#                                               3 = growth)
#   6. Previous 24 hour flare activity code    (1 = nothing as big as an M1,
#                                               2 = one M1,
#                                               3 = more activity than one M1)
#   7. Historically-complex                    (1 = Yes, 2 = No)
#   8. Did region become historically complex  (1 = yes, 2 = no) 
#      on this pass across the sun's disk
#   9. Area                                    (1 = small, 2 = large)
#  10. Area of the largest spot                (1 = <=5, 2 = >5)
# 
#  From all these predictors three classes of flares are predicted, which are 
#  represented in the last three columns.
# 
#   11. C-class flares production by this region    Number  
#       in the following 24 hours (common flares)
#   12. M-class flares production by this region    Number
#       in the following 24 hours (moderate flares)
#   13. X-class flares production by this region    Number
#       in the following 24 hours (severe flares)

colnames <- c("class", "lss", "spotdist", "activity", "evol", "fac", "hc",
              "hcp", "area", "area_ls", "cclass", "mclass", "xclass")

final_test_set  <- read.table("flare.data1", sep = " ", skip = 1, col.names = colnames)
root_train_set  <- read.table("flare.data2", sep = " ", skip = 1, col.names = colnames)

## Factorized data sets
final_test_set <- final_test_set %>% mutate(class    = as.factor(class),
                                            lss      = as.factor(lss),
                                            spotdist = as.factor(spotdist),
                                            cclass   = as.factor(cclass),
                                            mclass   = as.factor(mclass),
                                            xclass   = as.factor(xclass))

root_train_set <- root_train_set %>% mutate(class    = as.factor(class),
                                            lss      = as.factor(lss),
                                            spotdist = as.factor(spotdist),
                                            cclass   = as.factor(cclass),
                                            mclass   = as.factor(mclass),
                                            xclass   = as.factor(xclass))

## Replacing [,1:3] from letters to numbers
# Unfortunately I can't find a way to reduce the number of lines in this next chunk
levels(final_test_set[,1]) <- 1:6
levels(final_test_set[,2]) <- 1:6
levels(final_test_set[,3]) <- 1:4

levels(root_train_set[,1]) <- 1:6
levels(root_train_set[,2]) <- 1:6
levels(root_train_set[,3]) <- 1:4

## Binary data sets
# can't think a way to reduce this part either
binary_test_set                <- final_test_set 
levels(binary_test_set$cclass) <- c("0", "1", "1")
levels(binary_test_set$mclass) <- c("0", "1", "1", "1")

binary_train_set                <- root_train_set
levels(binary_train_set$cclass) <- c("0", "1", "1", "1", "1", "1", "1", "1", "1")
levels(binary_train_set$mclass) <- c("0", "1", "1", "1", "1", "1")
levels(binary_train_set$xclass) <- c("0", "1", "1")

binary_test_set <- binary_test_set %>% mutate(class    = as.numeric(as.character(class)),
                                              lss      = as.numeric(as.character(lss)),
                                              spotdist = as.numeric(as.character(spotdist)),
                                              cclass    = as.numeric(as.character(cclass)),
                                              mclass    = as.numeric(as.character(mclass)),
                                              xclass    = as.numeric(as.character(xclass)))

binary_train_set <- binary_train_set %>% mutate(class    = as.numeric(as.character(class)),
                                                lss      = as.numeric(as.character(lss)),
                                                spotdist = as.numeric(as.character(spotdist)),
                                                cclass    = as.numeric(as.character(cclass)),
                                                mclass    = as.numeric(as.character(mclass)),
                                                xclass    = as.numeric(as.character(xclass)))
rm(repo, url, colnames)
### Data Exploration: ###########################################################################################
## Class
# E and F have significantly fewer occurrences
round(prop.table(table(root_train_set$class)),2)

## Largest spot size
# H and K have significantly fewer occurrences
round(prop.table(table(root_train_set$lss)),2)

## Spot distributions
# C has significantly fewer occurrences
round(prop.table(table(root_train_set$spotdist)),2)

## Activity
# The majority of solar flares have reduced activity
round(prop.table(table(root_train_set$activity)),2)

## Evolution
# Only a minority of cases have decayed growth
round(prop.table(table(root_train_set$evol)),2)

## Flare activity code
# The majority of flare activity is lesser than an M1
round(prop.table(table(root_train_set$fac)),2)

## Historically-complex 
# Balanced distribution
round(prop.table(table(root_train_set$hc)),2)

## Historically-complex on this pass to Sun's disc
# The majority of regions did not became historically complex on the pass to Sun's disc
round(prop.table(table(root_train_set$hcp)),2)

## Area
# The majority of flares occur in small areas
round(prop.table(table(root_train_set$area)),2)

## Area of the largest spot
# All areas of the largest spot are less than 5 
round(prop.table(table(root_train_set$area_ls)),2)

## Flare Classes on root_train_set
#                  0    1  2  3  4  5  6  7  8  Total
# C-class flares  884 112 33 20  9  4  3  0  1  1066
# M-class flares 1030  29  3  2  1  0  1  0  0  1066
# X-class flares 1061   4  1  0  0  0  0  0  0  1066

### C-Class Prediction: #########################################################################################
## Removing single event of 8 C-Class flare:
cbinary_train_set      <- binary_train_set[-which(root_train_set$cclass == "8"),]

croot_train_set        <- root_train_set[-which(root_train_set$cclass == "8"),]
croot_train_set$cclass <- droplevels(croot_train_set$cclass)

# Data partitioning:
set.seed(2021, sample.kind = "Rounding")
cindex <- createDataPartition(croot_train_set$cclass, times = 1, p = .85, list = FALSE)

cttrain_set <- croot_train_set[cindex,]
cttest_set  <- croot_train_set[-cindex,]

cbttrain_set <- cbinary_train_set[cindex,]
cbttest_set  <- cbinary_train_set[-cindex,]

rm(cindex)
## Variable importance with Boruta
set.seed(2021, sample.kind = "Rounding")
cboruta <- Boruta(cttrain_set[,1:10],
                  cttrain_set[,11],
                  maxRuns = 300,
                  doTrace = 1)
cimp <- c(cboruta$finalDecision == "Confirmed" | cboruta$finalDecision == "Tentative")

## Random Forest
set.seed(2021, sample.kind = "Rounding")
ctrain_rf <- train(cttrain_set[,1:10],
                   cttrain_set$cclass,
                   method = "rf",
                   ntree = 50,
                   trControl = trainControl(method = "cv", number = 5),
                   tuneGrid = data.frame(mtry = seq(10,300,1)),
                   nSamp = 200)

set.seed(2021, sample.kind = "Rounding")
cfit_rf <- randomForest(cttrain_set[,1:10][cimp],
                        cttrain_set[,11],
                        ntree = 50,
                        minNode = ctrain_rf$bestTune$mtry)

crf_cm <- confusionMatrix(predict(cfit_rf, cttest_set), cttest_set$cclass)

## Random Forest for Binary data sets
set.seed(2021, sample.kind = "Rounding")
bctrain_rf <- train(cbttrain_set[,1:10],
                   cbttrain_set$cclass,
                   method = "rf",
                   ntree = 50,
                   trControl = trainControl(method = "cv", number = 5),
                   tuneGrid = data.frame(mtry = seq(10,300,1)),
                   nSamp = 200)

set.seed(2021, sample.kind = "Rounding")
bcfit_rf <- randomForest(cbttrain_set[,1:10][cimp],
                         as.factor(cbttrain_set[,11]),
                         ntree = 50,
                         minNode = bctrain_rf$bestTune$mtry)

bcrf_cm <- confusionMatrix(as.factor(ifelse(predict(bcfit_rf, cbttest_set, type="prob")[,2] >= .45, 1, 0)),
                           as.factor(cbttest_set$cclass))

## Logistic Regression
set.seed(2021, sample.kind = "Rounding")
cfit_log <- glm(cclass ~ .,
                data = cbttrain_set[,1:11][cimp],
                family = "binomial")

clog_cm <- confusionMatrix(as.factor(ifelse(predict(cfit_log, cbttest_set, type = "response") > .5, 1, 0)),
                            as.factor(cbttest_set$cclass))

## K-Nearest Neighbors
set.seed(2021, sample.kind = "Rounding")
ctrain_knn <- train(as.factor(cclass) ~ .,
                    method = "knn",
                    data = cbttrain_set[,1:11],
                    tuneGrid = data.frame(k = 10:100))

cfit_knn <- knn3(as.factor(cclass) ~ .,
                 data = cbttrain_set[,1:11][cimp],
                 k = ctrain_knn$bestTune)

set.seed(2021, sample.kind = "Rounding")
c_knn_p <- as.factor(ifelse(predict(cfit_knn, cbttest_set)[,2] >= .4, 1, 0))
cknn_cm <- confusionMatrix(c_knn_p, as.factor(cbttest_set$cclass))

## Positive Occurrence training with Random Forest Ensemble
# I tried ensemble methods to predict the number of C-Class occurrences given
# they could be detected but didn't work as expected. In other words, it turns
# out that wasn't worth 100 more lines to predict pretty much the same output,
# therefore this section could be worked in the future.

## Accuracy table:
c_acc_tbl <- data.frame(RF_Raw         = crf_cm$overall["Accuracy"],
                        RF_Binary      = bcrf_cm$overall["Accuracy"],
                        Log_Reg_Binary = clog_cm$overall["Accuracy"],
                        KNN_Binary     = cknn_cm$overall["Accuracy"])

## ROC/AUC table:
c_auc_tbl <- data.frame(RF_Binary      = roc(ifelse(predict(bcfit_rf, cbttest_set, type="prob")[,2] >= .45, 1, 0),
                                             cbttest_set$cclass)$auc,
                        Log_Reg_Binary = roc(ifelse(predict(cfit_log, cbttest_set, type = "response") > .5, 1, 0),
                                             cbttest_set$cclass)$auc,
                        KNN_Binary     = roc(ifelse(predict(cfit_knn, cbttest_set, type="prob")[,2] >= .4, 1, 0),
                                             cbttest_set$cclass)$auc)
## Verdict:
# Random forest on the Binary data set has more accuracy and area_under_curve, therefore will be used
# to predict occurrences of C-Class Solar Flares.

rm(bcfit_rf, bcrf_cm, bctrain_rf, c_auc_tbl, c_acc_tbl, cbinary_train_set, cboruta, cbttest_set, cbttrain_set,
   cfit_knn, cfit_log, cfit_rf, cknn_cm, clog_cm, crf_cm, croot_train_set, ctrain_knn, ctrain_rf, cttest_set,
   cttrain_set, c_knn_p, cimp)

### M-Class Prediction: #########################################################################################
## Data partitioning:
# There are only 3% of M-Class occurrences, therefore we will only use the binary data set for ML.
set.seed(2021, sample.kind = "Rounding")
mindex <- createDataPartition(binary_train_set$mclass, times = 1, p = .85, list = FALSE)

mbttrain_set <- binary_train_set[mindex,]
mbttest_set  <- binary_train_set[-mindex,]

rm(mindex)
## Variable importance with Boruta
set.seed(2021, sample.kind = "Rounding")
mboruta <- Boruta(mbttrain_set[,1:10],
                  mbttrain_set[,12],
                  maxRuns = 300,
                  doTrace = 1)
mimp <- c(mboruta$finalDecision == "Confirmed" | mboruta$finalDecision == "Tentative")

## Random Forest for Binary data sets
set.seed(2021, sample.kind = "Rounding")
bmtrain_rf <- train(mbttrain_set[,1:10],
                    mbttrain_set$mclass,
                    method = "rf",
                    ntree = 50,
                    trControl = trainControl(method = "cv", number = 5),
                    tuneGrid = data.frame(mtry = seq(10,300,1)),
                    nSamp = 200)

set.seed(2021, sample.kind = "Rounding")
bmfit_rf <- randomForest(mbttrain_set[,1:10][mimp],
                         as.factor(mbttrain_set[,12]),
                         ntree = 50,
                         minNode = bmtrain_rf$bestTune$mtry)

bmrf_cm <- confusionMatrix(as.factor(ifelse(predict(bmfit_rf, mbttest_set, type="prob")[,2] >= .45, 1, 0)),
                           as.factor(mbttest_set$mclass))

## Logistic Regression
set.seed(2021, sample.kind = "Rounding")
mfit_log <- glm(mclass ~ .,
                data = mbttrain_set[,c(1:10,12)][mimp],
                family = "binomial")

set.seed(2021, sample.kind = "Rounding")
m_log_p <- as.factor(ifelse(predict(mfit_log, mbttest_set, type = "response") > .3, 1, 0))

mlog_cm <- confusionMatrix(m_log_p, as.factor(mbttest_set$mclass))

## K-Nearest Neighbors
set.seed(2021, sample.kind = "Rounding")
mtrain_knn <- train(as.factor(mclass) ~ .,
                    method = "knn",
                    data = mbttrain_set[,c(1:10,12)],
                    tuneGrid = data.frame(k = 10:100))

mfit_knn <- knn3(as.factor(mclass) ~ .,
                 data = mbttrain_set[,c(1:10,12)][mimp],
                 k = mtrain_knn$bestTune)

set.seed(2021, sample.kind = "Rounding")
m_knn_p <- as.factor(ifelse(predict(mfit_knn, mbttest_set)[,2] >= .1, 1, 0))

mknn_cm <- confusionMatrix(m_knn_p, as.factor(mbttest_set$mclass))

## Accuracy table:
m_acc_tbl <- data.frame(RF_Binary      = bmrf_cm$overall["Accuracy"],
                        Log_Reg_Binary = mlog_cm$overall["Accuracy"],
                        KNN_Binary     = mknn_cm$overall["Accuracy"])

## ROC/AUC table:
m_auc_tbl <- data.frame(RF_Binary      = roc(ifelse(predict(bmfit_rf, mbttest_set, type="prob")[,2] >= .45, 1, 0),
                                             mbttest_set$mclass)$auc,
                        Log_Reg_Binary = roc(m_log_p, mbttest_set$mclass)$auc,
                        KNN_Binary     = roc(m_knn_p, mbttest_set$mclass)$auc)

## Verdict:
# K-nearest neighbors has more accuracy and area_under_curve, therefore will be used to predict occurrences of
# M-Class Solar Flares.

rm(bmfit_rf, bmrf_cm, bmtrain_rf, m_auc_tbl, m_acc_tbl, mboruta, mbttest_set, mbttrain_set,
   mfit_knn, mfit_log, mknn_cm, mlog_cm, mtrain_knn, m_knn_p, mimp, m_log_p)

### X-Class Prediction: #########################################################################################
## Data partitioning:
# There are less than 1% of X-Class occurrences, therefore we will only use the binary data set for ML.
set.seed(2021, sample.kind = "Rounding")
xindex <- createDataPartition(binary_train_set$xclass, times = 1, p = .85, list = FALSE)

xbttrain_set <- binary_train_set[xindex,]
xbttest_set  <- binary_train_set[-xindex,]

rm(xindex)
## Variable importance with Boruta
set.seed(2021, sample.kind = "Rounding")
xboruta <- Boruta(xbttrain_set[,1:10],
                  xbttrain_set[,12],
                  maxRuns = 300,
                  doTrace = 1)
ximp <- c(xboruta$finalDecision == "Confirmed" | xboruta$finalDecision == "Tentative")

## Random Forest
set.seed(2021, sample.kind = "Rounding")
bxtrain_rf <- train(xbttrain_set[,1:10],
                    xbttrain_set$xclass,
                    method = "rf",
                    ntree = 50,
                    trControl = trainControl(method = "cv", number = 5),
                    tuneGrid = data.frame(mtry = seq(10,300,1)),
                    nSamp = 200)

set.seed(2021, sample.kind = "Rounding")
bxfit_rf <- randomForest(xbttrain_set[,1:10][ximp],
                         as.factor(xbttrain_set[,12]),
                         ntree = 50,
                         minNode = bxtrain_rf$bestTune$mtry)

bxrf_cm <- confusionMatrix(as.factor(ifelse(predict(bxfit_rf, xbttest_set, type="prob")[,2] >= .45, 1, 0)),
                           as.factor(xbttest_set$xclass))

## Logistic Regression
set.seed(2021, sample.kind = "Rounding")
xfit_log <- glm(xclass ~ .,
                data = xbttrain_set[,c(1:10,13)][ximp],
                family = "binomial")

set.seed(2021, sample.kind = "Rounding")
x_log_p <- as.factor(ifelse(predict(xfit_log, xbttest_set, type = "response") > .2, 1, 0))
levels(x_log_p) <- as.factor(c(0,1))

xlog_cm <- confusionMatrix(x_log_p, as.factor(xbttest_set$xclass))

## K-Nearest Neighbors
set.seed(2021, sample.kind = "Rounding")
xtrain_knn <- train(as.factor(xclass) ~ .,
                    method = "knn",
                    data = xbttrain_set[,c(1:10,13)],
                    tuneGrid = data.frame(k = 10:100))

xfit_knn <- knn3(as.factor(xclass) ~ .,
                 data = xbttrain_set[,c(1:10,13)][ximp],
                 k = xtrain_knn$bestTune)

set.seed(2021, sample.kind = "Rounding")
x_knn_p <- as.factor(ifelse(predict(xfit_knn, xbttest_set)[,2] >= .03, 1, 0))
levels(x_knn_p) <- as.factor(c(0,1))

xknn_cm <- confusionMatrix(x_knn_p, as.factor(xbttest_set$mclass))

## Accuracy table:
x_acc_tbl <- data.frame(RF_Binary      = bxrf_cm$overall["Accuracy"],
                        Log_Reg_Binary = xlog_cm$overall["Accuracy"],
                        KNN_Binary     = xknn_cm$overall["Accuracy"])

## ROC/AUC table:
x_auc_tbl <- data.frame(RF_Binary      = roc(ifelse(predict(bxfit_rf, xbttest_set, type="prob")[,2] >= .45, 1, 0),
                                             xbttest_set$xclass)$auc,
                        Log_Reg_Binary = roc(x_log_p, xbttest_set$xclass)$auc,
                        KNN_Binary     = roc(x_knn_p, xbttest_set$xclass)$auc)

## Verdict:
# Random forest has more accuracy and area_under_curve, therefore will be used to predict occurrences of X-Class
# Solar Flares.

rm(bxfit_rf, bxrf_cm, bxtrain_rf, x_auc_tbl, x_acc_tbl, xboruta, xbttest_set, xbttrain_set,
   xfit_knn, xfit_log, xknn_cm, xlog_cm, xtrain_knn, x_knn_p, ximp, x_log_p)
### Final Class Prediction: #####################################################################################
# All models worked better on the binary data sets, therefore they will be used.
rm(root_train_set, final_test_set)

## C-Class
# Boruta
set.seed(2021, sample.kind = "Rounding")
cboruta <- Boruta(binary_train_set[,1:10],
                  binary_train_set[,11],
                  maxRuns = 300,
                  doTrace = 1)
cimp <- c(cboruta$finalDecision == "Confirmed" | cboruta$finalDecision == "Tentative")
rm(cboruta)

# Random Forest
set.seed(2021, sample.kind = "Rounding")
train_rf <- train(binary_train_set[,1:10],
                  binary_train_set$cclass,
                  method = "rf",
                  ntree = 50,
                  trControl = trainControl(method = "cv", number = 5),
                  tuneGrid = data.frame(mtry = seq(10,300,1)),
                  nSamp = 200)

set.seed(2021, sample.kind = "Rounding")
fit_rf <- randomForest(binary_train_set[,1:10][cimp],
                       as.factor(binary_train_set[,11]),
                       ntree = 50,
                       minNode = train_rf$bestTune$mtry)

c_pred <- as.factor(ifelse(predict(fit_rf, binary_test_set, type="prob")[,2] >= .45, 1, 0))
#rm(train_rf, fit_rf)

## M-Class
# Boruta
set.seed(2021, sample.kind = "Rounding")
mboruta <- Boruta(binary_train_set[,1:10],
                  binary_train_set[,12],
                  maxRuns = 300,
                  doTrace = 1)
mimp <- c(mboruta$finalDecision == "Confirmed" | mboruta$finalDecision == "Tentative")
rm(mboruta)

# K-Nearest Neighbors
set.seed(2021, sample.kind = "Rounding")
train_knn <- train(as.factor(mclass) ~ .,
                    method = "knn",
                    data = binary_train_set[,c(1:10,12)],
                    tuneGrid = data.frame(k = 10:100))

fit_knn <- knn3(as.factor(mclass) ~ .,
                 data = binary_train_set[,c(1:10,12)][mimp],
                 k = train_knn$bestTune)

set.seed(2021, sample.kind = "Rounding")
m_pred <- as.factor(ifelse(predict(fit_knn, binary_test_set)[,2] >= .1, 1, 0))
#rm(train_knn, fit_knn)

## X-Class
# Boruta
set.seed(2021, sample.kind = "Rounding")
xboruta <- Boruta(binary_train_set[,1:10],
                  binary_train_set[,13],
                  maxRuns = 300,
                  doTrace = 1)
ximp <- c(xboruta$finalDecision == "Confirmed" | xboruta$finalDecision == "Tentative")
rm(xboruta)

# Random Forest
set.seed(2021, sample.kind = "Rounding")
xtrain_rf <- train(binary_train_set[,1:10],
                   binary_train_set$xclass,
                   method = "rf",
                   ntree = 50,
                   trControl = trainControl(method = "cv", number = 5),
                   tuneGrid = data.frame(mtry = seq(10,300,1)),
                   nSamp = 200)

set.seed(2021, sample.kind = "Rounding")
xfit_rf <- randomForest(binary_train_set[,1:10][ximp],
                        as.factor(binary_train_set[,12]),
                        ntree = 50,
                        minNode = xtrain_rf$bestTune$mtry)

set.seed(2021, sample.kind = "Rounding")
x_pred <- as.factor(ifelse(predict(xfit_rf, binary_test_set, type="prob")[,2] >= .45, 1, 0))
#rm(xtrain_rf, xfit_rf)
#rm(cimp, mimp, ximp)

### Final Testing: ##############################################################################################
acc_cclass <- confusionMatrix(c_pred, as.factor(binary_test_set$cclass))$overall["Accuracy"]
acc_mclass <- confusionMatrix(m_pred, as.factor(binary_test_set$mclass))$overall["Accuracy"]
acc_xclass <- confusionMatrix(x_pred, as.factor(binary_test_set$xclass))$overall["Accuracy"]
