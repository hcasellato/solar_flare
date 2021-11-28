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

binary_test_set <- final_test_set %>% mutate(class    = as.numeric(as.character(class)),
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

### C-Class prediction: #########################################################################################
## Data preparation: 
# Removing single event of 8 C-Class flare:
binary_train_set      <- binary_train_set[-which(root_train_set$cclass == "8"),]

root_train_set        <- root_train_set[-which(root_train_set$cclass == "8"),]
root_train_set$cclass <- droplevels(root_train_set$cclass)


## Data partitioning:
set.seed(2021, sample.kind = "Rounding")
index <- createDataPartition(root_train_set$cclass, times = 1, p = .85, list = FALSE)

ttrain_set <- root_train_set[index,]
ttest_set  <- root_train_set[-index,]

bttrain_set <- binary_train_set[index,]
bttest_set  <- binary_train_set[-index,]

## Variable importance with Boruta
set.seed(2021, sample.kind = "Rounding")
cboruta <- Boruta(ttrain_set[,1:10],
                  ttrain_set[,11],
                  maxRuns = 300,
                  doTrace = 1)
cimp <- c(cboruta$finalDecision == "Confirmed" | cboruta$finalDecision == "Tentative")

## Random Forest
set.seed(2021, sample.kind = "Rounding")
ctrain_rf <- train(ttrain_set[,1:10],
                   ttrain_set$cclass,
                   method = "rf",
                   ntree = 50,
                   trControl = trainControl(method = "cv", number = 5),
                   tuneGrid = data.frame(mtry = seq(10,300,1)),
                   nSamp = 200)

set.seed(2021, sample.kind = "Rounding")
cfit_rf <- randomForest(ttrain_set[,1:10][cimp],
                        ttrain_set[,11],
                        ntree = 50,
                        minNode = ctrain_rf$bestTune$mtry)

crf_cm <- confusionMatrix(predict(cfit_rf, ttest_set), ttest_set$cclass)

## Random Forest for Binary data sets
set.seed(2021, sample.kind = "Rounding")
bctrain_rf <- train(bttrain_set[,1:10],
                   bttrain_set$cclass,
                   method = "rf",
                   ntree = 50,
                   trControl = trainControl(method = "cv", number = 5),
                   tuneGrid = data.frame(mtry = seq(10,300,1)),
                   nSamp = 200)

set.seed(2021, sample.kind = "Rounding")
bcfit_rf <- randomForest(bttrain_set[,1:10][cimp],
                         as.factor(bttrain_set[,11]),
                         ntree = 50,
                         minNode = bctrain_rf$bestTune$mtry)

bcrf_cm <- confusionMatrix(as.factor(ifelse(predict(bcfit_rf, bttest_set, type="prob")[,2] >= .45, 1, 0)),
                           as.factor(bttest_set$cclass))

## Logistic Regression
set.seed(2021, sample.kind = "Rounding")
cfit_log <- glm(cclass ~ .,
                data = bttrain_set[,1:11][cimp],
                family = "binomial")

clog_cm <- confusionMatrix(as.factor(ifelse(predict(cfit_log, bttest_set, type = "response") > .5, 1, 0)),
                            as.factor(bttest_set$cclass))

## K-Nearest Neighbors
set.seed(2021, sample.kind = "Rounding")
ctrain_knn <- train(as.factor(cclass) ~ .,
                    method = "knn",
                    data = bttrain_set[,1:11],
                    tuneGrid = data.frame(k = 10:100))

cfit_knn <- knn3(as.factor(cclass) ~ .,
                 data = bttrain_set[,1:11][cimp],
                 k = ctrain_knn$bestTune)

set.seed(2021, sample.kind = "Rounding")
c_knn_p <- as.factor(ifelse(predict(cfit_knn, bttest_set)[,2] >= .4, 1, 0))
cknn_cm <- confusionMatrix(c_knn_p, as.factor(bttest_set$cclass))

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
c_auc_tbl <- data.frame(RF_Binary      = roc(ifelse(predict(bcfit_rf, bttest_set, type="prob")[,2] >= .45, 1, 0),
                                             bttest_set$cclass)$auc,
                        Log_Reg_Binary = roc(ifelse(predict(cfit_log, bttest_set, type = "response") > .5, 1, 0),
                                             bttest_set$cclass)$auc,
                        KNN_Binary     = roc(ifelse(predict(cfit_knn, bttest_set, type="prob")[,2] >= .4, 1, 0),
                                             bttest_set$cclass)$auc)
## Verdict:
# Random forest on the Binary data set has more accuracy and area_under_curve, therefore will be used
# for predicting occurrences of C-Class Solar Flares

### M-Class prediction: #########################################################################################

#################################################################################################################