library(rpart)
library(rpart.plot)
path <- '~/Box Sync/Others/Data/SampleData'
setwd(path)

load("labData.Rda")

# 把90例死亡重复15次，使得数据平衡
test <- dataFinal[dataFinal$TREAT_RESULT == 'dead', ]
test <- test[rep(seq_len(nrow(test)), each = 15), ]
dataFinal <- rbind(dataFinal, test)


# Fit decision tree with different predictors
test1 <- rpart(TREAT_RESULT ~ SEX + AGE, data = dataFinal)
rpart.plot(test1, type = 2)
test2 <- rpart(TREAT_RESULT ~ SEX + AGE + Hemoglobin_A1C + BNP_Test + BloodCoagulation_1 + BloodCoagulation_2 + BloodCoagulation_3, data = dataFinal)
rpart.plot(test2, type = 2)
test3 <- rpart(TREAT_RESULT ~ . - PATIENT_ID, data = dataFinal)
rpart.plot(test3, type = 2)
plotcp(test3)

test4 <- rpart(TREAT_RESULT ~ . - PATIENT_ID, data = dataFinal, cp = 0.005)
rpart.plot(test4, type = 2)
plotcp(test4)

# Compute prediction error
testpred <- predict(test4, newdata = dataFinal, type = 'class')
confusion_mat <- table(dataFinal$TREAT_RESULT, testpred)
err <- 1 - sum(diag(confusion_mat)) / sum(confusion_mat)
err