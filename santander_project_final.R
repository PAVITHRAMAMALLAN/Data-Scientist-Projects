rm(list=ls(all=T))
setwd("F:\\MBA\\Edwisor\\project_santander")

#Loading the Libraries
x <- c("ggplot2","ggpubr","randomForest","caret", "class", "e1071", 
              "rpart", "DMwR","usdm","dplyr","caTools",
              "C50","car","DataCombine","inTrees","pROC","xgboost","data.table", "mlr")

lapply(x, library, character.only = TRUE)


#Reading the data
data=read.csv("train.csv", header = T, na.strings = c(" ", "", NA))

#-----------------------------------------------Exploratory Data Analysis--------------------------------------

#Understanding the data
str(data)
unique(data$target)
class(data)
dim(data)
head(data)
names(data)
str(data)
summary(data)

#-----------------------------------------------Missing value Analysis-----------------------------------------

#Total number of Missing values in the data
sum(is.na(data))
#There are no missing values in the dataset.

#------------------------------------------------Outlier Analysis----------------------------------------------

ggplot(data=data,aes(y=data$var_0))+
  stat_boxplot(geom = "errorbar", width = 0.5) +
  geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18, outlier.size=1, notch=FALSE)+
  labs(y="Var_0", x="values")+
  ggtitle("Outlier Analysis")

ggplot(data=data,aes(y=data$var_15))+
  stat_boxplot(geom = "errorbar", width = 0.5) +
  geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18, outlier.size=1, notch=FALSE)+
  labs(y="Var_15", x="values")+
  ggtitle("Outlier Analysis")

ggplot(data=data,aes(y=data$var_109))+
  stat_boxplot(geom = "errorbar", width = 0.5) +
  geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18, outlier.size=1, notch=FALSE)+
  labs(y="Var_109", x="values")+
  ggtitle("Outlier Analysis")

ggplot(data=data,aes(y=data$var_154))+
  stat_boxplot(geom = "errorbar", width = 0.5) +
  geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18, outlier.size=1, notch=FALSE)+
  labs(y="Var_154", x="values")+
  ggtitle("Outlier Analysis")

#----------------------------------------------Feature Selection-----------------------------------------------

#PCA - Principle Component Analysis
data.pca = prcomp(data[,3:202], scale = TRUE)
summary(data.pca)

#----------------------------------------------Feature Scaling-------------------------------------------------

#Selecting the Predictors which are continous in nature
cnames = colnames(data[ , 3:202])

#Normalization
for(i in cnames){
  data[,i] = (data[,i] - min(data[,i])) / (max(data[,i])-min(data[,i]))
}

#Deselecting the ID_code variable
data=data[,-1]

#-------------------------------------------------Sampling-----------------------------------------------------

#Simple Random Sampling
set.seed(2405)
train.index = createDataPartition(data$target, p = .80, list = FALSE)
train = data[train.index,]
test =  data[-train.index,]

#-------------------------------------------------Modeling------------------------------------------------------
#Logistic Regression
log_model = glm(target~., data=train, family="binomial")
summary(log_model)

log_model1 = glm(target~var_0 + var_2 + var_3 + var_4 + var_5 + var_6 + var_8 + var_9 + var_11 +
                  var_12 + var_13 + var_15 + var_18 + var_20 + var_21 + var_22+ var_23 + var_24 +
                  var_25 + var_26 + var_28 + var_31 + var_32 + var_33 + var_34 + var_35 + var_36 +
                  var_40 + var_43 + var_44 + var_45 + var_48 + var_49 + var_50 + var_51 + var_52 +
                  var_53 + var_54 + var_55 + var_56 + var_57 + var_58 + var_59 + var_62 + var_63 +
                  var_64 + var_66 + var_67 + var_68 + var_70 + var_71 + var_72 + var_74 + var_75 +
                  var_76 + var_77 + var_78 + var_80 + var_81 + var_82 + var_83 + var_84 + var_85 +
                  var_86 + var_87 + var_88 + var_89 + var_90 + var_91 + var_92 + var_93 + var_94 +
                  var_95 + var_97 + var_99 + var_101 + var_102 + var_104 + var_105 + var_106 + var_107 +
                  var_108 + var_109 + var_110 + var_111 + var_112 + var_113 + var_114 + var_115 + var_116 +
                  var_118 + var_119 + var_121 + var_122 + var_123 + var_125 + var_127 + var_128 + var_130 +
                  var_131 + var_132 + var_133 + var_134 + var_135 + var_137 + var_138 + var_139 + var_140 +
                  var_141 + var_142 + var_143 + var_144 + var_145 + var_146 + var_147 + var_148 + var_149 +
                  var_150 + var_151 + var_154 + var_155 + var_156 + var_157 + var_159 + var_162 + var_163 +
                  var_164 + var_165 + var_166 + var_167 + var_168 + var_169 + var_170 + var_171 + var_172 +
                  var_173 + var_174 + var_175 + var_177 + var_178 + var_179 + var_180 + var_181 + var_182 +
                  var_184 + var_186 + var_187 + var_188 + var_190 + var_191 + var_192 + var_193 + var_194 +
                  var_195 + var_196 + var_197 + var_198 + var_199, data=train, family="binomial")

#predict using logistic regression
log_Predictions = predict(log_model, newdata = test, type = "response")
log_Predictions1 = predict(log_model1, newdata = test, type = "response")

#converting probabilities
log_Predictions = ifelse(log_Predictions > 0.5, 1, 0)
log_Predictions1 = ifelse(log_Predictions1 > 0.5, 1, 0)

#Evaluate the performance of classification model
ConfMatrix_LR = table(test$target, log_Predictions)
ConfMatrix_LR

ConfMatrix_LR1 = table(test$target, log_Predictions1)
ConfMatrix_LR1

#Accuracy
Accuracy_LR = (ConfMatrix_LR[1,1] + ConfMatrix_LR[2,2]) / 
  (ConfMatrix_LR[1,1] + ConfMatrix_LR[1,2] + ConfMatrix_LR[2,1] + ConfMatrix_LR[2,2])
Accuracy_LR

Accuracy_LR1 = (ConfMatrix_LR1[1,1] + ConfMatrix_LR1[2,2]) / 
  (ConfMatrix_LR1[1,1] + ConfMatrix_LR1[1,2] + ConfMatrix_LR1[2,1] + ConfMatrix_LR1[2,2])
Accuracy_LR1

#False Negative rate
#FNR = FN/FN+TP 
FNR_LR = ConfMatrix_LR[2,1] /(ConfMatrix_LR[2,1] + ConfMatrix_LR[2,2])
FNR_LR

FNR_LR1 = ConfMatrix_LR1[2,1] /(ConfMatrix_LR1[2,1] + ConfMatrix_LR1[2,2])
FNR_LR1
#Accuracy = 91.40%
#FNR = 72.54%

#Model performance
test$target = as.numeric(test$target)
plot.roc(test$target, log_Predictions)
auc(roc(test$target, log_Predictions)) 
#AUC value = 0.63

#----------------------------------------------------XGBoost----------------------------------------------
#XGBoost Algorithm
setDT(train)
setDT(test)
labels <- train$target 
ts_label <- test$target
new_tr <- model.matrix(~.+0,data = train[,-c("target"),with=F]) 
new_ts <- model.matrix(~.+0,data = test[,-c("target"),with=F])

#convert factor to numeric 
dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)
params <- list(booster = "gbtree", objective = "binary:logistic",
               eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)
xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 5, showsd = T, stratified = T,
                 print_every_n = 10, early_stop_round = 20, maximize = F)

#first default - model training
xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 100, watchlist = list(val=dtest,train=dtrain),
                   print_every_n = 10, earlY_stop_round = 10, maximize = F , eval_metric = "error")

#model prediction
XG_Predictions <- predict (xgb1,dtest)
XG_Predictions <- ifelse (XG_Predictions > 0.5,1,0)
XG_Predictions <- as.factor(XG_Predictions)
ts_label <-as.factor(ts_label)

#confusion matrix
ConfMatrix_XG = table(XG_Predictions, ts_label)
ConfMatrix_XG

#Accuracy
Accuracy_XG = (ConfMatrix_XG[1,1] + ConfMatrix_XG[2,2]) / 
  (ConfMatrix_XG[1,1] + ConfMatrix_XG[1,2] + ConfMatrix_XG[2,1] + ConfMatrix_XG[2,2])
Accuracy_XG

#False Negative rate
#FNR = FN/FN+TP 
FNR_XG = ConfMatrix_XG[2,1] /(ConfMatrix_XG[2,1] + ConfMatrix_XG[2,2])
FNR_XG
#Accuracy = 91.5%` 
#FNR = 27.97%

#Model performance
plot.roc(test$target, XG_Predictions)
auc(roc(test$target, XG_Predictions)) 
#AUC value = 0.821

#----------------------------------------------------Naive Bayes-------------------------------------------

#Naive Bayes Classification Model
NB_model = naiveBayes(as.factor(train$target) ~ ., data = train)

#predict on test cases
NB_Predictions = predict(NB_model, test[,2:201], type = 'class')

#Confusion matrix
ConfMatrix_NB = table(observed = test$target, predicted = NB_Predictions)
ConfMatrix =confusionMatrix(Conf_matrix)

#Accuracy
Accuracy_NB = (ConfMatrix_NB[1,1] + ConfMatrix_NB[2,2]) / 
  (ConfMatrix_NB[1,1] + ConfMatrix_NB[1,2] + ConfMatrix_NB[2,1] + ConfMatrix_NB[2,2])
Accuracy_NB

#False Negative rate
#FNR = FN/FN+TP 
FNR_NB = ConfMatrix_NB[2,1] /(ConfMatrix_NB[2,1] + ConfMatrix_NB[2,2])
FNR_NB
#Accuracy: 92.31%
#FNR: 62.%59

#Model performance
NB_Predictions=as.numeric(NB_Predictions)
plot.roc(test$target, NB_Predictions)
auc(roc(test$target, NB_Predictions)) 
#AUC value = 0.6793

#----------------------------------------------------Decision Tree------------------------------------------

#Decision tree for classification
#Develop Model on training data
C50_model = C5.0(as.factor(train$target) ~., train, trials = 10, rules = TRUE)

#Summary of DT model
summary(C50_model)
plot(C50_model)
#Lets predict for test cases
C50_Predictions = predict(C50_model, test[,-1], type = "class")

##Evaluate the performance of classification model
ConfMatrix_C50 = table(test$target, C50_Predictions)
confusionMatrix(ConfMatrix_C50)

Accuracy_C50 = (ConfMatrix_C50[1,1]+ConfMatrix_C50[2,2])/
  (ConfMatrix_C50[1,1]+ConfMatrix_C50[1,2]+ConfMatrix_C50[2,1]+ConfMatrix_C50[2,2])
Accuracy_C50 

#False Negative rate
#FNR = FN/FN+TP 
FNR_C50 = ConfMatrix_C50[2,1] /(ConfMatrix_C50[2,1]+ConfMatrix_C50[2,2])
FNR_C50
#Accuracy: 89.745%
#FNR: 83.86%

#Model performance
plot.roc(test$target, C50_Predictions)
auc(roc(test$target, C50_Predictions)) 
#AUC value = 0.536

#-------------------------------------------------Random Forest------------------------------------------

#Random Forest
RF_model = randomForest(as.factor(train$target) ~ ., train, importance = TRUE, ntree = 500)

#Presdict test data using random forest model
RF_Predictions = predict(RF_model, test[,-1])


##Evaluate the performance of classification model
ConfMatrix_RF = table(test$target, RF_Predictions)
ConfMatrix_RF

Accuracy_RF = (ConfMatrix_RF[1,1]+ConfMatrix_RF[2,2])/
  (ConfMatrix_RF[1,1]+ConfMatrix_RF[1,2]+ConfMatrix_RF[2,1]+ConfMatrix_RF[2,2])
Accuracy_RF 

#False Negative rate
#FNR = FN/FN+TP 
FNR_RF = ConfMatrix_RF[2,1] / (ConfMatrix_RF[2,1]+ConfMatrix_RF[2,2])
FNR_RF
#Accuracy: 89.934%
#FNR: 99.88093%

#Model performance
RF_Predictions = as.numeric(RF_Predictions)
plot.roc(test$target, RF_Predictions)
auc(roc(test$target, RF_Predictions)) 
#AUC value = 0.5006

#-------------------------------------------------Predictions--------------------------------------------------

#Read the Test data
data_test = read.csv("test.csv", header = T, na.strings = c(" ", "", NA))

cnames = colnames(data_test[ ,2:201])

for(i in cnames){
  data_test[,i] = (data_test[,i] - min(data_test[,i])) / (max(data_test[,i])-min(data_test[,i]))
}

data_test=data_test[,-1]

#Using XGBoost algorithm for prediction
setDT(train)
setDT(data_test)
labels <- train$target 
dt_label <- data_test$var_0
new_tr <- model.matrix(~.+0,data = train[,-c("target"),with=F]) 
new_dt <- model.matrix(~.+0,data = data_test)

#convert factor to numeric 
dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
d_test <- xgb.DMatrix(data = new_dt,label = dt_label)
params <- list(booster = "gbtree", objective = "binary:logistic",
               eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

#first default - model training
xgb2 <- xgb.train (params = params, data = dtrain, nrounds = 100, watchlist = list(val=d_test,train=dtrain),
                   print_every_n = 10, earlY_stop_round = 10, maximize = F , eval_metric = "error")

#model prediction
Predictions <- predict (xgb2,d_test)
Predictions <- ifelse (Predictions > 0.5,1,0)
Predictions <- as.factor(Predictions)
data_test = cbind(data_test, Predictions)
colnames(data_test)

#Writing the results into the hard disk
write.csv(data_test, file="test_ans.csv", row.names = F)
