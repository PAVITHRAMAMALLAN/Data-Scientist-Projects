#-------------------------------Loading the Libraries-----------------------
rm(list=ls())
library(ggplot2)
library(caret)
library(DMwR)
library(MASS)
library(car)
library(rpart)
library(dummies)
library(corrgram)
library(randomForest)
library(e1071)
library(rpart.plot)
setwd("F:\\MBA\\Edwisor")
getwd()

#----------------------------------Reading the data-----------------------------

data <- read.csv("day.csv",header = T, na.strings = c("","", NA))

str(data)
sum(is.na(data))

#----------------------------------Understanding the Data------------------------
ggplot(data=data,aes(y=cnt, x = season))+
  geom_point(mapping=NULL, data= NULL, size=0.5, color = "RED")+
  labs(x="Season", y="Count")+
  ggtitle("Exploratory Data Analysis")+
  geom_abline()

ggplot(data=data,aes(y=cnt, x = mnth))+
  geom_point(mapping=NULL, data= NULL, size=0.5, color = "BLUE")+
  labs(x="Month", y="Count")+
  ggtitle("Exploratory Data Analysis")+
  geom_abline()

ggplot(data=data,aes(y=cnt, x = weathersit))+
  geom_point(mapping=NULL, data= NULL, size=0.5, color = "RED")+
  labs(x="Weathersit", y="Count")+
  ggtitle("Exploratory Data Analysis")+
  geom_abline()

ggplot(data=data,aes(y=cnt, x = atemp))+
  geom_point(mapping=NULL, data= NULL, size=0.5, color = "RED")+
  labs(x="atemperature", y="Count")+
  ggtitle("Exploratory Data Analysis")+
  geom_abline()

ggplot(data=data,aes(y=cnt, x = hum))+
  geom_point(mapping=NULL, data= NULL, size=0.5, color = "RED")+
  labs(x="Humidity", y="Count")+
  ggtitle("Exploratory Data Analysis")+
  geom_abline()

ggplot(data=data,aes(y=cnt, x = windspeed))+
  geom_point(mapping=NULL, data= NULL, size=0.5, color = "RED")+
  labs(x="Windspeed", y="Count")+
  ggtitle("Exploratory Data Analysis")+
  geom_abline()

ggplot(data=data,aes(y=cnt, x = casual))+
  geom_point(mapping=NULL, data= NULL, size=0.5, color = "RED")+
  labs(x="No of Casual users", y="Count")+
  ggtitle("Exploratory Data Analysis")+
  geom_abline()

ggplot(data=data,aes(y=cnt, x = registered))+
  geom_point(mapping=NULL, data= NULL, size=0.5, color = "RED")+
  labs(x="No of Registered users", y="Count")+
  ggtitle("Exploratory Data Analysis")+
  geom_abline()

histogram(data$hum,data)
histogram(data$windspeed,data)
histogram(data$temp,data)
histogram(data$cnt,data)
#------------------------------------Outliers Analysis---------------------------

ggplot(data=data,aes(y=data$hum))+
  stat_boxplot(geom = "errorbar", width = 0.5) +
  geom_boxplot(outlier.colour="red", fill = "yellow" ,outlier.shape=18, outlier.size=1, notch=FALSE)+
  labs(y="humidity", x="values")+
  ggtitle("Outlier Analysis")
ggplot(data=data,aes(y=data$weekday))+
  stat_boxplot(geom = "errorbar", width = 0.5) +
  geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18, outlier.size=1, notch=FALSE)+
  labs(y="Weekday", x="values")+
  ggtitle("Outlier Analysis")

#----------------------------------Feature Engineering--------------------------

data$season=as.factor(data$season)
data$mnth=as.factor(data$mnth)
data$yr=as.factor(data$yr)
data$holiday=as.factor(data$holiday)
data$weekday=as.factor(data$weekday)
data$workingday=as.factor(data$workingday)
data$weathersit=as.factor(data$weathersit)
d1=unique(data$dteday)
df=data.frame(d1)
data$dteday=as.Date(df$d1,format="%Y-%m-%d")
df$d1=as.Date(df$d1,format="%Y-%m-%d")
data$dteday=format(as.Date(df$d1,format="%Y-%m-%d"), "%d")
data$dteday=as.factor(data$dteday)

#-----------------------------------Feature Selection-----------------------------
library(usdm)
vif(data[,3:15])
data=subset(data,select = -c(instant,casual,registered))

numeric_index = sapply(data,is.numeric)
## Correlation Plot 
corrgram(data[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")
cor(data[,numeric_index])

## Dimension Reduction
data = subset(data,select = -c(atemp))


#--------------------------------------Sampling-----------------------------------
set.seed(11994)
train.index = sample(1:nrow(data), 0.8 * nrow(data))

train = data[ train.index,]
test  = data[-train.index,]

#-------------------------------------Model Development---------------------------

detach("package:usdm", unload=TRUE)
lm.fit=lm(cnt~.,data=train)
summary(lm.fit)
vif(lm.fit)
step=stepAIC(lm.fit)

lm.fit1=lm(cnt~.-dteday,data=train)
summary(lm.fit1)
vif(lm.fit1)

lm.fit2=lm(cnt~season + yr + mnth + holiday + weekday + weathersit + temp + hum + windspeed, data=train)
summary(lm.fit2)
vif(lm.fit2)

predictions_LM = predict(lm.fit2, test[,-12])
regr.eval(test[,12], predictions_LM)
rss <- sum((predictions_LM - test$cnt) ^ 2)
tss <- sum((test$cnt - mean(test$cnt)) ^ 2)  
rsq <- 1 - rss/tss
rsq

#MAE = 587.07
#MSE = 692444
#RMSE = 831
#MAPE = 0.183539
#Rsq = 0.8078589

#-----------------------------------Support Vector Machine------------------------

svm_fit = svm(cnt ~ ., data = train)
summary(svm_fit)
prediction_SVM = predict(svm_fit, test[-12])
regr.eval(test[,12], prediction_SVM)
rss <- sum((prediction_SVM - test$cnt) ^ 2)
tss <- sum((test$cnt - mean(test$cnt)) ^ 2)  
rsq <- 1 - rss/tss
rsq

#MAE = 543.4
#MSE = 524626
#RMSE = 724.31
#MAPE = 0.165052
#Rsq = 0.854425

#------------------------------------Decision Tree---------------------------------

rpart_fit = rpart(cnt ~ ., data = train, method = "anova")
rpart.plot(rpart_fit)
predictions_DT = predict(rpart_fit, test[,-12])
regr.eval(test[,12], predictions_DT)
rss <- sum((predictions_DT - test$cnt) ^ 2)
tss <- sum((test$cnt - mean(test$cnt)) ^ 2)  
rsq <- 1 - rss/tss
rsq

#MAE = 587.8
#MSE = 660668
#RMSE = 812.8
#MAPE = 0.188450
#Rsq = 0.816676

#------------------------------------Random Forest---------------------------------

RF_model = randomForest(cnt ~ ., train, importance = TRUE, ntree = 150)
predictions_RF = predict(RF_model, test[,-12])
plot(RF_model)
regr.eval(test[,12], predictions_RF)
rss <- sum((predictions_RF - test$cnt) ^ 2)
tss <- sum((test$cnt - mean(test$cnt)) ^ 2)  
rsq <- 1 - rss/tss
rsq

#MAE = 573.9
#MSE = 556472
#RMSE = 742.6
#MAPE = 0.2053828
#Rsq = 0.8455886


write.csv(test, file="sampleoutput.csv", row.names = F)
