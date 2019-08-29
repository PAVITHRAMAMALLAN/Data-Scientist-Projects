rm(list=ls())
library(ggplot2)
library(caret)
library(DMwR)
library(MASS)
library(geosphere)
library(car)
library(rpart)
library(dummies)
setwd("F:\\MBA\\Edwisor")
getwd()

#----------------------------------Reading the data-----------------------------

data <- read.csv("train_cab.csv",header = T, na.strings = c("","", 0, NA))

str(data)
data$fare_amount = as.numeric(as.character(data$fare_amount))
data = data[data$fare_amount >0,]
sum(is.na(data))


#--------------------------------Missing Value ANalysis--------------------------
missing_val = data.frame(apply(data,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(data)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]
missing_val

sum(is.na(data$pickup_latitude))
data[33,4]
data[33,4]=NA


#data$pickup_latitude[is.na(data$pickup_latitude)] = mean(data$pickup_latitude, na.rm = T)
#data$pickup_latitude[is.na(data$pickup_latitude)] = median(data$pickup_latitude, na.rm = T)

#data = knnImputation(data, k = 3)
#data = knnImputation(data, k = 5)
data = knnImputation(data, k = 10)
data[33,4]
#Actual Value : 40.77388
#MEAN : 40.71291
#MEDIAN : 40.7533
#knn (k=3): 40.77069
#knn (k=5): 40.77187
#knn (k=10): 40.77254

str(data)
length_passenger_count = data.frame(table(data$passenger_count))

#------------------------------------Outliers Analysis---------------------------
ggplot(data=data,aes(y=data$passenger_count))+
  stat_boxplot(geom = "errorbar", width = 0.5) +
  geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18, outlier.size=1, notch=FALSE)+
  labs(y="passenger_count", x="values")+
  ggtitle("Outlier Analysis")
ggplot(data=data,aes(y=data$fare_amount))+
  stat_boxplot(geom = "errorbar", width = 0.5) +
  geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18, outlier.size=1, notch=FALSE)+
  labs(y="fare_amount", x="values")+
  ggtitle("Outlier Analysis")

data$passenger_count = round(data$passenger_count,0)
data$passenger_count=ifelse(data$passenger_count>6|data$passenger_count<1,NA,data$passenger_count)
data$pickup_latitude = ifelse(data$pickup_latitude > 90 , NA, data$pickup_latitude)
numeric_index = sapply(data,function(x){is.numeric(x)})
numeric_data = data[,numeric_index]
numeric_data_name=colnames(numeric_data)
numeric_data_name
numeric_data_name=numeric_data_name[-6]
numeric_data_name
for (i in numeric_data_name){
  remove=data[,i][data[,i]%in% boxplot.stats(data[,i])$out] 
  data[,i][data[,i]%in% remove]=NA
}
sum(is.na(data))
data = na.omit(data)

ggplot(data, aes(x=data$fare_amount)) +
  geom_histogram(color="Black", fill="White") +
  ggtitle("After removing Outliers")
ggplot(data, aes(x=data$passenger_count)) +
  geom_histogram(color="Black", fill="White") +
  ggtitle("After removing Outliers")
  
histogram(data$fare_amount,data)
histogram(data$passenger_count,data)
histogram(data$pickup_longitude,data)
histogram(data$pickup_latitude,data)
histogram(data$dropoff_longitude,data)
histogram(data$dropoff_latitude,data)

#------------------------------------Feature Selection-------------------------------

data$date = as.Date(data$pickup_datetime)
data$time = format(as.POSIXlt(strptime(data$pickup_datetime, format="%Y-%m-%d %H:%M:%S", tz = "UTC")),format="%H:%M:%S")
data$hour = as.numeric(format(as.POSIXlt(strptime(data$time, format ="%H:%M:%S")), format = "%H"))
data$minutes = as.numeric(format(as.POSIXlt(strptime(data$time, format = "%H:%M:%S")),format = "%M"))
data$hour=ifelse(data$minutes>30,data$hour+1,data$hour)
data$hour=ifelse(data$hour<5,"Early-morning",ifelse(data$hour<10,"Morning",ifelse(data$hour<17,"Daytime",ifelse(data$hour<20,"Night","Late Night"))))
data$hour=as.factor(data$hour)
data$year = as.factor(format(data$date, format = "%Y"))
data$month = as.factor(format(data$date, format = "%m"))
data$day = as.factor(format(data$date, format = "%d"))
data$wday <- as.factor(weekdays(as.Date(data$date)))

data$passenger_count = as.factor(data$passenger_count)
str(data)

for (i in 1:nrow(data)){
  data$distance[i]=distGeo(c(data$pickup_longitude[i], data$pickup_latitude[i]), c(data$dropoff_longitude[i], data$dropoff_latitude[i]))
  data$distance[i]=data$distance[i]/1000
}
data = na.omit(data)
data_new = subset(data, select = -c(minutes, day, pickup_latitude, pickup_longitude, dropoff_longitude, dropoff_latitude, date, time, pickup_datetime))

d_hour = dummy(data_new$hour, sep = "_")
d_year = dummy(data_new$year, sep = "_")
d_month = dummy(data_new$month, sep = "_")
d_pass = dummy(data_new$passenger_count, sep = "")

data_train = cbind(data_new, d_hour, d_year, d_month, d_pass)
data_train = subset(data_train, select = -c(hour, year, month, wday, passenger_count))


#-------------------------------------Exploratory Data Analysis------------------------

ggplot(data_train, aes(x=data_train$fare_amount)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="Blue")+
  geom_density(alpha=.2)
ggplot(data_new, aes(x=data_new$distance)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="Green")+
  geom_density(alpha=.2)

ggplot(data=data_new,aes(y=fare_amount, x = passenger_count))+
  geom_point(mapping=NULL, data= NULL, size=0.5, color = "RED")+
  labs(x="passenger_count", y="Fare_amount")+
  geom_abline()+
  ggtitle("Exploratory Data Analysis")

ggplot(data=data_train,aes(y=fare_amount, x = distance))+
  geom_point(mapping=NULL, data= NULL, size=0.5, color = "RED")+
  labs(x="distance", y="Fare_amount")+
  ggtitle("Exploratory Data Analysis")+
  geom_abline()

ggplot(data=data,aes(y=data$fare_amount, x = wday))+
  geom_point(mapping=NULL, data= NULL, size=2, color = "BLACK")+
  labs(x="weekday", y="Fare_amount")+
  ggtitle("Exploratory Data Analysis")+
  geom_abline()
str(data_new)

histogram(x=data_new$hour, data=data_new)
barchart(x=data_new$year, data=data_new)
barchart(x=data_new$month, data=data_new)
barchart(x=data_new$wday, data=data_new)
barchart(x=data_new$distance, data=data_new)

#-----------------------------------------Sampling--------------------------------------

set.seed(11994)
train.index = sample(1:nrow(data_train), 0.8 * nrow(data_train))

train = data_train[ train.index,]
test  = data_train[-train.index,]

#--------------------------------------Model Development--------------------------------
## Multiple Linear Regression Model
library(usdm)
vif(train[,-1])

vifcor(train[,-1], th = 0.9)
detach("package:usdm", unload=TRUE)
lm.fit=lm(fare_amount~.,data=train)
summary(lm.fit)
vif(lm.fit)

variables = stepAIC(lm.fit)
lm.fit1 = lm(fare_amount ~ distance + hour_Daytime + `hour_Early-morning` + 
               `hour_Late Night` + hour_Morning + hour_Night + year_2009 + 
               year_2010 + year_2011 + year_2012 + year_2013 + year_2014 + 
               year_2015 + month_01 + month_02 + month_03 + month_04 + month_05 + 
               month_06 + month_07 + month_08 + month_09 + month_10 + month_11 + 
               month_12 + passenger_count1 + passenger_count2 + passenger_count3 + 
               passenger_count4 + passenger_count5 + passenger_count6, data=train)
summary(lm.fit1)
vif(lm.fit1)
lm.fit2 = lm(fare_amount ~ distance + hour_Daytime + `hour_Early-morning` + 
               `hour_Late Night` + hour_Morning + hour_Night + year_2009 + 
               year_2010 + year_2011 + year_2012 + year_2013 + year_2014 + 
               year_2015 + month_01 + month_02 + month_03 + month_04 + month_05 + 
               month_06 + month_07 + month_08 + month_09 + month_10 + month_11 + 
               month_12 + passenger_count1 + passenger_count2 + 
               passenger_count4 + passenger_count5 + passenger_count6, data=train)
summary(lm.fit2)
vif(lm.fit2)
lm.fit3 = lm(fare_amount ~ distance + hour_Daytime + `hour_Early-morning` + 
               `hour_Late Night` + hour_Morning + hour_Night + year_2009 + 
               year_2010 + year_2011 + year_2012 + year_2013 + year_2014 + 
               year_2015 + month_01 + month_02 + month_03 + month_04 + month_05 + 
               month_06 + month_07 + month_08 + month_09 + month_10 + month_11 + 
               month_12 + passenger_count1 + passenger_count2 + 
               passenger_count4 + passenger_count5, data=train)
summary(lm.fit3)
vif(lm.fit3)
lm.fit4 = lm(fare_amount ~ distance + hour_Daytime + `hour_Early-morning` + 
               `hour_Late Night` + hour_Morning + hour_Night + year_2009 + 
               year_2010 + year_2011 + year_2012 + year_2013 + year_2014 + 
               year_2015 + month_01 + month_02 + month_03 + month_04 + 
               month_06 + month_07 + month_08 + month_09 + month_10 + month_11 + 
               month_12 + passenger_count1 + passenger_count2 + 
               passenger_count4 + passenger_count5, data=train)
summary(lm.fit4)
vif(lm.fit4)
lm.fit5 = lm(fare_amount ~ distance + hour_Daytime + `hour_Early-morning` + 
               `hour_Late Night` + hour_Morning + hour_Night + year_2009 + 
               year_2010 + year_2011 + year_2012 + year_2013 + year_2014 + 
               year_2015 + month_01 + month_02 + month_03 + month_04 + 
               month_06 + month_07 + month_08 + month_09 + month_10 + month_11 + 
               month_12 + passenger_count1 + passenger_count2 + 
               passenger_count5, data=train)
summary(lm.fit5)
vif(lm.fit5)
lm.fit6 = lm(fare_amount ~ distance + hour_Daytime + `hour_Early-morning` + 
               `hour_Late Night` + hour_Morning + hour_Night + year_2009 + 
               year_2010 + year_2011 + year_2012 + year_2013 + year_2014 + 
               year_2015 + month_01 + month_02 + month_03 + month_04 + 
               month_06 + month_07 + month_08 + month_09 + month_10 + month_11 + 
               passenger_count1 + passenger_count2 + 
               passenger_count5, data=train)
summary(lm.fit6)
vif(lm.fit6)
lm.fit7 = lm(fare_amount ~ distance + hour_Daytime + `hour_Early-morning` + 
               `hour_Late Night` + hour_Morning + hour_Night + year_2009 + 
               year_2010 + year_2011 + year_2012 + year_2013 + year_2014 + 
               year_2015 + month_01 + month_02 + month_03 + month_04 + 
               month_06 + month_07 + month_08 + month_09 + month_10 + month_11+
               passenger_count1 + passenger_count2, data=train)
summary(lm.fit7)
vif(lm.fit7)
lm.fit8 = lm(fare_amount ~ distance + hour_Daytime + `hour_Early-morning` + 
               `hour_Late Night` + hour_Morning + year_2009 + 
               year_2010 + year_2011 + year_2012 + year_2013 + year_2014 + 
               month_01 + month_02 + month_03 + month_04 + 
               month_06 + month_07 + month_08 + month_09 + month_10 + month_11+
               passenger_count1, data=train)
summary(lm.fit8)
vif(lm.fit8)
lm.fit9 = lm(fare_amount ~ distance + hour_Daytime + `hour_Early-morning` + 
               `hour_Late Night` + hour_Morning + year_2009 + 
               year_2010 + year_2011 + year_2012 + year_2013 + year_2014 + 
               month_01 + month_02 + month_03 + month_04 + 
               month_07 + month_08 + month_09 + month_10 + month_11+
               passenger_count1, data=train)
summary(lm.fit9)
vif(lm.fit9)
lm.fit10 = lm(fare_amount ~ distance + hour_Daytime + `hour_Early-morning` + 
                `hour_Late Night` + hour_Morning + year_2009 + 
                year_2010 + year_2011 + year_2012 + year_2013 + year_2014 + 
                month_01 + month_02 + month_03 + month_04 + 
                month_07 + month_09 + month_10 + month_11+
                passenger_count1, data=train)
summary(lm.fit10)
vif(lm.fit10)
lm.fit11 = lm(fare_amount ~ distance + hour_Daytime + `hour_Early-morning` + 
                `hour_Late Night` + hour_Morning + year_2009 + 
                year_2010 + year_2011 + year_2012 + year_2013 + year_2014 + 
                month_01 + month_02 + month_03 + month_04 + 
                month_09 + month_10 + month_11+
                passenger_count1, data=train)
summary(lm.fit11)
vif(lm.fit11)
lm.fit12 = lm(fare_amount ~ distance + hour_Daytime + `hour_Early-morning` + 
                `hour_Late Night` + hour_Morning + year_2009 + 
                year_2010 + year_2011 + year_2012 + year_2013 + year_2014 + 
                month_01 + month_02 + month_03 +
                month_09 + month_10 + month_11+
                passenger_count1, data=train)
summary(lm.fit12)
vif(lm.fit12)
lm.fit13 = lm(fare_amount ~ distance + hour_Daytime + `hour_Early-morning` + 
                `hour_Late Night` + hour_Morning + year_2009 + 
                year_2010 + year_2011 + year_2012 + year_2013 + year_2014 + 
                month_01 + month_02 +
                month_09 + month_10 + month_11+
                passenger_count1, data=train)
summary(lm.fit13)
vif(lm.fit13)
lm.fit14 = lm(fare_amount ~ distance + `hour_Early-morning` + 
                `hour_Late Night` + hour_Morning + year_2009 + 
                year_2010 + year_2011 + year_2012 + year_2013 + year_2014 + 
                month_01 + month_02 + month_09 + month_10 + month_11+
                passenger_count1, data=train)
summary(lm.fit14)
vif(lm.fit14)

lm.fit15 = lm(fare_amount ~ distance + `hour_Early-morning` + 
                `hour_Late Night` + hour_Morning + year_2009 + 
                year_2010 + year_2011 + year_2012 + year_2013 + year_2014 + 
                month_01 + month_09 + month_10 + month_11+
                passenger_count1, data=train)
summary(lm.fit15)
vif(lm.fit15)
lm.fit16 = lm(fare_amount ~ distance + `hour_Early-morning` + 
                `hour_Late Night` + hour_Morning + year_2009 + 
                year_2010 + year_2011 + year_2012 + year_2013 + year_2014 + 
                month_01 + month_09 + month_10 + month_11, data=train)
summary(lm.fit16)
vif(lm.fit16)
lm.fit17 = lm(fare_amount ~ distance + `hour_Early-morning` + 
                `hour_Late Night` + hour_Morning + year_2009 + 
                year_2010 + year_2011 + year_2012 + year_2013 + year_2014 + 
                month_09 + month_10 + month_11, data=train)
summary(lm.fit17)
vif(lm.fit17)

##Decision Tree Regression Model
rpart_fit = rpart(fare_amount ~ ., data = train, method = "anova")
rpart_pred = predict(rpart_fit, test[,-1])
rss <- sum((rpart_pred - test$fare_amount) ^ 2)
tss <- sum((test$fare_amount - mean(test$fare_amount)) ^ 2)  
rsq <- 1 - rss/tss
rsq

#-----------------------------------------Model Acuracy----------------------------------

pred = predict(lm.fit16, test[,-1])
regr.eval(test[,1], pred)
#r2 = 0.6929
#MAE = 1.5862063
#MSE = 4.9341653
#RMSE = 2.221298
#MAPE = 0.1951627

regr.eval(test[,1], rpart_pred)
#r2 = 0.6223519
#MAE = 1.7679119
#MSE = 5.8136112
#RMSE = 2.4111431
#MAPE = 0.2196054

#-------------------------------------------Prediction----------------------------------

data_test <- read.csv("test.csv",header = T, na.strings = c(" ","",0,NA))
str(data_test)
data_test$date = as.Date(data_test$pickup_datetime)
data_test$time = format(as.POSIXlt(strptime(data_test$pickup_datetime, format="%Y-%m-%d %H:%M:%S", tz = "UTC")),format="%H:%M:%S")
data_test$hour = as.numeric(format(as.POSIXlt(strptime(data_test$time, format ="%H:%M:%S")), format = "%H"))
data_test$minutes = as.numeric(format(as.POSIXlt(strptime(data_test$time, format = "%H:%M:%S")),format = "%M"))
data_test$hour=ifelse(data_test$minutes>30,data_test$hour+1,data_test$hour)
data_test$hour=ifelse(data_test$hour<5,"Early-morning",ifelse(data_test$hour<10,"Morning",ifelse(data_test$hour<17,"Daytime",ifelse(data_test$hour<20,"Night","Late Night"))))
data_test$hour=as.factor(data_test$hour)
data_test$year = as.factor(format(data_test$date, format = "%Y"))
data_test$month = as.factor(format(data_test$date, format = "%m"))
data_test$day = as.factor(format(data_test$date, format = "%d"))
data_test$wday <- as.factor(weekdays(as.Date(data_test$date)))
data_test$passenger_count<-as.factor(data_test$passenger_count)
dt_hour = data.frame(model.matrix(~hour,data_test))
dt_hour =dt_hour[,-1]
dt_year = data.frame(model.matrix(~year,data_test))
dt_year = dt_year[,-1]
dt_month = data.frame(model.matrix(~month,data_test))
dt_month = dt_month[,-1]
dt_pass = data.frame(model.matrix(~passenger_count,data_test))
dt_pass = dt_pass[,-1]
data_test = cbind(data_test, dt_hour, dt_year, dt_month, dt_pass)
data_test = subset(data_test, select = -c(date, time, day, minutes, hour, month))
str(data_test)

for (i in 1:nrow(data_test)){
  data_test$distance[i]=distGeo(c(data_test$pickup_longitude[i], data_test$pickup_latitude[i]), c(data_test$dropoff_longitude[i], data_test$dropoff_latitude[i]))
  data_test$distance[i]=data_test$distance[i]/1000
}
data_test = subset(data_test, select = -c(passenger_count, pickup_datetime, pickup_latitude, pickup_longitude, dropoff_longitude, dropoff_latitude))
testsubmission <- read.csv("test.csv",header = T, na.strings = c(" ","",0,NA))
testsubmission$result <- data.frame(predict(lm.fit13,data_test))
write.csv(testsubmission$result, file="test_ans.csv", row.names = F)
