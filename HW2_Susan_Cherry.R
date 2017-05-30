###### Homework 2 ######
#### Susan Cherry ####

### Question 1 ###

#import data
data = as.matrix(read.csv('/Users/susancherry/Desktop/Machine_Learning/Homework_2/dataset.csv'))
#set parameters 
theta=matrix(c(.05,-3,2.5),nrow=1,ncol=3)
theta_0=.3

#### Part A) Calculate g(x) for each data point
g=rep(0,dim(data)[1])

for (i in 1:dim(data)[1]){
  print(i)
  g[i]=theta%*%data[i,1:3]+theta_0
}

#Threshold: If g(x)<=0, then it is predicted as "0". If g(x)>0, then it is predicted as "1"

#### Part B) Calculate g(x) for each data point
f=rep(0,dim(data)[1])

for (i in 1:dim(data)[1]){
  z=theta%*%data[i,1:3]+theta_0
  f[i]=(1+exp(1)^(-z))^(-1)
}


#### Part C) Calculate ROC curve
#set valuse that we will sweep over
range=c(min(f),max(f))
x=range[1]

#caluclate number of labels in each category
y_1=sum(data[,4]==1)
y_0=sum(data[,4]==0)
data=cbind(data,f)

#Calculate ROC coordinates
coordinates=matrix(0,10,2)
ROC_f=data.frame(cbind(f,data[,4]))
ROC_f=ROC_f[order(ROC_f$f),]
#plot the ROC curve

FP=c(5,4,3,2,1,0,0,0,0,0,0)
TP=c(5,5,5,5,5,5,4,3,2,1,0)
FPR=FP/y_0
TPR=TP/y_1
plot(FPR,TPR,ylab="TPR",xlab="FPR",main="ROC Curve for f(x)",pch=16,cex=2)

##AUC

### Question 2 ###
#











