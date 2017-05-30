######### Machine Learning ######
######### HW 1 ###################
##########Susan Cherry############

######## Code Perceptron Algorithm 

#Read in the data
mnist_test_labels=as.matrix(read.table('/Users/susancherry/Desktop/Machine\ Learning/Homework\ 1/mnist_test_labels.txt'))
mnist_test=as.matrix(read.table('/Users/susancherry/Desktop/Machine\ Learning/Homework\ 1/mnist_test.txt'))
mnist_train=as.matrix(read.table('/Users/susancherry/Desktop/Machine\ Learning/Homework\ 1/mnist_train.txt'))
mnist_train_labels=as.matrix(read.table('/Users/susancherry/Desktop/Machine\ Learning/Homework\ 1/mnist_train_labels.txt'))


#Filter out everything but 0s and 1s.
test_filter=mnist_test_labels==1 | mnist_test_labels==0
train_filter=mnist_train_labels==1 | mnist_train_labels==0
mnist_test=mnist_test[test_filter,]
mnist_test_labels=mnist_test_labels[test_filter]
mnist_train=mnist_train[train_filter,]
mnist_train_labels=mnist_train_labels[train_filter]

# 0 will be assinged as "-1" and 1 will be assigned as "1"
mnist_test_labels[mnist_test_labels==0]=-1
mnist_train_labels[mnist_train_labels==0]=-1

#Convert to matrices
mnist_train_labels=as.matrix(mnist_train_labels)
mnist_train=as.matrix(mnist_train)
mnist_test_labels=as.matrix(mnist_test_labels)
mnist_test=as.matrix(mnist_test)

#Perceptron Algorithm
perceptron=function(x,y){
  t=1
  w=matrix(0,dim(x)[2],1)
  keepgoing=TRUE
  while(keepgoing){
    i=1
    run=TRUE
    while(run & i<=dim(x)[1] ){
      if(y[i]*(t(x[i,])%*%w)<=0){
        w=w+y[i]*x[i,]
        t=t+1
        run=FALSE
      }
      else{
        i=i+1
      }
    }
    print(i)
    if(i>dim(x)[1]){
      print('here')
      return(w)
      keepgoing=FALSE
    }
    
  }
}

w=perceptron(mnist_train,mnist_train_labels)

######## Testing 
#test train data
test_train=mnist_train%*%w
ones_train=test_train>0
test_train[ones_train]=1
neg_train=test_train<0
test_train[neg_train]=-1
Accuracy_train=sum(test_train==mnist_train_labels)/dim(mnist_train)[1]

#test test data
test=mnist_test%*%w
ones=test>0
test[ones]=1
neg=test<0
test[neg]=-1

Accuracy=sum(test==mnist_test_labels)/dim(mnist_test)[1]

########## Feature Reduction
#Method 1: Get rid of all features with weights that are 0

#remove features that have 0 weight
weight_not0=w!=0
not0_test=mnist_test[,weight_not0]
not0_train=mnist_train[,weight_not0]

#run perceptron on only the remaining weights
not_zero_weights=perceptron(not0_train,mnist_train_labels)


#test new weights using test data
test_not0=not0_test%*%not_zero_weights
ones_not0=test_not0>0
test_not0[ones_not0]=1
neg_not0=test_not0<0
test_not0[neg_not0]=-1

Accuracy_not0=sum(test_not0==mnist_test_labels)/dim(not0_test)[1]


#Method 2: Get rid of all features with low variance

#Calculate variance by columns in the training data
sd_by_feature=apply(mnist_train,2,sd)

#Discard features with sd less than 5
keep_sd=sd_by_feature>60
sd_test=mnist_test[,keep_sd]
sd_train=mnist_train[,keep_sd]

#run perceptron
sd_weights=perceptron(sd_train,mnist_train_labels)

#test sd weights using test data
test_sd=sd_test%*%sd_weights
ones_sd=test_sd>0
test_sd[ones_not0]=1
neg_=test_sd<0
test_sd[neg_]=-1

Accuracy_sd=sum(test_sd==mnist_test_labels)/dim(test_sd)[1]






