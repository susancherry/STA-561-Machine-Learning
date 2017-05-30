%%%%%%%%%%%%%% Machine Learning %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Homework 4 %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% Susan Cherry %%%%%%%%%%%%%%%%

%% Question 3, Part A
%Testing my functions on the Iris Data set
load fisheriris
species=ones(1,100);
%format the data
species(1:50)=-1;
dataset=horzcat(meas(1:100,:),species');
data=meas(1:100,:);

%try out my functions
[lambda_star, lambda_0]=train(dataset);
iris_predictions=predict(lambda_star, lambda_0,dataset(:,1:(end-1)));

%see if it worked. It did!
sum(iris_predictions==species)

%% Question 3, Part B

%For this question, I'm just using the inner product k(x,z) = x'z

%load credit card dataset and split into tran versus test

%randomly shuffle
creditCard=creditCard(randsample(1:length(creditCard),length(creditCard)),:);
creditCard=double(creditCard);

%split into test and train
Index=round(9*(length(creditCard)/10));
train_credit=creditCard(1:Index,:);
test_credit=creditCard((Index+1):end,:);

%train the svm
svmModel = fitcsvm(train_credit(:,1:9), logical(double(train_credit(:,10))), 'KernelFunction', 'linear','Standardize',true);
svmModel = fitPosterior(svmModel);

%predict
[~, test_predictions] = predict(svmModel, test_credit(:,1:9));
[Xsvm, Ysvm, Tsvm, AUCsvm] = perfcurve(logical(double(test_credit(:,10))), test_predictions(:, 2), 'true');

%misclassification
svm_misclass=loss(svmModel,test_credit(:,1:9),logical(double(test_credit(:,10))));

%plot the ROC Curve
figure   
plot(Xsvm, Ysvm)
title('ROC Curve for k(x,z)=x^Tz');
print -dpdf P1

%% Question 3, Part C

%Now use radial base kernel with variance 2 
svmModel = fitcsvm(train_credit(:,1:9), logical(double(train_credit(:,10))), 'KernelFunction', 'rbf','KernelScale',sqrt(2),'Standardize',true);
svmModel = fitPosterior(svmModel);

%predict
[~, test_predictions] = predict(svmModel, test_credit(:,1:9));
[Xsvm, Ysvm, Tsvm, AUCsvm] = perfcurve(logical(double(test_credit(:,10))), test_predictions(:, 2), 'true');

%misclassification
svm_misclass_2=loss(svmModel,test_credit(:,1:9),logical(double(test_credit(:,10))));

%plot the ROC Curve
figure   
plot(Xsvm, Ysvm)
title('ROC Curve for RBF kernel and \sigma^2 = 2');
print -dpdf P2


%Now use radial base kernel with variance 20 
svmModel = fitcsvm(train_credit(:,1:9), logical(double(train_credit(:,10))), 'KernelFunction', 'rbf','KernelScale',sqrt(20),'Standardize',true);
svmModel = fitPosterior(svmModel);

%predict
[~, test_predictions] = predict(svmModel, test_credit(:,1:9));
[Xsvm, Ysvm, Tsvm, AUCsvm] = perfcurve(logical(double(test_credit(:,10))), test_predictions(:, 2), 'true');

%misclassification
svm_misclass_3=loss(svmModel,test_credit(:,1:9),logical(double(test_credit(:,10))));

%plot the ROC Curve
figure   
plot(Xsvm, Ysvm)
title('ROC Curve for RBF kernel and \sigma^2 = 20');
print -dpdf P3

%%%% Question 1 plots
x=linspace(-1,6);
g=log(1+exp(-x));
eq1=(1/(log(2)))*g;
eq2=max(0,1-x);

figure
plot(x,eq1,'LineWidth',2)
hold on
ylabel('Loss')
xlabel('Zeta Value')
title('(1/ln(2))*Logistic and Hinge Loss')
plot(x,eq2,'LineWidth',2)
legend('(1/ln(2))*Logistic','Hinge Loss')

x=linspace(-1,50);
g=log(1+exp(-x));
eq1=(1/(log(2)))*g;
eq2=max(0,1-x);
figure
plot(x,eq1,'LineWidth',2)
hold on
ylabel('Loss')
xlabel('Zeta Value')
title('(1/ln(2))*Logistic and Hinge Loss')
plot(x,eq2,'LineWidth',2)
legend('(1/ln(2))*Logistic','Hinge Loss')

