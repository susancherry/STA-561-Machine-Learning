%%%%%%%% Susan Cherry %%%%%%%%
%%%%%% Machine Learning %%%%%%
%%%%%%%%% Homework 3 %%%%%%%%%

%Loading the Data
haberman1=load('/Users/susancherry/Desktop/Machine_Learning/Homework_3/haberman.txt');
data=haberman1;
died=data(:,4)==2;%Change the labels for people who died to 0
data(died,4)=0;
haberman1(died,4)=0;
data=mat2dataset(data); %Convert to dataset 

sequence=mat2dataset((1:1:length(data))');
data=horzcat(sequence,data);

%Divide into 10 Test Folds Randomly
data=data(randsample(1:length(data),length(data)),:);

AUCmatrix=zeros(10,5); %initialize matrices
chosen_svm=zeros(1,10);
chosen_tree=zeros(1,10);
svm_loss=zeros(1,10);
tree_loss=zeros(1,10);

%%Cross Validation
for i=1:10;
    %Separate into folds. I randomly reorginized the data, so now I'm just
    %taking sections of my dataset. 
    num=round(length(data)/10);
    index1=(i-1)*num+1; %find appropriate index
    index2=(i-1)*num+num;
    if index2>length(data);
        index2=length(data);
    end;
    fold=data(index1:index2,:); %take that section of the dataset
    Xtest=double(fold(:,2:4)); %split into test and fold sections
    Ytest=logical(double(fold(:,5)));
    used=fold(:,1);
    used=ismember(data(:,1),used);
    X=double(data(~used,2:4));
    Y=logical(double(data(~used,5)));
    
    %%ROC Curve for Each Feature
    %Feature 1 Age
    Yscores = Xtest(:,1);
    [Xroc1, Yroc1, Tcoc, AUC] = perfcurve(Ytest, Yscores, 'true');

    %Feature 2 Year 
    Yscores = Xtest(:,2);
    [Xroc2, Yroc2, Tcoc, AUC] = perfcurve(Ytest, Yscores, 'true');
    
    %Feature 3 Positive Nodes
    Yscores = -Xtest(:,3);
    [Xroc3, Yroc3, Tcoc, AUC] = perfcurve(Ytest, Yscores, 'true');
   
    %Plot
    figure   
    plot(Xroc1, Yroc1)
    hold on
    plot(Xroc2, Yroc2)
    plot(Xroc3, Yroc3)
    legend('Feature 1: Age', 'Feature 2; Year', 'Feature 3: Positive Nodes','Location','southeast')
    xlabel('false positive rate');
    ylabel('true positive rate');
    output=sprintf('Fold %d, ROC Curves for Features',i);
    title(output)
    hold off
    name=sprintf('Features%d',i);
    print(name, '-dpdf');
  
    %%%Test Each Algorithm on the Folds
    %Logistic Regression
    glmModel = fitglm(X, Y, 'Distribution', 'binomial', 'Link', 'logit');
    Yscores = predict(glmModel, Xtest); % these are the posterior probabilities
                                    % of class 1 for the test data
    [Xglm, Yglm, Tglm, AUCglm] = perfcurve(Ytest, Yscores, 'true');
    %store the values from this iteration
    AUCmatrix(i,1)=AUCglm;
 
    %Random Forest
    rfModel = fitensemble(X, Y, 'Bag', 100, 'Tree', 'Type', 'Classification');
    [~, Yscores] = predict(rfModel, Xtest);
    [Xrf, Yrf, Trf, AUCrf] = perfcurve(Ytest, Yscores(:, 2), 'true');
    AUCmatrix(i,4)=AUCrf;
    %Boosted Trees
    btModel = fitensemble(X, Y, 'AdaBoostM1', 100, 'Tree');
    [~, Yscores] = predict(btModel, Xtest);

    [Xbt, Ybt, Tbt, AUCbt] = perfcurve(Ytest, sigmf(Yscores(:, 2), [1 0]), ...
                                   'true');
    AUCmatrix(i,5)=AUCbt;
    
    
    %Values I will try for the SVM Kernel Scale Parameter
    kvalues=[.5 1 1.5 2]; %I also try the auto feature in matlab
    %values I try for CART's min leaf size
    minvalues=[1 2 3 4 5];
      
    %%%Nested Cross Validation
    
    svmloss=zeros(9,5);
    treeloss=zeros(9,5);
    
    for j=1:9;
        %Split into folds
       num=round(length(X)/9);
       index1=(j-1)*num+1;
       index2=(j-1)*num+num;
       if index2>length(X);
            index2=length(X);
       end;
       sequence=mat2dataset((1:1:length(X))');
       Xseq=horzcat(sequence,mat2dataset(X)); 
       Yseq=horzcat(sequence,mat2dataset(Y));
       Yvalidation=Y(index1:index2);
       Xvalidation=X(index1:index2,:);
       used=Xseq(index1:index2,1);
       used=ismember(Xseq(:,1),used);
       X_Nested_Train=double(Xseq(~used,2:4));
       Y_Nested_Train=double(Yseq(~used,2));
       
     
       %Try each parameter value for SVM
       for k=1:4;
           svmModel = fitcsvm(X_Nested_Train, Y_Nested_Train, 'Standardize', true, 'KernelScale',kvalues(k),'KernelFunction', 'rbf');
           svmModel = fitPosterior(svmModel);
           svmloss(j,k)=loss(svmModel,Xvalidation,Yvalidation);
       end;

       svmModel = fitcsvm(X_Nested_Train, Y_Nested_Train, 'Standardize', true, 'KernelScale','auto','KernelFunction', 'rbf');
       svmModel = fitPosterior(svmModel);
       svmloss(j,5)=loss(svmModel,Xvalidation,Yvalidation);

        %Try each parameter value for CART
       for k=1:5;
           ctreeModel = fitctree(X_Nested_Train, Y_Nested_Train,'MinLeafSize',minvalues(k));
           treeloss(j,k)=loss(ctreeModel,Xvalidation,Yvalidation);
        end;
       
    end;
    
    mean_svm=mean(svmloss,1); %find the one that minimized the mean loss
    mean_tree=mean(treeloss,1);
    chosen_svm(i)=find(mean_svm==min(mean_svm));
    chosen_tree(i)=find(mean_tree==min(mean_tree));
    
    % Use the chosen parameter value over the test fold
    %SVM
    if chosen_svm(i)==5;
        svmModel = fitcsvm(X, Y, 'Standardize', true, 'KernelScale','auto','KernelFunction', 'rbf');
    else
         svmModel = fitcsvm(X, Y, 'Standardize', true, 'KernelScale',kvalues(chosen_svm(i)),'KernelFunction', 'rbf');

    end;
        svmModel = fitPosterior(svmModel);
        
    [~, Yscores] = predict(svmModel, Xtest);
    [Xsvm, Ysvm, Tsvm, AUCsvm] = perfcurve(Ytest, Yscores(:, 2), 'true');
    AUCmatrix(i,2)=AUCsvm;
    svm_loss(i)=loss(svmModel,Xtest,Ytest);
    
    %CART
    ctreeModel = fitctree(X, Y,'MinLeafSize',minvalues(chosen_tree(i)));
    tree_loss(i)=loss(ctreeModel,Xtest,Ytest);
    [~, Yscores, ~, ~] = predict(ctreeModel, Xtest);
    [Xcart, Ycart, Tcart, AUCcart] = perfcurve(Ytest, Yscores(:, 2), 'true');
    AUCmatrix(i,3)=AUCcart;

    
   %plot ROC curves 
figure   
plot(Xglm, Yglm)
hold on
plot(Xsvm, Ysvm)
plot(Xcart, Ycart)
plot(Xrf, Yrf)
plot(Xbt, Ybt)

legend('Logistic Regression', 'Support Vector Machine', 'CART', ...
       'Random Forest', 'Boosted Trees','Location','southeast')
xlabel('false positive rate');
ylabel('true positive rate');
output=sprintf('Fold %d, ROC Curves for Classification Algorithms',i);
title(output)
hold off
name=sprintf('Graph%d',i);
print(name, '-dpdf');

 
end;
