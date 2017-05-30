function [lambda_star,lambda_zero] = train(matrix)
%Train function for hard margin SVM. Takes in a matrix of features and labels
% Returns the optimal lambda values
%separate into features and labels
X=matrix(:,1:(end-1));
Y=matrix(:,end);

%if a label is 0, turn it into -1
zerolab=Y==0;
Y(zerolab)=-1;
Size=size(X);

%format the data for svm
Z=X;
for i=1:Size(1)
    Z(i,:)=X(i,:)*Y(i);
end;
D=Z*Z';

Ones=ones(Size(1),1) ;
%use quadprog to find the optimal solution given our constraints
solution=quadprog(D,-Ones,[],[],Y',0,zeros(Size(1),1) ,inf*ones(Size(1),1));

Matrix=zeros(Size(1),Size(2));
for i=1:Size(1)
    Matrix(i,:)=solution(i)*X(i,:)*Y(i);
end;

%use solution found above to find the correct values for lambda star and
%lambdazero
lambda_star=sum(Matrix,1);
num=X*lambda_star';
one_labels=Y==1;
Minimum=min(num(one_labels));
lambda_zero=1-Minimum;

%the function returns the optimal lambda values
end

