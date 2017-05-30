function [predictions] = predict(lambda_star,lambda_0,data)
%This function takes in optimal lambda values and a dataset
%   returns the predictions using the results from the train function

predictions=zeros(1,size(data,1));
%use lambda values to create optimal x
for j=1:size(data,1)
    Sum=0;
    for i=1:size(data,2)
        Sum=Sum+lambda_star(i)*data(j,i);
    end;
    predictions(j)=Sum;
end;
predictions=predictions+lambda_0;

%assign predictions based off of the sign of f
pos=predictions>=0;
predictions(pos)=1;
predictions(~pos)=-1;

%the function returns the predictions.

end

