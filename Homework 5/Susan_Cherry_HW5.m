
%%%%%%%%%%% Question 2 %%%%%%%%%%%%
m=5;
n=100;
%choose which coin for each sample. <0.5 is Coin A. >0.5 is Coin B.
which_coin=rand(5,1);

%Basied on the coins chosen above, randomly generate heads or tails. 
m1=rand(100,1);
m1(m1>.8)=0;
m1(m1~=0)=1;

m2=rand(100,1);
m2(m2>.35)=0;
m2(m2~=0)=1;

m3=rand(100,1);
m3(m3>.8)=0;
m3(m3~=0)=1;

m4=rand(100,1);
m4(m4>.8)=0;
m4(m4~=0)=1;

m5=rand(100,1);
m5(m5>.35)=0;
m5(m5~=0)=1;

%randomly initialize 
theta_A=rand(1,1);
theta_B=rand(1,1);
prev_theta_A=inf;
prev_theta_B=inf;

blist=theta_B;
alist=theta_A;

%Continue until converges 
while prev_theta_A~=theta_A || prev_theta_B~=theta_B
    prev_theta_A=theta_A;
    prev_theta_B=theta_B;
    
    %E Step
    p_1=theta_A^sum(m1)*(1-theta_A)^(100-sum(m1))/(theta_A^sum(m1)*(1-theta_A)^(100-sum(m1))+theta_B^sum(m1)*(1-theta_B)^(100-sum(m1)));
    p_2=theta_A^sum(m2)*(1-theta_A)^(100-sum(m2))/(theta_A^sum(m2)*(1-theta_A)^(100-sum(m2))+theta_B^sum(m2)*(1-theta_B)^(100-sum(m2)));
    p_3=theta_A^sum(m3)*(1-theta_A)^(100-sum(m3))/(theta_A^sum(m3)*(1-theta_A)^(100-sum(m3))+theta_B^sum(m3)*(1-theta_B)^(100-sum(m3)));
    p_4=theta_A^sum(m4)*(1-theta_A)^(100-sum(m4))/(theta_A^sum(m4)*(1-theta_A)^(100-sum(m4))+theta_B^sum(m4)*(1-theta_B)^(100-sum(m4)));
    p_5=theta_A^sum(m5)*(1-theta_A)^(100-sum(m5))/(theta_A^sum(m5)*(1-theta_A)^(100-sum(m5))+theta_B^sum(m5)*(1-theta_B)^(100-sum(m5)));
    
    d_1=theta_B^sum(m1)*(1-theta_B)^(100-sum(m1))/(theta_A^sum(m1)*(1-theta_A)^(100-sum(m1))+theta_B^sum(m1)*(1-theta_B)^(100-sum(m1)));
    d_2=theta_B^sum(m2)*(1-theta_B)^(100-sum(m2))/(theta_A^sum(m2)*(1-theta_A)^(100-sum(m2))+theta_B^sum(m2)*(1-theta_B)^(100-sum(m2)));
    d_3=theta_B^sum(m3)*(1-theta_B)^(100-sum(m3))/(theta_A^sum(m3)*(1-theta_A)^(100-sum(m3))+theta_B^sum(m3)*(1-theta_B)^(100-sum(m3)));
    d_4=theta_B^sum(m4)*(1-theta_B)^(100-sum(m4))/(theta_A^sum(m4)*(1-theta_A)^(100-sum(m4))+theta_B^sum(m4)*(1-theta_B)^(100-sum(m4)));
    d_5=theta_B^sum(m5)*(1-theta_B)^(100-sum(m5))/(theta_A^sum(m5)*(1-theta_A)^(100-sum(m5))+theta_B^sum(m5)*(1-theta_B)^(100-sum(m5)));
    
    %M Step 
    theta_A=(p_1*sum(m1)+p_2*sum(m2)+p_3*sum(m3)+p_4*sum(m4)+p_5*sum(m5))/((p_1+p_2+p_3+p_4+p_5)*100);
    theta_B=(d_1*sum(m1)+d_2*sum(m2)+d_3*sum(m3)+d_4*sum(m4)+d_5*sum(m5))/((d_1+d_2+d_3+d_4+d_5)*100);
    blist=[blist theta_B];
    alist=[alist theta_A];

    
end

plot(alist)
ylim([.6 1])
xlabel('Iteration')
ylabel('\Theta_A Value')
title('\Theta_A')

plot(blist)
ylim([.2 .8])
xlabel('Iteration')
ylabel('\Theta_B Value')
title('\Theta_B')


%%%%%%%%%%% Question 3 %%%%%%%%%%%%

%%% Read in Raccon
raccoon=imread('/Users/susancherry/Desktop/Machine_Learning/Homework 5/raccoon.png');
image=raccoon;

%%% K==2 
[a b]=K_Means_Code(raccoon,2);

for i=1:2
      a(a==i)=b(i);
end

a=uint8(a);
imshow(a)  


%%% K==4

[a b]=K_Means_Code(raccoon,4);

for i=1:4
      a(a==i)=b(i);
end

a=uint8(a);
imshow(a)  


%%% K==10

[a b]=K_Means_Code(raccoon,10);

for i=1:10
      a(a==i)=b(i);
end

a=uint8(a);
imshow(a)  