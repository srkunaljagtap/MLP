%% MLP with Error Back Propagation 
% This Matlab code is for one hidden layer with N number of neuron
% and 1 output neuron
% Input could of any dimension
% Also, If number of neurons in Hidden Layer are 1
% It calculates the L2 norm of the differences between the EBP-calculated gradients
% and their approximations.
clc;
clear;
close all;
%% Set up parameters
%number of nodes in the hidden layer
%change the value of number of hidden layer here
nHiddenLayer = 1;
%number of nodes in output
nOutput = 1; 
% learning rate
epsilon = 0.01; 
% load the data
X = load('dataMLP.mat');
X =X.X2;  
 [m,n] = size(X);
MLPtrain = X(1:50,:);
MLPval = X(51:500,:);
MLPtest = X(501:1000,:);
t = load('tMLP.mat');
t = t.t;
T = t';
Ttrain = T(:,1:50);
Tval = T(:,51:500);
Ttest = T(:,501:1000);

% load the data you want to use to train here
trainExamples = MLPtest';
[dimension,nTrain]=size(trainExamples);
% load the appropriate labels as per training data selected here
T = Ttest;

%set the last input to be 1 as a bias
trainExamples(dimension+1,:) = ones(1,nTrain); 

%Initilize the weights
S1(nHiddenLayer,nTrain) = 0;
S2(nOutput,nTrain) = 0;
w1 = randi([-3,3],dimension+1,nHiddenLayer);
w2 = randi([-2,2],nHiddenLayer+1,nOutput);


%% Initialise values for looping
nEpochs = 0;
Epoch = [];
nCount = 0;
nWrong = Inf;
gradient = zeros(dimension+1,nHiddenLayer);
nEpochs = 1;
nIterations = 1000;
for j=1:nIterations
% as long as more than 25% of outputs are wrong
%while(nWrong >= (nTrain*0.25)) 
    for i=1:nTrain
        
        x = trainExamples(:,i);
        S1(1:nHiddenLayer,i) = w1'*x;
        S2(:,i) = w2'*[tanh(S1(:,i));1];
        
        % back propagate the error
        delta1 = ( 1 ./ (1 + (exp(-2*(S2(:,i)))))) - T(:,i); 
        delta2 = (1-tanh(S1(:,i)).^2).*(w2(1:nHiddenLayer,:)*delta1);      
        
        % update weights
        w1 = w1 - epsilon*x*delta2';
        w2 = w2 - epsilon*[tanh(S1(:,i));1]*delta1'; 
           
    end

outputNN = ( 1 ./ (1 + (exp(-2*S2)))) ;

% Cross_Entrophy
E = -((T.*log(outputNN))+((1-T).*log(1-outputNN)));
E = (1/nTrain)*sum(E);
plot(j,E,'*');
hold on  
title('cross Entrophy VS number of Iterations')
xlabel('Epochs')
ylabel('Cross Entrophy')
hold on
% clculate the number of correct labels
for i = 1:nTrain
if(outputNN(1,i) > 0.5)
    Tpred(i,1) = 1;
else 
    Tpred(i,1) = 0;
end
end
nCount = 0;
nWrong = 0;

for i = 1: nTrain 
if(Tpred(i,1) == T(1,i))
nCount = nCount+1;
else
    nWrong= nWrong+1;
end
end
 nEpochs = nEpochs + 1;
end

%%Backward Difference Approximation
% we will consider the optimal weights obtained after training 
% and one of the samples to find backward diffrence and gradient
if(nHiddenLayer==1)
    precision = 10^-8;
    x1 =  trainExamples(:,1);
    a = w1'*x1;
    z = tanh(a);
    a2 = [z,1]*w2;
    y = 1/(1+exp(-a2));
    delta_2 = y-T(1);
    delta_1 = (1-z^2)*(delta_2*w2(1))';
    gradient_w1 = delta_1*x1';
    gradient_w2 = delta_2*[z,1];
    gradient = vertcat(gradient_w1',gradient_w2');
    CE = -(T(1)*log(y)+(1-T(1))*log(1-y));
    
    a = [w1(1)-precision ;w1(2); w1(3)]'*x1;
    z = tanh(a);
    youtw1 = [z,1]*w2;
    
    a = [w1(1);w1(2)-precision ; w1(3)]'*x1;
    z = tanh(a);
    youtw2 = [z,1]*w2;
    
    a = [w1(1) ;w1(2); w1(3)-precision]'*x1;
    z = tanh(a);
    youtw3 = [z,1]*[w2];
    
    a = w1'*x1;
    z = tanh(a);
    youtw4 = [z,1]*[w2(1)-precision; w2(2)];
    
    a = w1'*x1;
    z = tanh(a);
    youtw5 = [z,1]*[w2(1);w2(2)-precision];
   
    y_precisionw1 = 1/(1+exp(-youtw1));
    y_precisionw2 = 1/(1+exp(-youtw2));
    y_precisionw3 = 1/(1+exp(-youtw3));
    y_precisionw4 = 1/(1+exp(-youtw4));
    y_precisionw5 = 1/(1+exp(-youtw5));
        
    CE_precision1 = -(T(1)*log(y_precisionw1)+(1-T(1))*log(1-y_precisionw1));
    CE_precision2 = -(T(1)*log(y_precisionw2)+(1-T(1))*log(1-y_precisionw2));
    CE_precision3 = -(T(1)*log(y_precisionw3)+(1-T(1))*log(1-y_precisionw3));
    CE_precision4 = -(T(1)*log(y_precisionw4)+(1-T(1))*log(1-y_precisionw4));
    CE_precision5 = -(T(1)*log(y_precisionw5)+(1-T(1))*log(1-y_precisionw5));
    
    gradient_approx1 = (CE-CE_precision1)/precision;
    gradient_approx2 = (CE-CE_precision2)/precision;
    gradient_approx3 = (CE-CE_precision3)/precision;
    gradient_approx4 = (CE-CE_precision4)/precision;
    gradient_approx5 = (CE-CE_precision5)/precision;
    
    gradient_approx = vertcat(gradient_approx1,gradient_approx2,gradient_approx3,gradient_approx4,gradient_approx5);
    L2_norm = norm(gradient-gradient_approx);
    disp('L2_norm');
    disp(L2_norm);
end

% Misclassification rate is ratio of number of misclassified samples 
% over total number of samples in training set
misclassification_rate = nWrong/nTrain;
disp('misclassification_rate');
disp(misclassification_rate);