function[misclassification_rate]= MLP(nHiddenLayer,nOutput,epsilon,MLPtrain,Labeltrain)
%% Input arguments 
% nHiddenLayer: number of hidden layers (1-20)
% nOutput: number of outputs (1)
% epsilon: learning rate (0.1 - 0.01 (standard = 0.01))
% MLPtrain: trianing samples 
% Labeltrain: labels of training samples
%%

trainExamples = MLPtrain';
[dimension,nTrain]=size(trainExamples);
T = Labeltrain;

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

nEpochs = 1;
for j=1:1000
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
plot(j,E,'r*');
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

misclassification_rate = nWrong/nTrain;
%disp('misclassification_rate');
%disp(misclassification_rate);
end
