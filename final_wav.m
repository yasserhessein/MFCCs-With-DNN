 
    
load dAtA.mat
%% set the folds configureations 
Val=10;

cd I:\SR_Yasir\wav
C = cvpartition(length(dAtA),'KFold',Val);



%% set neural network results 
NET=cell(Val,1);
TR=cell(Val,1);

accurcy=[];
MSE=[];
RMSE=[];
MAE=[];


for i=1:Val


 
 

%% apply neural network 
   

net = feedforwardnet([12 12 12 12 12 2]);
net = configure(net, fdAtA_train', Target_train);

% train net
net.divideFcn='divideint';
net.divideParam.trainRatio = 0.6; % training set [%]
net.divideParam.valRatio = 0.1; % validation set [%]
net.divideParam.testRatio = 0.3; % test set [%]

% train a neural network

net.performFcn='mae';


net.trainParam.showWindow= true;
net.trainParam.showCommandLine= false;
net.trainParam.show = NaN;
net.trainParam.min_grad=  1e-300;         %minimum gradance     
 
net.trainParam.max_fail=200;
net.trainParam.goal=  1e-555 ;         %Minimum Performance Value
net.trainParam.epochs= 1000  ;       %Maximum Number of Training Epochs (Iterations)
net.trainParam.time=  10*60;          %Maximum Training Time
  
[net,tr] = trainlm(net,fdAtA_train', Target_train);


NET{i,1}=net;
TR{i,1}=tr;
 
ressst=NET{i} (fdAtA_test');

error=(Target_test)-round (ressst);
accurcy(i)=length (find (error==0))/length (error) *100
MSE(i)=mean(error.^2);
MAE(i)=mean(abs(error));           
RMSE(i)=sqrt(mse(i));


end


figure;
boxplot (accurcy)
ylabel("Accuracy %");
xlabel("10-folds");

figure;
boxplot (MSE)
ylabel("mse ");
xlabel("10-folds");

figure;
boxplot (MAE)
ylabel("mae ");
xlabel("10-folds");

figure;
boxplot (RMSE)
ylabel("rmse ");
xlabel("10-folds");





