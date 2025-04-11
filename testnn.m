NetVoitx={};
TRVoitx={};
accurcy=[];
%cc=1:575;
%cc=576:1151;
%cc=1152:1709;
%cc=1710:2134;
%cc=2135:2943
%cc=2944:3384
%cc=3385:4167
cc=4168:4873;
load dAtA.mat
for ii=1:10
net = feedforwardnet([12 12 12 12 12 2] );
net = configure(net, dAtA(cc,1:14)',dAtA(cc,15)' );

% train net
net.divideFcn='divideint';
net.divideParam.trainRatio = 0.6; % training set [%]
net.divideParam.valRatio = 0.3; % validation set [%]
net.divideParam.testRatio = 0.1; % test set [%]

% train a neural network

net.performFcn='mae';


net.trainParam.showWindow= true;
net.trainParam.showCommandLine= false;
net.trainParam.show = NaN;
net.trainParam.min_grad=  1e-300;         %minimum gradance     
 
net.trainParam.max_fail=100;
net.trainParam.goal=  1e-300 ;         %Minimum Performance Value
net.trainParam.epochs= 500  ;       %Maximum Number of Training Epochs (Iterations)
net.trainParam.time=  10*60;          %Maximum Training Time
  
[net,tr] = trainlm( net, dAtA(cc,1:14)',dAtA(cc,15)' );


NetVoitx{ii,1}=net;
TRVoitx{ii,1}=tr;
 
ressst=NetVoitx{ii} (dAtA(cc,1:14)');

error=(dAtA(cc,15)'-round (ressst));
accurcy(ii)=length (find (error==0))/length (error) *100
MSE(ii)= mean(error.^2);
MAE(ii)=mean(abs(error));           
RMSE(ii)=sqrt(mse(ii));
end