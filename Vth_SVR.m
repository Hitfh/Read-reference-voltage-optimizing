clear,clc
%% Get table data (Need to supplement the path of svr_pe)
error0=xlsread('C:\xxxx\svr_pe','pe','b2:iw900');
error10=xlsread('C:\xxxx\svr_pe\svr_pe','Retention_10day','b2:iw900');
error20=xlsread('C:\xxxx\svr_pe\svr_pe','Retention_20day','b2:iw900');
error30=xlsread('C:\xxxx\svr_pe\svr_pe','Retention_30day','b2:iw900');
error45=xlsread('C:\xxxx\svr_pe\svr_pe','Retention_45day','b2:iw900');
error60=xlsread('C:\xxxx\svr_pe\svr_pe','Retention_60day','b2:iw900');
error75=xlsread('C:\xxxx\svr_pe\svr_pe','Retention_75day','b2:iw900');
error90=xlsread('C:\xxxx\svr_pe\svr_pe','Retention_90day','b2:iw900');
error105=xlsread('C:\xxxx\svr_pe\svr_pe','Retention_105day','b2:iw900');
error120=xlsread('C:\xxxx\svr_pe\svr_pe','Retention_120day','b2:iw900');
pe=xlsread('C:\xxxx\svr_pe\svr_pe','RBER','b3:b62');
ret=[0 10 20 30 45 60 75 90 105 120];
pecycles_train=55;
retation=10;
pecycles_test=5;
% %% Build training set
% %Randomize the block number to facilitate the construction of test set and training set
for i=1:10
    temp(i,:)=randperm(size(pe,1));
end
% 
% % Cumulative distribution function input
step=-960:7.5:952.5;    %read reference voltage level
step_train=[];
for i=1:pecycles_train*retation
    step_train((i-1)*256+1:i*256)=step(:);
end
ret_train=[];
for i=1:retation
    for j=1:pecycles_train*256
        ret_train((i-1)*pecycles_train*256+j)=ret(i);
    end
end
pe_train=[];
for i=1:retation
    for j=1:pecycles_train
        for k=1:256
        pe_train((i-1)*pecycles_train*256+(j-1)*256+k)=pe(temp(i,j));
        end
    end
end
for i=1:13
   for k=1:pecycles_train
        error0_train(i,(k-1)*256+1:k*256)=error0(i+1+(temp(1,k)-1)*15,:);
        error10_train(i,(k-1)*256+1:k*256)=error10(i+1+(temp(2,k)-1)*15,:);
        error20_train(i,(k-1)*256+1:k*256)=error20(i+1+(temp(3,k)-1)*15,:);
        error30_train(i,(k-1)*256+1:k*256)=error30(i+1+(temp(4,k)-1)*15,:);
        error45_train(i,(k-1)*256+1:k*256)=error45(i+1+(temp(5,k)-1)*15,:);
        error60_train(i,(k-1)*256+1:k*256)=error60(i+1+(temp(6,k)-1)*15,:);
        error75_train(i,(k-1)*256+1:k*256)=error75(i+1+(temp(7,k)-1)*15,:);
        error90_train(i,(k-1)*256+1:k*256)=error90(i+1+(temp(8,k)-1)*15,:);
        error105_train(i,(k-1)*256+1:k*256)=error90(i+1+(temp(9,k)-1)*15,:);
        error120_train(i,(k-1)*256+1:k*256)=error90(i+1+(temp(10,k)-1)*15,:);
   end
end
error=[error0_train';error10_train';error20_train';error30_train';error45_train';error60_train';error75_train';error90_train';error105_train';error120_train'];
error_train=error';

% % % %Probability density function input
step1=-952.5:7.5:952.5;
step_train1=[];
for i=1:pecycles_train*retation
    step_train1((i-1)*255+1:i*255)=step1(:);
end
ret_train1=[];
for i=1:retation
    for j=1:pecycles_train*255
        ret_train1((i-1)*pecycles_train*255+j)=ret(i);
    end
end
pe_train1=[];
for i=1:retation
    for j=1:pecycles_train
        for k=1:255
        pe_train1((i-1)*pecycles_train*255+(j-1)*255+k)=pe(temp(i,j));
        end
    end
end
for i=1:pecycles_train*retation
    for j=1:255
    Vth_train((i-1)*255+j,:)=abs(error((i-1)*256+j+1,:)-error((i-1)*256+j,:));
    end
end

%%%/*
% %% Build test set
% % Cumulative distribution function input
step_test=[];
for i=1:pecycles_test*retation
   step_test((i-1)*256+1:i*256)=step(:);
end
ret_test=[];
for i=1:retation
    for j=1:pecycles_test*256
        ret_test((i-1)*pecycles_test*256+j)=ret(i);
    end
end
pe_test=[];
for i=1:retation
    for j=1:pecycles_test
        for k=1:256
        pe_test((i-1)*pecycles_test*256+(j-1)*256+k)=pe(temp(i,pecycles_train+j));
        end
    end
end
for i=1:13
   for k=1:pecycles_test
        error0_test(i,(k-1)*256+1:k*256)=error0(i+1+(temp(1,pecycles_train+k)-1)*15,:);
        error10_test(i,(k-1)*256+1:k*256)=error10(i+1+(temp(2,pecycles_train+k)-1)*15,:);
        error20_test(i,(k-1)*256+1:k*256)=error20(i+1+(temp(3,pecycles_train+k)-1)*15,:);
        error30_test(i,(k-1)*256+1:k*256)=error30(i+1+(temp(4,pecycles_train+k)-1)*15,:);
        error45_test(i,(k-1)*256+1:k*256)=error45(i+1+(temp(5,pecycles_train+k)-1)*15,:);
        error60_test(i,(k-1)*256+1:k*256)=error60(i+1+(temp(6,pecycles_train+k)-1)*15,:);
        error75_test(i,(k-1)*256+1:k*256)=error75(i+1+(temp(7,pecycles_train+k)-1)*15,:);
        error90_test(i,(k-1)*256+1:k*256)=error90(i+1+(temp(8,pecycles_train+k)-1)*15,:);
        error105_test(i,(k-1)*256+1:k*256)=error75(i+1+(temp(9,pecycles_train+k)-1)*15,:);
        error120_test(i,(k-1)*256+1:k*256)=error90(i+1+(temp(10,pecycles_train+k)-1)*15,:);
   end
end
error1=[error0_test';error10_test';error20_test';error30_test';error45_test';error60_test';error75_test';error90_test';error105_test';error120_test'];
% error_test=error1';

%Probability density function input
step_test1=[];
for i=1:pecycles_test*retation
   step_test1((i-1)*255+1:i*255)=step1(:);
end
ret_test1=[];
for i=1:retation
    for j=1:pecycles_test*255
        ret_test1((i-1)*pecycles_test*255+j)=ret(i);
    end
end
pe_test1=[];
for i=1:retation
    for j=1:pecycles_test
        for k=1:255
        pe_test1((i-1)*pecycles_test*255+(j-1)*255+k)=pe(temp(i,pecycles_train+j));
        end
    end
end
for i=1:pecycles_test*retation
    for j=1:255
    Vth_test((i-1)*255+j,:)=abs(error1((i-1)*256+j+1,:)-error1((i-1)*256+j,:));
    end
end

P_train=[step_train;ret_train;pe_train]';
P_test=[step_test;ret_test;pe_test]';
P_train1=[step_train1;ret_train1;pe_train1]';
P_test1=[step_test1;ret_test1;pe_test1]';
save train_data.mat P_train error P_test error1 Vth_train Vth_test P_train1 P_test1

load train_data.mat
%%*/
%% Cumulative distribution function fitting
%Randomize input data
s=2;
temp_train=randperm(size(P_train,1));
T_train=error(temp_train(:),s)';
train_in=P_train(temp_train(:),:)';
train_out=[];
train_out=T_train;
%Normalization of sample input and output data
[inputn,inputps]=mapminmax(train_in);
% mapstd
[outputn,outputps]=mapminmax(train_out);

%Test data construction
T_test=error1(:,s)';
test_out=T_test;
test_in=mapminmax('apply',P_test',inputps);
pecycles_train=55;
retation=10;
pecycles_test=5;
rber=8000;
% for m=3:12
%     for n=3:12
%         for k=1:5
%Initialize network structure
net=newff(inputn,outputn,[7,8],{'logsig','logsig','purelin'});
% {'logsig','radbas','purelin'}
net.trainParam.epochs=700;%Maximum number of epochs
net.trainParam.lr=0.1;%learning rate
net.trainParam.goal=0.0001;%Target Error

%Network Training
net.trainFcn='trainbr';  %Bayesian regularization backpropagation
net=train(net,inputn,outputn);

%Network Prediction Output
an=sim(net,inputn);
%Network Output Denormalization
ty=mapminmax('reverse',an,outputps);
%RMSE
for i=1:pecycles_train*retation
    for j=1:255
    Ty(:,(i-1)*255+j)=abs(ty(:,(i-1)*256+j+1)-ty(:,(i-1)*256+j));
    end
end
RMSE_nerve=RMSE(Vth_train',Ty);
ER=Ty-Vth_train(:,s)';
MSE=mean((ER./Vth_train(:,s)').^2)*100;
% save model_7.1747.mat net
%Prediction data normalization
% temp_test=randperm(size(P_test,1));
% T_test=error1(temp_test(:),:)';
% test_in=P_train(temp_test(:),:)';
% test_out=[];
% test_out=T_test(1,:);

%Network Prediction Output
bn=sim(net,test_in);
%Network Output Denormalization
BPoutput=mapminmax('reverse',bn,outputps);
BPoutput=BPoutput';
%subtract
bpout=[];
for i=1:pecycles_test*retation
    for j=1:255
    bpout((i-1)*255+j)=abs(BPoutput((i-1)*256+j+1)-BPoutput((i-1)*256+j));
    end
end
tp=Vth_test(:,s)';
RMSE_test=RMSE(tp,bpout);
ER_test=bpout-tp;
MSE_test=mean((ER_test./tp).^2)*100;
 save('modelref2_011.mat', 'net' );
% % save('model1.mat', 'net1' );
%  if RMSE_test<rber
%      RB(m,n)=RMSE_test;
%      rber=RMSE_test;
%      save('ref1_011_1.mat', 'net' );
%      hiddennum1=m;
%      hiddenum2=n;
% %      if RMSE_test<7000
% %      break;
% %      end
%  end
%         end
%     end
% end

%% 概率密度函数拟合
% %将输入数据随机化
% temp_train1=randperm(size(P_train1,1));
% T_train1=Vth_train(temp_train1(:),:)';
% train_in1=P_train1(temp_train1(:),:)';
% train_out1=[];
% train_out1=T_train1(1,:);
% %样本输入输出数据归一化
% [inputn1,inputps1]=mapminmax(train_in1);
% % mapstd
% [outputn1,outputps1]=mapstd(train_out1);
% 
% %初始化网络结构
% net1=newff(inputn1,outputn1,[6,3],{'logsig','radbas','purelin'});
% net1.trainParam.epochs=700;%最大迭代次数
% net1.trainParam.lr=0.1;%学习率
% net1.trainParam.goal=0.0001;%神经网络训练的目标误差
% 
% %网络训练
% net1.trainFcn='trainbr';
% net1=train(net1,inputn1,outputn1);
% 
% %网络预测输出
% an1=sim(net1,inputn1);
% %网络输出反归一化
% ty1=mapstd('reverse',an1,outputps1);
% RMSE_nerve1=RMSE(train_out1,ty1);
% ER1=ty1-train_out1;
% MSE1=mean((ER1./train_out1).^2);
% 
% % BP网络预测
% %预测数据归一化
% T_test1=Vth_test';
% test_out1=T_test1(1,:);
% test_in1=mapminmax('apply',P_test1',inputps1);
% %网络预测输出
% bn1=sim(net1,test_in1);
% %网络输出反归一化
% BPoutput1=mapstd('reverse',bn1',outputps1);
% BPoutput1=BPoutput1';
% RMSE_test1=RMSE(test_out1,BPoutput1);
% ER_test1=BPoutput1-test_out1;
% MSE_test1=mean((ER_test1./test_out1).^2);
% 
% save model1.mat net 




% Step=256*13;
% pecycles=31;
% vth(1,:)=step(1,:);
% vth(2,:)=step(2,:);
% vth(3,:)=step(2,:);
% vth(4,:)=step(3,:);
% vth(5,:)=step(3,:);
% vth(6,:)=step(4,:);
% vth(7,:)=step(4,:);
% vth(8,:)=step(5,:);
% vth(9,:)=step(5,:);
% vth(10,:)=step(6,:);
% vth(11,:)=step(6,:);
% vth(12,:)=step(7,:);
% vth(13,:)=step(7,:);
% Vth=reshape(vth', 1 ,Step);
% a=[];
% b=[];
% for i=1:Step  %天数
%     for j=1:pecycles
%         a(i,j)=Vth(1,i);
%         b(i,j)=pe(1,j);
%     end
% end
% re=reshape(a, 1 ,Step*pecycles); %reshape成1行Step*pecycles列数据
% pe=reshape(b, 1 ,Step*pecycles);
% train_in=[re;pe]';%神经网络训练输入，转置成列

%% 提取输出数据
% train_out=reshape(error', 1 ,Step*pecycles);
% train_out=train_out';
%% BP网络的数据
% %找出训练数据和预测数据
% input_train=train_in';
% output_train=train_out;
% % input_test=pe_ret2';
% 
% %样本输入输出数据归一化
% [inputn,inputps]=mapminmax(input_train);
% [outputn,outputps]=mapminmax(output_train);

%% BP网络训练
% %初始化网络结构
% net=newff(inputn,outputn,[2,4,3,1],{'tansig','tansig','tansig','purelin'});
% net.trainParam.epochs=1000;%最大迭代次数
% net.trainParam.lr=0.001;%学习率
% net.trainParam.goal=0.001;%神经网络训练的目标误差
% 
% %网络训练
% % net.trainFcn='trainbr';
% net.trainFcn='trainlm';
% net=train(net,inputn,outputn);

% %% 预测误差
% %网络预测输出
% an=sim(net,inputn);
% %网络输出反归一化
% ty=mapminmax('reverse',an,outputps);
% RMSE_nerve=RMSE(output_train,ty);
% error=ty-output_train;
% MSE=mean((error./output_train).^2);


%% SVR训练和预测
%找出训练数据和预测数据
% input=train_in;
% output=train_out;
% p_test=train_in(26624:43264,:);
% t_test=train_out(26624:43264,1);
% %%数据归一化
% [p_train,pn_test,ps] = scaleForSVM(input,p_test,-1,1);
% [t_train,tn_test,ts] = scaleForSVM(output,t_test,-1,1);
% %找出训练数据和预测数据
% pn_train=p_train;
% tn_train=t_train;
% 
% %%创建/训练SVR模型
% %[bestmse,bestc,bestg]= SVMcgForRegress(tn_train,pn_train);
% [bestMSE,bestc,bestg,ga_option] = gaSVMcgForRegress(tn_train,pn_train);
% cmd=['-c ',num2str(bestc),' -g ',num2str(bestg),' -t 2 -s 3 -p 0.01'];
% model=svmtrain(tn_train,pn_train,cmd);
% [Predict,mse,dec_value]=svmpredict(tn_test,pn_test,model);
% predict=mapminmax('reverse',Predict,ts);
% save('model1.mat', 'net1' );