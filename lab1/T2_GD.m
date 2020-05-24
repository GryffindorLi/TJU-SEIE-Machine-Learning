clear;
clc;
%data
data = readtable('Diabetes_Data.txt');
data = table2array(data);
train_set = data(1:400,:);
test_set = data(401:442,:);
train_label = train_set(:,11); 
train_data = train_set(:,1:10);
one_train = ones(400,1);
one_test = ones(42,1);
train_data = [train_data one_train];

% parameter
theta = rand(size(train_data,2),1);
iter = 10000;
learning_rate = 0.0000000004;
loss = zeros(iter, 1);
m = 1;

% Gradient Descend
for i = 1:iter
    y_ = sum(train_data*theta, 2);
    test_error = 1/(2*m) * sum(train_label-y_)^2;
    loss(i) = test_error;
    % 求导
    delta = -1/m * (train_data'*(train_label-y_));
    theta = theta - learning_rate*delta;
end
figure
plot(loss);
title('loss');
xlabel('epoch');
ylabel('loss')
test_label = test_set(:,11);
test_data = test_set(:,1:10);
test_data = [test_data one_test];
test_predict = test_data*theta;
test_error = sqrt(sum((test_label-test_predict).^2)./size(test_label,2));
log = fopen('log.txt','w');
fprintf(log,'本次训练学习率为:%16.15f\n,',learning_rate);
fprintf(log,'测试集测试误差为：%8.5f\n',test_error);
fclose(log);
