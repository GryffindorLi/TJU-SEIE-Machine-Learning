clear;
clc;

%prepare data
file = importdata('Diabetes_Data.txt');
data = file.data;
train_data = data(1:400, 1:10);
train_tag = data(1:400, 11);
test_data = data(401:442, 1:10);
test_tag = data(401:442, 11);

%insert ones in each dataset
one_train = ones(400,1);
one_test = ones(42,1);
train_data = [train_data one_train];
test_data = [test_data one_test];

%Norm equation
omega = inv(train_data'*train_data)*train_data'*train_tag;
predict = test_data*omega;
error = sum((test_tag-predict).^2)./size(test_tag,1);
disp('计算所得参数为：')
disp(omega)
fprintf('测试集测试误差为：%8.5f\n',error)