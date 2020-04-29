clear;
clc;
%prepare data
file = load('SyntheticData.mat');
data = file.data;
blue_data = data(1:2:end,:);
red_data = data(2:2:end,:);

%visualize original data
subplot(2,2,1)
plot3(blue_data(:,1),blue_data(:,2),blue_data(:,3),'b+',red_data(:,1),red_data(:,2),red_data(:,3),'r*')
title('原始数据')
xlabel('x')
ylabel('y')
zlabel('z')
axis([-130 130 -130 130 -130 130])

%Applying PCA to data
[Z_blue, MU_blue, SIGMA_blue] = zscore(blue_data);
[PC_blue,SCORE_blue,latent_blue,tsquare_blue] = pca(Z_blue);
[Z_red, MU_red, SIGMA_red] = zscore(red_data);
[PC_red,SCORE_red,latent_red,tsquare_red] = pca(Z_red);
%blue_acc_contri = cumsum(latent_blue)./sum(latent_blue);
%red_acc_contri = cumsum(latent_red)./sum(latent_red);
%tran_blue = PC_blue(:,1:2);
%tran_red = PC_red(:,1:2);
blue_after_PCA = SCORE_blue(:,1:2);
red_after_PCA = SCORE_red(:,1:2);
subplot(2,2,2)
plot(blue_after_PCA(:,1),blue_after_PCA(:,2),'b+',red_after_PCA(:,1),red_after_PCA(:,2),'r*')
title('PCA之后的数据')
xlabel('x')
ylabel('y')

%Applying Kernel PCA
[feature_blue, eigValue_blue, eigVector_blue] = kernel_PCA(blue_data, 1000000, 'Gaussian');
[feature_red, eigValue_red, eigVector_red] = kernel_PCA(red_data, 1000000, 'Gaussian');
subplot(2,2,3)
plot(feature_blue(:,1),feature_blue(:,2),'b+',feature_red(:,1),feature_red(:,2),'r*')
title('高斯KPCA之后的数据')
xlabel('x')
ylabel('y')




