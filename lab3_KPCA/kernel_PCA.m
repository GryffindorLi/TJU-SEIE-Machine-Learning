function [feature, eigenValue, eigenvector] = kernel_PCA(data, C, kernel_function)
%C是高斯核函数的参数

[row,col]=size(data);

Xmean = mean(data);
Xstd = std(data);
%归一化
X0 = (data-ones(row,1)*Xmean) ./ (ones(row,1)*Xstd);

%核函数矩阵
if (strcmp(kernel_function, 'Gaussian')~=0)
    for i = 1:row
        for j = 1:row
            K(i,j) = exp(-(norm(X0(i,:) - X0(j,:)))^2/C);
        end
    end
else
    fprintf('input "Gaussian"')
    feature = [];
    eigValue = [];
    eigVector = [];
    return
end
%%中心化矩阵
unit = (1/row) * ones(row, row);
Kp = K - unit*K - K*unit + unit*K*unit;
%特征向量和特征矩阵
[eigenvector, eigenvalue] = eig(Kp);
%单位化特征向量
for m =1 : row
    for n =1 : row
        Normvector(n,m) = eigenvector(n,m)/sum(eigenvector(:,m));
    end
end
eigenValue = diag(eigenvalue);
[eigenvalue_sort, index] = sort(eigenValue, 'descend'); % 特征值按降序排列，eigenvalue_sort是排列后的数组，index是序号
pcn = 2;
pcIndex = index(1:pcn);
feat_mat = zeros(size(eigenvalue , 1) , pcn);
for i = 1:pcn
    cur_pos = pcIndex(i);
    feat_vec = eigenvector(: , cur_pos);
    feat_mat(: , i) = feat_vec;
end
feature = K * feat_mat;
%disp(size(P))
%disp(size(X0))
end

