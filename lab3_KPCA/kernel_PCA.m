function [feature, eigenValue, eigenvector] = kernel_PCA(data, C, kernel_function)
%C�Ǹ�˹�˺����Ĳ���

[row,col]=size(data);

Xmean = mean(data);
Xstd = std(data);
%��һ��
X0 = (data-ones(row,1)*Xmean) ./ (ones(row,1)*Xstd);

%�˺�������
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
%%���Ļ�����
unit = (1/row) * ones(row, row);
Kp = K - unit*K - K*unit + unit*K*unit;
%������������������
[eigenvector, eigenvalue] = eig(Kp);
%��λ����������
for m =1 : row
    for n =1 : row
        Normvector(n,m) = eigenvector(n,m)/sum(eigenvector(:,m));
    end
end
eigenValue = diag(eigenvalue);
[eigenvalue_sort, index] = sort(eigenValue, 'descend'); % ����ֵ���������У�eigenvalue_sort�����к�����飬index�����
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

