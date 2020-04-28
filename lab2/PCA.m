clear;
clc;
data = load('data.txt');

%归一化
[Z, MU, SIGMA] = zscore(data);

%协方差
correlation = cov(Z);

%特征值，特征向量
[eigV, eigD] = eig(correlation);

%特征值排序，计算贡献和累计贡献
d = diag(eigD);
[D, pos] = sort(d, 'descend');
contribute = D ./ sum(D);
accumulate_contribute = cumsum(D) ./ sum(D)

%寻找最大k特征值
k = 0;
for i = 1:size(accumulate_contribute,1)
    if accumulate_contribute(i) > 0.85
        k = i;
        break
    end
end

%最大k位置
max_pos = pos(1:k);

%新特征基矩阵
feat_mat = zeros(size(eigV , 1) , size(max_pos , 2));
for i = 1:size(max_pos,1)
    cur_pos = max_pos(i);
    feat_vec = eigV(: , cur_pos);
    feat_mat(: , i) = feat_vec;
end

%新特征矩阵
final_feat = Z * feat_mat;

%对比
[coeff, latent1, explained] = pcacov(correlation);
[PC,SCORE,latent2,tsquare] = pca(Z);