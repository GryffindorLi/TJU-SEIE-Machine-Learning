clear;
clc;
data = load('data.txt');

%��һ��
[Z, MU, SIGMA] = zscore(data);

%Э����
correlation = cov(Z);

%����ֵ����������
[eigV, eigD] = eig(correlation);

%����ֵ���򣬼��㹱�׺��ۼƹ���
d = diag(eigD);
[D, pos] = sort(d, 'descend');
contribute = D ./ sum(D);
accumulate_contribute = cumsum(D) ./ sum(D)

%Ѱ�����k����ֵ
k = 0;
for i = 1:size(accumulate_contribute,1)
    if accumulate_contribute(i) > 0.85
        k = i;
        break
    end
end

%���kλ��
max_pos = pos(1:k);

%������������
feat_mat = zeros(size(eigV , 1) , size(max_pos , 2));
for i = 1:size(max_pos,1)
    cur_pos = max_pos(i);
    feat_vec = eigV(: , cur_pos);
    feat_mat(: , i) = feat_vec;
end

%����������
final_feat = Z * feat_mat;

%�Ա�
[coeff, latent1, explained] = pcacov(correlation);
[PC,SCORE,latent2,tsquare] = pca(Z);