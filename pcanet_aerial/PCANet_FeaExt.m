function [f] = PCANet_FeaExt(InImg, V, centroids, train_norm, PCANet)
% =======INPUT=============
% InImg     Input images (cell)
% V         given PCA filter banks (cell)
% PCANet    PCANet parameters (struct)
%       .PCANet.NumStages
%           the number of stages in PCANet; e.g., 2
%       .PatchSize
%           the patch size (filter size) for square patches; e.g., [5 3]
%           means patch size equalt to 5 and 3 in the first stage and second stage, respectively
%       .NumFilters
%           the number of filters in each stage; e.g., [16 8] means 16 and
%           8 filters in the first stage and second stage, respectively
%       .HistBlockSize
%           the size of each block for local histogram; e.g., [10 10]
%       .BlkOverLapRatio
%           overlapped block region ratio; e.g., 0 means no overlapped
%           between blocks, and 0.3 means 30% of blocksize is overlapped
%       .Pyramid
%           spatial pyramid matching; e.g., [1 2 4], and [] if no Pyramid
%           is applied
% =======OUTPUT============
% f         PCANet features (each column corresponds to feature of each image)
% =========================

addpath('./Utils')

if length(PCANet.NumFilters)~= PCANet.NumStages;
    display('Length(PCANet.NumFilters)~=PCANet.NumStages')
    return
end

NumImg = length(InImg);

OutImg = InImg;
ImgIdx = (1:NumImg)';
clear InImg;
for stage = 1:PCANet.NumStages
     [OutImg ImgIdx] = PCA_output(OutImg, ImgIdx, ...
           PCANet.PatchSize(stage), PCANet.NumFilters(stage), V{stage});
end

f = cell(NumImg,1); % compute the PCANet training feature one by one

for idx = 1:NumImg
    % if 0==mod(idx,100); display(['Extracting PCANet feature of the ' num2str(idx) 'th test sample...']); end

    f{idx} = CentroidHist(OutImg{idx}, centroids, PCANet.HistBlockSize, PCANet.BlkOverLapRatio);

    OutImg{idx} = [];

end

f = sparse([f{:}]);
f = sparse(f * (1 / train_norm));
