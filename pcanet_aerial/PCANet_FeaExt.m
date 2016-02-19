function [f BlkIdx] = PCANet_FeaExt(InImg,V,PCANet, ImgFormat)
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
% BlkIdx    index of local block from which the histogram is compuated
% =========================

addpath('./Utils')

if length(PCANet.NumFilters)~= PCANet.NumStages;
    display('Length(PCANet.NumFilters)~=PCANet.NumStages')
    return
end

NumImg = length(InImg);

ImgIdx = (1:NumImg)';

% if exist('ImgFormat') & strcmp(ImgFormat,'color')
%   [OutImg ImgIdx] = separate_image_layers(InImg, ImgIdx);
% else
  OutImg = InImg;
% end

clear InImg;

[OutImg ImgIdx] = PCA_output(OutImg, ImgIdx, ...
     PCANet.PatchSize(1), PCANet.NumFilters(1), V.filters{1}, PCANet.PoolingPatchSize(1));

stage = 2;
InImg = OutImg;
InImgIdx = ImgIdx;
clear OutImg;
clear ImgIdx;
V = V.next_stage{1};


prevNumFilters = PCANet.NumFilters(stage - 1);
ImgSubsets = cell(prevNumFilters, 1);
ImgIdxSubsets = cell(prevNumFilters, 1);
ImgPerSubset = length(InImg)/prevNumFilters;

% initialize subsets:
for i = 1:prevNumFilters
 ImgSubsets{i} = cell(ImgPerSubset, 1);
 ImgIdxSubsets{i} = zeros(ImgPerSubset, 1);
 for j = 1:ImgPerSubset
     ImgSubsets{i}{j} = zeros(size(InImg{1}));
     ImgIdxSubsets{i}(j) = InImgIdx(j*prevNumFilters);
 end

end

% put input images in right subsets:
for i = 1:length(InImg)
 fnum = mod(i, prevNumFilters) + 1; % ith image is produced by fnumth filter from prevous layer
 imgorig = ceil(i / prevNumFilters); % ith image is produced by convoluting imgorigth input image from previous layer

 for j = 1:prevNumFilters
     if PCANet.MappingMatrices{stage - 1}(fnum, j)
         ImgSubsets{j}{imgorig} = ImgSubsets{j}{imgorig} + InImg{i};
     end
 end
end

clear InImg;
clear InImgIdx;

ft = cell(prevNumFilters,1);

% compute filters' outputs:
for i = 1:prevNumFilters
 [ImgSubsets{i} ImgIdxSubsets{i}] = PCA_output(ImgSubsets{i}, ImgIdxSubsets{i}, ...
         PCANet.PatchSize(stage), PCANet.NumFilters(stage), V.filters{i}, PCANet.PoolingPatchSize(stage));

 [ft{i}] = sparse(HashingHist(PCANet, ImgIdxSubsets{i}, ImgSubsets{i}));

 ImgIdxSubsets{i} = [];
 ImgSubsets{i} = [];
end


clear ImgSubsets;
clear ImgIdxSubsets;

f = sparse(vertcat(ft{:}));

BlkIdx = [];

end
