function [f V BlkIdx] = PCANet_train(InImg,PCANet,IdtExt, ImgFormat)
% =======INPUT=============
% InImg     Input images (cell); each cell can be either a matrix (Gray) or a 3D tensor (RGB)
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
% IdtExt    a number in {0,1}; 1 do feature extraction, and 0 otherwise
% =======OUTPUT============
% f         PCANet features (each column corresponds to feature of each image)
% V         learned PCA filter banks (cell)
% BlkIdx    index of local block from which the histogram is compuated
% =========================

addpath('./Utils')


if length(PCANet.NumFilters)~= PCANet.NumStages;
    display('Length(PCANet.NumFilters)~=PCANet.NumStages')
    return
end


NumImg = length(InImg);

ImgIdx = (1:NumImg)';


if exist('ImgFormat') & strcmp(ImgFormat,'color')
  [OutImg ImgIdx] = separate_image_layers(InImg, ImgIdx);
else
  OutImg = InImg;
end

function [OutImg ImgIdx V] = process_stage(stage, PCANet, InImg, InImgIdx)

    prevNumFilters = PCANet.NumFilters(stage - 1);
    InImgSubsets = cell(prevNumFilters, 1);
    InImgIdxSubsets = cell(prevNumFilters, 1);
    ImgPerSubset = length(InImg)/prevNumFilters;

    % initialize subsets:
    for i = 1:prevNumFilters
      InImgSubsets{i} = cell(ImgPerSubset, 1);
      InImgIdxSubsets{i} = zeros(ImgPerSubset, 1);
      for j = 1:ImgPerSubset
        InImgSubsets{i}{j} = zeros(size(InImg{1}));
        InImgIdxSubsets{i}(j) = InImgIdx(j*prevNumFilters);
      end

    end

    % put input images in right subsets:
    for i = 1:length(InImg)
      fnum = mod(i, prevNumFilters) + 1; % ith image is produced by fnumth filter from prevous layer
      imgorig = ceil(i / prevNumFilters); % ith image is produced by convoluting imgorigth input image from previous layer

      for j = 1:prevNumFilters
        if PCANet.MappingMatrices{stage - 1}(fnum, j)
          InImgSubsets{j}{imgorig} = InImgSubsets{j}{imgorig} + InImg{i};
        end
      end
    end

    % create filters:
    V.filters = cell(prevNumFilters, 1);
    for i = 1:prevNumFilters
      V.filters{i} = PCA_FilterBank(InImgSubsets{i}, PCANet.PatchSize(stage), PCANet.NumFilters(stage));
    end

  OutImgSubsets = cell(prevNumFilters, 1);
  OutImgIdxSubsets = cell(prevNumFilters, 1);

  % compute filters' outputs:
  for i = 1:prevNumFilters
    [OutImgSubsets{i} OutImgIdxSubsets{i}] = PCA_output(InImgSubsets{i}, InImgIdxSubsets{i}, ...
         PCANet.PatchSize(stage), PCANet.NumFilters(stage), V.filters{i}, PCANet.PoolingPatchSize);
  end

  % propagate to the next layer if required:
  if stage < PCANet.NumStages
    display(['Processing layer ' num2str(stage + 1) '...'])
    V.next_stage = cell(prevNumFilters, 1);
    for i = 1:prevNumFilters
      [OutImgSubsets{i} OutImgIdxSubsets{i} V.next_stage{i}] = process_stage(stage + 1, PCANet, OutImgSubsets{i}, OutImgIdxSubsets{i});
    end
  end

  % assemble output:
  OutImg = vertcat(OutImgSubsets{:});
  ImgIdx = vertcat(OutImgIdxSubsets{:});

end % end of process_stage

display(['Processing layer ' num2str(1) '...'])

V.filters = cell(1,1);
V.next_stage = cell(1,1);
V.filters{1} = PCA_FilterBank(OutImg, PCANet.PatchSize(1), PCANet.NumFilters(1));

[OutImg ImgIdx] = PCA_output(OutImg, ImgIdx, ...
     PCANet.PatchSize(1), PCANet.NumFilters(1), V.filters{1}, PCANet.PoolingPatchSize);

if PCANet.NumStages > 1
  display(['Processing layer ' num2str(2) '...'])
  [OutImg ImgIdx V.next_stage{1}] = process_stage(2, PCANet, OutImg, ImgIdx);
end

[f BlkIdx] = HashingHist(PCANet,ImgIdx,OutImg);

end
