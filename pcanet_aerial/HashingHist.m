function [f] = HashingHist(PCANet,ImgIdx,OutImg)
% Output layer of PCANet (Hashing plus local histogram)
% ========= INPUT ============
% PCANet  PCANet parameters (struct)
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
% ImgIdx  Image index for OutImg (column vector)
% OutImg  PCA filter output before the last stage (cell structure)
% ========= OUTPUT ===========
% f       PCANet features (each column corresponds to feature of each image)
% ============================
addpath('./Utils')


NumImg = max(ImgIdx);
f = cell(NumImg,1);
map_weights = 2.^((PCANet.NumFilters(end)-1):-1:0); % weights for binary to decimal conversion


for Idx = 1:NumImg
    im = OutImg{Idx};

    hash = zeros(size(im, 1), size(im, 2));
    for i = 1:length(map_weights)
        hash = hash + map_weights(i)*Heaviside(im(:,:,i));
    end

    if isempty(PCANet.HistBlockSize)
        NumBlk = ceil((PCANet.ImgBlkRatio - 1)./PCANet.BlkOverLapRatio) + 1;
        HistBlockSize = ceil(size(hash)./PCANet.ImgBlkRatio);
        OverLapinPixel = ceil((size(hash) - HistBlockSize)./(NumBlk - 1));
        NImgSize = (NumBlk-1).*OverLapinPixel + HistBlockSize;
        Tmp = zeros(NImgSize);
        Tmp(1:size(hash,1), 1:size(hash,2)) = hash;
        f{Idx} = vec(sparse(histc(im2col_general(Tmp,HistBlockSize,...
            OverLapinPixel),(0:2^PCANet.NumFilters(end)-1)')));
    else

        stride = round((1-PCANet.BlkOverLapRatio)*PCANet.HistBlockSize);
        blkwise_fea = sparse(histc(im2col_general(hash,PCANet.HistBlockSize,...
          stride),(0:2^PCANet.NumFilters(end)-1)'));
        f{Idx} = vec(blkwise_fea);
    end
    OutImg{Idx} = [];
end

f = [f{:}];
%-------------------------------
function X = Heaviside(X) % binary quantization
X = sign(X);
X(X<=0) = 0;

function x = vec(X) % vectorization
x = X(:);


function beta = spp(blkwise_fea, sam_coordinate, ImgSize, pyramid)

[dSize, ~] = size(blkwise_fea);

img_width = ImgSize(2);
img_height = ImgSize(1);

% spatial levels
pyramid_Levels = length(pyramid);
pyramid_Bins = pyramid.^2;
tBins = sum(pyramid_Bins);

beta = zeros(dSize, tBins);
cnt = 0;

for i1 = 1:pyramid_Levels,

    Num_Bins = pyramid_Bins(i1);

    wUnit = img_width / pyramid(i1);
    hUnit = img_height / pyramid(i1);

    % find to which spatial bin each local descriptor belongs
    xBin = ceil(sam_coordinate(1,:) / wUnit);
    yBin = ceil(sam_coordinate(2,:) / hUnit);
    idxBin = (yBin - 1)*pyramid(i1) + xBin;

    for i2 = 1:Num_Bins,
        cnt = cnt + 1;
        sidxBin = find(idxBin == i2);
        if isempty(sidxBin),
            continue;
        end
        beta(:, cnt) = max(blkwise_fea(:, sidxBin), [], 2);
    end
end
