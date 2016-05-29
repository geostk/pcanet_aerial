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
  
    Idx_span = find(ImgIdx == Idx);
    NumOs = length(Idx_span)/PCANet.NumFilters(end); % the number of "O"s
    Bhist = cell(NumOs,1);
    
    for i = 1:NumOs 
        
        T = 0;
        ImgSize = size(OutImg{Idx_span(PCANet.NumFilters(end)*(i-1) + 1)});
        for j = 1:PCANet.NumFilters(end)
            T = T + map_weights(j)*Heaviside(OutImg{Idx_span(PCANet.NumFilters(end)*(i-1)+j)}); 
            % weighted combination; hashing codes to decimal number conversion
            
            OutImg{Idx_span(PCANet.NumFilters(end)*(i-1)+j)} = [];
        end
          
        stride = round((1-PCANet.BlkOverLapRatio)*PCANet.HistBlockSize); 
        blkwise_fea = sparse_hist(im2col_general(T,PCANet.HistBlockSize,...
          stride), 2^PCANet.NumFilters(end)); 
        % calculate histogram for each local block in "T"
        Bhist{i} = blkwise_fea; 
    end           
    f{Idx} = vertcat(Bhist{:});
    f{Idx} = sparse(f{Idx}/norm(f{Idx}, 'fro'));
end
f = vertcat(f{:});
%-------------------------------
function h = sparse_hist( input, hvalues )

if size(input, 2) == 1
    h = sparse(input + 1, 1, 1, hvalues, 1, length(input));
else
    ca = cell(size(input, 2), 1);
    for idx = 1:size(input, 2)
        ca{idx} = sparse( input(:, idx) + 1, 1, 1, hvalues, 1, size(input,1));
    end
    h = sparse(vertcat(ca{:}));
end

function X = Heaviside(X) % binary quantization
X = sign(X);
X(X<=0) = 0;

function x = vec(X) % vectorization
x = X(:);

