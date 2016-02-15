function [out idxdummy] = AvgPooling(in, blocksize)

idxdummy = 0;

numRows = size(in, 1) / blocksize(1);
numCols = size(in, 2) / blocksize(2);

cells = mat2cell(in, blocksize(1)*ones(1, numRows), blocksize(2)*ones(1, numCols));
out = cellfun(@mean2, cells);

% out = zeros(numRows, numCols);
%
% for i  = 1:numRows
%     for j = 1:numCols
%             out(i, j) = mean2(in((i-1)*blocksize(1)+1:i*blocksize(1), (j-1)*blocksize(2)+1:j*blocksize(2)));
%     end
% end


end
