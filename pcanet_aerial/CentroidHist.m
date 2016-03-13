function [h] = CentroidHist(img, centroids, HistBlockSize, BlkOverLapRatio)

    strideY = round((1 - BlkOverLapRatio) * HistBlockSize(1));
    strideX = round((1 - BlkOverLapRatio) * HistBlockSize(2));

    ypos = 1:strideX:size(img, 1) - HistBlockSize(1) + strideX;
    xpos = 1:strideY:size(img, 2) - HistBlockSize(2) + strideY;

    h = cell(1, length(xpos)*length(ypos));
    idx = 1;
    for i = ypos
        for j = xpos
            tmp = img(i:i+HistBlockSize(1)-1, j:j+HistBlockSize(2)-1, :);
            tmp = reshape(permute(tmp, [3 2 1]), size(img, 3), []);
            tmp = find_closest_centroids(tmp, centroids);
            h{idx} = sparse(histc(tmp, 1:length(centroids)));
            idx = idx + 1;
        end
    end

    h = sparse(vertcat(h{:}));

end
