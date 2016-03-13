function idx = find_closest_centroids(X, centroids)

numCentroids = size(centroids, 1);

idx = zeros(size(X,2), 1);

for i = 1:length(idx)

  dists = zeros(numCentroids, 1);

  for j = 1:numCentroids
    diffs = X(:, i) - centroids(:, j);
    dists(j) = sum( diffs .* diffs );
  end

  [dummy idx(i)] = min(dists);

end

end
