function  [OutImg ImgIdx] = separate_image_layers(InImg, ImgIdx)

  OutImg = cell(4 * length(InImg));

  for i = 1:length(InImg)
    p = (i - 1) * 4;
    OutImg{p + 1} = InImg{i}(:,:,1);
    OutImg{p + 2} = InImg{i}(:,:,2);
    OutImg{p + 3} = InImg{i}(:,:,3);
    OutImg{p + 4} = rgb2gray(InImg{i});
  end

  ImgIdx = kron(ImgIdx, ones(4, 1));

end
