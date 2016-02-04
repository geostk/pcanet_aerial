function create_training_data(indeces)

%% %%%%%%%%%%%%%%% Configuration Constants %%%%%%%%%%%%%

IMAGES_PATH = '../UCMerced_LandUse/Images/';

CATEGORIES = {
  'agricultural';
  'airplane';
  'baseballdiamond';
  'beach';
  'buildings';
  'chaparral';
  'denseresidential';
  'forest';
  'freeway';
  'golfcourse';
  'harbor';
  'intersection';
  'mediumresidential';
  'mobilehomepark';
  'overpass';
  'parkinglot';
  'river';
  'runway';
  'sparseresidential';
  'storagetanks';
  'tenniscourt'
};

IMAGES_PER_CATEGORY = 100;
IMAGEFILES_PER_CATEGORY = 100;

TRAINGING_SET_PERCENTAGE = 0.8;
CROSS_VALIDATION_SET_PERCENTAGE = 0.0;
TEST_SET_PERCENTAGE = 0.2;

TRAINGING_SET_X_FILE = 'training_set_x.mat';
CROSS_VALIDATION_SET_X_FILE = 'cross_validation_set_x.mat';
TEST_SET_X_FILE = 'test_set_x.mat';

TRAINGING_SET_Y_FILE = 'training_set_y.mat';
CROSS_VALIDATION_SET_Y_FILE = 'cross_validation_set_y.mat';
TEST_SET_Y_FILE = 'test_set_y.mat';

IMAGES_WIDTH = 128;
IMAGES_HEIGHT = 128;
NUMS_PER_PIXEL = 3;


%% %%%%%%%%%%%%%%   Helper Functions %%%%%%%%%%%%%%%%%%%%%%%%

function [name] = create_path_name(category, number)

  name = sprintf('%s%s/%s%02d.tif', IMAGES_PATH, category, category, number);

end

function [out] = precision_conv(in)
  out = in;
end

function [out] = loadFilesInCategory_impl(category_name, indeces)

  out = precision_conv(zeros(length(indeces), IMAGES_WIDTH*IMAGES_HEIGHT*NUMS_PER_PIXEL));

  for i = 1:length(indeces)
    image_path = create_path_name(category_name, indeces(i));

    full_image = im2double(imread(image_path));

    full_image = imresize(full_image, 0.5);

    out(i, :) = full_image(:)';
    % p = (i-1)*4;
    % out(p + 1, :) = full_image(:)';
    % full_image = imrotate(full_image, 90);
    % out(p + 2, :) = full_image(:)';
    % full_image = imrotate(full_image, 90);
    % out(p + 3, :) = full_image(:)';
    % full_image = imrotate(full_image, 90);
    % out(p + 4, :) = full_image(:)';

  end

end

function [trainging_set, cross_validation_set, test_set] = loadFilesInCategory(category_name)
  trs_last_index = round(IMAGEFILES_PER_CATEGORY * TRAINGING_SET_PERCENTAGE);
  cvs_last_index = trs_last_index + round(IMAGEFILES_PER_CATEGORY * CROSS_VALIDATION_SET_PERCENTAGE);
  tss_last_index = cvs_last_index + round(IMAGEFILES_PER_CATEGORY * TEST_SET_PERCENTAGE);

  if tss_last_index ~= IMAGEFILES_PER_CATEGORY
    error('Percentages do not give us correct indeces!');
  end

  trainging_set = loadFilesInCategory_impl(category_name, indeces(1 : trs_last_index));
  cross_validation_set = loadFilesInCategory_impl(category_name, indeces(trs_last_index + 1 : cvs_last_index));
  test_set = loadFilesInCategory_impl(category_name, indeces(cvs_last_index + 1 : IMAGEFILES_PER_CATEGORY));

end

function [X, X_cv, X_t, y, y_cv, y_t] = loadDataset()

  X = precision_conv(zeros(length(CATEGORIES) * IMAGES_PER_CATEGORY * TRAINGING_SET_PERCENTAGE, IMAGES_WIDTH * IMAGES_HEIGHT * NUMS_PER_PIXEL));
  X_cv = precision_conv(zeros(length(CATEGORIES) * IMAGES_PER_CATEGORY * CROSS_VALIDATION_SET_PERCENTAGE, IMAGES_WIDTH * IMAGES_HEIGHT * NUMS_PER_PIXEL));
  X_t = precision_conv(zeros(length(CATEGORIES) * IMAGES_PER_CATEGORY * TEST_SET_PERCENTAGE, IMAGES_WIDTH * IMAGES_HEIGHT * NUMS_PER_PIXEL));

  y = precision_conv(zeros(length(CATEGORIES) * IMAGES_PER_CATEGORY * TRAINGING_SET_PERCENTAGE, 1));
  y_cv = precision_conv(zeros(length(CATEGORIES) * IMAGES_PER_CATEGORY * CROSS_VALIDATION_SET_PERCENTAGE, 1));
  y_t = precision_conv(zeros(length(CATEGORIES) * IMAGES_PER_CATEGORY * TEST_SET_PERCENTAGE, 1));

  x_index = 1;
  xcv_index = 1;
  xt_index = 1;

  fprintf('preparing for load...');

  for cid = 1:length(CATEGORIES)
    fprintf('\r[');
    for b = 1:cid
      fprintf('*');
    end
    for b = cid:length(CATEGORIES)
      fprintf(' ');
    end
    fprintf('] %g%% done, processing: %s', cid/length(CATEGORIES) * 100, CATEGORIES{cid});

    [trs cvs tss] = loadFilesInCategory(CATEGORIES{cid});

    X(x_index : x_index + IMAGES_PER_CATEGORY * TRAINGING_SET_PERCENTAGE - 1, :) = trs;
    X_cv(xcv_index : xcv_index + IMAGES_PER_CATEGORY * CROSS_VALIDATION_SET_PERCENTAGE - 1, :) = cvs;
    X_t(xt_index : xt_index + IMAGES_PER_CATEGORY * TEST_SET_PERCENTAGE - 1, :) = tss;

    y(x_index : x_index + IMAGES_PER_CATEGORY * TRAINGING_SET_PERCENTAGE - 1, :) = cid;
    y_cv(xcv_index : xcv_index + IMAGES_PER_CATEGORY * CROSS_VALIDATION_SET_PERCENTAGE - 1, :) = cid;
    y_t(xt_index : xt_index + IMAGES_PER_CATEGORY * TEST_SET_PERCENTAGE - 1, :) = cid;

    x_index = IMAGES_PER_CATEGORY * TRAINGING_SET_PERCENTAGE + x_index;
    xcv_index = IMAGES_PER_CATEGORY * CROSS_VALIDATION_SET_PERCENTAGE + xcv_index;
    xt_index = IMAGES_PER_CATEGORY * TEST_SET_PERCENTAGE + xt_index;

  end



end

[X, X_cv, X_t, y, y_cv, y_t] = loadDataset();

fprintf('\n');

save('../datasets/UCMerced_LandUse.mat', 'y', 'y_cv', 'y_t', 'X', 'X_cv', 'X_t', '-v7.3');

fprintf('\nDone.\n');

end
