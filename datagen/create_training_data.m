% create matrices representing grayscale images, devide them in trainging set,
% cross validation set, and test set.
% create y vectors.

function create_training_data()

%% %%%%%%%%%%%%%%% Configuration Constants %%%%%%%%%%%%%

IMAGES_PATH = '../UCMerced_LandUse/Images/';

CATEGORIES = {
  'agricultural';
  % 'airplane';
  % 'baseballdiamond';
  % 'beach';
  % 'buildings';
  % 'chaparral';
  'denseresidential';
  % 'forest';
  % 'freeway';
  % 'golfcourse';
  % 'harbor';
  % 'intersection';
  % 'mediumresidential';
  % 'mobilehomepark';
  % 'overpass';
  % 'parkinglot';
  'river';
  % 'runway';
  % 'sparseresidential';
  % 'storagetanks';
  'tenniscourt'
};

IMAGES_PER_CATEGORY = 100;

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
  out = single(in);
end

function [out] = loadFilesInCategory_impl(category_name, numbers)

  out = precision_conv(zeros(length(numbers), IMAGES_WIDTH*IMAGES_HEIGHT*NUMS_PER_PIXEL));


  for i = 1:length(numbers)
    image_path = create_path_name(category_name, numbers(i));

    % load image
    % convert to grayscale
    % convert to uint8 representation
    % unfold it and save it to matrix

    full_image = imread(image_path);
    %fprintf('<%s> [%d %d %d]\n', create_path_name(category_name, numbers(i)), size(full_image, 1), size(full_image, 2), size(full_image, 3));
    %image_grayscale =  rgb2gray(full_image);

    %image_uint8 = im2uint8(image_grayscale);
    full_image = imresize(full_image, 0.5);
    % image_uint8 = im2uint8(full_image);
    full_image = im2single(full_image);

    %fprintf('%s [%d %d] <= [%d %d]\n', create_path_name(category_name, numbers(i)), size(out(i, :), 1),  size(out(i, :), 2), size(image_uint8(:)', 1), size(image_uint8(:)', 2));
    %out(i, :) = image_uint8(:)';kjjjjlkmklm
    out(i, :) = full_image(:)';

  end

end

function [trainging_set, cross_validation_set, test_set] = loadFilesInCategory(category_name)
  numbers = [0 randperm(IMAGES_PER_CATEGORY - 1)];

  trs_last_index = round(IMAGES_PER_CATEGORY * TRAINGING_SET_PERCENTAGE);
  cvs_last_index = trs_last_index + round(IMAGES_PER_CATEGORY * CROSS_VALIDATION_SET_PERCENTAGE);
  tss_last_index = cvs_last_index + round(IMAGES_PER_CATEGORY * TEST_SET_PERCENTAGE);

  if tss_last_index ~= IMAGES_PER_CATEGORY
    error('Percentages do not give us correct indecies!');
  end

  trainging_set = loadFilesInCategory_impl(category_name, numbers(1 : trs_last_index));
  cross_validation_set = loadFilesInCategory_impl(category_name, numbers(trs_last_index + 1 : cvs_last_index));
  test_set = loadFilesInCategory_impl(category_name, numbers(cvs_last_index + 1 : IMAGES_PER_CATEGORY));

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

function [] = saveMatrix(filename, matrix)
  fprintf('saving matrix [%d %d] to file %s \n', size(matrix, 1), size(matrix, 2), filename);
  fileID = fopen(filename, 'w');
  fwrite(fileID, matrix, 'uint8');

end


[X, X_cv, X_t, y, y_cv, y_t] = loadDataset();



fprintf('\n');

% saveMatrix(TRAINGING_SET_X_FILE, X);
% saveMatrix(CROSS_VALIDATION_SET_X_FILE, X_cv);
% saveMatrix(TEST_SET_X_FILE, X_t);
%
% saveMatrix(TRAINGING_SET_Y_FILE, y);
% saveMatrix(CROSS_VALIDATION_SET_Y_FILE, y_cv);
% saveMatrix(TEST_SET_Y_FILE, y_t);

save('../datasets/UCMerced_LandUse.mat', 'y', 'y_cv', 'y_t', 'X', 'X_cv', 'X_t');

fprintf('\nDone.\n');

end
