
clear all; close all; clc;
addpath('./Utils');
addpath('./Liblinear');


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

load('../datasets/UCMerced_LandUse');
load('../datasets/models_and_predict_labels');

TrnSize = size(X, 2);
ImgSize = 64; %28;
ImgFormat = 'color'; %'color' or 'gray'

TrnData = X;
TrnLabels = y;
clear X;
clear y;

TestData = X_t;
TestLabels = y_t;
clear X_t;
clear y_t;


TestData_ImgCell = mat2imgcell(TestData,ImgSize,ImgSize,ImgFormat); % convert columns in TestData to cells
clear TestData;

showidx = ceil(rand(50, 1) * (length(TestLabels) - 1));

for i = showidx'
    imshow(TestData_ImgCell{i});
    title(CATEGORIES{test_pred_labels(i)});
    pause
end
