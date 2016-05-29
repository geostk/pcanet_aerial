% ==== PCANet Demo =======
% T.-H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma,
% "PCANet: A simple deep learning baseline for image classification?" submitted to IEEE TIP.
% ArXiv eprint: http://arxiv.org/abs/1404.3606

% Tsung-Han Chan [thchan@ieee.org]
% Please email me if you find bugs, or have suggestions or questions!
% ========================

clear all; close all; clc;
addpath('./Utils');
addpath('./Liblinear');

%% Loading data from MNIST Basic (10000 training, 2000 validation, 50000 testing)
% load mnist_basic data
load('../datasets/UCMerced_LandUse');

TrnSize = size(X, 2);
ImgSize = 256; %28;
ImgFormat = 'color'; %'color' or 'gray'


% ===== Reshuffle the training data =====
% Randnidx = randperm(size(mnist_train,1));
% mnist_train = mnist_train(Randnidx,:);
% =======================================

TrnData = X; %mnist_train(1:TrnSize,1:end-1)';  % partition the data into training set and validation set
TrnLabels = y; %mnist_train(1:TrnSize,end);
% ValData = mnist_train(TrnSize+1:end,1:end-1)';
% ValLabels = mnist_train(TrnSize+1:end,end);
clear X; % clear mnist_train;
clear y;

TestData = X_t; %mnist_test(:,1:end-1)';
TestLabels = y_t; %mnist_test(:,end);
clear X_t; % clear mnist_test;
clear y_t;

% ==== Subsampling the Training and Testing sets ============
% (comment out the following four lines for a complete test)
% every_nth_example = 80;
% TrnData = TrnData(1:every_nth_example:end,:);  % sample around 2500 training samples
% TrnLabels = TrnLabels(1:every_nth_example:end); %
% TestData = TestData(1:every_nth_example:end,:);  % sample around 1000 test samples
% TestLabels = TestLabels(1:every_nth_example:end);
% ===========================================================

nTestImg = length(TestLabels);

%% PCANet parameters (they should be funed based on validation set; i.e., ValData & ValLabel)
% We use the parameters in our IEEE TPAMI submission
PCANet.NumStages = 2;
PCANet.PatchSize = [7 7];
PCANet.NumFilters = [16 16];
PCANet.HistBlockSize = [128 128];
PCANet.BlkOverLapRatio = 0.25;
PCANet.Pyramid = [];

fprintf('\n ====== PCANet Parameters ======= \n')
PCANet

%% PCANet Training with 10000 samples
fprintf('\n ====== PCANet Training ======= \n')
TrnData_ImgCell = mat2imgcell(TrnData,ImgSize,ImgSize,ImgFormat); % convert columns in TrnData to cells
clear TrnData;


fprintf('Number of training samples: %d \n', length(TrnData_ImgCell))
tic;
[ftrain V] = PCANet_train(TrnData_ImgCell,PCANet,1);
PCANet_TrnTime = toc;
clear TrnData_ImgCell;

fprintf('\n ====== Training Linear SVM Classifier ======= \n')
tic;
ftrain = ftrain';
models = train(TrnLabels, ftrain, '-s 1 -q'); % we use linear SVM classifier (C = 1), calling libsvm library
LinearSVM_TrnTime = toc;
[predict_labels] = predict(TrnLabels, ftrain, models, '-q');
clear ftrain;

trn_accuracy = sum(predict_labels == TrnLabels) / length(TrnLabels);
fprintf('Accuracy for trainging set is %g.\n', trn_accuracy);


%% PCANet Feature Extraction and Testing

TestData_ImgCell = mat2imgcell(TestData,ImgSize,ImgSize,ImgFormat); % convert columns in TestData to cells
clear TestData;

fprintf('\n ====== PCANet Testing ======= \n')

nCorrRecog = 0;
RecHistory = zeros(nTestImg,1);

tic;
for idx = 1:1:nTestImg

    ftest = PCANet_FeaExt(TestData_ImgCell(idx),V,PCANet); % extract a test feature using trained PCANet model

    [xLabel_est, accuracy, decision_values] = predict(TestLabels(idx),...
        sparse(ftest'), models, '-q'); % label predictoin by libsvm

    if xLabel_est == TestLabels(idx)
        RecHistory(idx) = 1;
        nCorrRecog = nCorrRecog + 1;
    end

    if 0==mod(idx,nTestImg/100);
        fprintf('Accuracy up to %d tests is %.2f%%; taking %.2f secs per testing sample on average. \n',...
            [idx 100*nCorrRecog/idx toc/idx]);
    end

    TestData_ImgCell{idx} = [];

end

rix = ceil(rand(90, 1) * length(TestData_ImgCell))';
% 
% figure
% for i = rix
%     imshow(TestData_ImgCell{i}(:,:,1:3));
%     title(sprintf('[%s] %s', CATEGORIES{TestLabels(i)}, CATEGORIES{predLabels(i)}));
%     pause
% end

Averaged_TimeperTest = toc/nTestImg;
Accuracy = nCorrRecog/nTestImg;
ErRate = 1 - Accuracy;

%% Results display
fprintf('\n ===== Results of PCANet, followed by a linear SVM classifier =====');
fprintf('\n     PCANet training time: %.2f secs.', PCANet_TrnTime);
fprintf('\n     Linear SVM training time: %.2f secs.', LinearSVM_TrnTime);
fprintf('\n     Testing error rate: %.2f%%', 100*ErRate);
fprintf('\n     Average testing time %.2f secs per test sample. \n\n',Averaged_TimeperTest);
