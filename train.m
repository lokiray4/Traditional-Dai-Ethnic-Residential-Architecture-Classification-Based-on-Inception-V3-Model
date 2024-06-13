clc
clear
close all
digitDatasetPath = fullfile('/Users/camus/Desktop/傣族论文/l2');
digitData = imageDatastore(digitDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');
figure;
perm = randperm(numel(digitData.Files), 20)
%perm = randperm(644,20);
for i = 1:20
    subplot(4,5,i);
    imshow(digitData.Files{perm(i)});
end

labelCount = countEachLabel(digitData);

img = imread(digitData.Files{1});

size(img)

% trainNumFiles =500;

[trainDigitData,valDigitData] = splitEachLabel(digitData,0.8,'randomize');
layers = [
    imageInputLayer([256 256 3])
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
   reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding',1)
   batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,64,'Padding',1)
 batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm',...
    'MiniBatchSize',128, ...
    'MaxEpochs',100, ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
   'ValidationData',valDigitData,...
    'ValidationFrequency',30,...
    'Shuffle','every-epoch',...
    'Verbose',true,...
    'Plots','training-progress');


net = trainNetwork(trainDigitData,layers,options);
predictedLabels = classify(net,valDigitData);
valLabels = valDigitData.Labels;

accuracy = sum(predictedLabels == valLabels)/numel(valLabels);
figure
trueLabels=valLabels ;
cm = confusionchart(trueLabels,predictedLabels, ...
    'Title','My Title', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
% load('','trueLabels','predictedLabels');
%  
% figure %创建混淆矩阵图
% cm = confusionchart(trueLabels,predictedLabels);