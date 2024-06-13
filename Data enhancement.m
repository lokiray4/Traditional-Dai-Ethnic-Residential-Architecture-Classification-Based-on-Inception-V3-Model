originalFolderPath = '/Users/camus/Desktop/test/土掌房';
augmentedFolderPath = '/Users/camus/Desktop/augmented_data/土掌房';

if ~exist(augmentedFolderPath, 'dir')
    mkdir(augmentedFolderPath);
end

imageFiles = dir(fullfile(originalFolderPath, '*.jpg'));

augmenter = imageDataAugmenter(...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandRotation', [90, 90], ... 
    'RandXTranslation', [-20, 20], ... 
    'RandYTranslation', [-20, 20], ...
    'RandXScale', [0.8, 1.2], ... 
    'RandYScale', [0.8, 1.2] ... 
);

for i = 1:numel(imageFiles)
    originalImg = imread(fullfile(originalFolderPath, imageFiles(i).name));
    
    [~, baseFileName, ext] = fileparts(imageFiles(i).name);
    originalFileName = fullfile(augmentedFolderPath, [baseFileName, '_original', ext]);
    imwrite(originalImg, originalFileName);
    
    augmentedImages = cell(5, 1);
    augmentedImages{1} = augment(augmenter, originalImg); 
    augmentedImages{2} = flipdim(originalImg, 2); 
    augmentedImages{3} = imtranslate(originalImg, [randi([-20, 20]), randi([-20, 20])]); 
    augmentedImages{4} = imresize(originalImg, [randi([round(size(originalImg,1)*0.8), round(size(originalImg,1)*1.2)]), randi([round(size(originalImg,2)*0.8), round(size(originalImg,2)*1.2)])]); 
    augmentedImages{5} = originalImg; 
    
    for j = 1:5
        augmentedFileName = fullfile(augmentedFolderPath, [baseFileName, sprintf('_augmented_%d', j), ext]);
        imwrite(augmentedImages{j}, augmentedFileName);
    end
end

disp('数据增强完成');
