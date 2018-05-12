clear; close all; clc

%link download database
%http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

%convention: Col - x - width
%            Row - y - height

num_eigenface = 20; %number of eigenfaces
num_weight = 20; %number of weight parameters

%% read all images in database into a 3D array

currentFolder = pwd; %identify current directory
path = fullfile(currentFolder,'\ORL'); %path to image database folder
folderInfo = dir(path); %list files and folders in current path
folderInfo = folderInfo(3:end); %remove the '.' and '..' directories
num_folder = length(folderInfo); %total number of files and folders in current path

depth = 1; %initialize counter
for folder_idx = 1:num_folder %go through each folder
    
    imagePath = fullfile(path,folderInfo(folder_idx).name); 
    %path to each folder in current path
    %folder contains images
    imageInfo = dir(imagePath); %list all images in current path
    imageInfo = imageInfo(3:end);
    num_image = length(imageInfo); %number of images per directory - 10 images
    
    for image_idx = 1:num_image %go through each image
        deck(:,:,depth) = imread([imagePath,'\',imageInfo(image_idx).name]); %3D array
        depth = depth+1; %update counter
    end
end

%% divide the database into testing image sets and training image sets 

%use the first image (1.pgm) of each person for testing  
testingImageSet = deck(:,:,1:10:end); %step = 10

%use the last 9 images (2.pgm - 10.pgm) of each person for training
trainingImageSet = deck; 
trainingImageSet(:,:,1:10:end) = []; %delete items of testing set

%convert to double for later computation
testingImageSet = double(testingImageSet);
trainingImageSet = double(trainingImageSet);

%% resize image dimension

[imageHeight,imageWidth,imageNum] = size(trainingImageSet); %3D size
M = imageHeight*imageWidth; 
TotalImage = zeros(M,imageNum); %2D size
sum = zeros(M,1); %1D size

for cnt = 1:imageNum
    OneImage = trainingImageSet(:,:,cnt); %2D size
    TotalImage(:,cnt) = OneImage(:); %1D size
    sum = sum + OneImage(:); %sum all data in trainingImageSet
end

%% calculate average face

avgFaceVec = sum/imageNum; 
avgFace = reshape(avgFaceVec,imageHeight,imageWidth); %reshape 1D array to 2D array
figure; imagesc(avgFace); colormap(gray(256)); title('Average Face'); axis image;

%% compute each face's difference from the average face

A = zeros(M,imageNum);
for cnt = 1:imageNum
    A(:,cnt) = TotalImage(:,cnt) - avgFaceVec;
end

%% compute a covariance matrix for dataset 

C = A'*A;

%---PCA---
[eigVec,eigVal] = eig(C); 
%eigVal: diagonal matrix of eigenvalues
%eigVec: whose columns are the corresponding right eigenvectors

V = A*eigVec;

%% calculate eigenfaces
eigenFace = [];
ghostFace = []; %store eigenfaces with the highest eigenvalues

for cnt = 1:imageNum
    eigenFace{cnt} = reshape(V(:,cnt),imageHeight,imageWidth); %cell array
end

element = diag(eigVal); %get diagonal elements of matrix eigVal
[element_sort,element_index] = sort(element,'descend'); %sort array elements
%element_sort: sorted the elements in descending order
%element_index: describes the arrangement of the elements
%element_sort = element(element_index)

for n = 1:num_eigenface %eigenfaces with the highest eigenvalues
    ghostFace{n} = eigenFace{element_index(n)};
    eigenFaceVec(:,n) = ghostFace{n}(:); %create reduced eigenface space
end

eigenFace20 = [ghostFace{1} ghostFace{2} ghostFace{3} ghostFace{4} ghostFace{5} ...
               ghostFace{6} ghostFace{7} ghostFace{8} ghostFace{9} ghostFace{10}; ...
               ghostFace{11} ghostFace{12} ghostFace{13} ghostFace{14} ghostFace{15} ...
               ghostFace{16} ghostFace{17} ghostFace{18} ghostFace{19} ghostFace{20}];
figure; imagesc(eigenFace20); colormap(gray); title('20 Eigenfaces'); axis image;

%% weight vector is computed for each face in the training dataset

for i = 1:imageNum 
    for k = 1:num_weight
        w_train(k,i) = dot(A(:,i),eigenFaceVec(:,k));
    end
end

%% weight vector is computed for each face in the testing dataset

[~,~,numTest] = size(testingImageSet); 

for i = 1:numTest
    newFace = testingImageSet(:,:,i);
    newA = newFace(:) - avgFaceVec;
    for k = 1:num_weight
        w_test(k,i) = dot(newA,eigenFaceVec(:,k));
    end
end

%% matching

%matching is done by finding the weight vector in the training set 
%that has the minimum Euclidean distance to that of image being tested

dist = distance_mx(w_test,w_train);

for i = 1:numTest
    newFace = testingImageSet(:,:,i); %image being tested
    [mi,idx] = min(dist(i,:)); %find minimum
    recognizedFace = trainingImageSet(:,:,idx); %matching
    figure; 
    subplot(121); imagesc(newFace); colormap(gray); axis image; title('Testing Face');
    subplot(122); imagesc(recognizedFace); colormap(gray); axis image; title('Recognized Face');
end
