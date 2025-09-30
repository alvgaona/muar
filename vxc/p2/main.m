% Read the RGB image
img = imread('GOOD.png');
[rows, cols, channels] = size(img);

%% Translation Transformation
% Define translation parameters
tx = 0;  % horizontal translation (pixels)
ty = 100;  % vertical translation (pixels)

translated_img = translateimage(img, tx, ty);

% Display original and translated images
figure;
subplot(1,2,1);
imshow(img);
title('Original Image');

subplot(1,2,2);
imshow(translated_img);
title(sprintf('Translated (tx=%d, ty=%d)', tx, ty));

%% Rotation Transformation


%% Euclidean Transformation


%% Similarity Transformation

%% Affine Transformation

%% Proyective Transformation