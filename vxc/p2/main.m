% Read the RGB image
img = imread('GOOD.png');
[rows, cols, channels] = size(img);

%% Translation Transformation
tx = 0;  % Horizontal translation (pixels)
ty = 100;  % Vertical translation (pixels)

out = translateimage(img, tx, ty);

% Display original and translated images
figure;
subplot(3,4,1);
imshow(img);
title('Original Image');

subplot(3,4,2);
imshow(out);
title(sprintf('Translated (tx=%d, ty=%d)', tx, ty));

%% Rotation Transformation
theta = pi/4;  % Rotation angle (radians)

out = rotateimage(img, theta);

subplot(3,4,3);
imshow(out);
title(sprintf('Rotated theta=%d deg', theta * 180 / pi));

%% Euclidean Transformation
theta = pi/4;  % Rotation angle (radians)
tx = 0;
ty = 300;
out = euclideantf(img, theta, tx, ty);

subplot(3,4,4);
imshow(out);
title(sprintf('Euclidean Transform theta=%d deg', theta * 180 / pi));

%% Similarity Transformation
theta = pi/4;  % Rotation angle (radians)
tx = 0;
ty = 100;
s = 0.75;
out = similaritytf(img, s, theta, tx, ty);

subplot(3,4,5);
imshow(out);
title(sprintf('Similarity Transform theta=%d deg', theta * 180 / pi));

%% Affine Transformation
a00 = 1;
a01 = 0.3;
a10 = 0.2;
a11 = 0.9;
tx = 50;
ty = 100;
out = affinetf(img, a00, a01, a10, a11, tx, ty);

subplot(3,4,6);
imshow(out);
title(sprintf('Affine Transform (Shear)'));
%% Projective Transformation
h00 = 2;
h01 = 0.5;
h02 = -100;
h10 = 0;
h11 = 2;
h12 = 0;
h20 = 0;
h21 = 0.0005;
out = projectivetf(img, h00, h01, h02, h10, h11, h12, h20, h21);

subplot(3,4,7);
imshow(out);
title(sprintf('Projective Transform (Perspective)'));
