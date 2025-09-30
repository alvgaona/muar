clc;
clear;

figure(1)
good_info = imfinfo("IMG_1525.heic")
imshow("IMG_1525.heic");

figure(2)
bad_info = imfinfo("IMG_1527.heic")
imshow("IMG_1527.heic");