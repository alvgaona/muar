clear;
clc

m = mobiledev;
cam = camera(m, 'back');
good_img = snapshot(cam, 'manual');

figure(1)
imshow(good_img)

% Capture another image for comparison
cam.Flash = 'on';

bad_img = snapshot(cam, 'manual');

figure(2)
imshow(bad_img)

