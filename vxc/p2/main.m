% Read the RGB image
img = imread('BAD.png');
[rows, cols, channels] = size(img);

% Create figure with better layout
figure('Position', [100, 100, 1400, 900]);
sgtitle('\textbf{Image 2D Transformations}', 'Interpreter', 'latex', 'FontSize', 16);

%% Original Image
subplot(3,4,1);
imshow(img);
axis on;
axis tight;
xtickangle(45);
xlabel('$x$ (pixels)', 'Interpreter', 'latex', 'FontSize', 10);
ylabel('$y$ (pixels)', 'Interpreter', 'latex', 'FontSize', 10);
title('Original Image', 'Interpreter', 'latex', 'FontSize', 12);

%% Translation Transformation
tx = 0;  % Horizontal translation (pixels)
ty = 100;  % Vertical translation (pixels)

out = translateimage(img, tx, ty);

subplot(3,4,2);
imshow(out);
axis on;
axis tight;
xtickangle(45);
xlabel('$x$ (pixels)', 'Interpreter', 'latex', 'FontSize', 10);
ylabel('$y$ (pixels)', 'Interpreter', 'latex', 'FontSize', 10);
title(sprintf('\\textbf{Translation}\n$t_x = %d$, $t_y = %d$ px', tx, ty), ...
    'Interpreter', 'latex', 'FontSize', 11);

%% Rotation Transformation
theta = pi/4;  % Rotation angle (radians)
out = rotateimage(img, theta);

subplot(3,4,3);
imshow(out);
axis on;
axis tight;
xtickangle(45);
xlabel('$x$ (pixels)', 'Interpreter', 'latex', 'FontSize', 10);
ylabel('$y$ (pixels)', 'Interpreter', 'latex', 'FontSize', 10);
title(sprintf('\\textbf{Rotation}\n$\\theta = %.0f^\\circ$ (%.3f rad)', theta * 180 / pi, theta), ...
    'Interpreter', 'latex', 'FontSize', 11);

%% Euclidean Transformation
theta = pi/4;  % Rotation angle (radians)
tx = 0;
ty = 300;
out = euclideantf(img, theta, tx, ty);

subplot(3,4,4);
imshow(out);
axis on;
axis tight;
xtickangle(45);
xlabel('$x$ (pixels)', 'Interpreter', 'latex', 'FontSize', 10);
ylabel('$y$ (pixels)', 'Interpreter', 'latex', 'FontSize', 10);
title(sprintf('\\textbf{Euclidean Transform}\n$\\theta = %.0f^\\circ$, $t_x = %d$, $t_y = %d$ px', ...
    theta * 180 / pi, tx, ty), 'Interpreter', 'latex', 'FontSize', 11);

%% Similarity Transformation
theta = pi/4;  % Rotation angle (radians)
tx = 0;
ty = 100;
s = 0.75;
out = similaritytf(img, s, theta, tx, ty);

subplot(3,4,5);
imshow(out);
axis on;
axis tight;
xtickangle(45);
xlabel('$x$ (pixels)', 'Interpreter', 'latex', 'FontSize', 10);
ylabel('$y$ (pixels)', 'Interpreter', 'latex', 'FontSize', 10);
title(sprintf('\\textbf{Similarity Transform}\n$s = %.2f$, $\\theta = %.0f^\\circ$, $t_x = %d$, $t_y = %d$ px', ...
    s, theta * 180 / pi, tx, ty), 'Interpreter', 'latex', 'FontSize', 11);

%% Affine Transformation
a00 = 1;
a01 = 0.2;
a10 = 0.1;
a11 = 1;
tx = 0;
ty = 0;
out = affinetf(img, a00, a01, a10, a11, tx, ty);

subplot(3,4,6);
imshow(out);
axis on;
axis tight;
xtickangle(45);
xlabel('$x$ (pixels)', 'Interpreter', 'latex', 'FontSize', 10);
ylabel('$y$ (pixels)', 'Interpreter', 'latex', 'FontSize', 10);
title(sprintf('\\textbf{Affine Transform}\n$a_{00}=%.1f$, $a_{01}=%.1f$, $a_{10}=%.1f$, $a_{11}=%.1f$', ...
    a00, a01, a10, a11), 'Interpreter', 'latex', 'FontSize', 11);

%% Projective Transformation
T = [
    1 0 0;
    0 1 0;
    0 0.0009 1
];

out = transformimage(img, T);

subplot(3,4,7);
imshow(out);
axis on;
axis tight;
xtickangle(45);
xlabel('$x$ (pixels)', 'Interpreter', 'latex', 'FontSize', 10);
ylabel('$y$ (pixels)', 'Interpreter', 'latex', 'FontSize', 10);
title(sprintf('\\textbf{Projective Transform}\n(Perspective: $h_{21} = %.4f$)', T(3,2)), ...
    'Interpreter', 'latex', 'FontSize', 11);

subplot(3,4,[9:12]);
axis off;
text(0, 0.9, '\textbf{Transformation Hierarchy:}', ...
    'Interpreter', 'latex', 'FontSize', 13, 'FontWeight', 'bold');
text(0, 0.75, '$\bullet$ \textbf{Translation:} \quad $\mathbf{x}'' = \mathbf{x} + \mathbf{t}$ \quad (2 DOF)', ...
    'Interpreter', 'latex', 'FontSize', 11);
text(0, 0.60, '$\bullet$ \textbf{Euclidean (Rigid):} \quad Rotation + Translation \quad (3 DOF)', ...
    'Interpreter', 'latex', 'FontSize', 11);
text(0, 0.45, '$\bullet$ \textbf{Similarity:} \quad Scale + Rotation + Translation \quad (4 DOF)', ...
    'Interpreter', 'latex', 'FontSize', 11);
text(0, 0.30, '$\bullet$ \textbf{Affine:} \quad Linear transform + Translation \quad (6 DOF)', ...
    'Interpreter', 'latex', 'FontSize', 11);
text(0, 0.15, '$\bullet$ \textbf{Projective (Homography):} \quad Most general linear transform \quad (8 DOF)', ...
    'Interpreter', 'latex', 'FontSize', 11);

set(gcf, 'Color', 'w');
