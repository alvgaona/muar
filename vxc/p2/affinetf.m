function out=affinetf(img, a00, a01, a10, a11, tx, ty)
    % AFFINETF Apply affine transformation to an image
    %
    %   out = AFFINETF(img, a00, a01, a10, a11, tx, ty) applies an affine
    %   transformation to the input image using the specified transformation
    %   matrix coefficients.
    %
    %   Inputs:
    %       img - Input image to be transformed
    %       a00 - Element (1,1) of the affine transformation matrix
    %       a01 - Element (1,2) of the affine transformation matrix
    %       a10 - Element (2,1) of the affine transformation matrix
    %       a11 - Element (2,2) of the affine transformation matrix
    %       tx  - Translation in x direction
    %       ty  - Translation in y direction
    %
    %   Output:
    %       out - Transformed image
    %
    %   The transformation matrix is:
    %       [a00 a01 tx]
    %       [a10 a11 ty]
    %       [ 0   0   1]

    % Homogeneous transformation matrix
    T = [ a00 a01 tx;
        a10 a11 ty;
        0  0  1
    ];

    out = transformimage(img, T);
end
