function out=similaritytf(img, s, theta, tx, ty)
    % SIMILARITYTF Apply a similarity transformation to an image
    %
    % Syntax:
    %   out = similaritytf(img, s, theta, tx, ty)
    %
    % Description:
    %   Applies a similarity transformation (rotation, scaling, and translation)
    %   to an input image using a homogeneous transformation matrix.
    %
    % Input Arguments:
    %   img   - Input image to be transformed
    %   s     - Scale factor
    %   theta - Rotation angle in radians
    %   tx    - Translation in x-direction
    %   ty    - Translation in y-direction
    %
    % Output Arguments:
    %   out   - Transformed image
    %
    % See also: transformimage

    % Homogeneous transformation matrix
    T = [
        s*cos(theta)  -s*sin(theta)  tx;
        s*sin(theta)  s*cos(theta)  ty;
        0  0  1
    ];

    out = transformimage(img, T);
end
