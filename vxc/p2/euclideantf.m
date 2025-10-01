function out=euclideantf(img, theta, tx, ty)
    % EUCLIDEANTF Apply Euclidean transformation to an image
    %
    %   out = euclideantf(img, theta, tx, ty)
    %
    %   Applies a Euclidean transformation (rotation + translation) to an image.
    %   A Euclidean transformation preserves distances and angles.
    %
    %   Inputs:
    %       img   - Input image to transform
    %       theta - Rotation angle (in radians or degrees, depending on similaritytf)
    %       tx    - Translation in x-direction
    %       ty    - Translation in y-direction
    %
    %   Output:
    %       out   - Transformed image
    %
    %   See also: similaritytf

    out = similaritytf(img, 1, theta, tx, ty);
end
