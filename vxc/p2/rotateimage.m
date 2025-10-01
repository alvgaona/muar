function out=rotateimage(img, theta)
    % ROTATEIMAGE Rotate an image by a specified angle
    %
    % Syntax:
    %   out = rotateimage(img, theta)
    %
    % Inputs:
    %   img   - Input image to be rotated
    %   theta - Rotation angle in radians (positive for counter-clockwise)
    %
    % Outputs:
    %   out   - Rotated image
    %
    % Description:
    %   This function rotates an image by the specified angle theta
    %   without any translation. It uses euclideantf for the transformation.

    out = euclideantf(img, theta, 0, 0);
end
