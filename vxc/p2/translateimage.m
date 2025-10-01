function out=translateimage(img, tx, ty)
    % TRANSLATEIMAGE Translate an image by specified pixel amounts
    %
    %   out = translateimage(img, tx, ty)
    %
    % Inputs:
    %   img - Input image to be translated
    %   tx  - Translation in x direction (horizontal, in pixels)
    %   ty  - Translation in y direction (vertical, in pixels)
    %
    % Output:
    %   out - Translated image
    %
    % This function translates an image by the specified amounts using
    % Euclidean transformation with zero rotation.

    out = euclideantf(img, 0, tx, ty);
end
