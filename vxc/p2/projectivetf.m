function out = projectivetf(img, h00, h01, h02, h10, h11, h12, h20, h21)
    % PROJECTIVETF Apply a projective transformation to an image
    %
    %   out = PROJECTIVETF(img, h00, h01, h02, h10, h11, h12, h20, h21)
    %   applies a projective (homography) transformation to the input image.
    %
    %   Inputs:
    %       img - Input image to be transformed
    %       h00, h01, h02 - First row of the 3x3 homography matrix
    %       h10, h11, h12 - Second row of the 3x3 homography matrix
    %       h20, h21 - First two elements of third row (h22 is fixed to 1)
    %
    %   Output:
    %       out - Transformed image
    %
    %   The transformation matrix T is constructed as:
    %       [h00 h01 h02]
    %       [h10 h11 h12]
    %       [h20 h21  1 ]

    T = [
        h00 h01 h02;
        h10 h11 h12;
        h20 h21 1
    ];

    out = transformimage(img, T);
end
