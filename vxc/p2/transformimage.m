function out=transformimage(img, T)
    % TRANSFORMIMAGE Apply a geometric transformation to an image using inverse mapping
    %
    %   out = TRANSFORMIMAGE(img, T) applies the transformation matrix T to the
    %   input image img using inverse mapping.
    %
    %   Inputs:
    %       img - Input image (rows x cols x channels), can be grayscale or color
    %       T   - 3x3 transformation matrix in homogeneous coordinates
    %
    %   Output:
    %       out - Transformed image with the same size as the input image.
    %             Pixels that map outside the source image bounds are set to
    %             white (255).
    %
    %   The function uses inverse mapping to avoid holes in the output image.
    %   For each pixel in the output image, it computes the corresponding
    %   location in the input image using the inverse of T.

    [rows, cols, channels] = size(img);

    out = 255 * ones(rows, cols, channels, 'uint8');

    for i = 1:rows
        for j = 1:cols
            % Swapping the indices due as the image point is defined
            % as [x; y; 1] and x spans through the columns of the image
            % and y spans through the rows of the image
            dest_point = [j; i; 1];

            source_point = T \ dest_point;

            source_j = round(source_point(1) / source_point(3));
            source_i = round(source_point(2) / source_point(3));

            % Check if new coordinates are within image bounds
            if source_i >= 1 && source_i <= rows && source_j >= 1 && source_j <= cols
                out(i, j, :) = img(source_i, source_j, :);
            end
        end
    end
end
