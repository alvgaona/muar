function out=transformimage(img, T)
   [rows, cols, channels] = size(img);

    out = 255 * ones(rows, cols, channels, 'uint8');
    
    for i = 1:rows
        for j = 1:cols
            % Swapping the indices due as the image point is defined
            % as [x; y; 1]
            point = [j; i; 1];
            
            new_point = T * point;
               
            new_j = round(new_point(1));
            new_i = round(new_point(2));
            
            % Check if new coordinates are within image bounds
            if new_i >= 1 && new_i <= rows && new_j >= 1 && new_j <= cols
                out(new_i, new_j, :) = img(i, j, :);
            end
        end
    end
end