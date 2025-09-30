function out=translateimage(img, tx, ty)
    % Homogeneous transformation matrix
    T = [
        1  0  tx;
        0  1  ty;
        0  0  1
    ];
    
    out = transformimage(img, T);
end