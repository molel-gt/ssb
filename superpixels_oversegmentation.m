function [BW, L] = superpixels_oversegmentation(img_path)
    pkg load image
    img = imread(img_path);
    img = single(img);
    L = vl_slic(img, 10, 10);
    %[L, N] = superpixels(img, 500);
    %figure
    BW = bwboundaries(L);
%     imshow(imoverlay(img,BW,'red'),'InitialMagnification',67)
end