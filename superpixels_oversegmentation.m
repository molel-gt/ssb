function [BW, L, N] = superpixels_oversegmentation(img_path)
    img = imread(img_path);
    [L, N] = superpixels(img, 500);
    figure
    BW = boundarymask(L);
%     imshow(imoverlay(img,BW,'red'),'InitialMagnification',67)
end