clc;
clear;

% Load image
[filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp'}, 'Select an image');
if isequal(filename, 0)
    disp('User canceled.');
    return;
end

img_rgb = im2double(imread(fullfile(pathname, filename)));

% Convert to HSV and get Brightness channel
img_hsv = rgb2hsv(img_rgb);
V = img_hsv(:, :, 3);
V_uint8 = im2uint8(V); % Work in uint8 for histogram functions

% ========== 1. Global Histogram Equalization ==========
ghe_V = manual_global_hist_eq(V_uint8);
img_hsv(:, :, 3) = im2double(ghe_V);
ghe_rgb = hsv2rgb(img_hsv);

% ========== 2. CLAHE ==========
clahe_V = manual_clahe(V_uint8, 2.0);
img_hsv(:, :, 3) = im2double(clahe_V);
clahe_rgb = hsv2rgb(img_hsv);

% ========== 3. BBHE ==========
bbhe_V = manual_bbhe(V_uint8);
img_hsv(:, :, 3) = im2double(bbhe_V);
bbhe_rgb = hsv2rgb(img_hsv);

% ========== 4. DSIHE ==========
dsihe_V = manual_dsihe(V_uint8);
img_hsv(:, :, 3) = im2double(dsihe_V);
dsihe_rgb = hsv2rgb(img_hsv);

% ========== 5. Local Histogram Equalization ==========
local_V = manual_local_hist_eq(V_uint8, 15);
img_hsv(:, :, 3) = im2double(local_V);
local_rgb = hsv2rgb(img_hsv);

% ========== Show All ==========
figure('Name','Comparison of Histogram Methods', 'NumberTitle','off');
subplot(2,3,1); imshow(img_rgb); title('Original Image');
subplot(2,3,2); imshow(ghe_rgb); title('Global Hist Eq');
subplot(2,3,3); imshow(clahe_rgb); title('CLAHE');
subplot(2,3,4); imshow(bbhe_rgb); title('BBHE');
subplot(2,3,5); imshow(dsihe_rgb); title('DSIHE');
subplot(2,3,6); imshow(local_rgb); title('Local Hist Eq');
