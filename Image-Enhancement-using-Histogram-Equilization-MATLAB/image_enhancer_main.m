function image_enhancer_main()
    clc;
    clear;

    [filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp'}, 'Select an image');
    if isequal(filename,0)
        return;
    end

    img_rgb = im2double(imread(fullfile(pathname, filename)));
    img_hsv = rgb2hsv(img_rgb);
    V = img_hsv(:,:,3);
    V_uint8 = im2uint8(V);

    contrast_val = std2(V);
    entropy_val = entropy(V);
    haze_level = measure_haze(V);

    fprintf('Contrast: %.4f\nEntropy: %.4f\nHaze Level: %.4f\n', contrast_val, entropy_val, haze_level);

    bbhe = manual_bbhe(V_uint8);
    dsihe = safe_manual_dsihe(V_uint8);
    clahe = manual_clahe(V_uint8, 2.0);
    local = manual_local_hist_eq(V_uint8, 15);

    imgs = {
        hsv2rgb(cat(3, img_hsv(:,:,1:2), im2double(bbhe))), 'BBHE';
        hsv2rgb(cat(3, img_hsv(:,:,1:2), im2double(dsihe))), 'DSIHE';
        hsv2rgb(cat(3, img_hsv(:,:,1:2), im2double(clahe))), 'CLAHE';
        hsv2rgb(cat(3, img_hsv(:,:,1:2), im2double(local))), 'Local HE'
    };

    figure('Name', 'Preview All Enhancement Methods');
    for k = 1:4
        subplot(2, 2, k);
        imshow(imgs{k, 1});
        title(imgs{k, 2});
    end

    choice = inputdlg('Choose method (1-BBHE, 2-DSIHE, 3-CLAHE, 4-Local HE):', 'Select Method', [1 50], {'1'});
    if isempty(choice), return; end
    selected_img = imgs{str2double(choice{1}), 1};

    prompt = {'Brightness [-1 to 1]:','Contrast [0.1 to 3]:','Blur Sigma:', ...
              'Sharpen [0-2]:','Add Noise? [0/1]:','Median Filter? [0/1]:'};
    def = {'0','1','0','0','0','0'};
    adjust = inputdlg(prompt, 'Postprocessing Adjustments', [1 50], def);
    if isempty(adjust), return; end

    final_img = postprocess(selected_img, adjust);

    figure; imshow(final_img); title('Final Enhanced Image');
    choice = questdlg('Do you want to save the final image?', 'Save Image', 'Yes', 'No', 'Yes');
    if strcmp(choice, 'Yes')
        [f, p] = uiputfile({'*.png'; '*.jpg'}, 'Save Image As');
        if f ~= 0
            imwrite(final_img, fullfile(p, f));
            disp('Image saved successfully.');
        end
    end
end

function final = postprocess(img, adj)
    b = str2double(adj{1});
    c = str2double(adj{2});
    blur = str2double(adj{3});
    sharp = str2double(adj{4});
    noise = str2double(adj{5});
    median_filt = str2double(adj{6});

    hsv = rgb2hsv(img);
    V = hsv(:,:,3);
    V = imadjust(V, [], [], c);
    V = V + b;
    V = min(max(V, 0), 1);
    hsv(:,:,3) = V;
    final = hsv2rgb(hsv);

    if blur > 0
        h = fspecial('gaussian', [5 5], blur);
        final = imfilter(final, h, 'replicate');
    end
    if sharp > 0
        final = imsharpen(final, 'Amount', sharp);
    end
    if noise == 1
        final = imnoise(final, 'gaussian', 0, 0.005);
    end
    if median_filt == 1
        final = medfilt3(final);
    end
end

function out = safe_manual_dsihe(img)
    median_val = median(double(img(:)));
    lower_mask = img <= median_val;
    upper_mask = img > median_val;

    lower = img; lower(~lower_mask) = 0;
    upper = img; upper(~upper_mask) = 0;

    lower_eq = manual_global_hist_eq(uint8(lower));
    upper_eq = manual_global_hist_eq(uint8(upper));

    out = zeros(size(img), 'uint8');
    out(lower_mask) = lower_eq(lower_mask);
    out(upper_mask) = upper_eq(upper_mask);
end
