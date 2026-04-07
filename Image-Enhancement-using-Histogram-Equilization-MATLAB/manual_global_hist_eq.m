function out = manual_global_hist_eq(img)
    img = double(img);
    hist_counts = imhist(uint8(img));
    cdf = cumsum(hist_counts) / numel(img);
    out = uint8(255 * cdf(img + 1));
end