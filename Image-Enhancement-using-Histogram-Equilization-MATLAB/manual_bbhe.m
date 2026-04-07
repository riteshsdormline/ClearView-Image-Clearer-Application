function out = manual_bbhe(img)
    mean_val = mean(img(:));
    lower_mask = img <= mean_val;
    upper_mask = img > mean_val;

    lower = img;
    lower(~lower_mask) = 0;

    upper = img;
    upper(~upper_mask) = 0;

    lower_eq = manual_global_hist_eq(lower);
    upper_eq = manual_global_hist_eq(upper);

    % Use logical indexing to merge
    out = zeros(size(img), 'uint8');
    out(lower_mask) = lower_eq(lower_mask);
    out(upper_mask) = upper_eq(upper_mask);
end
