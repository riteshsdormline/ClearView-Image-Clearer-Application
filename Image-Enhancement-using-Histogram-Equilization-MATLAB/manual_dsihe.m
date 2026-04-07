function out = manual_dsihe(img)
median_val = median(double(img(:)));
lower_mask = img <= median_val;
upper_mask = img > median_val;

lower = img .* uint8(lower_mask);
upper = img .* uint8(upper_mask);

lower_eq = manual_global_hist_eq(lower);
upper_eq = manual_global_hist_eq(upper);

out = lower_mask .* lower_eq + upper_mask .* upper_eq;
end
