function out = manual_local_hist_eq(img, win_size)
    pad_size = floor(win_size / 2);
    padded_img = padarray(img, [pad_size pad_size], 'symmetric');
    out = zeros(size(img), 'like', img);

    for i = 1:size(img, 1)
        for j = 1:size(img, 2)
            local = padded_img(i:i+win_size-1, j:j+win_size-1);
            local_eq = manual_global_hist_eq(local);
            out(i, j) = local_eq(pad_size+1, pad_size+1);
        end
    end

    out = uint8(out);
end
