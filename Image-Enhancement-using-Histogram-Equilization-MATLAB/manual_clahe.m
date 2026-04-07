function out = manual_clahe(img, clip_limit)
    img = double(img);
    num_bins = 256;
    tile_size = [floor(size(img,1)/8), floor(size(img,2)/8)];
    out = adapthisteq(uint8(img), 'ClipLimit', clip_limit/100, 'NumTiles', tile_size);
end
