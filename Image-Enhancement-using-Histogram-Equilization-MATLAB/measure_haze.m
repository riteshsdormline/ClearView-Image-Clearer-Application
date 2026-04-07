function haze_level = measure_haze(V)

edges = edge(V, 'sobel');
edge_density = sum(edges(:)) / numel(V);

haze_level = 1 - edge_density;
end
