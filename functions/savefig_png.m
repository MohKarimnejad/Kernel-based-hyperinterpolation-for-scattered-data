function savefig_png(fig_id, tag, out_dir)
% Saves the current figure as <out_dir>/Figure_<tag>_<fig_id>.png.
% tag and out_dir are optional.
    if nargin < 2 || isempty(tag)
        filename = sprintf('Figure_%d.png', fig_id);
    else
        filename = sprintf('Figure_%s_%d.png', tag, fig_id);
    end
    if nargin >= 3 && ~isempty(out_dir)
        filename = fullfile(out_dir, filename);
    end
    exportgraphics(gcf, filename, 'Resolution', 300, 'BackgroundColor', 'white');
end
