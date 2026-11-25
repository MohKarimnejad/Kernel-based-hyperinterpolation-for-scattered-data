function savefig_png(fig_id, tag)
% Saves the current figure as Figure_<tag>_<fig_id>.png (or Figure_<fig_id>.png
% if no tag is given). The tag avoids overwriting the figures of the
% "scaled" and "fixed" shape-parameter runs.
    if nargin < 2 || isempty(tag)
        filename = sprintf('Figure_%d.png', fig_id);
    else
        filename = sprintf('Figure_%s_%d.png', tag, fig_id);
    end
    exportgraphics(gcf, filename, 'Resolution', 300, 'BackgroundColor', 'white');
end
