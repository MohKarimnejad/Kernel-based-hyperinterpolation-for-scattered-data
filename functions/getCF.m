function [X, w, info] = getCF(N, meas, y, opts)
% Revised Algorithm 2: construction of a positive cubature formula that is
% exact on the space spanned by given generators.
%
% Nodes are the first N points of a nested sequence in Omega. At each
% iteration the weights are computed by BG_solver (dependent rows removed
% by pivoted QR) and the loop stops when
%
%     ||A W - b||_inf / ||b||_inf <= tol   and   min(W) > 0.
%
% The dimension of the space spanned by the generators is not estimated.
%
% Inputs
%   N     : initial number of nodes M_0
%   meas  : |Omega|
%   y     : moment vector b (K x 1)
%   opts  : struct with fields
%           .nodes  function handle N -> first N nodes (N x dim)   (required)
%           .gen    function handle X -> generator matrix A (K x N) (required)
%           .tol    exactness tolerance        (default 1e-10)
%           .delta  rank tolerance for BG       (default 1e-10)
%           .dN     nodes added per iteration   (default 1, as in the paper)
%           .N_max  maximum number of nodes     (default 1e5)
%
% Outputs
%   X     : nodes (N x dim)
%   w     : weights (N x 1)
%   info  : struct with fields residual, w_min, w_max, rank, N,
%           iterations, converged, tol, delta

    tol   = get_opt(opts, 'tol',   1e-10);
    delta = get_opt(opts, 'delta', 1e-10);
    dN    = get_opt(opts, 'dN',    1);
    N_max = get_opt(opts, 'N_max', 1e5);
    nodes_fun = opts.nodes;
    gen_fun   = opts.gen;

    residual  = Inf;
    w_min     = -Inf;
    iter      = 0;
    converged = true;

    while (residual > tol) || (w_min <= 0)

        if N > N_max
            converged = false;
            warning('getCF:noConvergence', ...
                ['getCF: N_max = %d reached without an exact positive rule ', ...
                 '(residual = %.2e, min(w) = %.2e).'], N_max, residual, w_min);
            break
        end

        X   = nodes_fun(N);
        Phi = gen_fun(X);
        [w, residual, r] = BG_solver(Phi, meas, y, delta);
        w_min = min(w);

        iter = iter + 1;
        N    = N + dN;
    end

    info = struct('residual', residual, 'w_min', w_min, 'w_max', max(w), ...
                  'rank', r, 'N', size(X, 1), 'iterations', iter, ...
                  'converged', converged, 'tol', tol, 'delta', delta);
end

function v = get_opt(opts, name, default)
    if isfield(opts, name) && ~isempty(opts.(name))
        v = opts.(name);
    else
        v = default;
    end
end
