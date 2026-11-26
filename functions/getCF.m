function [Phi, X, w, N, r, info] = getCF(~, N, x_ctrs, ker, meas, y, reduced_param, dim, opts)
% Revised Algorithm 2: construction of the positive cubature formula.
%
% Nodes are the first N points of the Halton sequence (nested). At each
% iteration the weights are computed by BG_solver (dependent rows removed
% by pivoted QR) and the loop stops when
%
%     ||A W - b||_inf / ||b||_inf <= tol   and   min(W) > 0.
%
% The dimension of the product space is NOT estimated: the first input
% (former K) is ignored and kept only for backward compatibility.
%
% Inputs
%   N      : initial number of nodes M_0
%   opts   : struct with optional fields
%            .tol    exactness tolerance        (default 1e-10)
%            .delta  rank tolerance for BG       (default 1e-10)
%            .dN     nodes added per iteration   (default 1, as in the paper)
%            .N_max  maximum number of nodes     (default 1e5)
%            .nodes  function handle N -> first N nodes (N x dim) of a
%                    nested sequence in Omega (default: Halton in [0,1]^dim)
%            .gen    function handle X -> generator matrix A (K x N)
%                    (default: prod_basis(X, x_ctrs, ker, reduced_param))
%
% Outputs
%   Phi, X, w : generator matrix, nodes (N x dim), weights (N x 1)
%   N         : number of nodes returned
%   r         : numerical rank of Phi at termination
%   info      : struct with fields residual, w_min, w_max, rank, N,
%               iterations, converged, tol, delta

    if nargin < 9 || isempty(opts), opts = struct(); end
    tol   = get_opt(opts, 'tol',   1e-10);
    delta = get_opt(opts, 'delta', 1e-10);
    dN    = get_opt(opts, 'dN',    1);
    N_max = get_opt(opts, 'N_max', 1e5);
    nodes_fun = get_opt(opts, 'nodes', []);
    gen_fun   = get_opt(opts, 'gen',   []);

    % Nested Halton nodes, generated once and extended when needed
    H = [];
    if isempty(get_opt(opts, 'nodes', []))
        H = haltonseq(max(2 * N, 1000), dim);
    end

    residual = Inf;
    w_min    = -Inf;
    iter     = 0;
    converged = true;

    while (residual > tol) || (w_min <= 0)

        if N > N_max
            converged = false;
            warning('getCF:noConvergence', ...
                ['getCF: N_max = %d reached without an exact positive rule ', ...
                 '(residual = %.2e, min(w) = %.2e).'], N_max, residual, w_min);
            break
        end

        if isempty(nodes_fun)
            if N > size(H, 1)
                H = haltonseq(2 * N, dim);
            end
            X = H(1:N, :);
        else
            X = nodes_fun(N);
        end

        if isempty(gen_fun)
            Phi = prod_basis(X, x_ctrs, ker, reduced_param);
        else
            Phi = gen_fun(X);
        end
        [w, residual, r] = BG_solver(Phi, meas, y, delta);
        w_min = min(w);

        iter = iter + 1;
        N    = N + dN;
    end

    N = size(X, 1);

    info = struct('residual', residual, 'w_min', w_min, 'w_max', max(w), ...
                  'rank', r, 'N', N, 'iterations', iter, ...
                  'converged', converged, 'tol', tol, 'delta', delta);
end

function v = get_opt(opts, name, default)
    if isfield(opts, name) && ~isempty(opts.(name))
        v = opts.(name);
    else
        v = default;
    end
end
