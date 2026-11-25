function D = cf_diagnostics(info, VZ, w, G, K_test, solver)
% Diagnostics requested by Reviewer 2 (point 7) for one cubature rule.
%
% Inputs
%   info    : info struct returned by getCF.m
%   VZ      : M x n kernel translates at the cubature nodes
%   w       : M x 1 cubature weights
%   G       : n x n L2 Gram matrix of the kernel translates (moment.m)
%   K_test  : n_test x n kernel translates on the error grid
%   solver  : "QR" or "SVD", the solver used in the test (LS_stable_solver)
%
% Output struct D
%   M          number of nodes (= function evaluations)
%   residual   ||A W - b||_inf / ||b||_inf       (cubature residual)
%   w_min      min weight   (no weight is modified)
%   w_max      max weight
%   rank_A     numerical rank of the generator matrix A
%   rank_VZ    numerical rank of sqrt(W) VZ (singular values > 1e-12*max)
%   repro      reproduction error on V_n:  ||K_test (a - c)||_2 / ||K_test c||_2
%              for f = sum_j c_j K(., x_j), c random (fixed seed)
%   eta        MZ constant, eq. (23):  max |lambda - 1| over the eigenvalues
%              of the discrete Gram matrix V_Z' W V_Z relative to G, computed
%              in an L2-orthonormal basis of the numerically resolved part of
%              V_n (eigenvalues of G below 1e-12*max are discarded)
%   n_eta      dimension of that resolved subspace

    if nargin < 6 || isempty(solver), solver = "QR"; end
    svd_tol = 1e-12;

    n = size(VZ, 2);
    W = diag(w);

    % numerical rank of sqrt(W) VZ
    s = svd(sqrt(W) * VZ);
    rank_VZ = sum(s > svd_tol * s(1));

    % reproduction of V_n
    rng_state = rng;  rng(1);
    c = randn(n, 1);
    rng(rng_state);
    a  = LS_stable_solver(W, VZ, VZ * c, solver);
    repro = norm(K_test * (a - c)) / norm(K_test * c);

    % MZ constant
    G  = (G + G') / 2;
    [U, S] = eig(G);
    lam = diag(S);
    keep = lam > 1e-12 * max(lam);
    T  = U(:, keep) ./ sqrt(lam(keep))';      % L2-orthonormal basis
    Gw = VZ' * W * VZ;
    Mz = T' * Gw * T;
    Mz = (Mz + Mz') / 2;
    eta = max(abs(eig(Mz) - 1));

    D = struct('M', numel(w), 'residual', info.residual, ...
               'w_min', min(w), 'w_max', max(w), 'rank_A', info.rank, ...
               'rank_VZ', rank_VZ, 'repro', repro, 'eta', eta, ...
               'n_eta', sum(keep), 'converged', info.converged);
end
