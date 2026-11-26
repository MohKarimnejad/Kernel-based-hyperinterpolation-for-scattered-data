function val = leb_const(K_leb, VZ, W)
% Lebesgue constant (ell^inf(Z) -> L^inf(Omega) operator norm) of the
% weighted least-squares operator
%
%     f(Z)  ->  a = argmin || sqrt(W) (VZ a - f(Z)) ||_2  ->  K_leb * a,
%
% computed with the same economy QR factorization of sqrt(W)*VZ that
% LS_stable_solver.m uses for the coefficients (Algorithm 1):
%
%     sqrt(W)*VZ = Q R,   a = R \ (Q' sqrt(W) f(Z)),
%     H = K_leb * R^{-1} Q' sqrt(W)   (n_leb x M),
%     val = max_x sum_k |H(x,k)|.
%
% Inputs
%   K_leb : n_leb x n  kernel translates on the evaluation grid
%   VZ    : M x n      kernel translates at the nodes Z
%   W     : M x M      diagonal weight matrix

    sW = sqrt(W);
    [Q, R] = qr(sW * VZ, 0);
    B   = R \ (Q' * sW);          % n x M
    H   = K_leb * B;              % n_leb x M
    val = max(sum(abs(H), 2));
end
