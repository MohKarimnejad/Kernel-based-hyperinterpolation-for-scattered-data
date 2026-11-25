function [w, residual, r] = BG_solver(Phi, meas, y, delta)
% Backus-Gilbert cubature weights, eq. (14)-(15), with removal of
% numerically dependent rows (revised Algorithm 2, lines 5-7).
%
%   W = argmin ||R^{-1/2} W||_2  subject to  A~ W = b~,
%
% where A~, b~ are the rows of A = Phi and b = y selected by a pivoted QR
% factorization of A' (numerical rank r: |U_kk| > delta |U_11|).
% The prior weights are uniform, R = (meas/M) I (Theorem 5 with omega = 1),
% so the solution is the minimum-norm solution of A~ W = b~; it is computed
% from the same QR factorization, without forming A~ R A~'.
%
% Inputs
%   Phi   : K x M generator matrix (prod_basis(...)')
%   meas  : |Omega| (kept for the interface; a scalar R does not change W)
%   y     : K x 1 moment vector
%   delta : relative rank tolerance (default 1e-10)
%
% Outputs
%   w        : M x 1 weights (NOT modified; negative weights are returned
%              as they are)
%   residual : ||Phi*w - y||_inf / ||y||_inf on ALL K constraints
%   r        : numerical rank of Phi

    if nargin < 4 || isempty(delta)
        delta = 1e-10;
    end

    % Pivoted QR of A':  A'(:,p) = Q U
    [Q, U, p] = qr(Phi', 0);
    d = abs(diag(U));
    r = sum(d > delta * d(1));

    % A~' = Q(:,1:r) U(1:r,1:r)  ->  min-norm solution of A~ W = b~
    b_sel = y(p(1:r));
    w = Q(:, 1:r) * (U(1:r, 1:r)' \ b_sel);

    residual = norm(Phi * w - y, inf) / norm(y, inf);
end
