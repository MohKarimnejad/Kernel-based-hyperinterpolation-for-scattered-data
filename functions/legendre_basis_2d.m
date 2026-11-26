function P = legendre_basis_2d(X, q, box)
% Tensor Legendre polynomials of total degree <= q on the box
% [a1,b1] x [a2,b2], L2-orthonormal on the box (not on the domain):
%   p_ij(x,y) = Lt_i(x) Lt_j(y),  i + j <= q,
%   Lt_k(x)   = sqrt((2k+1)/(b-a)) * P_k( (2x - a - b)/(b - a) ).
% Ordered by total degree d = 0..q, and for each d by i = d..0.
%
% Inputs:  X (N x 2), q (degree), box = [a1 b1; a2 b2] (default [0 1;0 1])
% Output:  P (N x (q+1)(q+2)/2)
    if nargin < 3 || isempty(box), box = [0 1; 0 1]; end
    Lx = leg1d(X(:,1), q, box(1,:));
    Ly = leg1d(X(:,2), q, box(2,:));
    P  = zeros(size(X,1), (q+1)*(q+2)/2);
    c  = 0;
    for d = 0:q
        for i = d:-1:0
            c = c + 1;
            P(:,c) = Lx(:,i+1) .* Ly(:,d-i+1);
        end
    end
end

function L = leg1d(x, q, ab)
% Orthonormal Legendre polynomials of degree 0..q on [ab(1), ab(2)].
    a = ab(1); b = ab(2);
    t = (2*x - a - b) / (b - a);
    L = zeros(numel(x), q+1);
    L(:,1) = 1;
    if q >= 1, L(:,2) = t; end
    for k = 1:q-1                       % (k+1) P_{k+1} = (2k+1) t P_k - k P_{k-1}
        L(:,k+2) = ((2*k+1) * t .* L(:,k+1) - k * L(:,k)) / (k+1);
    end
    L = L .* sqrt((2*(0:q) + 1) / (b - a));
end
