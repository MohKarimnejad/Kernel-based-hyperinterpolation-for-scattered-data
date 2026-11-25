function K = prod_basis(X, x_ctrs, ker, reduced_param)
% Generator matrix of the cubature system A W = b  (eq. (13)).
%
% Rows are the generators evaluated at the nodes X (M x d):
%   reduced_param == 1 :  [1; K(.,x_1); ...; K(.,x_n)]                (n+1) x M
%   reduced_param == 0 :  [1; K(.,x_i), i=1..n; K(.,x_i)K(.,x_j), i<=j] x M
% The products are ordered as in the loop  for i = 1:n, for j = i:n.
% This ordering must match moment.m.

    m = size(X, 1);
    n = size(x_ctrs, 1);
    A = ker(X, x_ctrs);                       % m x n

    if reduced_param == 1
        K = [ones(m, 1), A];
    else
        [I, J] = prod_index(n);
        K = [ones(m, 1), A, A(:, I) .* A(:, J)];
    end

    K = K';
end
