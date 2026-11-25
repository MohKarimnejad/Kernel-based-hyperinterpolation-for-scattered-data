function [y, err_qmc, G] = moment(ker, x_ctrs, reduced_param, L)
% Moments b_i = int_Omega g_i(x) dx on Omega = [0,1]^d (uniform measure),
% for the generators of prod_basis.m, computed by quasi-Monte Carlo (QMC)
% on the first L Halton points.
%
% Outputs
%   y       : moment vector, same ordering as the rows of prod_basis(...)'
%   err_qmc : estimate of the QMC error, ||y_L - y_{L/2}||_inf / ||y_L||_inf,
%             i.e. the difference between the rule on L and on L/2 points.
%             It is used to set the exactness tolerance in getCF.m.
%   G       : n x n Gram matrix G_ij = int_Omega K(x,x_i) K(x,x_j) dx (same
%             QMC rule). It is returned for both values of reduced_param and
%             is used to estimate the MZ constant (cf_diagnostics.m).
%
% The sums are accumulated in chunks, so the K x L generator matrix is
% never formed.

    if nargin < 4 || isempty(L)
        L = 1e5;
    end
    L_half = floor(L / 2);
    chunk  = 1e4;

    dim = size(x_ctrs, 2);
    n   = size(x_ctrs, 1);
    X   = haltonseq(L, dim);

    s1      = zeros(n, 1);   % sum of K(x, x_i)
    S2      = zeros(n, n);   % sum of K(x, x_i) K(x, x_j)
    s1_half = [];
    S2_half = [];

    for i0 = 1:chunk:L
        i1 = min(i0 + chunk - 1, L);
        % split the chunk at L/2, so that the half sums are exact
        if i0 <= L_half && i1 > L_half
            blocks = [i0, L_half; L_half + 1, i1];
        else
            blocks = [i0, i1];
        end
        for b = 1:size(blocks, 1)
            A  = ker(X(blocks(b,1):blocks(b,2), :), x_ctrs);
            s1 = s1 + sum(A, 1)';
            S2 = S2 + A' * A;
            if blocks(b,2) == L_half
                s1_half = s1;
                S2_half = S2;
            end
        end
    end

    G = S2 / L;
    y      = assemble(s1 / L,           G,                reduced_param, n);
    y_half = assemble(s1_half / L_half, S2_half / L_half, reduced_param, n);

    err_qmc = norm(y - y_half, inf) / norm(y, inf);
end

function y = assemble(m1, G, reduced_param, n)
    if reduced_param == 1
        y = [1; m1];
    else
        [I, J] = prod_index(n);
        y = [1; m1; G(sub2ind([n, n], I, J))'];
    end
end
