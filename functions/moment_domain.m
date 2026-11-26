function [y, err_qmc, G] = moment_domain(Bfun, Xq, meas)
% Moments for REDUCED exactness on a general domain, for any basis.
%
% Generators: [1; B_1; ...; B_n], where B(X) = Bfun(X) is N x n
% (kernel translates or polynomials). Integrals over Omega are computed by
% QMC on the points Xq (L x d, a Halton sequence in Omega, see
% halton_domain.m):  int_Omega g  ~  meas * mean(g(Xq)).
%
% Outputs
%   y       : (n+1) x 1 moments [meas; int B_1; ...; int B_n]
%   err_qmc : ||y_L - y_{L/2}||_inf / ||y_L||_inf  (QMC error estimate)
%   G       : n x n Gram matrix int_Omega B_i B_j  (for the MZ constant)

    L      = size(Xq, 1);
    L_half = floor(L / 2);
    chunk  = 1e4;

    s1 = []; S2 = []; s1_half = []; S2_half = [];
    for i0 = 1:chunk:L
        i1 = min(i0 + chunk - 1, L);
        if i0 <= L_half && i1 > L_half
            blocks = [i0, L_half; L_half + 1, i1];
        else
            blocks = [i0, i1];
        end
        for b = 1:size(blocks, 1)
            B = Bfun(Xq(blocks(b,1):blocks(b,2), :));
            if isempty(s1)
                s1 = zeros(size(B,2), 1);  S2 = zeros(size(B,2));
            end
            s1 = s1 + sum(B, 1)';
            S2 = S2 + B' * B;
            if blocks(b,2) == L_half
                s1_half = s1;  S2_half = S2;
            end
        end
    end

    y       = [meas; meas * s1 / L];
    y_half  = [meas; meas * s1_half / L_half];
    G       = meas * S2 / L;
    err_qmc = norm(y - y_half, inf) / norm(y, inf);
end
