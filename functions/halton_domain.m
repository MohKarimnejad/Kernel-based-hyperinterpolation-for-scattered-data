function H = halton_domain(D, N_box)
% Nested Halton sequence in the domain D: the 2D Halton sequence on the
% bounding box D.box, keeping only the points inside D (order preserved).
% The first N points H(1:N,:) are therefore nested in N.
%
% Inputs:  D (see domain_Lshape.m), N_box = number of Halton points in the box
% Output:  H (N_in x 2)
    B = haltonseq(N_box, 2);
    B = D.box(:,1)' + B .* (D.box(:,2) - D.box(:,1))';
    H = B(D.inside(B), :);
end
