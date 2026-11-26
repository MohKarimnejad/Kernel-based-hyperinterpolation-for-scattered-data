function D = domain_Lshape()
% L-shaped domain  Omega = [0,1]^2 \ (1/2,1]^2,  |Omega| = 3/4.
% A non-tensor domain, on which no standard (Gauss-type) cubature is used:
% both the kernel and the polynomial rules are built with Algorithm 2.
%
% Output struct
%   D.name    : 'Lshape'
%   D.inside  : X (N x 2) -> logical N x 1
%   D.meas    : |Omega|
%   D.box     : bounding box [xmin xmax; ymin ymax]
    D.name   = 'Lshape';
    D.inside = @(X) all(X >= 0 & X <= 1, 2) & ~(X(:,1) > 0.5 & X(:,2) > 0.5);
    D.meas   = 0.75;
    D.box    = [0 1; 0 1];
end
