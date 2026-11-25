function opts = cf_opts(err_qmc)
% Options for getCF.m used in all tests.
%
% The moments b are computed by QMC (moment.m), so the cubature residual
% cannot be required below the QMC error. The exactness tolerance is set to
% 10 times the estimated QMC error, and never below 1e-10.

    opts = struct();
    opts.tol   = max(1e-10, 10 * err_qmc);   % exactness tolerance
    opts.delta = 1e-10;                      % rank tolerance (pivoted QR)
    opts.dN    = 1;                          % nodes added per iteration
    opts.N_max = 1e5;                        % maximum number of nodes
end
