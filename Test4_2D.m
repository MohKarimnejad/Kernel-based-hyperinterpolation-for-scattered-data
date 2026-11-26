clear; clc; close all

%% =========================================================================
%  TEST 4 — FRANKE FUNCTION ON AN L-SHAPED DOMAIN  (dim = 2)
%
%  Omega = [0,1]^2 \ (1/2,1]^2  (non-tensor domain, |Omega| = 3/4).
%  No standard cubature is used: the kernel AND the polynomial methods build
%  their positive cubature rule with the same Algorithm 2 (Backus-Gilbert on
%  a nested Halton sequence in Omega, reduced exactness on (V + 1) x 1).
%  The comparison between kernel and polynomial spaces is therefore made
%  with the same cubature construction and the same type of nodes.
%
%  Methods:
%     (A) Kernel interpolation at X                 (n function values)
%     (B) Kernel least squares, unweighted, at Z    (M function values)
%     (C) Kernel hyperinterpolation, reduced        (M function values)
%     (D) Polynomial hyperinterpolation, reduced    (M_p function values),
%         tensor Legendre basis of total degree q, n_p = (q+1)(q+2)/2
%
%  Moments are computed by QMC on the first L Halton points in Omega.
%  Coefficients and Lebesgue constants: economy QR (Algorithm 1).
%  Fixed shape parameter ep for all n and all kernels.
%  Output: figures, CSV tables and .mat file in test4_results/.
% =========================================================================
addpath('functions')

%% -------------------------------------------------------------------------
%  PARAMETERS
% -------------------------------------------------------------------------
dim  = 2;
Dom  = domain_Lshape();

rbf_names = ["LMatern", "QMatern"];
tau_vec   = [5/2, 7/2];          % d = 2: tau = nu + d/2

ep = 4;                          % fixed shape parameter (Reviewer 2, point 9)

n_ctrsVec = [80, 105, 140, 180, 240, 300, 380, 460, 560, 680, 800, 1000, 1200];
q_vec     = [11, 13, 16, 19, 22, 26, 30, 34, 38, 43, 47];   % n_p = 78 ... 1176
n_poly    = (q_vec + 1) .* (q_vec + 2) / 2;

L_qmc = 1e5;                     % QMC points for the moments
N_box = 4e5;                     % Halton points in the bounding box

fig_tag = 'test4';
out_dir = 'test4_results';       % figures, CSV tables and .mat file
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% -------------------------------------------------------------------------
%  Nested Halton sequence in Omega, test and Lebesgue grids
% -------------------------------------------------------------------------
H     = halton_domain(Dom, N_box);      % nodes, centres and QMC points
Xq    = H(1:L_qmc, :);
nodes = @(N) H(1:N, :);

f = @(x) franke(x(:,1), x(:,2));

[G1, G2] = meshgrid(linspace(0, 1, 200));
X_test   = [G1(:), G2(:)];  X_test = X_test(Dom.inside(X_test), :);
f_test   = f(X_test);

[G1, G2] = meshgrid(linspace(0, 1, 60));
X_leb    = [G1(:), G2(:)];  X_leb = X_leb(Dom.inside(X_leb), :);

%% -------------------------------------------------------------------------
%  STORAGE
% -------------------------------------------------------------------------
nk = numel(rbf_names);
nn = numel(n_ctrsVec);
nq = numel(q_vec);

rms_interp = zeros(nk, nn);  rms_uls = zeros(nk, nn);  rms_red = zeros(nk, nn);
cond_interp = zeros(nk, nn); cond_uls = zeros(nk, nn); cond_red = zeros(nk, nn);
leb_interp = zeros(nk, nn);  leb_uls = zeros(nk, nn);  leb_red = zeros(nk, nn);
M_red = zeros(nk, nn);
diag_red = cell(nk, nn);     % cubature diagnostics, kernel (Reviewer 2, point 7)

rms_poly = zeros(1, nq); cond_poly = zeros(1, nq); leb_poly = zeros(1, nq);
M_poly   = zeros(1, nq);
diag_poly = cell(1, nq);     % cubature diagnostics, polynomial

%% =========================================================================
%  (D) POLYNOMIAL HYPERINTERPOLATION, REDUCED EXACTNESS
% =========================================================================
fprintf('=== Polynomial hyperinterpolation on %s ===\n', Dom.name);
for qi = 1:nq
    q    = q_vec(qi);
    Pfun = @(X) legendre_basis_2d(X, q, Dom.box);

    % moments: the basis contains the constant, so drop the leading |Omega|
    [y, err_qmc, G] = moment_domain(Pfun, Xq, Dom.meas);
    y = y(2:end);

    opts       = cf_opts(err_qmc);
    opts.nodes = nodes;
    opts.gen   = @(X) Pfun(X)';
    opts.N_max = size(H, 1);

    [~, Z, w, ~, ~, info] = getCF([], 2 * n_poly(qi), [], [], Dom.meas, y, 1, dim, opts);

    VZ = Pfun(Z);  W = diag(w);
    a  = LS_stable_solver(W, VZ, f(Z));

    rms_poly(qi)  = rmse(Pfun(X_test) * a, f_test);
    cond_poly(qi) = cond(sqrt(W) * VZ);
    leb_poly(qi)  = leb_const(Pfun(X_leb), VZ, W);
    M_poly(qi)    = numel(w);
    diag_poly{1, qi} = cf_diagnostics(info, VZ, w, G, Pfun(X_test));

    fprintf('  q = %2d  n_p = %4d  M = %5d  |  RMSE = %.3e  Leb = %.2f\n', ...
        q, n_poly(qi), M_poly(qi), rms_poly(qi), leb_poly(qi));
end

%% =========================================================================
%  KERNEL METHODS
% =========================================================================
for k = 1:nk
    rbf_name = rbf_names(k);
    rbf      = select_rbf(rbf_name);
    ker      = @(x, y) rbf(pdist2(x, y), ep);

    fprintf('\n=== Kernel: %s | ep = %.2f | tau = %.2f ===\n', rbf_name, ep, tau_vec(k));

    for idx = 1:nn
        n_ctrs = n_ctrsVec(idx);
        x_ctrs = H(1:n_ctrs, :);                 % Halton centres in Omega
        Kfun   = @(X) ker(X, x_ctrs);

        K_test = Kfun(X_test);
        K_leb  = Kfun(X_leb);

        % ---- reduced-exactness cubature (shared by (B) and (C)) ---------
        [y, err_qmc, G] = moment_domain(Kfun, Xq, Dom.meas);

        opts       = cf_opts(err_qmc);
        opts.nodes = nodes;
        opts.gen   = @(X) [ones(size(X,1), 1), Kfun(X)]';
        opts.N_max = size(H, 1);

        [~, Z, w, ~, ~, info] = getCF([], 2 * (n_ctrs + 1), x_ctrs, ker, ...
            Dom.meas, y, 1, dim, opts);

        VZ = Kfun(Z);  W = diag(w);  yZ = f(Z);
        M_red(k, idx)    = numel(w);
        diag_red{k, idx} = cf_diagnostics(info, VZ, w, G, K_test);

        % ---- (A) interpolation at X -------------------------------------
        K_XX = ker(x_ctrs, x_ctrs);
        [L_X, flag_X] = chol(K_XX, 'lower');
        if flag_X ~= 0
            warning('K(X,X) not SPD at n = %d', n_ctrs);
            a_i = K_XX \ f(x_ctrs);   Phi_leb = (K_XX \ K_leb')';
        else
            a_i = L_X' \ (L_X \ f(x_ctrs));   Phi_leb = (L_X' \ (L_X \ K_leb'))';
        end
        rms_interp(k, idx)  = rmse(K_test * a_i, f_test);
        cond_interp(k, idx) = cond(K_XX);
        leb_interp(k, idx)  = max(sum(abs(Phi_leb), 2));

        % ---- (B) unweighted least squares at Z --------------------------
        W_u = eye(numel(w));
        a_u = LS_stable_solver(W_u, VZ, yZ);
        rms_uls(k, idx)  = rmse(K_test * a_u, f_test);
        cond_uls(k, idx) = cond(VZ);
        leb_uls(k, idx)  = leb_const(K_leb, VZ, W_u);

        % ---- (C) hyperinterpolation, reduced exactness ------------------
        a_r = LS_stable_solver(W, VZ, yZ);
        rms_red(k, idx)  = rmse(K_test * a_r, f_test);
        cond_red(k, idx) = cond(sqrt(W) * VZ);
        leb_red(k, idx)  = leb_const(K_leb, VZ, W);

        fprintf('  n = %4d  M = %5d  |  RMSE: int %.2e  LS %.2e  hyp %.2e  |  Leb hyp %.2f\n', ...
            n_ctrs, M_red(k, idx), rms_interp(k, idx), rms_uls(k, idx), ...
            rms_red(k, idx), leb_red(k, idx));
    end
end

%% =========================================================================
%  PLOTTING
% =========================================================================
fig_id = 1;
C.interp = [0.47 0.67 0.19];  C.uls = [0.00 0.45 0.74];
C.red    = [0.93 0.69 0.13];  C.poly = [0.49 0.18 0.56];  C.ref = [0 0 0];

for k = 1:nk
    ttl = sprintf('%s kernel - %s domain', rbf_names(k), Dom.name);

    % ---- BLOCK I: condition number vs dimension --------------------------
    figure(fig_id); clf; set(gcf, 'Color', 'w'); hold on;
    plot_series(n_ctrsVec, cond_interp(k,:), '-^',  C.interp);
    plot_series(n_ctrsVec, cond_uls(k,:),    '-o',  C.uls);
    plot_series(n_ctrsVec, cond_red(k,:),    '-.d', C.red);
    plot_series(n_poly,    cond_poly,        '--s', C.poly);
    set(gca, 'XScale', 'log', 'YScale', 'log'); pub_axes(gca);
    xlabel('dimension n (kernel) / n_p (polynomial)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Condition number', 'FontSize', 13, 'FontWeight', 'bold');
    pub_legend(legend('Kernel interpolation', 'Kernel LS', 'Kernel hyper reduced', ...
        'Poly hyper reduced', 'Location', 'northwest'));
    title(['f_4 - Condition number - ' ttl], 'FontSize', 12, 'FontWeight', 'bold');
    savefig_png(fig_id, fig_tag, out_dir); fig_id = fig_id + 1;

    % ---- BLOCK II: RMSE vs dimension -------------------------------------
    slope_theory = -(tau_vec(k) - dim/2) / dim;   % O(h^{tau-d/2}), h ~ n^{-1/d}
    s_i = fit_slope(log(n_ctrsVec'), log(rms_interp(k,:)'));
    s_u = fit_slope(log(n_ctrsVec'), log(rms_uls(k,:)'));
    s_r = fit_slope(log(n_ctrsVec'), log(rms_red(k,:)'));
    s_p = fit_slope(log(n_poly'),    log(rms_poly'));

    figure(fig_id); clf; set(gcf, 'Color', 'w'); hold on;
    plot_series(n_ctrsVec, rms_interp(k,:), '-^',  C.interp);
    plot_series(n_ctrsVec, rms_uls(k,:),    '-o',  C.uls);
    plot_series(n_ctrsVec, rms_red(k,:),    '-.d', C.red);
    plot_series(n_poly,    rms_poly,        '--s', C.poly);
    plot(n_ctrsVec, rms_red(k,1) * (n_ctrsVec / n_ctrsVec(1)).^slope_theory, ...
        '-', 'Color', C.ref, 'LineWidth', 1.2);
    set(gca, 'XScale', 'log', 'YScale', 'log'); pub_axes(gca);
    xlabel('dimension n (kernel) / n_p (polynomial)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('RMSE', 'FontSize', 13, 'FontWeight', 'bold');
    pub_legend(legend(sprintf('Kernel interpolation  O(n^{%.2f})', s_i), ...
        sprintf('Kernel LS  O(n^{%.2f})', s_u), ...
        sprintf('Kernel hyper reduced  O(n^{%.2f})', s_r), ...
        sprintf('Poly hyper reduced  O(n_p^{%.2f})', s_p), ...
        sprintf('Ref  O(n^{%.2f})', slope_theory), 'Location', 'southwest'));
    title(['f_4 - RMSE vs dimension - ' ttl], 'FontSize', 12, 'FontWeight', 'bold');
    savefig_png(fig_id, fig_tag, out_dir); fig_id = fig_id + 1;

    % ---- BLOCK III: RMSE vs number of function evaluations (R2, point 10)
    figure(fig_id); clf; set(gcf, 'Color', 'w'); hold on;
    plot_series(n_ctrsVec,    rms_interp(k,:), '-^',  C.interp);
    plot_series(M_red(k,:),   rms_uls(k,:),    '-o',  C.uls);
    plot_series(M_red(k,:),   rms_red(k,:),    '-.d', C.red);
    plot_series(M_poly,       rms_poly,        '--s', C.poly);
    set(gca, 'XScale', 'log', 'YScale', 'log'); pub_axes(gca);
    xlabel('number of function evaluations', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('RMSE', 'FontSize', 13, 'FontWeight', 'bold');
    pub_legend(legend('Kernel interpolation (n)', 'Kernel LS (M)', ...
        'Kernel hyper reduced (M)', 'Poly hyper reduced (M_p)', 'Location', 'southwest'));
    title(['f_4 - RMSE vs function evaluations - ' ttl], 'FontSize', 12, 'FontWeight', 'bold');
    savefig_png(fig_id, fig_tag, out_dir); fig_id = fig_id + 1;

    % ---- BLOCK IV: Lebesgue constant vs dimension ------------------------
    l_i = fit_slope(log(n_ctrsVec'), log(leb_interp(k,:)'));
    l_u = fit_slope(log(n_ctrsVec'), log(leb_uls(k,:)'));
    l_r = fit_slope(log(n_ctrsVec'), log(leb_red(k,:)'));
    l_p = fit_slope(log(n_poly'),    log(leb_poly'));

    figure(fig_id); clf; set(gcf, 'Color', 'w'); hold on;
    plot_series(n_ctrsVec, leb_interp(k,:), '-^',  C.interp);
    plot_series(n_ctrsVec, leb_uls(k,:),    '-o',  C.uls);
    plot_series(n_ctrsVec, leb_red(k,:),    '-.d', C.red);
    plot_series(n_poly,    leb_poly,        '--s', C.poly);
    plot(n_ctrsVec, leb_interp(k,1) * (n_ctrsVec / n_ctrsVec(1)).^0.5, ...
        '-', 'Color', C.ref, 'LineWidth', 1.2);
    set(gca, 'XScale', 'log', 'YScale', 'log'); pub_axes(gca);
    xlabel('dimension n (kernel) / n_p (polynomial)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Lebesgue constant', 'FontSize', 13, 'FontWeight', 'bold');
    pub_legend(legend(sprintf('Kernel interpolation  O(n^{%+.2f})', l_i), ...
        sprintf('Kernel LS  O(n^{%+.2f})', l_u), ...
        sprintf('Kernel hyper reduced  O(n^{%+.2f})', l_r), ...
        sprintf('Poly hyper reduced  O(n_p^{%+.2f})', l_p), ...
        'Ref  O(n^{+0.50})', 'Location', 'northwest'));
    title(['f_4 - Lebesgue constant - ' ttl], 'FontSize', 12, 'FontWeight', 'bold');
    savefig_png(fig_id, fig_tag, out_dir); fig_id = fig_id + 1;

    fprintf('\n[Slopes]  %s | ep = %.2f   (RMSE vs dimension | Lebesgue)\n', rbf_names(k), ep);
    fprintf('  %-28s  %+.4f  |  %+.4f\n', 'Kernel interpolation:', s_i, l_i);
    fprintf('  %-28s  %+.4f  |  %+.4f\n', 'Kernel LS:',            s_u, l_u);
    fprintf('  %-28s  %+.4f  |  %+.4f   (theory RMSE: %+.4f)\n', ...
        'Kernel hyper reduced:', s_r, l_r, slope_theory);
    fprintf('  %-28s  %+.4f  |  %+.4f\n', 'Poly hyper reduced:',   s_p, l_p);
end

fprintf('\nDone. %d figures saved.\n', fig_id - 1);
print_cf_diagnostics(diag_red, rbf_names, n_ctrsVec, ...
    sprintf('kernel, reduced exactness, ep = %g', ep), ...
    fullfile(out_dir, 'cubature_diagnostics_kernel_red.csv'));
print_cf_diagnostics(diag_poly, "Poly", n_poly, ...
    'polynomial, reduced exactness (n = n_p)', ...
    fullfile(out_dir, 'cubature_diagnostics_poly_red.csv'));
save(fullfile(out_dir, 'diagnostics_test4.mat'), 'diag_red', 'diag_poly', ...
    'n_ctrsVec', 'q_vec', 'n_poly', 'rbf_names', 'ep');

%% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================
function plot_series(x, y, style, col)
    plot(x, y, style, 'Color', col, 'LineWidth', 1.8, 'MarkerSize', 6, ...
        'MarkerFaceColor', col);
end

function pub_axes(ax)
    set(ax, 'FontSize', 13, 'FontWeight', 'bold', 'LineWidth', 1.5, ...
        'TickLength', [0.015 0.025], 'TickDir', 'out', 'Box', 'on', ...
        'Color', 'w', 'XColor', 'k', 'YColor', 'k');
end

function pub_legend(lg)
    set(lg, 'Color', 'w', 'EdgeColor', 'k', 'TextColor', 'k', 'FontSize', 9.5, ...
        'FontWeight', 'normal', 'LineWidth', 1, 'Interpreter', 'tex');
end

function v = franke(x, y)
% Standard Franke function on [0,1]^2 (vectorized).
    v =  0.75 * exp( -((9*x - 2).^2 + (9*y - 2).^2) / 4 ) ...
       + 0.75 * exp( -((9*x + 1).^2) / 49 - (9*y + 1) / 10 ) ...
       + 0.50 * exp( -((9*x - 7).^2 + (9*y - 3).^2) / 4 ) ...
       - 0.20 * exp( -( 9*x - 4).^2 - (9*y - 7).^2 );
end
