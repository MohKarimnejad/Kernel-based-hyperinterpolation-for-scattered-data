clear; clc; close all
addpath('functions')

%% =========================================================================
%  TEST 5 — THE OVERSAMPLED REGIME  M >> n
%
%  In Tests 1-4 the number of cubature nodes M is whatever getCF returns:
%  the smallest M for which the exactness system has full rank and positive
%  weights.  M is therefore tied to n and the two cannot be varied
%  independently.  This test decouples them.
%
%  PART A : n fixed, M swept from n+1 up to 100n.
%  PART B : M/n ratio fixed, n swept over the Test-1 grid.
%
%  Both parts are run on clean and on noisy samples, and are compared
%  against (i) kernel interpolation at X and (ii) the exact L2 projection
%  onto N_K(X), computed with a high-order Gauss-Legendre rule.  The latter
%  is the M -> infinity limit of the hyperinterpolant and is the floor that
%  the RMSE curve of Part A must approach.
%
%  Cubature: reduced exactness on (N_K(X) + 1) x 1, so that the number of
%  constraints is n+1 and any M >= n+1 is admissible.  Weights come from
%  BG_solver (pivoted QR, no regularization, no clipping of negative
%  weights; a warning is issued if a weight is nonpositive); moments are
%  taken in closed form on [0,1] via moment_closed_1d.
% =========================================================================

%% -------------------------------------------------------------------------
%  PARAMETERS
% -------------------------------------------------------------------------
dim               = 1;
data_distribution = "Halton";
solver            = "SVD";

rbf_names = ["LMatern", "QMatern"];
tau_vec   = [2, 3];              % d = 1

C_ep   = 500;
ep_min = 3;
ep_max = 50;

sigma    = 1e-2;                 % noise level
n_trials = 30;                   % Monte-Carlo repetitions for the noisy runs
rng(0);

f = @(x) sin(2*pi*x) + 0.5*sin(4*pi*x);

n_test = 2000;
X_test = get_ctrs(n_test, "Uniform", dim);
f_test = f(X_test);

n_leb = 1000;
X_leb = linspace(0, 1, n_leb)';

nk = numel(rbf_names);

%% =========================================================================
%  PART A  —  n FIXED, M SWEPT
% =========================================================================
nA      = 40;
ratioA  = [1, 2, 5, 10, 25, 50, 100];

x_A  = get_ctrs(nA, data_distribution, dim);
h_A  = fill_distance(x_A, 5000);
ep_A = min(max(ep_min, C_ep * h_A), ep_max);
fprintf('PART A:  n = %d  |  h_X = %.4f  |  ep = %.3f\n', nA, h_A, ep_A);

nM = numel(ratioA);
M_A         = zeros(nk, nM);
SM_A        = zeros(nk, nM);
wmin_A      = zeros(nk, nM);
rmsA_clean  = zeros(nk, nM);
rmsA_noisy  = zeros(nk, nM);
lebA        = zeros(nk, nM);
condA       = zeros(nk, nM);

rmsA_interp_clean = zeros(nk, 1);
rmsA_interp_noisy = zeros(nk, 1);
lebA_interp       = zeros(nk, 1);
rmsA_proj         = zeros(nk, 1);

for k = 1:nk

    rbf_name = rbf_names(k);
    rbf      = select_rbf(rbf_name);
    ker      = @(x, y) rbf(pdist2(x, y), ep_A);

    K_XX   = ker(x_A, x_A);
    K_test = ker(X_test, x_A);
    K_leb  = ker(X_leb,  x_A);
    b_A    = moment_closed_1d(rbf_name, x_A, ep_A);   % (n+1) x 1

    % ---- (i) interpolation at X --------------------------------------
    a_i = K_XX \ f(x_A);
    rmsA_interp_clean(k) = rmse(K_test * a_i, f_test);

    acc = zeros(n_trials, 1);
    for t = 1:n_trials
        a_t    = K_XX \ (f(x_A) + sigma * randn(nA, 1));
        acc(t) = rmse(K_test * a_t, f_test);
    end
    rmsA_interp_noisy(k) = mean(acc);
    lebA_interp(k)       = max(sum(abs((K_XX \ K_leb')'), 2));

    % ---- (ii) exact L2 projection onto N_K(X)  (M -> infinity) -------
    [xq, wq] = lgwt(4000, 0, 1);
    Vq  = ker(xq, x_A);
    a_p = (Vq' * (wq .* Vq)) \ (Vq' * (wq .* f(xq)));
    rmsA_proj(k) = rmse(K_test * a_p, f_test);

    fprintf('\n=== %s ===  interp = %.3e | interp+noise = %.3e | L2proj = %.3e | leb_interp = %.2f\n', ...
        rbf_name, rmsA_interp_clean(k), rmsA_interp_noisy(k), rmsA_proj(k), lebA_interp(k));
    fprintf('%8s %10s %10s %12s %12s %9s %11s\n', ...
        'M', 'S_M', 'min(w)', 'RMSE clean', 'RMSE noisy', 'Lambda', 'cond');

    % ---- (iii) hyperinterpolation, M swept ---------------------------
    for j = 1:nM

        M = max(nA + 1, ratioA(j) * nA);
        Z = haltonseq(M, dim);

        Phi = prod_basis(Z, x_A, ker, 1);         % (n+1) x M, reduced
        [w, res_w] = BG_solver(Phi, 1, b_A);   % no weight is modified
        if min(w) <= 0
            warning('Test5: nonpositive weight, min(w) = %.2e, M = %d, residual = %.2e', min(w), M, res_w);
        end

        W  = diag(w);
        VZ = ker(Z, x_A);
        yZ = f(Z);

        a = LS_stable_solver(W, VZ, yZ, solver);

        M_A(k,j)        = M;
        SM_A(k,j)       = sum(w);
        wmin_A(k,j)     = min(w);
        rmsA_clean(k,j) = rmse(K_test * a, f_test);
        lebA(k,j)       = leb_svd(K_leb, VZ, W, 1e-12);
        condA(k,j)      = cond(sqrt(W) * VZ);

        acc = zeros(n_trials, 1);
        for t = 1:n_trials
            a_t    = LS_stable_solver(W, VZ, yZ + sigma * randn(M, 1), solver);
            acc(t) = rmse(K_test * a_t, f_test);
        end
        rmsA_noisy(k,j) = mean(acc);

        fprintf('%8d %10.4f %10.2e %12.3e %12.3e %9.2f %11.2e\n', ...
            M, SM_A(k,j), wmin_A(k,j), rmsA_clean(k,j), rmsA_noisy(k,j), ...
            lebA(k,j), condA(k,j));
    end
end

%% =========================================================================
%  PART B  —  M/n FIXED, n SWEPT
% =========================================================================
ratioB    = 100;
n_ctrsVec = [20, 30, 45, 60, 80, 105, 140];
nn        = numel(n_ctrsVec);

rmsB_interp_clean = zeros(nk, nn);
rmsB_interp_noisy = zeros(nk, nn);
lebB_interp       = zeros(nk, nn);
rmsB_hyper_clean  = zeros(nk, nn);
rmsB_hyper_noisy  = zeros(nk, nn);
lebB_hyper        = zeros(nk, nn);
ep_usedB          = zeros(nk, nn);

fprintf('\n\nPART B:  M = %d n\n', ratioB);

for k = 1:nk

    rbf_name = rbf_names(k);
    rbf      = select_rbf(rbf_name);

    fprintf('\n=== %s ===\n', rbf_name);
    fprintf('%6s %8s %10s %10s %12s %12s %12s %12s\n', ...
        'n', 'ep', 'Lam int', 'Lam hyp', 'RMSE int', 'RMSE hyp', 'int+noise', 'hyp+noise');

    for idx = 1:nn

        n_ctrs = n_ctrsVec(idx);
        x_ctrs = get_ctrs(n_ctrs, data_distribution, dim);
        h_X    = fill_distance(x_ctrs, 1000);
        ep     = min(max(ep_min, C_ep * h_X), ep_max);
        ep_usedB(k,idx) = ep;

        ker    = @(x, y) rbf(pdist2(x, y), ep);
        K_XX   = ker(x_ctrs, x_ctrs);
        K_test = ker(X_test, x_ctrs);
        K_leb  = ker(X_leb,  x_ctrs);
        b      = moment_closed_1d(rbf_name, x_ctrs, ep);

        % interpolation
        a_i = K_XX \ f(x_ctrs);
        rmsB_interp_clean(k,idx) = rmse(K_test * a_i, f_test);
        acc = zeros(n_trials, 1);
        for t = 1:n_trials
            a_t    = K_XX \ (f(x_ctrs) + sigma * randn(n_ctrs, 1));
            acc(t) = rmse(K_test * a_t, f_test);
        end
        rmsB_interp_noisy(k,idx) = mean(acc);
        lebB_interp(k,idx)       = max(sum(abs((K_XX \ K_leb')'), 2));

        % hyperinterpolation with M = ratioB * n
        M   = ratioB * n_ctrs;
        Z   = haltonseq(M, dim);
        Phi = prod_basis(Z, x_ctrs, ker, 1);
        [w, res_w] = BG_solver(Phi, 1, b);   % no weight is modified
        if min(w) <= 0
            warning('Test5: nonpositive weight, min(w) = %.2e, M = %d, residual = %.2e', min(w), M, res_w);
        end

        W  = diag(w);
        VZ = ker(Z, x_ctrs);
        yZ = f(Z);

        a = LS_stable_solver(W, VZ, yZ, solver);
        rmsB_hyper_clean(k,idx) = rmse(K_test * a, f_test);
        lebB_hyper(k,idx)       = leb_svd(K_leb, VZ, W, 1e-12);

        acc = zeros(n_trials, 1);
        for t = 1:n_trials
            a_t    = LS_stable_solver(W, VZ, yZ + sigma * randn(M, 1), solver);
            acc(t) = rmse(K_test * a_t, f_test);
        end
        rmsB_hyper_noisy(k,idx) = mean(acc);

        fprintf('%6d %8.2f %10.2f %10.2f %12.3e %12.3e %12.3e %12.3e\n', ...
            n_ctrs, ep, lebB_interp(k,idx), lebB_hyper(k,idx), ...
            rmsB_interp_clean(k,idx), rmsB_hyper_clean(k,idx), ...
            rmsB_interp_noisy(k,idx), rmsB_hyper_noisy(k,idx));
    end
end

%% =========================================================================
%  PLOTTING
% =========================================================================
fig_id = 1;

col_interp = [0.47 0.67 0.19];
col_hyper  = [0.85 0.33 0.10];
col_noisy  = [0.93 0.69 0.13];
col_ref    = [0.00 0.00 0.00];

    function pub_axes(ax)
        set(ax, 'FontSize',13,'FontWeight','bold','LineWidth',1.5, ...
            'TickLength',[0.015 0.025],'TickDir','out','Box','on', ...
            'Color','w','XColor','k','YColor','k', ...
            'GridAlpha',0.25,'MinorGridAlpha',0.12);
        set(get(ax,'Parent'),'Color','w');
    end
    function pub_legend(lg)
        set(lg,'Color','w','EdgeColor','k','TextColor','k','FontSize',9.5, ...
            'FontWeight','normal','LineWidth',1,'Interpreter','tex');
    end

% ---- BLOCK I : RMSE vs M (Part A) ---------------------------------------
for k = 1:nk
    figure(fig_id); clf; set(gcf,'Color','w'); hold on;
    plot(M_A(k,:), rmsA_clean(k,:), '--s','Color',col_hyper,'LineWidth',1.8, ...
        'MarkerSize',6,'MarkerFaceColor',col_hyper);
    plot(M_A(k,:), rmsA_noisy(k,:), '-.d','Color',col_noisy,'LineWidth',1.8, ...
        'MarkerSize',6,'MarkerFaceColor',col_noisy);
    yline(rmsA_interp_clean(k), '-',  'Color',col_interp,'LineWidth',1.8);
    yline(rmsA_interp_noisy(k), ':',  'Color',col_interp,'LineWidth',1.8);
    yline(rmsA_proj(k),         '-',  'Color',col_ref,   'LineWidth',1.2);
    set(gca,'XScale','log','YScale','log'); pub_axes(gca);
    xlabel('M (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    ylabel('RMSE (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    lg = legend('Hyper (clean)','Hyper (noisy)','Interpolation (clean)', ...
        'Interpolation (noisy)','L_2 projection onto N_K(X)','Location','northeast');
    pub_legend(lg);
    title(sprintf('f_1 - RMSE vs M - %s kernel - n = %d', rbf_names(k), nA), ...
        'FontSize',12,'FontWeight','bold','Color','k');
    savefig_png(fig_id); fig_id = fig_id + 1;
end

% ---- BLOCK II : Lebesgue constant vs M (Part A) -------------------------
for k = 1:nk
    figure(fig_id); clf; set(gcf,'Color','w'); hold on;
    plot(M_A(k,:), lebA(k,:), '--s','Color',col_hyper,'LineWidth',1.8, ...
        'MarkerSize',6,'MarkerFaceColor',col_hyper);
    yline(lebA_interp(k), '-', 'Color',col_interp,'LineWidth',1.8);
    set(gca,'XScale','log','YScale','log'); pub_axes(gca);
    xlabel('M (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    ylabel('Lebesgue constant (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    lg = legend('Hyperinterpolation','Interpolation','Location','northeast');
    pub_legend(lg);
    title(sprintf('f_1 - Lebesgue Constant vs M - %s kernel - n = %d', rbf_names(k), nA), ...
        'FontSize',12,'FontWeight','bold','Color','k');
    savefig_png(fig_id); fig_id = fig_id + 1;
end

% ---- BLOCK III : Lebesgue constant vs n at M = 20n (Part B) -------------
for k = 1:nk
    figure(fig_id); clf; set(gcf,'Color','w'); hold on;
    plot(n_ctrsVec, lebB_interp(k,:), '-^','Color',col_interp,'LineWidth',1.8, ...
        'MarkerSize',6,'MarkerFaceColor',col_interp);
    plot(n_ctrsVec, lebB_hyper(k,:), '--s','Color',col_hyper,'LineWidth',1.8, ...
        'MarkerSize',6,'MarkerFaceColor',col_hyper);
    ref = lebB_interp(k,1) * (n_ctrsVec / n_ctrsVec(1)).^0.5;
    plot(n_ctrsVec, ref, '-','Color',col_ref,'LineWidth',1.2);

    log_n = log(n_ctrsVec(2:end)');
    s_i = fit_slope(log_n, log(lebB_interp(k,2:end)'));
    s_h = fit_slope(log_n, log(lebB_hyper(k,2:end)'));

    set(gca,'XScale','log','YScale','log'); pub_axes(gca);
    xlabel('n (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    ylabel('Lebesgue constant (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    lg = legend(sprintf('Interpolation  O(n^{%.2f})', s_i), ...
                sprintf('Hyper, M = %dn  O(n^{%.2f})', ratioB, s_h), ...
                'Ref  O(n^{0.50})','Location','northwest');
    pub_legend(lg);
    title(sprintf('f_1 - Lebesgue Constant vs n - %s kernel - M = %dn', ...
        rbf_names(k), ratioB),'FontSize',12,'FontWeight','bold','Color','k');
    savefig_png(fig_id); fig_id = fig_id + 1;
end

% ---- BLOCK IV : RMSE vs n, clean and noisy, at M = 20n (Part B) ---------
for k = 1:nk
    tau = tau_vec(k);
    figure(fig_id); clf; set(gcf,'Color','w'); hold on;
    plot(n_ctrsVec, rmsB_interp_clean(k,:), '-^','Color',col_interp,'LineWidth',1.8, ...
        'MarkerSize',6,'MarkerFaceColor',col_interp);
    plot(n_ctrsVec, rmsB_hyper_clean(k,:), '--s','Color',col_hyper,'LineWidth',1.8, ...
        'MarkerSize',6,'MarkerFaceColor',col_hyper);
    plot(n_ctrsVec, rmsB_interp_noisy(k,:), '-v','Color',col_interp,'LineWidth',1.2, ...
        'MarkerSize',5,'LineStyle',':');
    plot(n_ctrsVec, rmsB_hyper_noisy(k,:), '-.d','Color',col_noisy,'LineWidth',1.8, ...
        'MarkerSize',6,'MarkerFaceColor',col_noisy);

    slope_theory = -(tau - 0.5);
    n_ref = logspace(log10(n_ctrsVec(2)), log10(n_ctrsVec(end)), 60);
    plot(n_ref, rmsB_hyper_clean(k,2) * (n_ref / n_ctrsVec(2)).^slope_theory, ...
        '-','Color',col_ref,'LineWidth',1.2);

    set(gca,'XScale','log','YScale','log'); pub_axes(gca);
    xlabel('n (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    ylabel('RMSE (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    lg = legend('Interpolation (clean)','Hyper (clean)', ...
                'Interpolation (noisy)','Hyper (noisy)', ...
                sprintf('Ref  O(n^{%.2f})', slope_theory),'Location','southwest');
    pub_legend(lg);
    title(sprintf('f_1 - RMSE vs n - %s kernel - M = %dn - \\sigma = %.0e', ...
        rbf_names(k), ratioB, sigma),'FontSize',12,'FontWeight','bold','Color','k');
    savefig_png(fig_id); fig_id = fig_id + 1;
end

fprintf('\nDone. %d figures saved.\n', fig_id - 1);
