clear; clc; close all

%% =========================================================================
%  TEST 4 — FRANKE FUNCTION  (dim = 2)   [REDUCED EXACTNESS ONLY]
%
%  Rebuilt from the former 2D Test (test2.m).  Full-exactness cubature has
%  been removed entirely:
%
%  Methods compared:
%     (A) Kernel interpolation at X            (baseline, n x n solve)
%     (B) Kernel least squares                 (sampling f at the cubature
%                                               nodes Z_red, W = I)
%     (C) Kernel hyperinterpolation, REDUCED   (cubature nodes Z_red, W_red)
%
%  All Lebesgue constants for the projection methods (LS, Hyper Reduced) are
%  computed through the same truncated-SVD pseudoinverse used to form the
%  coefficients (leb_svd.m), avoiding the kappa^2 amplification.
% =========================================================================

%% -------------------------------------------------------------------------
%  PARAMETERS
% -------------------------------------------------------------------------
dim               = 2;
data_distribution = "Halton";
solver            = "QR";

% Kernels and Sobolev smoothness orders for d=2:  tau = nu + d/2
%   LMatern (nu=3/2): tau = 3/2 + 1 = 5/2
%   QMatern (nu=5/2): tau = 5/2 + 1 = 7/2
rbf_names = ["LMatern", "QMatern"];
tau_vec   = [5/2, 7/2];


% n is now allowed to grow well past the old ceiling of 140, since the
% reduced cubature needs only O(n) nodes.
n_ctrsVec = [80, 105, 140, 180, 240, 300, 380, 460, 560, 680, ...
    800, 1000, 1200];

% Shape parameter: fixed, the same for all n and all kernels
% (setting of Corollary 2; Reviewer 2, point 9)
ep = 4;
fig_tag = 'test4';

%% ------------------------------------------------------------------------
%  Test function: Franke function on [0,1]^2  (local subfunction below)
% ------------------------------------------------------------------------
f = @(x) franke(x(:,1), x(:,2));

n_test = 300;
X_test = get_ctrs(n_test, "Uniform", dim);   % dense uniform error grid
f_test = f(X_test);

n_leb_1d = 50;
t_leb    = linspace(0, 1, n_leb_1d);
[G1, G2] = meshgrid(t_leb, t_leb);
X_leb    = [G1(:), G2(:)];
n_leb    = size(X_leb, 1);

%% -------------------------------------------------------------------------
%  STORAGE  —  (kernel x n)
% -------------------------------------------------------------------------
nk = length(rbf_names);
nn = length(n_ctrsVec);

rms_interp    = zeros(nk, nn);
rms_uls       = zeros(nk, nn);
rms_hyper_red = zeros(nk, nn);

cond_interp    = zeros(nk, nn);
cond_uls       = zeros(nk, nn);
cond_hyper_red = zeros(nk, nn);

leb_interp    = zeros(nk, nn);
leb_uls       = zeros(nk, nn);
leb_hyper_red = zeros(nk, nn);

M_red_vec = zeros(nk, nn);
ep_used   = zeros(nk, nn);
diag_red  = cell(nk, nn);   % cubature diagnostics (Reviewer 2, point 7)

%% -------------------------------------------------------------------------
%  MAIN LOOP
% -------------------------------------------------------------------------
for k = 1:nk

    rbf_name = rbf_names(k);
    rbf      = select_rbf(rbf_name);

    fprintf('\n=== Kernel: %s | ep=%.2f | tau=%.2f ===\n', ...
        rbf_name, ep, tau_vec(k));

    for idx = 1:nn

        n_ctrs = n_ctrsVec(idx);
        x_ctrs = get_ctrs(n_ctrs, data_distribution, dim);

        h_X            = fill_distance(x_ctrs, 100);
        ep_used(k,idx) = ep;

        ker = @(x, y) rbf(pdist2(x, y), ep);

        fprintf('  n = %3d  |  h_X = %.4f  |  ep = %6.2f\n', n_ctrs, h_X, ep);

        K_leb = ker(X_leb, x_ctrs);
        meas  = 1;

        % ------------------------------------------------------------------
        %  REDUCED-EXACTNESS CUBATURE  (exact on (V_n + 1) x 1)
        %  Shared by the least-squares and reduced-hyperinterpolation methods.
        % ------------------------------------------------------------------
        reduced_param = 1;
        [y_red, err_qmc, G] = moment(ker, x_ctrs, reduced_param);
        r_red    = size(y_red, 1);
        N_init_r = 2 * (n_ctrs + 1);

        [~, Z_red, w_red, ~, ~, info] = getCF(r_red, N_init_r, x_ctrs, ...
            ker, meas, y_red, reduced_param, dim, cf_opts(err_qmc));

        M_red            = length(w_red);
        M_red_vec(k,idx) = M_red;

        VZ_red  = ker(Z_red, x_ctrs);
        diag_red{k,idx} = cf_diagnostics(info, VZ_red, w_red, G, ker(X_test, x_ctrs), solver);
        W_red   = diag(w_red);
        rhs_red = f(Z_red);

        % ------------------------------------------------------------------
        %  (A) INTERPOLATION AT X  (cardinal-function Lebesgue, no cubature)
        % ------------------------------------------------------------------
        K_XX = ker(x_ctrs, x_ctrs);
        f_X  = f(x_ctrs);

        [L_X, flag_X] = chol(K_XX, 'lower');
        if flag_X ~= 0
            warning('K(X,X) not SPD at n=%d, ep=%.2f — using backslash', n_ctrs, ep);
            a_interp  = K_XX \ f_X;
            Phi_leb_X = (K_XX \ K_leb')';
        else
            a_interp  = L_X' \ (L_X \ f_X);
            Phi_leb_X = (L_X' \ (L_X \ K_leb'))';
        end

        f_pred_interp      = ker(X_test, x_ctrs) * a_interp;
        rms_interp(k,idx)  = rmse(f_pred_interp, f_test);
        cond_interp(k,idx) = cond(K_XX);
        leb_interp(k,idx)  = max(sum(abs(Phi_leb_X), 2));

        % ------------------------------------------------------------------
        %  (B) LEAST SQUARES  (unweighted, W = I, sampled at Z_red)
        % ------------------------------------------------------------------
        W_uls = eye(M_red);
        a_uls = LS_stable_solver(W_uls, VZ_red, rhs_red, solver);

        f_pred_uls      = ker(X_test, x_ctrs) * a_uls;
        rms_uls(k,idx)  = rmse(f_pred_uls, f_test);
        cond_uls(k,idx) = cond(VZ_red);
        leb_uls(k,idx)  = leb_svd(K_leb, VZ_red, W_uls, solver);

        % ------------------------------------------------------------------
        %  (C) HYPERINTERPOLATION — REDUCED EXACTNESS
        % ------------------------------------------------------------------
        a_red = LS_stable_solver(W_red, VZ_red, rhs_red, solver);

        f_pred_hr             = ker(X_test, x_ctrs) * a_red;
        rms_hyper_red(k,idx)  = rmse(f_pred_hr, f_test);
        cond_hyper_red(k,idx) = cond(sqrt(W_red) * VZ_red);
        leb_hyper_red(k,idx)  = leb_svd(K_leb, VZ_red, W_red, solver);

        fprintf('    M_red = %d\n', M_red);

    end % n loop
end % kernel loop

%% =========================================================================
%  PLOTTING
% =========================================================================
fig_id     = 1;
anchor_idx = 1;

col_interp = [0.47 0.67 0.19];
col_uls    = [0.00 0.45 0.74];
col_red    = [0.93 0.69 0.13];
col_ref1   = [0.00 0.00 0.00];

    function pub_axes(ax)
        set(ax, 'FontSize',13,'FontWeight','bold','LineWidth',1.5, ...
            'TickLength',[0.015 0.025],'TickDir','out','Box','on', ...
            'Color','w','XColor','k','YColor','k', ...
            'GridAlpha',0.25,'MinorGridAlpha',0.12);
        set(get(ax,'Parent'),'Color','w');
    end
    function pub_legend(lg)
        set(lg,'Color','w','EdgeColor','k','TextColor','k', ...
            'FontSize',9.5,'FontWeight','normal','LineWidth',1,'Interpreter','tex');
    end

% =========================================================================
%  BLOCK I — CONDITION NUMBER vs n
% =========================================================================
for k = 1:nk
    figure(fig_id); clf; set(gcf,'Color','w'); hold on;
    plot(n_ctrsVec, cond_interp(k,:),    '-^',  'Color',col_interp, ...
        'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_interp);
    plot(n_ctrsVec, cond_uls(k,:),       '-o',  'Color',col_uls,   ...
        'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_uls);
    plot(n_ctrsVec, cond_hyper_red(k,:), '-.d', 'Color',col_red,   ...
        'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_red);
    set(gca,'YScale','log'); pub_axes(gca);

    xlabel('n','FontSize',14,'FontWeight','bold','Color','k');
    ylabel('Condition Number (log scale)','FontSize',14,'FontWeight', ...
        'bold','Color','k');
    lg = legend('Interpolation','Least Squares','Hyper Reduced',...
        'Location','northwest');
    pub_legend(lg);
    title(sprintf('f_4 - Condition Number - %s kernel - %s points', ...
        rbf_names(k), data_distribution),'FontSize',12,'FontWeight', ...
        'bold','Color','k');
    savefig_png(fig_id, fig_tag); fig_id = fig_id + 1;
end

% =========================================================================
%  BLOCK II — RMSE log-log + empirical slopes
% =========================================================================
for k = 1:nk
    tau          = tau_vec(k);
    slope_theory = -(tau - 1) / 2;   % d=2: -(tau-d/2)/d  with d=2

    figure(fig_id); clf; set(gcf,'Color','w'); hold on;

    rmse_i  = rms_interp(k,:);
    rmse_u  = rms_uls(k,:);
    rmse_hr = rms_hyper_red(k,:);

    plot(n_ctrsVec, rmse_i,  '-^',  'Color',col_interp, ...
        'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_interp);
    plot(n_ctrsVec, rmse_u,  '-o',  'Color',col_uls,   ...
        'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_uls);
    plot(n_ctrsVec, rmse_hr, '-.d', 'Color',col_red,   ...
        'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_red);

    log_n_asym = log(n_ctrsVec(anchor_idx:end)');
    slope_i    = fit_slope(log_n_asym, log(rmse_i(anchor_idx:end)'));
    slope_u    = fit_slope(log_n_asym, log(rmse_u(anchor_idx:end)'));
    slope_hr   = fit_slope(log_n_asym, log(rmse_hr(anchor_idx:end)'));

    n_anchor  = n_ctrsVec(anchor_idx);
    n_ref_vec = logspace(log10(n_anchor), log10(n_ctrsVec(end)), 60);
    ref_line  = rmse_hr(anchor_idx) * (n_ref_vec / n_anchor).^slope_theory;
    plot(n_ref_vec, ref_line, '-', 'Color',col_ref1, 'LineWidth',1.2);

    set(gca,'XScale','log','YScale','log'); pub_axes(gca);
    xlabel('n (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    ylabel('RMSE (log scale)','FontSize',14,'FontWeight','bold','Color','k');

    leg1 = sprintf('Interpolation  O(n^{%.2f})', slope_i);
    leg2 = sprintf('Least Squares  O(n^{%.2f})', slope_u);
    leg3 = sprintf('Hyper Reduced  O(n^{%.2f})', slope_hr);
    leg4 = sprintf('Ref  O(n^{%.2f})',           slope_theory);
    lg   = legend(leg1,leg2,leg3,leg4,'Location','northeast');
    pub_legend(lg);
    title(sprintf('f_4 - RMSE - %s kernel - %s points', ...
        rbf_names(k), data_distribution),'FontSize',12,'FontWeight', ...
        'bold','Color','k');

    savefig_png(fig_id, fig_tag); fig_id = fig_id + 1;
end

% =========================================================================
%  BLOCK III — LEBESGUE CONSTANT log-log  (SVD-consistent values)
% =========================================================================
for k = 1:nk
    figure(fig_id); clf; set(gcf,'Color','w'); hold on;

    leb_i  = leb_interp(k,:);
    leb_u  = leb_uls(k,:);
    leb_hr = leb_hyper_red(k,:);

    plot(n_ctrsVec, leb_i,  '-^',  'Color',col_interp,...
        'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_interp);
    plot(n_ctrsVec, leb_u,  '-o',  'Color',col_uls,   ...
        'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_uls);
    plot(n_ctrsVec, leb_hr, '-.d', 'Color',col_red,   ...
        'LineWidth',1.8,'MarkerSize',6,'MarkerFaceColor',col_red);

    ref_leb = leb_i(1) * (n_ctrsVec / n_ctrsVec(1)).^0.5;
    plot(n_ctrsVec, ref_leb, '-', 'Color',col_ref1, 'LineWidth',1.2);

    log_n_asym = log(n_ctrsVec(anchor_idx:end)');
    slope_li   = fit_slope(log_n_asym, log(leb_i(anchor_idx:end)'));
    slope_lu   = fit_slope(log_n_asym, log(leb_u(anchor_idx:end)'));
    slope_lhr  = fit_slope(log_n_asym, log(leb_hr(anchor_idx:end)'));

    set(gca,'XScale','log','YScale','log'); pub_axes(gca);
    xlabel('n (log scale)','FontSize',14,'FontWeight','bold','Color','k');
    ylabel('Lebesgue constant (log scale)','FontSize',14,'FontWeight','bold','Color','k');

    leg1 = sprintf('Interpolation  O(n^{%.2f})', slope_li);
    leg2 = sprintf('Least Squares  O(n^{%.2f})', slope_lu);
    leg3 = sprintf('Hyper Reduced  O(n^{%.2f})', slope_lhr);
    leg4 = 'Ref  O(n^{+0.50})';
    lg   = legend(leg1,leg2,leg3,leg4,'Location','northwest');
    pub_legend(lg);
    title(sprintf('f_4 - Lebesgue Constant - %s kernel - %s points', ...
        rbf_names(k), data_distribution),'FontSize',12,'FontWeight','bold','Color','k');

    fprintf('\n[Lebesgue slopes]  %s | ep=%.2f\n', rbf_names(k), ep);
    fprintf('  %-48s  %+.4f\n', 'Interpolation:', slope_li);
    fprintf('  %-48s  %+.4f\n', 'Least Squares:', slope_lu);
    fprintf('  %-48s  %+.4f\n', 'Hyper Reduced:', slope_lhr);

    savefig_png(fig_id, fig_tag); fig_id = fig_id + 1;
end

fprintf('\nDone. %d figures saved.\n', fig_id - 1);
print_cf_diagnostics(diag_red,  rbf_names, n_ctrsVec, sprintf('reduced exactness, ep = %g', ep));
save(sprintf('diagnostics_%s.mat', fig_tag), 'diag_red', 'ep_used', 'n_ctrsVec', 'rbf_names');


%% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================
function v = franke(x, y)
% Standard Franke function on [0,1]^2 (vectorized).
    v =  0.75 * exp( -((9*x - 2).^2 + (9*y - 2).^2) / 4 ) ...
       + 0.75 * exp( -((9*x + 1).^2) / 49 - (9*y + 1) / 10 ) ...
       + 0.50 * exp( -((9*x - 7).^2 + (9*y - 3).^2) / 4 ) ...
       - 0.20 * exp( -( 9*x - 4).^2 - (9*y - 7).^2 );
end