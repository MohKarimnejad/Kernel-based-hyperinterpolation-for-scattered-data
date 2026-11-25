function print_cf_diagnostics(Dg, rbf_names, n_ctrsVec, label)
% Prints a table of cf_diagnostics structs, Dg{k, idx} (kernel x n).
% Empty cells are skipped.

    fprintf('\n--- Cubature diagnostics: %s ---\n', label);
    fprintf('%-9s %5s %7s %10s %10s %10s %7s %7s %10s %9s %5s\n', ...
        'kernel', 'n', 'M', 'residual', 'min(w)', 'max(w)', ...
        'rank A', 'rank VZ', 'repro', 'eta', 'conv');
    for k = 1:size(Dg, 1)
        for idx = 1:size(Dg, 2)
            D = Dg{k, idx};
            if isempty(D), continue; end
            fprintf('%-9s %5d %7d %10.2e %10.2e %10.2e %7d %7d %10.2e %9.2e %5d\n', ...
                rbf_names(k), n_ctrsVec(idx), D.M, D.residual, D.w_min, ...
                D.w_max, D.rank_A, D.rank_VZ, D.repro, D.eta, D.converged);
        end
    end
end
