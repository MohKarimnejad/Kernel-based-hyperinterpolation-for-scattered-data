function a = LS_stable_solver(W, V, y)
% Coefficients of the weighted least-squares approximant (eq. (10)),
%
%     a = argmin || sqrt(W) (V a - y) ||_2,
%
% computed by the economy QR factorization of sqrt(W)*V (Algorithm 1),
% which avoids forming the normal matrix V' W V.

    V_tild = sqrt(W) * V;
    y_tild = sqrt(W) * y;

    [Q, R] = qr(V_tild, 0);
    a = R \ (Q' * y_tild);
end
