function [I, J] = prod_index(n)
% Index pairs (i,j), i <= j, in the order  for i = 1:n, for j = i:n.
    cnt = n * (n + 1) / 2;
    I = zeros(1, cnt);
    J = zeros(1, cnt);
    c = 0;
    for i = 1:n
        len = n - i + 1;
        I(c+1:c+len) = i;
        J(c+1:c+len) = i:n;
        c = c + len;
    end
end
