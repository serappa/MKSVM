function y = unit_std_0_mu_normalize_columns(x)
    mu = mean(x);
    sigma = std(x);
    for i = 1: size(x,2)
        x(:,i) = (x(:,i) - mu(i));
        if(sigma(i) ~= 0)
            x(:,i) = (x(:,i) /sigma(i));
        end
    end
    y = x;
end