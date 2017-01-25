function [theta,cv_err, K, K0] = train_kernel_LLS(tra,val,krnl,C, krnl_param)
    %Gram matrix with the last column being all ones to subsume theta0
    %K has n x n+1 dimensions
    xtra = tra.x;
    ytra = tra.y;
    K = ones(size(xtra,1), size(xtra,1)+1);
    K(:,1:end-1) = krnl(xtra, xtra, krnl_param);

    %Gram matrix with the last column and last row being all zeros to
    %ignore bias in theta while computing theta' * K * theta

    %K has n x n+1 dimensions
    K0 = zeros(size(xtra,1)+1, size(xtra,1)+1);
    K0(1:end-1, 1:end-1) = K(1:end, 1:end-1);

    % Here ytra is n * k matrix and therefore theta will be n+1 x k matrix
    theta = pinv(K' * K + C * K0) * K' * ytra;

    %computing cross_validation error
    xval = val.x;
    yval = val.y;

    y = kernel_dot_product(theta, xtra, xval, krnl, krnl_param);
    ymax = max(y, [], 2);
    for i = 1:size(y,1)
        y(i,:) = y(i,:) == ymax(i);
    end

    %Compare the computed labels with actual labels in yval
    err = y ~= yval;
    err = sum(err,2);
    err(err>1) = 1;
    cv_err = mean(err);

end