function [theta,cv_err, K, K0] = train_mclass_kernel_SVM(tra,val,krnl,C, krnl_param)
    e = 1e-6;
    [theta, cv_err, K, K0] = train_kernel_LLS(tra, val, krnl, C, krnl_param);
    tra_cov_matrix = zeros(size(tra.x,1)+1, size(tra.x,1)+1, size(tra.x,1));
    for i = 1:size(tra.x,1)
       tra_cov_matrix(:, :, i) = K(i,:)' * K(i,:); 
    end
    
 
    for k = 1:size(tra.y,2)
    y = tra.y(:,k);
    cost = 0;
    itr = 0;
    while 1==1
        z = abs(1 - y .* (K * theta(:,k)));  
        filter = (z < e);
        z(filter,1) = e;
        
        hinge_loss_cost = 0;
        for i = 1:size(y,1)
            hinge_loss_cost = hinge_loss_cost + (1/(4*z(i))) * ((1- y(i) * K(i,:) * theta(:,k)) + z(i))^2;
        end
        new_cost = 0.5 * (theta(:,k)' * K0 * theta(:,k)) + C * hinge_loss_cost;
        cost_change = abs(cost - new_cost)/abs(cost);
        if(cost_change < 1e-3 || itr == 250)
            break;
        end
        cost = new_cost;

        inv_sum = 0;
        other_sum = 0;
        for i = 1:size(y,1)
%             inv_sum = inv_sum +  ( (K(i,:)' * K(i,:)) / (2*z(i)) );
              inv_sum = inv_sum +  ( (tra_cov_matrix(:, :,i)) / (2*z(i)) );
%             tra_cov_matrix(i, :, :)
            other_sum = other_sum +  ( ((1+z(i)) / (2*z(i))) * ( y(i) .* K(i,:)') );
        end
        theta(:,k) = (pinv (K0 + C * inv_sum) * (C * other_sum));
        itr = itr + 1;
    end 
   
    end
    yval_computed= kernel_dot_product(theta, tra.x, val.x, krnl, krnl_param);
    cv_err = compute_error(val.y, yval_computed);
end
 