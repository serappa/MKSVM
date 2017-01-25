C =  10;
C_LLS = 10;
sigma = 7;
e = 10e-5;
krnl = @rbf_k;
[tra, val, tes] = get_data(90,30, 'handwritten');

tra = tra(1);
xtra = tra.x;
val = val(1);

[theta,cv_err, K, K0] = train_kernel_LLS(tra,val,krnl,C_LLS, sigma);

z = zeros(size(tra.y,2),size(tra.y,2),size(tra.y,1));
cost = 0;
i = 0;
while 1==1
    %update z(k,l,n)
    K_dot_thetak= zeros(size(tra.y,2), size(tra.y,1));
    for k=1:size(tra.y,2)
        K_dot_thetak(k,:) =  K*theta(:,k);
    end
    
    for k=1:size(tra.y,2)
        for l=1:size(tra.y,2)
            if(k~=l)
                Z = abs(1 - K_dot_thetak(k,:) + K_dot_thetak(l,:));
                filter = (Z < e);
                Z(filter) = e;
                z(k,l,:) = Z;
            end
        end
    end
        
    hinge_loss_cost = 0;
    for k = 1:size(tra.y,2)
        for l = 1:size(tra.y,2)
            if(l ~= k)
                for n = 1:size(tra.y,1)
                    if(tra.y(n,k) ~=0)
                        hinge_loss_cost = hinge_loss_cost +( (1/(4*z(k,l,n))) * ((1 - K_dot_thetak(k,n) + K_dot_thetak(l,n) + z(k,l,n))^2) );
                    end
                end
            end
        end
    end
    
    new_cost = 0;
    for k = 1:size(tra.y,2)
        new_cost = new_cost + 0.5 * (theta(:,k)' * K0 * theta(:,k));
    end
    new_cost = new_cost + C * hinge_loss_cost;
    cost_change = abs(cost - new_cost)/abs(cost);
    if(cost_change < 1e-3 )
        break;
    end
    
    cost = new_cost;
    
    new_theta = theta;
    for k=1:size(tra.y,2)
        inv_sum = 0;
        C2_sum = 0;
 
        for n = 1:size(tra.y,1)
            if(tra.y(n,k) == 1)
                Kn_dot_Knt = K(n,:)' * K(n,:);
                Kn = K(n,:)';
                for l = 1:size(tra.y,2)
                    if(l ~= k)
                        inv_sum = inv_sum + ( Kn_dot_Knt ./ (2*z(k,l,n)) );
                        C2_sum = C2_sum + Kn .*  ( ((1+z(k,l,n)+K_dot_thetak(l,n)) / (2*z(k,l,n))) );
                    end
                end
            end
        end

        C1 = inv ( K0 + C * inv_sum);
        C2 = C * C2_sum;
        new_theta(:,k) = (C1 * C2);
    end
    i = i + 1;
    theta = new_theta;
end
 %computing cross_validation error
    xval = tra.x;
    yval = tra.y;

    y = kernel_dot_product(theta, xtra, xval, krnl, sigma);
    y_computed = y;
    ymax = max(y, [], 2);
    for i = 1:size(y,1)
        y_computed(i,:) = y_computed(i,:) == ymax(i);
    end

    %Compare the computed labels with actual labels in yval
    err = y_computed ~= yval;
    err = sum(err,2);
    err(err>1) = 1;
    cv_err = mean(err);
%     
