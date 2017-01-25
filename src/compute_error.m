function tes_err = compute_error(tes_y, ytes_computed)
    %Compute test error
    if (size(tes_y,2) == 1)
        filter = ytes_computed(:,1) <0;
        ytes_computed(filter ,1)  = -1;
        ytes_computed(not(filter) ,1)  = 1;
    else
        ymax = max(ytes_computed, [], 2);
        for i = 1:size(ytes_computed,1)
            ytes_computed(i,:) = ytes_computed(i,:) == ymax(i);
        end
        for k = 1:size(tes_y,2)
            ytes_computed(ytes_computed(:,k) == 0 ,k) = -1;
        end
    end
    
    %Compare the computed labels with actual labels in tes_y
    tes_err = ytes_computed ~= tes_y;
    tes_err = sum(tes_err,2);
    tes_err(tes_err>1) = 1;
    tes_err = mean(tes_err);

end