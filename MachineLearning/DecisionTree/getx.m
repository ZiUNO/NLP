function x = getx(D)
    for i = 1:size(D,1)
        x(i,:) = D(i).x;
    end