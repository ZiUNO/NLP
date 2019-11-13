function y = gety(D)
    y = zeros(size(D));
    for i = 1:size(D,1)
        y(i) = D(i).y;
    end