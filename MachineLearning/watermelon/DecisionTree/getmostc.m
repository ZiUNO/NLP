function c = getmostc(D)
    y = gety(D);
    uni = unique(y)';
    uni = [uni;zeros(size(uni))];
    for u = 1:size(uni,2)
        uni(2,u) = sum(y==uni(1,u));
    end
    [~,index] = max(uni(2,:));
    c = uni(1,index);