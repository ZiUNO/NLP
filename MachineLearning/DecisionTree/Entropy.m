function Ent = Entropy(D)
    y = gety(D);
    uni = unique(y);
    p = zeros(size(uni));
    for i = 1:size(uni, 1)
        p(i) = size(y(y==uni(i)),1)/size(y,1);
    end
    Ent = - sum(p.*log2(p));