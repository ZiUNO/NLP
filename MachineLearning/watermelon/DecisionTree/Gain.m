function G = Gain(D, a)
    assert(a<=size(D(1).x,1)&&a>0,'a > size of the attributes');
    x = getx(D);
    y = gety(D);
    uni = unique(x(:,a))';
    s = 0.0;
    for v = uni
        n = find(x(:,a)==v);
        D_v = makeD(x(n,:),y(n));
        s = s + size(D_v,1)/size(D,1)*Entropy(D_v);
    end
    G = Entropy(D) - s;
    