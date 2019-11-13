function D = makeD(x, y)
    assert(size(x,1)==size(y,1),'the length of x != the length of y');
    for i = 1:size(x,1)
        D(i).x = x(i,:)';
        D(i).y = y(i);
    end
    D = D';