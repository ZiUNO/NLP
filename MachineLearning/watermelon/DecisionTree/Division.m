function a = Division(D)
    a_ = zeros(size(D(1).x'));
    for i = 1:size(a_,2)
        a_(i) = Gain(D,i);
    end
    [~,a] = max(a_);
    