function root = DecisionTree(D, A)
    y = gety(D);
    x = getx(D);
    results = unique(y);
    root = struct;
    if size(results,1) == 1
        root.attri_resu = results(1);
        root.branch = struct;
        return
    end
    clear results;
    if isempty(A) || sum(max(x(:,A))-min(x(:,A)))==0
        root.attri_resu = getmostc(D);
        root.branch = struct;
        return
    end
    clear x_;
    a = Division(D);
    a_ = unique(x(:,a))';
    if ~isfield(root,'branch')
        root.branch = struct;
    end
    for a_v = a_
       if isempty(fieldnames(root.branch))
           b = 1;
       else
           b = size(root.branch,2)+1;
       end
       n = find(x(:,a)==a_v);
       D_v = makeD(x(n,:),y(n));
%        size(D_v)
%        for v = D_v'
%            t_x = (v.x)'
%            t_y = v.y'
%        end
       clear n;
       if isempty(fieldnames(D_v))
           root.branch(b).node.attri_value = a_v;
           root.branch(b).node.attri_resu = getmostc(D);
           root.branch(b).node.branch = struct;
           return
       else
           root.attri_resu = a;
           root.branch(b).node = DecisionTree(D_v,setdiff(A,a));
           root.branch(b).node.attri_value = a_v;
       end
    end