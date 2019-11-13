function D = loadData(xfile, yfile)
    x = load(xfile);
    y = load(yfile);
    D = makeD(x, y);
    