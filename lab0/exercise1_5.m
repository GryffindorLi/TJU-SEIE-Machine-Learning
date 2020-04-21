fun = @(x)100*(x(2) - x(1)^2)^2 + (1 - x(1)^2)^2;
x0 = [-10,-10];
x = fminsearch(fun,x0)
x1 = -2:0.05:2;
x2 = -1:0.05:3;
[X,Y]=meshgrid(x1,x2);
Z=100*(Y-X.^2).^2 + (1 - X.^2).^2;
mesh(X,Y,Z)

zi = griddata(x1,x2,Z,X,Y);
contour(X,Y,zi,30)
