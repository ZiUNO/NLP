des = 2;
x = rand(30,des);
y = 3*x(:,1)-2*x(:,2)+0.3; % y=3*x1-2*x2+0.3
y(y<0.5)=-1;
y(y>=0.5)=1;
figure;
plot(x(y==1,1),x(y==1,2),'bx');
hold on;
plot(x(y==-1,1),x(y==-1,2),'r+');
w = rand(1,des);
