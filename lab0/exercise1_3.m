x = 0:pi/100:pi*2;
y1 = 2*cos(x);
y2 = cos(x);
y3 = 0.5*cos(x);
plot(x,y1,'-.',x,y2,'-',x,y3,'--');
axis([0,6,-3,3]);
legend('2*cos(x)','cos(x)','0.5*cos(x)');