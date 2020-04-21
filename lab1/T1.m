clear;
clc;
data = csvread('Salary_Data.csv', 1, 0);
years = data(:,1);
salary = data(:,2);
scatter(years, salary)
hold on;
one_matrix = ones(30,1);
years = [one_matrix years];
%¼ÆËãw
omega = inv(years'*years)*years'*salary;
x = 0:0.2:12;
y = omega(1) + omega(2)*x;
plot(x,y)

