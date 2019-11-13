clear all; clc;
D = loadData('data/x.txt','data/y.txt');
A = makeA(D);
DT = DecisionTree(D,A);