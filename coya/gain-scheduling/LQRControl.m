theta0 = 0:0.1:pi;
g = 9.81;
l = 1;
b = 0.43;
m = 0.5;

B = [0; 1/(m*l^2)];
C = [1 0];
D = 0;

Q = [1 0; 0 1];
R = 1;

LUT = {};

for i=1:length(theta0)
    A = [0 1; -g/l*cos(theta0(i)) -b/(m*l^2)];
    sys = ss(A, B, C, D);
    K = lqr(sys, Q, R);

    LUT{i} = K;
end



