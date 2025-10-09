clc;
clear;

%% Descripción

% El simulador es levemente distinto al implementado en clase ya que me
% parece mejor no utilizar variables globales,
% y tratar de tener una mejor estructura para cada método.
% Con una interfaz similar a la que MATLAB muestra con ODE45.
%
% Mantenemos el sistema dada por la función `f`, el estado inicial,
% el paso, y trabajamos con un vector de tiempo calculado por cada método
% al utilizar la variable `tspan`.

%% Simulador
f = @(x,u) -x+u; % System
y0 = 0;          % Initial state
h = 0.1;         % Step of the method
tspan = [0 5];   % Time bounds for the simulation

% [t, y] = euler(f, tspan, y0, h);
% [t, y] = euler_trapezoidal(f, tspan, y0, h);
% [t, y] = rk2(f, tspan, y0, h);
% [t, y] = rk4(f, tspan, y0, h);
% [t, y] = adams_bashforth(f, tspan, y0, h);
[t, y] = adams_moulton(f, tspan, y0, h);

y_calc = 1 - exp(-t);

figure(1)
plot(t, y, 'b');
hold on
plot(t, y_calc, 'r');
legend( ...
    'Numerical Solution', 'Analytical Solution: $y = 1 - e^{-t}$', ...
    'Interpreter', 'latex', 'Location', 'east', 'FontSize', 14 ...
    );
xlabel('Time $[s]$', 'Interpreter', 'latex');
ylabel('$y(t)$', 'Interpreter', 'latex');
title('Numerical vs Analytical Solution', 'Interpreter', 'latex');
