function [t,y] = rk4(f, tspan, y0, h)
    % rk4 Solve ODE using Runge-Kutta 4th Order method
    %   [t,y] = rk4(f,tspan,y0,h) integrates the system of differential
    %   equations y' = f(x,u) using 4th Order Runge-Kutta method with step size h.
    %
    %   Inputs:
    %       f      - Function handle for the derivative function f(x,u)
    %       tspan  - Time span [t0, tf] for integration (default: [0, 1])
    %       y0     - Initial conditions as column vector (default: 0)
    %       h      - Step size for integration (default: 0.1)
    %
    %   Outputs:
    %       t      - Time vector
    %       y      - Solution matrix where each row is the solution at time t(i)
    %
    %   The function calls entrada(t) to compute the input u at each time step.
    %
    %   Example:
    %       f = @(x,u) -2*x + u;
    %       [t,y] = rk4(f, [0 10], 1, 0.01);

    arguments
        f function_handle
        tspan (1,2) double = [0, 1]
        y0 (:,1) double = 0
        h (1,1) double {mustBePositive} = 0.1
    end

    t = tspan(1):h:tspan(2);
    n = length(t);
    y = zeros(n, length(y0));
    y(1,:) = y0;

    for i = 1:n-1
        u0 = entrada(t(i));
        u1 = entrada(t(i) + h/2);
        u2 = entrada(t(i) + h);

        % RK4 stages
        R1 = h * f(y(i,:), u0);
        R2 = h * f(y(i,:) + R1/2, u1);
        R3 = h * f(y(i,:) + R2/2, u1);
        R4 = h * f(y(i,:) + R3, u2);

        y(i+1,:) = y(i,:) + (R1 + 2*R2 + 2*R3 + R4)/6;
    end
end
