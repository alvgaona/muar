function [t,y] = rk2(f, tspan, y0, h)
    % rk2 Solve ODE using Runge-Kutta 2nd Order method
    %   [t,y] = rk2(f,tspan,y0,h) integrates the system of differential
    %   equations y' = f(x,u) using 2nd Order Runge-Kutta method with step size h.
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
    %   The function calls ENTRADA(t) to compute the input u at each time step.
    %
    %   Example:
    %       f = @(x,u) -2*x + u;
    %       [t,y] = rk2(f, [0 10], 1, 0.01);

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

        R1 = h * f(y(i,:), u0);
        R2 = h * f(y(i,:) + R1/2, u1);

        y(i+1,:) = y(i,:) + R2;
    end
end
