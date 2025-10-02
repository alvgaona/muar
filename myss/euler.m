function [t,y] = euler(f, tspan, y0, h)
    % euler Solve ODE using Euler's method
    %   [t,y] = euler(f,tspan,y0,h) integrates the system of differential
    %   equations y' = f(x,u) using Euler's method with step size h.
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
    %       [t,y] = euler(f, [0 10], 1, 0.01);

    arguments
        f function_handle
        tspan (1,2) double = [0, 1]
        y0 (:,1) double = 0
        h (1,1) double {mustBePositive} = 0.1
    end

    t = tspan(1):h:tspan(2);
    y = zeros(length(t), length(y0));

    y(1,:) = y0;

    for i=1:length(t)-1
        u = entrada(t(i));
        x = y(i, :);
        y(i+1,:) = x + h * f(x, u);
    end
end
