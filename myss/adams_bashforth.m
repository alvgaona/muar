function [t,y] = adams_bashforth(f, tspan, y0, h)
    % adams_bashford Solve ODE using Adams-Bashforth method
    %   [t,y] = adams_bashford(f,tspan,y0,h) integrates the system of differential
    %   equations y' = f(y,u) using Adams-Bashforth method with step size h.
    %
    %   Inputs:
    %       f      - Function handle for the derivative function f(y,u)
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
    %       f = @(y,u) -2*y + u;
    %       [t,y] = adams_bashford(f, [0 10], 1, 0.01);

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

    % I'm working with Euler Trapezoidal for the first 4 points
    % as there are no previous values at these timesteps.
    [~, y(1:4, :)] = euler_trapezoidal(f, [t(1), t(4)], y(1,:)', h);

    for i = 4:n-1
        u1 = entrada(t(i));
        u2 = entrada(t(i-1));
        u3 = entrada(t(i-2));
        u4 = entrada(t(i-3));

        f1 = f(y(i,:), u1);
        f2 = f(y(i-1,:), u2);
        f3 = f(y(i-2,:), u3);
        f4 = f(y(i-3,:), u4);

        y(i+1,:) = y(i,:) + h/24 * (55*f1 - 59*f2 + 37*f3 - 9*f4);
    end
end
