clc;
clear;
close all;

%% Comparison of Adams-Bashforth vs Adams-Moulton for different step sizes
f = @(y,u) -y+u;
y0 = 0;
tspan = [0 5];
h_values = [1, 0.5, 0.2, 0.1, 0.01, 0.001];

results = cell(length(h_values), 1);

figure(1);
set(gcf, 'Position', [100, 100, 1400, 900]);

for i = 1:length(h_values)
    h = h_values(i);

    [t, y_ab] = adams_bashforth(f, tspan, y0, h);
    [~, y_am] = adams_moulton(f, tspan, y0, h);

    y_analytical = 1 - exp(-t);

    % Store for error plot
    results{i}.t = t;
    results{i}.error_ab = abs(y_ab - y_analytical');
    results{i}.error_am = abs(y_am - y_analytical');
    results{i}.h = h;

    subplot(2, 3, i);
    plot(t, y_ab, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Adams-Bashforth');
    hold on;
    plot(t, y_am, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Adams-Moulton');
    plot(t, y_analytical, 'k--', 'LineWidth', 2, 'DisplayName', 'Analytical');
    grid on;

    xlabel('$t$ [s]', 'Interpreter', 'latex', 'FontSize', 14);
    ylabel('$y(t)$', 'Interpreter', 'latex', 'FontSize', 14);
    title(sprintf('$h = %.3f$', h), 'Interpreter', 'latex', 'FontSize', 14);
    legend('Interpreter', 'latex', 'Location', 'best', 'FontSize', 14);
end

sgtitle(
    'Adams-Bashforth vs Adams-Moulton: $\dot{y} = -y + u$, $y(0) = 0$', ...
    'Interpreter', 'latex', 'FontSize', 16, 'FontWeight', 'bold'
);

figure(2);
set(gcf, 'Position', [150, 150, 1400, 900]);

for i = 1:length(h_values)
    subplot(2, 3, i);
    plot(results{i}.t, results{i}.error_ab', 'b-', 'LineWidth', 1.5, 'DisplayName', 'Adams-Bashforth');
    hold on;
    plot(results{i}.t, results{i}.error_am', 'r-', 'LineWidth', 1.5, 'DisplayName', 'Adams-Moulton');
    grid on;

    xlabel('$t$ [s]', 'Interpreter', 'latex', 'FontSize', 14);
    ylabel('Absolute Error', 'Interpreter', 'latex', 'FontSize', 14);
    title(sprintf('$h = %.3f$', results{i}.h), 'Interpreter', 'latex', 'FontSize', 14);
    legend('Interpreter', 'latex', 'Location', 'best', 'FontSize', 14);
end

sgtitle(
    'Absolute Error vs Analytical Solution', ...
    'Interpreter', 'latex', 'FontSize', 16, 'FontWeight', 'bold'
);
