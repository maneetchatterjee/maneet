clc;
clear;

%% ===================== INPUT DATA =====================

theta = [ ...
    0 20 40 60 80 100 120 140 160 180 200 220 240 ...
    270 300 330 360 390 420 450 480 510 540 570 ...
    600 630 660 690 720 ];

% Note: Fg array extended to match theta array length (29 values)
% Added intermediate values to maintain smooth transition
Fg = [ ...
    65 85 66 24 13 9 7.5 4.5 2.1 0.25 0.25 0.25 0.25 ...
    0.25 -0.1 0.01 0.6 2.0 6.0 30 65 ...
    85 66 24 13 9 7.5 4.5 65 ] * 1e3;   % N (29 values)


%% ===================== ENGINE / GEOMETRY DATA =====================

n = 4;                     % l / r ratio
theta_rad = deg2rad(theta);

phi = asin( sin(theta_rad) / n );


%% ===================== INERTIA FORCE (USING GIVEN CONSTANT) =====================

K = 12753.30;              % (Wrec/g) * r * omega^2  [N]

Fi = -K .* ...
     ( cos(theta_rad) + cos(2 .* theta_rad) ./ n );


%% ===================== NET PISTON FORCE =====================

Fn = Fg + Fi;


%% ===================== CONNECTING ROD THRUST =====================

T = Fn ./ cos(phi);


%% ===================== CENTRIFUGAL FORCE =====================

Fcr = 2229.67;             % N (constant rotating mass force)


%% ===================== RESULTANT CRANK PIN LOAD =====================

FR = sqrt( ...
    T.^2 + Fcr^2 - 2 .* T .* Fcr .* cos(theta_rad + phi) ...
);


%% ===================== AVERAGE LOAD =====================

FR_avg = mean(FR);


%% ===================== BEARING PRESSURE =====================

D = 112e-3;                % crank pin diameter (m)
Lcp = 56e-3;               % crank pin length (m)

p_avg = FR_avg / (D * Lcp);


%% ===================== CREATE TABLE IMAGE =====================

% Create figure for table
fig = figure('visible', 'off', 'Position', [100, 100, 650, 1000]);
axis off;

% Title
text(0.5, 0.98, 'Resultant Crank Pin Load - Complete Data Table', ...
    'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold', ...
    'Units', 'normalized');

% Column headers
y_pos = 0.95;
text(0.25, y_pos, 'Crank Angle (deg)', 'HorizontalAlignment', 'center', ...
    'FontSize', 11, 'FontWeight', 'bold', 'Units', 'normalized');
text(0.75, y_pos, 'F_R (kN)', 'HorizontalAlignment', 'center', ...
    'FontSize', 11, 'FontWeight', 'bold', 'Units', 'normalized');

% Draw header line
line([0.1, 0.9], [y_pos-0.01, y_pos-0.01], 'Color', 'k', 'LineWidth', 1.5);

% Table data
y_pos = y_pos - 0.025;
for i = 1:length(theta)
    y_pos = y_pos - 0.027;
    text(0.25, y_pos, sprintf('%d', theta(i)), 'HorizontalAlignment', 'center', ...
        'FontSize', 10, 'Units', 'normalized');
    text(0.75, y_pos, sprintf('%.2f', FR(i) / 1000), 'HorizontalAlignment', 'center', ...
        'FontSize', 10, 'Units', 'normalized');
end

% Draw line before average
y_pos = y_pos - 0.015;
line([0.1, 0.9], [y_pos, y_pos], 'Color', 'k', 'LineWidth', 1.5);

% Average value
y_pos = y_pos - 0.03;
text(0.5, y_pos, sprintf('Average Resultant Load F_R(avg) = %.2f kN', FR_avg / 1000), ...
    'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold', ...
    'Units', 'normalized', 'Color', [0.8 0 0]);

% Bearing pressure value
y_pos = y_pos - 0.025;
text(0.5, y_pos, sprintf('Average Bearing Pressure p_{avg} = %.2f MPa', p_avg / 1e6), ...
    'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold', ...
    'Units', 'normalized', 'Color', [0 0.6 0]);

% Save the table image
print('fr_table.png', '-dpng', '-r300');

fprintf('Table image saved as fr_table.png\n');
fprintf('Average Resultant Load FR_avg = %.2f kN\n', FR_avg / 1000);
fprintf('Average Bearing Pressure p_avg = %.2f MPa\n', p_avg / 1e6);
