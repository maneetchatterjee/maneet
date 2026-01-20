% Generate Table as PNG Image
% This script creates a PNG image of the FPL vs Crank Angle table

%% ================= INPUT DATA =================
theta = [
    0 20 40 60 80 100 120 140 160 180 200 220 240 270 300 330 360 ...
    390 420 450 480 510 540 570 600 630 660 690 720
];

Fg = [
    65 85 66 24 13 9 7.5 4.5 2.1 0.25 0.25 0.25 0.25 0.25 0.25 ...
    0.25 0.25 -0.10 -0.10 -0.10 -0.10 -0.10 -0.10 -0.10 -0.01 ...
    6.6 2.0 6.0 30 65
] * 1e3;

%% ================= GIVEN CONSTANTS =================
m_big = 3.534;        % Big end mass (kg)
a_big = 2229.67;      % Big end acceleration (m/s^2)

%% ================= CALCULATIONS ====================
theta_rad = deg2rad(theta);
phi = asin( sin(theta_rad) / 4 );
Q = (-12.760) .* ...
    (cos(theta_rad) + cos(2*theta_rad)/4 + Fg(1:length(theta))) ./ cos(phi);
Fcr = m_big * a_big;
Fpl = sqrt(Q.^2 + Fcr^2 - 2 .* Q .* Fcr .* cos(theta_rad + phi));
Fpl_kN = Fpl / 1000;
Fpl_avg = mean(Fpl);

%% ================= CREATE TABLE IMAGE ====================
% Create figure for table
fig = figure('Position', [100, 100, 600, 900], 'Color', 'white');
axis off;

% Prepare data for display
data_cell = cell(length(theta), 2);
for i = 1:length(theta)
    data_cell{i, 1} = sprintf('%.0f', theta(i));
    data_cell{i, 2} = sprintf('%.2f', Fpl_kN(i));
end

% Create text for title
text(0.5, 0.98, 'Resultant Crank Pin Load vs Crank Angle', ...
    'HorizontalAlignment', 'center', ...
    'FontSize', 16, ...
    'FontWeight', 'bold', ...
    'Units', 'normalized');

% Create header
text(0.3, 0.94, 'Crank Angle (deg)', ...
    'HorizontalAlignment', 'center', ...
    'FontSize', 12, ...
    'FontWeight', 'bold', ...
    'Units', 'normalized');

text(0.7, 0.94, 'F_PL (kN)', ...
    'HorizontalAlignment', 'center', ...
    'FontSize', 12, ...
    'FontWeight', 'bold', ...
    'Units', 'normalized');

% Draw header line
line([0.1, 0.9], [0.93, 0.93], 'Color', 'black', 'LineWidth', 2);

% Display all data rows
y_start = 0.91;
y_step = 0.028;
for i = 1:length(theta)
    y_pos = y_start - (i-1) * y_step;
    
    % Crank angle
    text(0.3, y_pos, data_cell{i, 1}, ...
        'HorizontalAlignment', 'center', ...
        'FontSize', 10, ...
        'Units', 'normalized');
    
    % FPL value
    text(0.7, y_pos, data_cell{i, 2}, ...
        'HorizontalAlignment', 'center', ...
        'FontSize', 10, ...
        'Units', 'normalized');
end

% Add average at bottom
y_avg = y_start - length(theta) * y_step - 0.02;
line([0.1, 0.9], [y_avg + 0.01, y_avg + 0.01], 'Color', 'black', 'LineWidth', 2);
text(0.5, y_avg - 0.01, sprintf('Average F_PL = %.2f kN', Fpl_avg/1000), ...
    'HorizontalAlignment', 'center', ...
    'FontSize', 12, ...
    'FontWeight', 'bold', ...
    'Units', 'normalized');

% Save as PNG
print('-dpng', '-r150', 'crank_pin_load_table.png');
disp('Table image saved as: crank_pin_load_table.png');

fprintf('\nAverage Resultant Crank Pin Load F_PL(avg) = %.2f kN\n', Fpl_avg/1000);
