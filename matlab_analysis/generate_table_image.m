% ===================== INPUT DATA =====================

theta = [ ...
    0 20 40 60 80 100 120 140 160 180 200 220 240 270 300 ...
    330 360 390 420 450 480 510 540 570 600 630 660 690 720 ];

Fg = [ ...
    65 85 66 24 13 9 7.5 4.5 2.1 0.25 0.25 0.25 0.25 ...
    0.25 0.25 0.25 0.25 -0.10 -0.10 -0.10 -0.10 ...
    -0.10 -0.10 -0.01 6.2 2.6 6.0 30 65 ] * 1e3;


% ===================== GIVEN CONSTANTS =====================

m_big = 3.534;        % Big end mass (kg)
a_big = 2229.67;      % Big end acceleration (m/s^2)


% ===================== CALCULATIONS =====================

theta_rad = deg2rad(theta);

phi = asin( sin(theta_rad) / 4 );

Q = 1278.79 * ...
    ( cos(theta_rad) + cos(2 * theta_rad) / 4 + Fg ) ...
    ./ cos(phi);

Fcr = m_big * a_big;

Fpl = sqrt( ...
    Q.^2 + Fcr^2 - 2 .* Q .* Fcr .* cos(theta_rad + phi) ...
);

Fpl_kN = Fpl / 1000;


% ===================== AVERAGE LOAD =====================

Fpl_avg = mean(Fpl);   % Average resultant load (N)
Fpl_avg_kN = Fpl_avg / 1000;


% ===================== CREATE TABLE IMAGE =====================

% Create figure for table
fig = figure('visible', 'off', 'Position', [100, 100, 600, 1000]);
axis off;

% Title
text(0.5, 0.98, 'Crank Pin Load Analysis - Complete Data Table', ...
    'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold', ...
    'Units', 'normalized');

% Column headers
y_pos = 0.95;
text(0.25, y_pos, 'Crank Angle (deg)', 'HorizontalAlignment', 'center', ...
    'FontSize', 11, 'FontWeight', 'bold', 'Units', 'normalized');
text(0.75, y_pos, 'F_{PL} (kN)', 'HorizontalAlignment', 'center', ...
    'FontSize', 11, 'FontWeight', 'bold', 'Units', 'normalized');

% Draw header line
line([0.1, 0.9], [y_pos-0.01, y_pos-0.01], 'Color', 'k', 'LineWidth', 1.5);

% Table data
y_pos = y_pos - 0.025;
for i = 1:length(theta)
    y_pos = y_pos - 0.027;
    text(0.25, y_pos, sprintf('%d', theta(i)), 'HorizontalAlignment', 'center', ...
        'FontSize', 10, 'Units', 'normalized');
    text(0.75, y_pos, sprintf('%.2f', Fpl_kN(i)), 'HorizontalAlignment', 'center', ...
        'FontSize', 10, 'Units', 'normalized');
end

% Draw line before average
y_pos = y_pos - 0.015;
line([0.1, 0.9], [y_pos, y_pos], 'Color', 'k', 'LineWidth', 1.5);

% Average value
y_pos = y_pos - 0.03;
text(0.5, y_pos, sprintf('Average Resultant Crank Pin Load F_{PL}(avg) = %.2f kN', Fpl_avg_kN), ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', ...
    'Units', 'normalized', 'Color', [0.8 0 0]);

% Save the table image
print('fpl_table.png', '-dpng', '-r300');

fprintf('Table image saved as fpl_table.png\n');
fprintf('Average Resultant Crank Pin Load F_PL(avg) = %.2f kN\n', Fpl_avg_kN);
