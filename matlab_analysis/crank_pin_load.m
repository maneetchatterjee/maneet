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


% ===================== TABLE OUTPUT =====================

disp('Table: FPL vs Crank Angle');
disp('===================================');
disp(' Crank Angle (deg)  |  FPL (kN)');
disp('===================================');
for i = 1:length(theta)
    fprintf('       %6.0f        |  %8.2f\n', theta(i), Fpl_kN(i));
end
disp('===================================');


% ===================== AVERAGE LOAD =====================

Fpl_avg = mean(Fpl);   % Average resultant load (N)


% ===================== PLOTTING =====================

figure('visible', 'off');
plot(theta, Fpl_kN, '-o', 'LineWidth', 1.5)
grid on

xlabel('Crank Angle (degrees)')
ylabel('Resultant Crank Pin Load, F_{PL} (kN)')
title('F_{PL} vs Crank Angle')

% Save the plot
print('fpl_plot.png', '-dpng', '-r300');


% ===================== PRINT RESULT =====================

fprintf( ...
    'Average Resultant Crank Pin Load F_PL(avg) = %.2f kN\n', ...
    Fpl_avg / 1000 ...
);
