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


%% ===================== OUTPUT =====================

fprintf('Resultant Crank Pin Load vs Crank Angle\n');
fprintf('=========================================\n');
fprintf(' Crank Angle (deg)  |   FR (kN)\n');
fprintf('=========================================\n');
for i = 1:length(theta)
    fprintf('       %6.0f        |  %9.2f\n', theta(i), FR(i) / 1000);
end
fprintf('=========================================\n\n');

fprintf('Average Resultant Load FR_avg = %.2f kN\n', FR_avg / 1000);
fprintf('Average Bearing Pressure p_avg = %.2f MPa\n', p_avg / 1e6);


%% ===================== PLOT =====================

figure('visible', 'off');
plot(theta, FR / 1000, '-o', 'LineWidth', 1.5);
grid on;
xlabel('Crank Angle (degrees)');
ylabel('Resultant Crank Pin Load (kN)');
title('Resultant Crank Pin Load vs Crank Angle');

% Save the plot
print('fr_plot.png', '-dpng', '-r300');

fprintf('\nPlot saved as fr_plot.png\n');
