clear;
clc;

% set(0, 'defaulttextinterpreter', 'Latex')
%% data here

T_xpoints = [2,3,4,5,6,7,8];
D_xpoints = [1000, 2000, 3000, 4000, 5000];
eps_xpoints = [0.01, 0.02, 0.03, 0.04, 0.05];
q_xpoints = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
R_xpoints = [pi/4, pi/5, pi/6, pi/7, pi/8];
R_xpoints_ticks = {'\pi/4', '\pi/5', '\pi/6', '\pi/7', '\pi/8'};

% OG topo different D and q
% for slots vs T
C2_M1_vs_T = [15295.3, 29748.5, 43784.8, 58402, 73277, 87557, 101810.4];
C2_M2_vs_T = [15295.4, 29958.06868, 32085.25682, 34496.91901, 48902.81494, 50984.38023, 65296.12323];
C2_Trep_vs_T = [27576, 42084, 56112, 70740, 85608, 99876, 114144];
% for slots vs D
C2_M1_vs_D_T5 = [58402, 59016.5, 59612, 59608, 59605.5];
C2_M2_vs_D_T5 = [34413.08227, 35520.8225, 35253.54136, 35751.83317, 36236.55754];
C2_Trep_vs_D_T5 = [70740, 71340, 71940, 71940, 71940];
% for slots vs eps
C2_M1_vs_eps_T5 = [462144, 117408, 58400.5, 37616, 26864];
C2_M2_vs_eps_T5 = [235908.9184, 64708.31127, 34562.91513, 23774.82403, 18136.55226];
C2_Trep_vs_eps_T5 = [575420, 144500, 70740, 44760, 31320];
% for slots vs q
C2_M1_vs_q_T5 = [58400.5, 58400.5, 58400, 58402, 58402, 58404.5, 58402];
C2_M2_vs_q_T5 = [33997.86424, 34460.30531, 34863.33347, 35353.90335, 36667.3515, 37473.65848, 38341.41039];
C2_Trep_vs_q_T5 = [70740, 70740, 70740, 70740, 70740, 70740, 70740];
% for slots vs Range (R)
C2_M1_vs_R_T5 = [58400.4, 58400.2, 58400.2, 58400.25, 58400];
C2_M2_vs_R_T5 = [34576.48081, 34182.29812, 34032.59269, 33904.30191, 33846.98268];
C2_Trep_vs_R_T5 = [70740, 70740, 70740, 70740, 70740];

% New topo, different D and q
% for slots vs T
C4_M1_vs_T = [30515.2, 59501.2, 87554.8, 116803, 146538.4, 175086, 203618.4];
C4_M2_vs_T = [30518.7, 59814.15357, 63834.03866, 68359.59499, 97686.57972, 101598.7647, 130225.6649];
C4_Trep_vs_T = [55152, 84168, 112224, 141480, 171216, 199752, 228288];
% for slots vs D (mean)
C4_M1_vs_D_T5 = [116800, 118005, 119216.5, 119211, 119219.5];
C4_M2_vs_D_T5 = [69070.1659, 70524.38928, 70498.72175, 71092.79329, 71790.10432];
C4_Trep_vs_D_T5 = [141480, 142680, 143880, 143880, 143880];
% for slots vs eps
C4_M1_vs_eps_T5 = [924289.5, 234825.5, 116807.5, 75233, 53728];
C4_M2_vs_eps_T5 = [471454.447, 127740.5466, 68262.37787, 47069.46373, 36059.87233];
C4_Trep_vs_eps_T5 = [1150840, 289000, 141480, 89520, 62640];
% for slots vs q (mean)
C4_M1_vs_q_T5 = [116802, 116803.5, 116806.5, 116814.5, 116816.5, 116836, 116843];
C4_M2_vs_q_T5 = [67757.6017, 68185.77323, 68858.10248, 69581.44008, 70712.41132, 71613.64148, 72929.31052];
C4_Trep_vs_q_T5 = [141480, 141480, 141480, 141480, 141480, 141480, 141480];
% for slots vs Range (R)
C4_M1_vs_R_T5 = [116804.5, 116801, 116801, 116800, 116801];
C4_M2_vs_R_T5 = [68359.59499, 67830.85151, 67659.55615, 67526.60184, 67519.64937];
C4_Trep_vs_R_T5 = [141480, 141480, 141480, 141480, 141480];

%% plotting begins
% p = 'test';
% p.Style = {Bold(true)};
plot(T_xpoints, C4_M1_vs_T,':.k', T_xpoints, C4_M2_vs_T,'--diamondk', T_xpoints, C4_Trep_vs_T,':*k');%,x,u,':.k');
set(gca,'FontSize',16,'FontName','Times New Roman');
title('Number of slots required vs T','FontName','Times New Roman', 'FontWeight','bold', 'FontSize', 14);
xlabel('T','FontSize',16);
ylabel('Number of slots required','FontSize',16);
% xticks([pi/8, pi/7, pi/6, pi/5, pi/4]);
% xticklabels({'\pi/8', '\pi/7', '\pi/6', '\pi/5', '\pi/4'});
% xlim([pi/8 pi/4]);
% ylim([60000 160000]);
sum1 = 'HSRC-M1';
sum2 = 'HSRC-M2';
sum3 = 'T Repetitions of SRC_M';
Legend = cell(3,1);
Legend{1,1} = (sprintf('%s', sum1));
Legend{2,1} = (sprintf('%s', sum2));
Legend{3,1} = (sprintf('%s', sum3));
legend(Legend,'FontSize',14, 'Location', 'Best')



