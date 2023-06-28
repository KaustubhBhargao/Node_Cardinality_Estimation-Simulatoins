% y = zeros(10000,1);
% l = 28321;  % number of blocks in step-1, corresponding to eps = 0.01
% for i=1:100
%     y(i)=random('bino',1,2^(-1)/l);
% end
% disp(sum(y));
%% test -2
% 
% M = 4; % stops by MBS
% mbs_stops = [0.75, 0.75; 0.25, 0.75; 0.25, 0.25; 0.75, 0.25];
% T = 3; % types of nodes
% D = 300; % total nodes per type
% % q = 0.1:0.1:0.5; % probability that node is active
% % q = 0.3; % qb = probability with which a node is active. For simplicity,
%          % qb = q for all b in {1, T}
% n_all = T * D; % total manufactured nodes
% t = floor(log2(n_all)); % maximum hash value (time slots), also the lenght
%                         % of every trial in phase-1 of SRC_M scheme
% x_pos = rand(T, D); % random x coordinates, in range (0,1)
% y_pos = rand(T, D); % random y coordinates, in range (0,1)
% active = zeros(T, D); % matrix of active nodes
% active = Active_Nodes(T, D, q, active); % creating active nodes
% 
% 
% for m=1:M
%     active_inrange = active;
%     for b=1:T
%         for act_nd=1:D
%             if active(b, act_nd) == 0
%               % do nothing, node is inactive   
%             else %node is active
%                 if sqrt((mbs_stops(m,1)-x_pos(b,act_nd))^2 + (mbs_stops(m,2)-y_pos(b,act_nd))^2)<=pi/4
%                     % do nothing, node is active and in range
%                 else 
%                     % node is active but out of range
%                     active_inrange(b,act_nd)=0;
%                 end
%             end
%         end
%     end
%     x_pos_inrange = x_pos.*active_inrange;
%     y_pos_inrange = y_pos.*active_inrange;
%     x_pos_inrange_vec = reshape(x_pos_inrange.', 1, []);
%     y_pos_inrange_vec = reshape(y_pos_inrange.', 1, []);
%     figure; 
%     axis([0 1 0 1]);
%     scatter(x_pos_inrange_vec, y_pos_inrange_vec);
%     hold on
%     scatter(mbs_stops(m,1),mbs_stops(m,2));
% end

%% test for in-range plotting
%% HSRC-M1

%% Network Design Variables 
M = 4; % stops by MBS
mbs_stops = [0.75, 0.75; 0.25, 0.75; 0.25, 0.25; 0.75, 0.25];
% T_mat = [3,4,5,6,7,8];
% Prop_Method_T = zeros(6);
% Prop_Method_T_Trep = zeros(6);
% for T_ind=1:6
T = 4; % types of nodes
% T = T_mat(T_ind);
D = 300; % total nodes per type
% q = 0.1:0.1:0.5; % probability that node is active
q = 0.3; % qb = probability with which a node is active. For simplicity,
         % qb = q for all b in {1, T}
n_all = T * D; % total manufactured nodes
t = floor(log2(n_all)); % maximum hash value (time slots), also the lenght
                        % of every trial in phase-1 of SRC_M scheme
%% User specified accuracy variables
eps = 0.03; % desired relative error bound
delta = 0.2; % desired error probability
W = 30; % trials for Phase 1
% each block divided into T-1 slots

% l = 28321;  % number of blocks in step-1, corresponding to eps = 0.01
% l = 6775;   % corresponding to eps = 0.02
l = 3087;   % corresponding to eps = 0.03
% l = 1788;   % corresponding to eps = 0.04
% l = 1116;   % corresponding to eps = 0.05
%% Even more variables, take values from literature
% slot_width = ;
% bit_rate = ;

%% Network Model
x_pos = rand(T, D); % random x coordinates, in range (0,1)
y_pos = rand(T, D); % random y coordinates, in range (0,1)
%for visualising ->
x_pos_vec = reshape(x_pos.', 1, []);
y_pos_vec = reshape(y_pos.', 1, []);
% figure; scatter(x_pos_vec, y_pos_vec, color="r")
figure;
scatter(x_pos(1,:) , y_pos(1,:), 'Marker','o', 'LineWidth', 1)
hold on
scatter(x_pos(2,:) , y_pos(2,:), 'Marker','square', 'LineWidth', 1)
scatter(x_pos(3,:) , y_pos(3,:), 'Marker','x', 'LineWidth', 1)
scatter(x_pos(4,:) , y_pos(4,:), 'Marker','diamond', 'LineWidth', 1)
% for iter=1:max_iter
%         q = 0.6;
        active = zeros(T, D); % matrix of active nodes
        active = Active_Nodes(T, D, q, active); % creating active nodes
        %disp(sum(sum(active)));
        for m=1:M % for every stop, execute both phase 1 and phase two
            %nodes in range of the MBS: consider network model as described
            %in the simulation section of the SPCOM paper.
            %range of MBS is a circle with radius = p1/4
            %for every active node, find its distance from the current MBS
            %stop
            active_inrange = active;
            for b=1:T
                for act_nd=1:D
                    if active(b, act_nd) == 0
                      % do nothing, node is inactive   
                    else %node is active
                        if sqrt((mbs_stops(m,1)-x_pos(b,act_nd))^2 + (mbs_stops(m,2)-y_pos(b,act_nd))^2)<=pi/4
                            % do nothing, node is active and in range
                        else 
                            % node is active but out of range
                            active_inrange(b,act_nd)=0;
                        end
                    end
                end
            end
            x_pos_inrange = x_pos.*active_inrange;
            y_pos_inrange = y_pos.*active_inrange;
            x_pos_inrange_vec = reshape(x_pos_inrange.', 1, []);
            y_pos_inrange_vec = reshape(y_pos_inrange.', 1, []);
            figure; 
            axis([0 1 0 1]);
            %scatter(x_pos_inrange_vec, y_pos_inrange_vec);
            scatter(active_inrange(1,:).*x_pos(1,:) , active_inrange(1,:).*y_pos(1,:), 'Marker','o', 'LineWidth', 1)
            hold on
            scatter(active_inrange(2,:).*x_pos(2,:) , active_inrange(2,:).*y_pos(2,:), 'Marker','square', 'LineWidth', 1)
            scatter(active_inrange(3,:).*x_pos(3,:) , active_inrange(3,:).*y_pos(3,:), 'Marker','x', 'LineWidth', 1)
            scatter(active_inrange(4,:).*x_pos(4,:) , active_inrange(4,:).*y_pos(4,:), 'Marker','diamond', 'LineWidth', 1)
            hold on
            scatter(mbs_stops(m,1),mbs_stops(m,2),'Marker', 'pentagram', 'MarkerEdgeColor',[0.1 1 .5], 'MarkerFaceColor',[1 0 0], 'LineWidth', 4);
        end
% end