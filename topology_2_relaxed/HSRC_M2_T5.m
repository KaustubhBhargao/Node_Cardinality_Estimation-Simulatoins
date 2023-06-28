%%% HSRC-M2
clc
clear
%% Network Design Variables 
M = 8; % stops by MBS
mbs_stops = [1.75, 0.75; 1.25, 0.75; 0.75, 0.75; 0.25, 0.75; 0.25, 0.25; 0.75, 0.25; 1.25, 0.25; 1.75, 0.25];
ncc = 5; % (alpha,C):{1,2}; (C,beta):{3,4); (C,C):{1,2,3,4}; (beta,C):{4,5}; (C,alpha):{2,5}
b = 6; % slot width
len = ceil(log2(ncc)); % length of BP
load M2T2.txt %Load values from 2 types of nodes' result
load M2T3.txt %Load values from 3 types of nodes' result
% load M2T2_vs_D.txt
% load M2T3_vs_D.txt
% load M2T2_vs_eps.txt
% load M2T3_vs_eps.txt
% load M2T2_vs_q.txt
% load M2T3_vs_q.txt
% T_mat = [3,4,5,6,7,8];
% Prop_Method_T = zeros(6);
% Prop_Method_T_Trep = zeros(6);
% for T_ind=1:6
T = 5; % types of nodes
% T = T_mat(T_ind);
D = 1000; % total nodes per type
% q = 0.1:0.1:0.5; % probability that node is active
q = 0.3; % qb = probability with which a node is active. For simplicity,
         % qb = q for all b in {1, T}
n_all = T * D; % total manufactured nodes
t = ceil(log2(n_all));
% t = floor(log2(n_all)); % maximum hash value (time slots), also the lenght
                        % of every trial in phase-1 of SRC_M scheme
Hmax = ceil(log2(max(n_all))); % Maximum Hash Value.
%% User specified accuracy variables
eps = 0.03; % desired relative error bound
delta = 0.2; % desired error probability
W = 30; % trials for Phase 1
% each block divided into T-1 slots

% l = 28321;  % number of blocks in phase-2, corresponding to eps = 0.01
% l = 6775;   % corresponding to eps = 0.02
l = 3087;   % corresponding to eps = 0.03
% l = 1788;   % corresponding to eps = 0.04
% l = 1116;   % corresponding to eps = 0.05
% l = 1; %test only
fprintf('T = %d ,epislon = %f, delta = %f, l = %d, W = %d\n', T, eps, delta, l, W);
%% Even more variables, take values from literature
% slot_width = ;
% bit_rate = ;

%% Network Model
x_pos =2*rand(T, D); % random x coordinates, in range (0,1)
y_pos = rand(T, D); % random y coordinates, in range (0,1)
%for visualising ->
% x_pos_vec = reshape(x_pos.', 1, []);
% y_pos_vec = reshape(y_pos.', 1, []);
% figure; scatter(x_pos_vec, y_pos_vec, color="r")
% active = zeros(T, D); % matrix of active nodes
% active = Active_Nodes(T, D, q, active); % creating active nodes

%% Plotting Slots vs D
% Prop_Method_D = zeros(1,5);
% Prop_Method_D_Trep = zeros(1,5);
% D_mat = [1000, 2000, 3000, 4000, 5000];
% for d=1:5
% D=D_mat(d);
% fprintf('D = %d\n', D);
% n_all = T * D; % total manufactured nodes
% t = floor(log2(n_all)); % maximum hash value (time slots), also the lenght
%                         % of every trial in phase-1 of SRC_M scheme
% x_pos = 2*rand(T, D); % random x coordinates, in range (0,2)
% y_pos = rand(T, D); % random y coordinates, in range (0,1)
%% plotting slots vs D ends here

%% Plotting slots vs eps
% eps_mat = [0.01, 0.02, 0.03, 0.04, 0.05];
% l_mat = [28321, 6775, 3087, 1788, 1116];
% Prop_Method_eps = zeros(1,5);
% Prop_Method_eps_Trep = zeros(1,5);
% for eps_ind=1:5
% eps = eps_mat(eps_ind);
% fprintf('eps = %f\n', eps);
% l = l_mat(eps_ind);
%% Plotting slots vs eps ends above

%% Plotting slots vs q
% q_mat = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
% Prop_Method_q = zeros(1,7);
% Prop_Method_q_Trep = zeros(1,7);
% for q_ind=1:7
%     q = q_mat(q_ind);
%     fprintf('q = %f\n', q);
% %     S_verify = 0;
% %     Y_verify = 0;
%     %q_max = 0.5;
%% Plotting slots vs q ends above
max_iter = 100;
Prop_Method_iter = zeros(1,max_iter);
for iter=1:max_iter
    n_tild = zeros(M, T); % rough estimates, updates every phase
    n_hat = zeros(M, T);
    n_hat_sum = zeros(1, T);
    Prop_Method = zeros(1,M); % time slots needed at each stop of the MBS
    temp = zeros(l, T, M); % block by type of node (as used by Dr Sachin)
    X_bit_pattern = zeros(l, T, M); % bit patterns, cumulative blocks, for calculating n_hat
    X_bit_pattern_1 = zeros(l, T, M); % cumulative bit patterns, cumulative blocks, for calculating n_hat
    z = zeros(M,T); % number of zeros in phase 2, for calculating n_hat
    %% Phase 1 of HSRC-M1
    p = zeros(T,M); % used in phase 2, step 1, p(b,m)
    I = zeros(T,M); % used in phase 2, step 1, I(b,m)
    % v = zeros(1,W); % for every trial w
    % Y = zeros(M, W, t); % Cumulative Bit vector
    % S = zeros(M, W, t); % the bit vector of lenght t
    %% Main code here
    active = zeros(T, D); % matrix of active nodes
    active = Active_Nodes(T, D, q, active); % creating active nodes
    %fprintf('no. of active nodes for q = %d is: %d\n',q, sum(sum(active)));
    %disp(sum(sum(active)));
    Y = zeros(W, t, M, T); % Cumulative Bit vector
    S = zeros(W, t, M, T); % the bit vector of lenght t
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
                        % try changing the range of MBS
                        % do nothing, node is active and in range
                    else 
                        % node is active but out of range
                        active_inrange(b,act_nd)=0;
                    end
                end
            end
        end
%         fprintf('no. of active nodes (in range) for q = %d is: %d\n',q, sum(sum(active_inrange)));
%             x_pos_inrange = x_pos.*active_inrange;
%             y_pos_inrange = y_pos.*active_inrange;
%             x_pos_inrange_vec = reshape(x_pos_inrange.', 1, []);
%             y_pos_inrange_vec = reshape(y_pos_inrange.', 1, []);
%             figure; 
%             axis([0 1 0 1]);
%             scatter(x_pos_inrange_vec, y_pos_inrange_vec);
%             hold on
%             scatter(mbs_stops(m,1),mbs_stops(m,2));
        %% Phase 1 of HSRC-M2 begins here
        for b=1:T % (for every type of node) repeat SRC_M protcol T times
%             Y = zeros(M, W, t); % Cumulative Bit vector
%             S = zeros(M, W, t); % the bit vector of lenght t
            v = zeros(1,W); % for every trial w
            for w=1:W % for every trial
                for act_nd=1:D % for every active node of this type
                    if active_inrange(b, act_nd) == 0
                        % do nothing, node is inactive
                    else % the node is active
                        tx = geornd(0.5); % retuns number of failure before a success
                        tx = tx + 1; % tx is now the slot in which the current node transmits
                        if tx >= t
                            tx = t;
                        end
                        %disp(tx);
                        S(w, tx, m, b) = 1; % set the corresponding bit to '1' 
                    end
                end
                % for this trial, S is know, we can evaluate Y
                if m==1
                    Y(w, :, m, b) = S(w, :, m, b);
                else
                    Y(w, :, m, b) = bitor(Y(w, :, m-1, b), S(w, :, m, b));
                end
                % bit vector Y for trial w, for node type b, at stop m is known
                % we can now find the largest interger v for this trial w.
    
                % search Yw(m) at bit positions that are powers of 2
                % find the power that gives first 0 bit
                % then search in that decade (2^(j-1) to 2^j) for max slot
                % with a bit 1, that will be our vw(m)
                j_dash = 0;
                for j=1:log2(t)
                    if Y(w, 2^(j-1), m, b)==0
                        j_dash = j;
                        %disp(j);
                        break;
                    end
                end
                % we now have the value j'=j where we encounter first 0 bit
                % now perform binary search over (2^(j-2) to 2^(j-1)), to find the
                % integer largest v s.t. Y(v) = 1
                if(j_dash==0) % no slot found with 0 bit
                    v(w) = t;
                else
                    % Permuting the Y matix
                    %Y_vec = permute(Y(m, w, (2^(j_dash-2):2^(j_dash-1)-1)), [2 3 1]);
                    %Y_vec = permute(Y(w, (2^(j_dash-2):2^(j_dash-1)-1), m, b), [2 3 1]);
                    %Y_vec = zeros(2^(j_dash-1));
                    Y_vec = zeros(1,t);
                    %Y_vec = Y(w, (2^(j_dash-2):2^(j_dash-1)-1), m, b);
                    Y_vec = Y(w, :, m, b);
                    % Y_vec is now a row vector
                    
                    %v(w) = Binary_Search(Y_vec, 2^(j_dash-2), 2^(j_dash-1)-1);
                    v(w) = Binary_Search(Y_vec, 2^(j_dash-2), 2^(j_dash-1)-1);
                    % Binary_Search(vector, start index, end index)
                    %disp(m)
                end 
            end % trial w ends here
            % calculating rough estimates for each type of node,at stop m
            %disp(sum(v));
            n_tild(m,b) = 0.794 * 2^(sum(v)/W);
%             S_verify = S_verify + sum(sum(sum(S)));
%             Y_verify = Y_verify + sum(sum(sum(Y)));
%             Y_vec_verify = Y_verify + sum(sum(sum(Y_vec)));
        end % end SRC_M for node type b
        Prop_Method(m) = T*W*t; %slots used in phase-1 at stop m
    
        %% Phase 1 ends here, and Phase 2 of HSRC-M1 begins
        % use n_tild and find better approximations
        % use earlier results (recursion)
        % case by case for each type of nodes
        % update Prop_Method(m)

        for b=1:T % for every type of node, we find p(b,m) I(b,m)
            p(b,m) = min(1,1.6*l/n_tild(m,b));
            %p(b,m) = min(1,0.02*l/n_tild(m,b)); %PROBLEMMMM!?
            j = 1;
            %I(b,m) = abs(2^(-j) - p(b,m)); % initial value of I(b,m)
            approx_init = abs(2^(-j) - p(b,m));
            while 1
                % loop and find a minimum j that works
                j = j+1;
                approx_new = abs(2^(-j) - p(b,m));
                if (approx_new < approx_init)
                    % do nothing, continue
                    approx_init = approx_new;
                else
                    I(b,m) = j-1;
                    break;
                end
            end
            %Next: choosing block uniformly randomly and transmitting in that block
            %with the calculated probability of 2^(-I(b,m))/l, Note that this is
            %done for each active node of each type. Once done, we go case by case
            %(depending on T) to check for collisions. Please refer to Dr Sachin's
            %simulation codes
    
            % and now transmitting in that block
            % p_block_tx = random('bino',1,2^(-I(b,m))/l);
            for i2=1:D % every active node of that type
                h = randi(l,1); % block chosen by this node, randomly
%                 temp(h,b,m) = temp(h,b,m) + active_inrange(b,i2)*random('bino',1,2^(-I(b,m)/l));
                temp(h,b,m) = temp(h,b,m) + active_inrange(b,i2)*random('bino',1,2^(-I(b,m)));
            end  
        end

        R1=0; % 1 extra slot required
        R2=0; % all C case
        for ind=1:l
            switch 1
                case ((temp(ind,1,m)==0) && (temp(ind,2,m)==1) && (temp(ind,3,m)>=1) && (temp(ind,4,m)==0) && (temp(ind,5,m)==0)) % (alpha,C) case
                    R1=R1+1;
                case ((temp(ind,1,m)==1) && (temp(ind,2,m)==0) && (temp(ind,3,m)>=2) && (temp(ind,4,m)==0) && (temp(ind,5,m)==0)) % (alpha,C) case
                    R1=R1+1;
                case ((temp(ind,1,m)>=1) && (temp(ind,2,m)==1) && (temp(ind,3,m)==0) && (temp(ind,4,m)==0) && (temp(ind,5,m)==0)) % (C,alpha) case
                    R1=R1+1;
                case ((temp(ind,1,m)>=1) && (temp(ind,2,m)==0) && (temp(ind,3,m)==0) && (temp(ind,4,m)==0) && (temp(ind,5,m)==1)) % (C,alpha) case
                    R1=R1+1;
                case ((temp(ind,1,m)>=1) && (temp(ind,2,m)==0) && (temp(ind,3,m)==0) && (temp(ind,4,m)==1) && (temp(ind,5,m)==0)) % (C, beta) case
                    R1=R1+1;
                case ((temp(ind,1,m)>=2) && (temp(ind,2,m)==0) && (temp(ind,3,m)==1) && (temp(ind,4,m)==0) && (temp(ind,5,m)==0)) % (C, beta) case
                    R1=R1+1;
                case ((temp(ind,1,m)==0) && (temp(ind,2,m)==0) && (temp(ind,3,m)>=1) && (temp(ind,4,m)==1) && (temp(ind,5,m)==0)) % (beta,C) case
                    R1=R1+1;
                case ((temp(ind,1,m)==0) && (temp(ind,2,m)==0) && (temp(ind,3,m)>=1) && (temp(ind,4,m)==0) && (temp(ind,5,m)==1)) % (beta,C) case
                    R1=R1+1;

                % none of no_zero_symbol_combination_type tx
                case ((temp(ind,1,m)>=2) && (temp(ind,2,m)==0) && (temp(ind,3,m)>=2) && (temp(ind,4,m)==0) && (temp(ind,5,m)==0)) % (C,C) case
                    R2=R2+1;

                % any one of no_zero_symbol_combination_type tx
                %Then nodes 1 and 3 must also tx for (C,C) to occus
                case ((temp(ind,1,m)>=1) && (temp(ind,3,m)>=1) && (temp(ind,2,m)==1)) % (C,C) case
                    R2=R2+1;
                case ((temp(ind,1,m)>=1) && (temp(ind,3,m)>=1) && (temp(ind,4,m)==1)) % (C,C) case
                    R2=R2+1;
                case ((temp(ind,1,m)>=1) && (temp(ind,3,m)>=1) && (temp(ind,5,m)==1)) % (C,C) case
                    R2=R2+1;

                % any two of no_zero_symbol_combination_type tx
                case ((temp(ind,2,m)==1) && (temp(ind,4,m)==1)) % (C,C) case
                    R2=R2+1;
                case ((temp(ind,2,m)==1) && (temp(ind,5,m)==1)) % (C,C) case
                    R2=R2+1;
                case ((temp(ind,5,m)==1) && (temp(ind,4,m)==1)) % (C,C) case
                    R2=R2+1;

                % all three of no_zero_symbol_combination_type tx
                case ((temp(ind,2,m)==1) && (temp(ind,4,m)==1) && (temp(ind,5,m)==1)) % (C,C) case
                    R2=R2+1;

                % multiple nodes of any one of no_zero_symbol_combination_type tx 
                case ((temp(ind,2,m)>=2)) % (C,C) case
                    R2=R2+1;
                case ((temp(ind,4,m)>=2)) % (C,C) case
                    R2=R2+1;
                case ((temp(ind,5,m)>=2)) % (C,C) case
                    R2=R2+1;
          
            end
            for b=1:T % finding the bit pattern
                if temp(ind,b,m)>=1
                    X_bit_pattern(ind,b,m) = 1;
                end
            end
        end

        %calculate cumulative bit pattern here::
        for b=1:T
            if (m==1)
                X_bit_pattern_1(:,b,m) = X_bit_pattern(:,b,m);
            else
                X_bit_pattern_1(:,b,m) = bitor(X_bit_pattern(:,b,m), X_bit_pattern(:,b,m-1));
            end
            z(m,b) = nnz(~X_bit_pattern_1(:,b,m));
            %disp(X_bit_pattern_1(:,b,m));
            n_hat(m,b) = log(z(m,b)/l)/log(1-(p(b,m)/l));
        end

        %calculate n_hat for this m here::
        n_hat_sum = sum(n_hat);
         
        % updating Prop_method for m_th stop
%         Prop_Method(m) = Prop_Method(m) + (2)*l+ (len*l)/b + R1 + R2*((M2T2/l) + (M2T3/l)); % with BP lenght
        Prop_Method(m) = Prop_Method(m) + 2*l+ 1 + R1+ R2*((M2T2/l) + (M2T3/l)); % without BP lenght  
%         Prop_Method(m) = Prop_Method(m) + 2*l+ 1 + R1+ R2*((M2T2_vs_eps(eps_ind)/l) + (M2T3_vs_eps(eps_ind)/l));
    end % end loop for m
    Prop_Method_iter(iter) = sum(Prop_Method);
%     fprintf('iteration: %d\n', iter);
    fprintf('slots using HSRC-M2 for iter = %d are : %d\n', iter, Prop_Method_iter(iter));
%     disp(iter);
end % end iteration
%%
avg_slots = mean(Prop_Method_iter);
fprintf('avg slots with HSRC-M2 = %f\n', mean(Prop_Method_iter));
fprintf('slots using T-rep SRCM = %d\n', M*T*(W*t + l))
% disp(mean(Prop_Method_iter));
% disp(M*T*(W*t + l));
%% Plotting Slots vs D
% Prop_Method_D(d) = mean(Prop_Method_iter);
% Prop_Method_D_Trep(d) = M*T*(W*t + l);
% end % end D loop
% figure;
% % plot(D_mat, Prop_Method_D, '--', 'MarkerSize', 2, D_mat, Prop_Method_D_Trep, ':', 'MarkerSize', 2);
% % plot(D_mat, Prop_Method_D,'-*', D_mat, Prop_Method_D_Trep,'--*')%,x,u,':.k')
% plot(D_mat, Prop_Method_D, '-', 'Marker', '*')
% hold on
% plot(D_mat, Prop_Method_D_Trep,'--', 'Marker', '*', 'Color','red')
% set(gca,'FontSize',14,'FontName','Times New Roman')
% xlim([1000 5000])
% ylim([3e4 7e4])
% title(sprintf('Number of slots required vs D'),'FontSize',16)
% xlabel('D','FontSize',16)%Probability with which a node is active
% ylabel('Number of slots required','FontSize',16)
% sum1 = 'HSRC-M2-T5';
% sum2 = 'T Repetitions of SRC_M';
% Legend = cell(2,1);
% Legend{1,1} = (sprintf('%s',sum1));
% Legend{2,1} = (sprintf('%s', sum2));
% pause(0.1);
% legend(Legend,'FontSize',12)
%% Plotting slots vs T
% Prop_Method_T(T_ind) = mean(Prop_Method_iter);
% Prop_Method_T_Trep(T_ind) = M*T*(W*t + l);
% end
% figure;
% % plot(T_mat, Prop_Method_T, T_mat, Prop_Method_T_Trep);
% plot(T_mat, Prop_Method_T, '-', 'Marker','*')
% hold on
% plot(T_mat, Prop_Method_T_Trep,'--', 'Marker','*')
% set(gca,'FontSize',14,'FontName','Times New Roman')
% % xlim([1000 5000])
% ylim([0 13e4])
% title(sprintf('Number of slots required vs T'),'FontSize',16)
% xlabel('T','FontSize',16)%Probability with which a node is active
% ylabel('Number of slots required','FontSize',16)
% sum1 = 'HSRC-M1';
% sum2 = 'T Repetitions of SRC_M';
% Legend = cell(2,1);
% Legend{1,1} = (sprintf('%s',sum1));
% Legend{2,1} = (sprintf('%s',sum2));
% pause(0.1);
% legend(Legend,'FontSize',12)
% % legend([h1, h2(1)], 'HSRC-M1', 'T Repetitions of SRC_M')
%% Plotting slots vs eps
% Prop_Method_eps(eps_ind) = mean(Prop_Method_iter);
% Prop_Method_eps_Trep(eps_ind) = M*T*(W*t + l);
% end %end eps loop
% figure;
% plot(eps_mat, Prop_Method_eps, '-', 'Marker', '*')
% hold on
% plot(eps_mat, Prop_Method_eps_Trep,'--', 'Marker', '*', 'Color','red')
% set(gca,'FontSize',14,'FontName','Times New Roman')
% % xlim([1000 5000])
% % ylim([3e4 7e4])
% title(sprintf('Number of slots required vs epsilon'),'FontSize',16)
% xlabel('epsilon','FontSize',16)%Probability with which a node is active
% ylabel('Number of slots required','FontSize',16)
% sum1 = 'HSRC-M2-T5';
% sum2 = 'T Repetitions of SRC_M';
% Legend = cell(2,1);
% Legend{1,1} = (sprintf('%s',sum1));
% Legend{2,1} = (sprintf('%s', sum2));
% pause(0.1);
% legend(Legend,'FontSize',12)
 %% Plotting slots vs q
% Prop_Method_q(q_ind) = mean(Prop_Method_iter);
% Prop_Method_q_Trep(q_ind) = M*T*(W*t + l);
% % fprintf('avg. S sum for q = %f is: %d\n', q, S_verify/T);
% % fprintf('avg. Y sum for q = %f is: %f\n', q, Y_verify/T);
% end %end q loop
% figure;
% plot(q_mat, Prop_Method_q, '-', 'Marker', '*')
% hold on
% plot(q_mat, Prop_Method_q_Trep,'--', 'Marker', '*', 'Color','red')
% set(gca,'FontSize',14,'FontName','Times New Roman')
% % xlim([1000 5000])
% % ylim([3e4 7e4])
% title(sprintf('Number of slots required vs q'),'FontSize',16)
% xlabel('q','FontSize',16)%Probability with which a node is active
% ylabel('Number of slots required','FontSize',16)
% sum1 = 'HSRC-M2-T5';
% sum2 = 'T Repetitions of SRC_M';
% Legend = cell(2,1);
% Legend{1,1} = (sprintf('%s',sum1));
% Legend{2,1} = (sprintf('%s', sum2));
% pause(0.1);
% legend(Legend,'FontSize',12)

% save M2T5.txt avg_slots -ascii %To save the values, which can be used in 4dev, 6dev etc.
% save M2T5_vs_D.txt Prop_Method_D -ascii
% save M2T5_vs_eps.txt Prop_Method_eps -ascii
% save M2T5_vs_q.txt Prop_Method_q -ascii