%% HSRC-M2

%% Network Design Variables 
M = 4; % stops by MBS
mbs_stops = [0.75, 0.75; 0.25, 0.75; 0.25, 0.25; 0.75, 0.25];
T = 4; % types of nodes
D = 300; % total nodes per type
% q = 0.1:0.1:0.5; % probability that node is active
% q = 0.3; % qb = probability with which a node is active. For simplicity,
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
% x_pos_vec = reshape(x_pos.', 1, []);
% y_pos_vec = reshape(y_pos.', 1, []);
% figure; scatter(x_pos_vec, y_pos_vec, color="r")
% active = zeros(T, D); % matrix of active nodes
% active = Active_Nodes(T, D, q, active); % creating active nodes
n_tild = zeros(M, T); % rough estimates, updates every phase
n_hat = zeros(M, T);
Prop_Method = zeros(1,M); % time slots needed at each stop of the MBS
q_max = 0.5;
Prop_Method_q = zeros(1,5);
max_iter = 10;
Prop_Method_iter = zeros(1,max_iter);
temp = zeros(l, T, M); % block by type of node (as used by Dr Sachin)
X_bit_pattern = zeros(l, T, M); % bit patterns, cumulative blocks, for calculating n_hat
X_bit_pattern_1 = zeros(l, T, M); % cumulative bit patterns, cumulative blocks, for calculating n_hat
z = zeros(M,T); % number of zeros in phase 2, for calculating n_hat
%% Code starts here

% TODO: number of iterations?
% TODO: plot slots vs q,a nd other plots as done in earlier paper
% TODO: overhead in computation, binary search etc.? Skipped for now
% TODO: can reduce space complexity by resusing v. YES! DONE!
% TODO: finding n_hat. Getting terrible estimates
% TODO: Account for the location of MBS stops! Which nodes will respond in
% each stop?

%% Phase 1 of HSRC-M2
p = zeros(T,M); % used in phase 2, step 1, p(b,m)
I = zeros(T,M); % used in phase 2, step 1, I(b,m)
% v = zeros(1,W); % for every trial w
% Y = zeros(M, W, t); % Cumulative Bit vector
% S = zeros(M, W, t); % the bit vector of lenght t

for iter=1:max_iter
%     for q_ind=1:5
%         q = q_ind/10;
        q = 0.3;
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
                Y = zeros(M, W, t); % Cumulative Bit vector
                S = zeros(M, W, t); % the bit vector of lenght t
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
                            S(m, w, tx) = 1; % set the corresponding bit to '1' 
                        end
                    end
                    % for this trial, S is know, we can evaluate Y
                    if m==1
                        Y(m, w, :) = S(m, w, :);
                    else
                        Y(m, w, :) = bitor(Y(m-1, w, :), S(m, w, :));
                    end
                    % bit vector Y for trial w, for node type b, at stop m is known
                    % we can now find the largest interger v for this trial w.
        
                    % search Yw(m) at bit positions that are powers of 2
                    % find the power that gives first 0 bit
                    % then search in that decade (2^(j-1) to 2^j) for max slot
                    % with a bit 1, that will be our vw(m)
                    j_dash = 0;
                    for j=1:log2(t)
                        if Y(m, w, 2^(j-1))==0
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
                        Y_vec = permute(Y(m, w, (2^(j_dash-2):2^(j_dash-1)-1)), [2 3 1]);
                        % Y_vec is now a row vector
                        
                        v(w) = Binary_Search(Y_vec, 2^(j_dash-2), 2^(j_dash-1)-1);
                        % Binary_Search(vector, start index, end index)
                        %disp(m)
                    end 
                end % trial w ends here
                % calculating rough estimates for each type of node,at stop m
                %disp(sum(v));
                n_tild(m,b) = 0.794 * 2^(sum(v)/W);
            end % end SRC_M for node type b
            Prop_Method(m) = T*W*t;
        
            %% Phase 1 ends here, and Phase 2 of HSRC-M2 begins
            
            



            %calculate cumulative bit pattern here::
            for b=1:T
                if (m==1)
                    X_bit_pattern_1(:,b,m) = X_bit_pattern(:,b,m);
                else
                    X_bit_pattern_1(:,b,m) = bitor(X_bit_pattern(:,b,m),X_bit_pattern(:,b,m-1));
                end
                z(m,b) = nnz(~X_bit_pattern_1(:,b:m));
                n_hat(m,b) = log(z(m,b)/l)/log(1-(p(b,m)/l));
            end

            %calculate n_hat for this m here::
        
            
            Prop_Method(m) = Prop_Method(m) + (T-1)*l+1+K+1+(T-1)*R; 
       
        end % end loop for m
%         fprintf('For q = %f, Slots with HSRC-M1 = %f\n', q, sum(Prop_Method));
%         fprintf('Slots with T-rep SRC_M = %f\n', M*T*(W*t + l));
%         Prop_Method_q(q_ind) = sum(Prop_Method);
%         % disp(sum(Prop_Method));
%         % disp(M*T*(W*t + l));
%     end % end loop for q
    Prop_Method_iter(iter) = sum(Prop_Method);
    %disp(iter);
end % end iteration
disp(mean(Prop_Method_iter));
disp(M*T*(W*t + l));



