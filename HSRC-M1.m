%% HSRC-M1 Figure 5a

%% Variables 
M = 4; % stops by MBS
T = 4; % types of nodes
D = 300; % total nodes per type
%q = [0.1:0.1:0.5]; % probability that node is active
q = 0.3; % qb = probability with which a node is active. For simplicity,
         % qb = q for all b in {1, T}
W = 30; % trials for Phase 1

n_all = T * D; % total manufactured nodes
t = floor(log2(n_all)); % maximum hash value (time slots), also the lenght
                        % of every trial in phase-1 of SRC_M scheme

%% Network Model
x_pos = rand(T, D); % random x coordinates
y_pos = rand(T, D); % random y coordinates
active = zeros(T, D); % matrix of active nodes
active = Active_Nodes(T, D, q, active); % creating active nodes
n_tild = zeros(1, T); % rough estimates, updates every phase
n_hat = zeros(1, T);

%% Code starts here

% TODO: number of iterations?
% TODO: overhead in computation, slot choosing, binary search etc.?
% TODO: can reduce space complexity by resusing v

%% Phase 1 of HSRC-M1
% Execute the phase-1 of SRC_M protocol seperatly T times (for each 
% node type). Hence, a total of T*W*t slots are needed.
p = zeros(1, t); % the pmf of transmiting in a slot
for slot=1:t     % assigning the probability of tx in each slot
    if slot==t
        p(slot) = 2^(1-t);
    else
        p(slot) = 2^(-slot);
    end
end
%v = zeros(M, W); % for every trial w and stop m, we find the
% largest interger v in (2^(j'-2),...,2^(j'-1)) s.t. Y(v) is 1. 
% We search Y at bit posistions which are power of 2, and 
% let i=2^(j'-1) be the position we encounter the first 0 bit.
v = zeros(W); % for every trial w
Y = zeros(M, W, t); % Cumulative Bit vector
S = zeros(M, W, t); % the bit vector of lenght t

for m=1:M % for every stop
    %% Phase 1 of HSRC-M1 begins here
    for b=1:T % (for every type of node) repeat SRC_M protcol T times
        for w=1:W % for every trial
            for act_nd=1:D % for every active node of this type
                if active(b, act_nd) == 0
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
            for j=1:log2(t)
                if Y(m, w, 2^(j-1))==0
                    break;
                end
            end
            % we now have the value j'=j where we encounter first 0 bit
            % now perform binary search over (2^(j-2) to 2^(j-1)), to find the
            % integer largest v s.t. Y(v) = 1
            v(w) = Binary_Search(Y(m, w, (2^(j-2):2^(j-1))));
        end
        % calculating rough estimates for each type of node,at stop m
        n_tild(b) = 0.794 * 2^(sum(v) / W);
    end
    %% Phase 1 ends here, and Phase 2 of HSRC-M1 begins

end
