% Nodes generation

function [active] = Active_Nodes(T, D, q, active)
    for j=1:T % for every node type
        for i=1:D(1,j) % for every node of that type
            if (random('bino',1,q(1,j))) % ~Bernoulli(q) distribution
%                 fprintf('q = %f\n', q(1,j));
                active(j, i) = active(j, i) + 1; % Node is marked '1', i.e. "active"
            end
        end
    end