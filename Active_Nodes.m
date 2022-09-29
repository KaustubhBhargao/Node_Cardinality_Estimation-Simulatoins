% Nodes generation

function [active] = Active_Nodes(T, N, q, active)
    for j=1:T % for every node type
        for i=1:N % for every node of that type
            if (random('bino',1,q)) % ~Bernoulli(q) distribution
                active(j, i) = active(j, i) + 1; % Node is marked as active, i.e. '1'
            end
        end
    end