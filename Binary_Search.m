function v = Binary_Search(Y) % Y is a 1x1x2^(something) dimensioned matrix
% returns a vector of lenght w
    [a, b, c] = size(Y); % a and b are 1, c is the useful value
    % end case of recursion when c = 1
    if c==1
        return; % ???? what am I returning?
    end
    low = 1;
    high = c;
    key = floor(c/2);
    flag = 0; % flag to indicate is a '1' is found
    for i=key:high % scan the later half for '1'  
        if Y(a, b, i)==1
            flag = 1;
            Y_new = Y(a, b, i:high);
            v = Binary_Search(Y_new);
            break;
        end
    end
    if flag==0 % if we do not find a '1' in the second half, go to first half
        Y_new = Y(a, b, low:key);
        v = Binary_Search(Y_new);
    end
end

