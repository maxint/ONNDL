function tmpl = samplePos(frame, param0, sz, notNorm)
    c = 1;
    for i = -1 : 1
        for j = -1 : 1
            temp = warpimg(frame, param0 + [i j 0 0 0 0], sz);
            tmpl(:, c) = temp(:);
            if ~exist('notNorm', 'var')
                tmpl(:, c) = temp(:) / norm(temp(:));
            end
            c = c + 1;
        end
    end
    temp = warpimg(frame, param0, sz);
    tmpl(:, c) = temp(:);
    if ~exist('notNorm', 'var')
        tmpl(:, c) = temp(:) / norm(temp(:));
    end
end