function negTmpl = sampleNeg(frame, param0, sz, num)
    c = 1;
    geom = affparam2geom(param0);
    h = round(sz(2)*geom(3));
    w = round(sz(1)*geom(3)*geom(5));
    
    for i = 1 : num
        a = randn() * h / 8;
        b = randn() * w / 8;
        while abs(a) < h / 4 && abs(b) < w / 4
            a = randn() * h / 8;
            b = randn() * w / 8;
        end
        temp = warpimg(frame, affparam2mat(geom + [a b 0 0 0 0]), sz);
        negTmpl(:, c) = temp(:) / norm(temp(:));
        c = c + 1;
    end
end