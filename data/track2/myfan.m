
function [out, replaced] = myfan(img)
    [ht, wd, c] = size(img);
    out = zeros(ht, wd, c);
    sz= wd*ht;
    % Generating mask
    MASK = zeros(ht, wd);
    MASK(2:2:end,2:2:end) = 1;
    MASK(2:3:end,2:3:end) = 1;
    numrandom = sum(sum(MASK));
    sigma = sqrt(double(sz)/(pi*double(numrandom)));
    i = fftfilt(MASK, sigma);
    for c_i = 1:c
        g = squeeze(img(:, :, c_i));
        
        g = fftfilt(g, sigma);
        g = g./i;
        
        out(:, :, c_i) = g;
    end
    
    replaced = out;
    replaced(2:2:end, 2:2:end, :) = img(2:2:end, 2:2:end, :);
    replaced(2:3:end, 2:3:end, :) = img(2:3:end, 2:3:end, :);
end

function out = fftfilt(in, sigma)
    [ht,wd] = size(in);
    [X,Y] = meshgrid(1:wd,1:ht);
    X = X-wd/2;
    Y = Y-ht/2;
    Z = exp(-0.5*(X.^2 + Y.^2)/sigma^2);
    Z = Z/sum(sum(Z));
    out = ifftshift(ifft2( fft2(in).*fft2(Z) ));
    %figure, imshow(uint8(out));
end