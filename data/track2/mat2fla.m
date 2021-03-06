% clear; clc;
I = FLAread('./sample.fla');

path = './stage_one/';

files = dir([path '*.mat']);
for k = 1:length(files)
    file = files(k).name;
    load([path file]);
    for i = 1:240
        for j = 1:480
            for l = 1:14
                I.I(i,j,l) = out_g(1, l, j, i) * 65535;
            end
        end
    end
    FLAwrite([path strrep(file, 'mat', 'fla')], I);
end
