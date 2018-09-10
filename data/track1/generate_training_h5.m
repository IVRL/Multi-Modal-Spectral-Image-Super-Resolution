% original data folder
lr2dir = 'training_data/';
lr3dir = 'training_data/';
hrdir = 'training_data/';
% configuration
psize = 32;

% stride
s3 = 8;
s2 = s3 / 2 * 3;
s = s3 * 3;


fnum = 1;
files = dir([lr2dir '/' '*lr2' '*.fla']);

h5create('hd5/out.h5', '/data', [96 96 14 Inf], 'Datatype', 'single', 'ChunkSize', [96 96 14 1]);
h5create('hd5/replaced.h5', '/data', [96 96 14 Inf], 'Datatype', 'single', 'ChunkSize', [96 96 14 1]);
h5create('hd5/hr_g.h5', '/data', [96 96 14 Inf],  'Datatype', 'single','ChunkSize', [96 96 14 1]);
h5create('hd5/hr.h5', '/data', [96 96 14 Inf],  'Datatype', 'single','ChunkSize', [96 96 14 1]);

for k = 1:length(files)
    file = files(k).name;
    L2I = FLAread([lr2dir '/' file]);
    HI = FLAread([hrdir '/' strrep(file, 'lr2', 'hr')]);
    L3I = FLAread([lr3dir '/' strrep(file, 'lr2', 'lr3')]);
    
    l3w = L3I.HDR.samples; l3h = L3I.HDR.lines;
    l2w = L2I.HDR.samples; l2h = L2I.HDR.lines;
    hw = HI.HDR.samples; hh = HI.HDR.lines;
    
    hw_g = zeros(size(HI.I));
    hw_g(2:2:end, 2:2:end, :) = L2I.I;
    hw_g(2:3:end, 2:3:end, :) = L3I.I;
    
    [out, replaced] = myfan(hw_g);
    
    hi = 1;
    l2i = 1;
    
    for l3i = 1:s3:L3I.HDR.samples
        hj = 1;
        l2j = 1;
        if ((l3i + psize > l3w) | (hi + psize * 3 > hw) | (l2i + psize * 3 / 2 > l2w))
               break;
        end
        
        for l3j = 1:s3:L3I.HDR.lines
            if ((l3j + psize > l3h) | (hj + psize * 3 > hh) | (l2j + psize * 3 / 2 > l2h))
                continue;
            end
            hgpatch = hw_g(hj:hj+psize*3-1, hi:hi+psize*3-1, :);
            outpatch = out(hj:hj+psize*3-1, hi:hi+psize*3-1, :);
            replacedpatch = replaced(hj:hj+psize*3-1, hi:hi+psize*3-1, :);
            hpatch = HI.I(hj:hj+psize*3-1, hi:hi+psize*3-1, :);
            
            h5write('hd5/out.h5', '/data', single(outpatch),  [1 1,1,fnum], [96, 96, 14 1]);
            h5write('hd5/hr.h5', '/data', single(hpatch),  [1,1,1,fnum], [96, 96, 14 1]);
            h5write('hd5/hr_g.h5', '/data', single(hgpatch),  [1,1,1,fnum], [96, 96, 14 1]);
            h5write('hd5/replaced.h5', '/data', single(replacedpatch),  [1,1,1,fnum], [96, 96, 14 1]);
            
            %save([lrpatch sprintf('%08d',fnum) '.mat'], 'lpatch');
            %save([hrpatch sprintf('%08d',fnum) '.mat'], 'hpatch');
            fnum = fnum + 1;
            if (mod(fnum,100) == 0)
                sprintf('already generated %d patches\n', fnum)
            end
            l2j = l2j + s2;
            hj = hj + s;
        end
        hi = hi + s;
        l2i = l2i + s2;
    end
end
sprintf('generated %d patches\n', fnum)
