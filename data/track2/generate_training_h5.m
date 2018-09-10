% original data folder
lr2dir = 'stage_one/';
hrdir = 'training_data/';

tifdir = 'training_data/';
% configuration
psize = 48;

% stride
s = 12;


fnum = 1;
files = dir([lr2dir '/' '*lr2' '*.fla']);

h5create('hd5/out.h5', '/data', [48 48 14 Inf], 'Datatype', 'single', 'ChunkSize', [48 48 14 1]);
h5create('hd5/hr.h5', '/data', [48 48 14 Inf],  'Datatype', 'single', 'ChunkSize', [48 48 14 1]);
h5create('hd5/tif.h5', '/data', [48 48 3 Inf],  'Datatype', 'single', 'ChunkSize', [48 48 3 1]);

for k = 1:length(files)
    file = files(k).name;
    tif = imread([tifdir '/' strrep(strrep(file, 'lr2', 'lr2_registered'), 'fla', 'tif')]);
    L2I = FLAread([lr2dir '/' file]);
    HI = FLAread([hrdir '/' strrep(file, 'lr2', 'hr')]);
    
    hw_g = L2I.I;

    hw = HI.HDR.samples; hh=HI.HDR.lines;
    
    for i = 1:s:L2I.HDR.samples
        if (i + psize > hw)
            break;
        end
        for j = 1:s:L2I.HDR.lines
            if (j + psize > hh)
                break;
            end
            outpatch = hw_g(j:j+psize-1, i:i+psize-1, :);
            tifpatch = tif(j:j+psize-1, i:i+psize-1, :);

            hpatch = HI.I(j:j+psize-1, i:i+psize-1, :);
            
            h5write('hd5/hr.h5', '/data', single(hpatch),  [1,1,1,fnum], [48, 48, 14 1]);
            h5write('hd5/out.h5', '/data', single(outpatch),  [1,1,1,fnum], [48, 48, 14 1]);
            h5write('hd5/tif.h5', '/data', single(tifpatch), [1,1,1,fnum], [48,48,3,1]);

            fnum = fnum + 1;
            if (mod(fnum,100) == 0)
                sprintf('already generated %d patches\n', fnum)
            end
        end
    end
end
sprintf('generated %d patches\n', fnum)
