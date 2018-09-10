% original data folder
lr2dir = 'testing_lr/';
lr3dir = 'testing_lr/';

files = dir([lr2dir '/' '*lr2' '*.fla']);

fnum = 1;

for k = 1:length(files)
    file = files(k).name;
    h5filename = ['testing_lr/' strrep(file, 'fla', 'h5')]
    h5create(h5filename,'/data',[240 480 14 Inf], 'Datatype', 'single', 'ChunkSize', [240 480 14 1]);
    L2I = FLAread([lr2dir '/' file]);
    L3I = FLAread([lr3dir '/' strrep(file, 'lr2', 'lr3')]);
    
    %LL2I = HI.I; LL3I = HI.I;

    hw_g = zeros([240,480,14]);
    hw_g(2:2:end, 2:2:end, :) = L2I.I;
    hw_g(2:3:end, 2:3:end, :) = L3I.I;
    
    [out, replaced] = myfan(hw_g);

            
    %h5write(h5filename, '/data', single(out), [1 1 1 1], [240 480 14 1]);
    %h5write(h5filename, '/data', single(hw_g),  [1 1,1,2], [240 480 14 1]);
    h5write(h5filename, '/data', single(replaced),  [1,1,1,1], [240 480 14 1]);
    fnum = fnum + 1
end
sprintf('generated %d patches\n', fnum)
