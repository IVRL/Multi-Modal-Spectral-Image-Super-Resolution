wget -O tr2_training_lr.zip http://users.cecs.anu.edu.au/~arobkell/PIRM2018/COLOR_Aided_HS_super_resolution/traning_lr.zip
wget -O tr2_training_hr.zip http://users.cecs.anu.edu.au/~arobkell/PIRM2018/COLOR_Aided_HS_super_resolution/training_hr.zip
wget -O tr2_validation_lr.zip http://users.cecs.anu.edu.au/~arobkell/PIRM2018/COLOR_Aided_HS_super_resolution/validation_lr.zip
wget -O tr2_validation_hr.zip https://competitions.codalab.org/my/datasets/download/0e8f6418-cdb7-4431-b967-96a9ac9eff9d
unzip tr2_training_lr.zip
unzip tr2_training_hr.zip
unzip tr2_validation_lr.zip
unzip tr2_validation_hr.zip

mkdir training_data

mv traning_lr/*.fla training_data/
mv traning_lr/*.hdr training_data/
mv traning_lr/*lr2_registered*.tif training_data/
mv training_hr/*.fla training_data/
mv training_hr/*.hdr training_data/
mv validation_lr/*.fla training_data/
mv validation_lr/*.hdr training_data/
mv validation_lr/*lr2_registered.tif training_data/
mv *.fla training_data/
mv *.hdr training_data/
rename -v 's/tr2/hr/g' training_data/*.fla
rename -v 's/tr2/hr/g' training_data/*.hdr

rm -rf traning_lr training_hr validation_lr
rm *.zip

mkdir hd5
mkdir 
