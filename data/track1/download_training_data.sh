# get Track 1 training data
wget -O tr1_training_lr.zip http://users.cecs.anu.edu.au/~arobkell/PIRM2018/Hyperspectral_super_resolution/training_lr.zip 
wget -O tr1_training_hr.zip http://users.cecs.anu.edu.au/~arobkell/PIRM2018/Hyperspectral_super_resolution/training_hr.zip
wget -O tr1_validation_lr.zip http://users.cecs.anu.edu.au/~arobkell/PIRM2018/Hyperspectral_super_resolution/validation_lr.zip
wget -O tr1_validation_hr.zip https://competitions.codalab.org/my/datasets/download/421584c6-756f-45f6-afeb-7e2029e0d6f7
unzip tr1_training_lr.zip
unzip tr1_training_hr.zip
unzip tr1_validation_lr.zip
unzip tr1_validation_hr.zip

mkdir training_data
mv training_lr/*.fla training_data/
mv training_lr/*.hdr training_data/
mv training_hr/*.fla training_data/
mv training_hr/*.hdr training_data/
mv validation_lr/*.fla training_data/
mv validation_lr/*.hdr training_data/
mv *.fla training_data/
mv *.hdr training_data/
rename -v 's/tr1/hr/g' training_data/*.fla
rename -v 's/tr1/hr/g' training_data/*.hdr

cd training_data
for f in * ; do mv -- "$f" "TR1_$f" ; done
cd ..

rm -rf training_lr training_hr validation_lr
rm *.zip

# get Track 2 training data
wget -O tr2_training_lr.zip http://users.cecs.anu.edu.au/~arobkell/PIRM2018/COLOR_Aided_HS_super_resolution/traning_lr.zip
wget -O tr2_training_hr.zip http://users.cecs.anu.edu.au/~arobkell/PIRM2018/COLOR_Aided_HS_super_resolution/training_hr.zip
wget -O tr2_validation_lr.zip http://users.cecs.anu.edu.au/~arobkell/PIRM2018/COLOR_Aided_HS_super_resolution/validation_lr.zip
wget -O tr2_validation_hr.zip https://competitions.codalab.org/my/datasets/download/0e8f6418-cdb7-4431-b967-96a9ac9eff9d
unzip tr2_training_lr.zip
unzip tr2_training_hr.zip
unzip tr2_validation_lr.zip
unzip tr2_validation_hr.zip

mv traning_lr/*.fla training_data/
mv traning_lr/*.hdr training_data/
mv training_hr/*.fla training_data/
mv training_hr/*.hdr training_data/
mv validation_lr/*.fla training_data/
mv validation_lr/*.hdr training_data/
mv *.fla training_data/
mv *.hdr training_data/
rename -v 's/tr2/hr/g' training_data/*.fla
rename -v 's/tr2/hr/g' training_data/*.hdr

rm -rf traning_lr training_hr validation_lr
rm *.zip

mkdir hd5
