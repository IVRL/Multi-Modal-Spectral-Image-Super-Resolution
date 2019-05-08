# Multi-Modal Spectral Image Super-Resolution

## Code
### Dependencies
- Pytorch 0.4.0
- cuDNN

Our code is tested under Ubuntu 14.04 environment with Titan X GPUs.

### Inference for Track1
1. goto folder: data/track1/
2. run download_testing_data.sh
3. run generate_testing_h5.m
4. goto folder: code/track1/
5. run: python validate.py
6. run: python npz2mat.py
7. run mat2fla.m

the reconstruction is in code/track1/validation/*.{hdr,fla}

### Inference for Track2
1. goto folder: data/track2/
2. run download_testing_data.sh
3. run generate_testing_h5.m
4. goto folder: code/track2/
5. run: python validate.py
6. run: python npz2mat.py
7. run mat2fla.m

the reconstruction is in code/track2/validation/*.{hdr,fla}

### Training for Stage-I (Track1):
1. goto folder: data/track1/
2. run download_training_data.sh
3. run generate_training_h5.m
4. goto folder: code/track1/
5. run: cp -r ../../data/track1/hd5 ./data
6. run: python main.py 

### Training for Stage-II (Track2):
1. train Stage-I
2. goto folder: data/track2/
3. run download_training_data.sh
4. run generate_stage_one_h5.m
5. run: python generate_stage_one_results.py
6. run: python npz2mat.py
7. run mat2flat.m
8. run generate_training_h5.m
9. goto folder: code/track2/
10. run: cp -r ../../data/track2/hd5 ./data
11. run: python main.py

## Authors
- [Fayez Lahoud](http://ivrl.epfl.ch/people/Lahoud) [[Home]](http://fayez.me)
- [Ruofan Zhou](https://ivrl.epfl.ch/people/Zhou)
- [Sabine Süsstrunk](https://ivrl.epfl.ch/people/susstrunk)

## License
- For academic and non-commercial use only.
