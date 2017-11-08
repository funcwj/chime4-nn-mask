# CHiME4 NN-based mask estimation

Implementation of BLSTM mask estimator in pytorch.

Updating...

### Usage
1. Split `dt05_simu.json`, `tr05_simu.json` in `CHiME4/data/annotations`(in order to accelarate step 2), which will be used in `CHiME4_simulate_data_patched.m`
```sh
$ for x in dt05_simu.json tr05_simu.json; do
$     python tools/split_json.py $CHiME4/data/annotations/$x $nj --output_dir $CHiME4/data/annotations
$ done
```

2. Using modified version of official `CHiME4_simulate_data.m` to generate noise/clean part of the simulated data in CHiME4(dt05/tr05), `simulate_nngev.sh` enable us to generate data parallelly(NOTE: modify the variable defined in `CHiME4_simulate_data_pactched.m` according
to your custom settings).
```sh
$ cp tools/CHiME4_simulate_data_pactched.m $CHiME4/simulation
$ cd $CHiME4/simulation
$ ./simulate_nngev.sh $nj
```

3. Prepare masks for NN training(NOTE: the program will take almost 140G disk space).
```sh
$ ./calculate_masks.sh $CHiME4/data $mask_dir &
```

4. Training models(`$nj` keeps same as step 1).
```sh
$ python train_estimator.py $mask_dir --nj $nj
```

5 ...

### Reference
* Heymann J, Drude L, Haebumbach R. Neural network based spectral mask estimation for acoustic beamforming.[J]. IEEE Transactions on Industrial Electronics, 2016, 46(3):544-553.
* https://github.com/fgnt/nn-gev