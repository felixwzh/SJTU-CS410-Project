#!/usr/bin/env bash

#choose best dimension for PCA
#python SVM.py -k 0 -dgr 3 -t 0 -rf False -dm 100
#python SVM.py -k 0 -dgr 3 -t 0 -rf False -dm 200
#python SVM.py -k 0 -dgr 3 -t 0 -rf False -dm 400
#python SVM.py -k 0 -dgr 3 -t 0 -rf False -dm 500
#python SVM.py -k 0 -dgr 3 -t 0 -rf False -dm 1000
#python SVM.py -k 0 -dgr 3 -t 0 -rf False -dm 1200
#python SVM.py -k 0 -dgr 3 -t 0 -rf False -dm 2000
#python SVM.py -k 0 -dgr 3 -t 0 -rf False -dm 3400
#python SVM.py -k 0 -dgr 3 -t 1 -rf False -dm 100
#python SVM.py -k 0 -dgr 3 -t 1 -rf False -dm 200
#python SVM.py -k 0 -dgr 3 -t 1 -rf False -dm 400
#python SVM.py -k 0 -dgr 3 -t 1 -rf False -dm 500
#python SVM.py -k 0 -dgr 3 -t 1 -rf False -dm 1000
#python SVM.py -k 0 -dgr 3 -t 1 -rf False -dm 1200
#python SVM.py -k 0 -dgr 3 -t 1 -rf False -dm 2000
#python SVM.py -k 0 -dgr 3 -t 1 -rf False -dm 3400
#python SVM.py -k 0 -dgr 3 -t 2 -rf False -dm 100
#python SVM.py -k 0 -dgr 3 -t 2 -rf False -dm 200
#python SVM.py -k 0 -dgr 3 -t 2 -rf False -dm 400
#python SVM.py -k 0 -dgr 3 -t 2 -rf False -dm 500
#python SVM.py -k 0 -dgr 3 -t 2 -rf False -dm 1000
#python SVM.py -k 0 -dgr 3 -t 2 -rf False -dm 1200
#python SVM.py -k 0 -dgr 3 -t 2 -rf False -dm 2000
#python SVM.py -k 0 -dgr 3 -t 2 -rf False -dm 3400
#python SVM.py -k 0 -dgr 3 -t 3 -rf False -dm 100
#python SVM.py -k 0 -dgr 3 -t 3 -rf False -dm 200
#python SVM.py -k 0 -dgr 3 -t 3 -rf False -dm 400
#python SVM.py -k 0 -dgr 3 -t 3 -rf False -dm 500
#python SVM.py -k 0 -dgr 3 -t 3 -rf False -dm 1000
#python SVM.py -k 0 -dgr 3 -t 3 -rf False -dm 1200
#python SVM.py -k 0 -dgr 3 -t 3 -rf False -dm 2000
#python SVM.py -k 0 -dgr 3 -t 3 -rf False -dm 3400

# choose best param for SVM
# poly kernel
python SVM.py -k 1 -dgr 1 -t 0 -rf False -dm 500
python SVM.py -k 1 -dgr 2 -t 0 -rf False -dm 500
python SVM.py -k 1 -dgr 3 -t 0 -rf False -dm 500
python SVM.py -k 1 -dgr 4 -t 0 -rf False -dm 500
python SVM.py -k 1 -dgr 5 -t 0 -rf False -dm 500

python SVM.py -k 1 -dgr 1 -t 1 -rf False -dm 500
python SVM.py -k 1 -dgr 2 -t 1 -rf False -dm 500
python SVM.py -k 1 -dgr 3 -t 1 -rf False -dm 500
python SVM.py -k 1 -dgr 4 -t 1 -rf False -dm 500
python SVM.py -k 1 -dgr 5 -t 1 -rf False -dm 500

python SVM.py -k 1 -dgr 1 -t 2 -rf False -dm 500
python SVM.py -k 1 -dgr 2 -t 2 -rf False -dm 500
python SVM.py -k 1 -dgr 3 -t 2 -rf False -dm 500
python SVM.py -k 1 -dgr 4 -t 2 -rf False -dm 500
python SVM.py -k 1 -dgr 5 -t 2 -rf False -dm 500

python SVM.py -k 1 -dgr 1 -t 3 -rf False -dm 500
python SVM.py -k 1 -dgr 2 -t 3 -rf False -dm 500
python SVM.py -k 1 -dgr 3 -t 3 -rf False -dm 500
python SVM.py -k 1 -dgr 4 -t 3 -rf False -dm 500
python SVM.py -k 1 -dgr 5 -t 3 -rf False -dm 500

# rbf kernel
#python SVM.py -k 2 -dgr 2 -t 0 -rf False -dm 1200 -gm 1e-1
#python SVM.py -k 2 -dgr 2 -t 0 -rf False -dm 1200 -gm 1e-2
#python SVM.py -k 2 -dgr 3 -t 0 -rf False -dm 1200 -gm 1e-3
#python SVM.py -k 2 -dgr 4 -t 0 -rf False -dm 1200 -gm 1e-4
#python SVM.py -k 2 -dgr 5 -t 0 -rf False -dm 1200 -gm 1e-5
#
#python SVM.py -k 2 -dgr 2 -t 1 -rf False -dm 1200 -gm 1e-1
#python SVM.py -k 2 -dgr 2 -t 1 -rf False -dm 1200 -gm 1e-2
#python SVM.py -k 2 -dgr 3 -t 1 -rf False -dm 1200 -gm 1e-3
#python SVM.py -k 2 -dgr 4 -t 1 -rf False -dm 1200 -gm 1e-4
#python SVM.py -k 2 -dgr 5 -t 1 -rf False -dm 1200 -gm 1e-5
#
#python SVM.py -k 2 -dgr 2 -t 2 -rf False -dm 1200 -gm 1e-1
#python SVM.py -k 2 -dgr 2 -t 2 -rf False -dm 1200 -gm 1e-2
#python SVM.py -k 2 -dgr 3 -t 2 -rf False -dm 1200 -gm 1e-3
#python SVM.py -k 2 -dgr 4 -t 2 -rf False -dm 1200 -gm 1e-4
#python SVM.py -k 2 -dgr 5 -t 2 -rf False -dm 1200 -gm 1e-5
#
#python SVM.py -k 2 -dgr 2 -t 3 -rf False -dm 1200 -gm 1e-1
#python SVM.py -k 2 -dgr 2 -t 3 -rf False -dm 1200 -gm 1e-2
#python SVM.py -k 2 -dgr 3 -t 3 -rf False -dm 1200 -gm 1e-3
#python SVM.py -k 2 -dgr 4 -t 3 -rf False -dm 1200 -gm 1e-4
#python SVM.py -k 2 -dgr 5 -t 3 -rf False -dm 1200 -gm 1e-5

# sigmoid kernel
python SVM.py -k 3 -dgr 2 -t 0 -rf False -dm 500 -gm 1e-1
python SVM.py -k 3 -dgr 2 -t 0 -rf False -dm 500 -gm 1e-2
python SVM.py -k 3 -dgr 3 -t 0 -rf False -dm 500 -gm 1e-3
python SVM.py -k 3 -dgr 4 -t 0 -rf False -dm 500 -gm 1e-4
python SVM.py -k 3 -dgr 5 -t 0 -rf False -dm 500 -gm 1e-5

python SVM.py -k 3 -dgr 2 -t 1 -rf False -dm 500 -gm 1e-1
python SVM.py -k 3 -dgr 2 -t 1 -rf False -dm 500 -gm 1e-2
python SVM.py -k 3 -dgr 3 -t 1 -rf False -dm 500 -gm 1e-3
python SVM.py -k 3 -dgr 4 -t 1 -rf False -dm 500 -gm 1e-4
python SVM.py -k 3 -dgr 5 -t 1 -rf False -dm 500 -gm 1e-5

python SVM.py -k 3 -dgr 2 -t 2 -rf False -dm 500 -gm 1e-1
python SVM.py -k 3 -dgr 2 -t 2 -rf False -dm 500 -gm 1e-2
python SVM.py -k 3 -dgr 3 -t 2 -rf False -dm 500 -gm 1e-3
python SVM.py -k 3 -dgr 4 -t 2 -rf False -dm 500 -gm 1e-4
python SVM.py -k 3 -dgr 5 -t 2 -rf False -dm 500 -gm 1e-5

python SVM.py -k 3 -dgr 2 -t 3 -rf False -dm 500 -gm 1e-1
python SVM.py -k 3 -dgr 2 -t 3 -rf False -dm 500 -gm 1e-2
python SVM.py -k 3 -dgr 3 -t 3 -rf False -dm 500 -gm 1e-3
python SVM.py -k 3 -dgr 4 -t 3 -rf False -dm 500 -gm 1e-4
python SVM.py -k 3 -dgr 5 -t 3 -rf False -dm 500 -gm 1e-5