# datamining assignment usage guide
* clone this project
* for assignment1
```
cd datamining
virtualenv assignment1 --system-site-packages
cd assignment1
# for ksh:
. bin/activate

# for bash, etc
source bin/activate

# if you don't have numpy and Pillow already installed on your machine
pip install numpy, Pillow
```
* put test image somewhere
* run program
```
script usage: python test_sceleton.py imagefile <algorithm> <alg_parms>
                         <algorithm>:   either <d> for dbscan or <k> for k-means
                         <alg_parms>:   * dbscan : epsilon: int value, min_pts: int value
                                        * k-means : k: int value
example 1: python test_sceleton.py inputimage.jpg k 5
example 2: python test_sceleton.py inputimage.jpg d 10 20
```
