# datamining assignment usage guide
* clone this project
* assignment was solved using python 2.7.15
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
pip install numpy Pillow
```
* put test image somewhere
* run program
```
script usage: python test_sceleton.py imagefile <algorithm> <alg_parms>
                         <algorithm>:   either <d> for dbscan or <k> for k-means
                         <alg_parms>:   * dbscan : epsilon: int value, min_pts: int value, distance_function 'e' or 'm'
                                        * k-means : k: int value
example 1: python test_sceleton.py inputimage.jpg k 5
example 2: python test_sceleton.py inputimage.jpg d 10 20 e
```
* for assignment2
```
cd datamining
virtualenv assignment2 --system-site-packages

cd assignment2
# for ksh:
. bin/activate

# for bash, etc
source bin/activate

pip install numpy pandas scikit-learn scipy imbalanced-learn matplotlib 

script usage: python test_sceleton.py 

the script will produce the output data for all experiments described in the documentation and prepare latex files and plots for inclusion or review.

```
