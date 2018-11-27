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
python test_sceleton.py <path/to/image/img.ext>
```
