# Cycle-GAN with Shape-Color-Similarity Regularization

## Installation

Clone the repository
```
git clone https://github.com/Jayliu227/cycle-gan-shape-color-regularization.git
```
Clone the helper repository from tensorflow
```
git clone https://github.com/tensorflow/models.git
```
Follow their instruction to install and setup the environment, and remember to download their pretrained model, the instruction of which can be found in the *jupyter notebook* in their *object_detection* folder.

Once finished, we only need to use one module from their repository
```
cp -rf models/research/object_detection/ cycle-gan-shape-color-regularization/models/
```
Navigate into our repository and create two folders
```
mkdir output
mkdir save
mkdir data
```
where output is used to store the testing results of the model, save is used for storing the trained model, and data is the directory for the dataset

Dataset is organized as follows:
```
Data
    |---test <testing set>
        |---X <X image set>
        |---Y <Y image set>
    |---train <training set>
        |---X <X image set>
        |---Y <Y image set>
```
Overall project organization:
```
cycle-gan-shape-color-regularization
    |---data
        |---test <testing set>
            |---X <X image set>
            |---Y <Y image set>
        |---train <training set>
            |---X <X image set>
            |---Y <Y image set>
    |---save
    |---output
    |---models
    |---object_detection
    |---other scripts
```
Train model (in *model* folder)
```
python train.py
```
Test model (in *model* folder)
```
python test.py
```
