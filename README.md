# cv-facial-key-points

Facial Key-point Detection and Real Time Filtering using Convolutional Neural Networks

![Facial Keypoint Detection][image1]

The project is broken down into the following main parts, all in one notebook "CV_project.ipynb:

__Part 1__ : Investigating OpenCV, Pre-processing, and Face Detection

__Part 2__ : Training a Convolutional Neural Network (CNN) to detect facial keypoints

__Part 3__ : Combining 1 and 2 together effectively to identify facial keypoints on any image!

## Project Overview 

All of the starting code and resources needed to complete the project are in this repo! Before you can get stared coding, you'll have to make sure that you have all the libraries and dependencies required to support this project.

### Environment

1. Clone the repository, and navigate to the downloaded folder.
```
git clone https://github.com/ssvasista/cv-facial-key-points.git
cd cv-facial-key-points
```

2. Create (and activate) a new environment with Python 3.5 and the `numpy` package.

	- __Linux__ or __Mac__: 
	```
	conda create --name cv-cnn python=3.5 numpy
	source activate cv-cnn
	```
	- __Windows__: 
	```
	conda create --name cv-cnn python=3.5 numpy scipy
	activate cv-cnn
	```

3. Install/Update TensorFlow (for this project, you may use CPU only).
	- Option 1: __To install TensorFlow with GPU support__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using the Udacity AMI, you can skip this step and only need to install the `tensorflow-gpu` package:
	```
	pip install tensorflow-gpu -U
	```
	- Option 2: __To install TensorFlow with CPU support only__:
	```
	pip install tensorflow -U
	```

4. Install/Update Keras.
 ```
pip install keras -U
```

5. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
	```
	KERAS_BACKEND=tensorflow python -c "from keras import backend"
	```
	- __Windows__: 
	```
	set KERAS_BACKEND=tensorflow
	python -c "from keras import backend"
	```

6. Install a few required pip packages (including OpenCV).
```
pip install -r requirements.txt
```


### Data

All of the data you'll need to train a neural network is in the the subdirectory `data`. In this folder are a zipped training and test set of data.

1. Navigate to the data directory
```
cd data
```

2. Unzip the training and test data (in that same location). If you are in Windows, you can download this data and unzip it by double-clicking the zipped files. In Mac, you can use the terminal commands below.
```
unzip training.zip
unzip test.zip
```

You should be left with two `.csv` files of the same name. You may delete the zipped files.

*Troubleshooting*: If you are having trouble unzipping this data, you can download that same training and test data on [Kaggle](https://www.kaggle.com/c/facial-keypoints-detection/data).

Now, with that data unzipped, you should have everything you need!

## Notebook

1. Navigate back to the repo. (Also your source environment should still be activated at this point)
```shell
cd
cd cv-facial-key-points
```

2. Open the notebook and follow the instructions.
```shell
jupyter notebook CV_project.ipynb
```
## Implemenatation

After the dependencies and the requirements of the environment are set right, it is good to go!
All of the code is written in one single python notebook "CV_project.ipynb" that has the following (High Level) flow of execution - 

Step 1: Add eye detections to the face detection setup

Step 2: De-noise an image for better face detection

Step 3: Blur and edge detect an image

Step 4: Automatically hide the identity of a person (blur a face)

Step 5: Specify the network architecture

Step 6: Compile and train the model

Step 7: Visualize the loss

Step 8: Complete a facial keypoints detector and finally the CV pipeline

## GG WP

## Scope - This model can be used for various applications involving face recognition, detection, emotion tracking/recognition and human behaviour analysis.
