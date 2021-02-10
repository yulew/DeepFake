
https://www.kaggle.com/c/deepfake-detection-challenge
deepfake-detection-challenge
Detecting whether a video has been deepfaked.         
          
#
Co-authors:
Yule Wang and Wenyi Wang

#
We implemented package facenet-pytorch to extract face embeddings of the frames in the videos (500 GB training dataset). Then we trained a convolutional LSTM model for temporal sequence analysis of the frames on Google Cloud Platform (GCP).
#

I am focusing on the whole pipeline framework: Loading the huge training dataset 500GB into batches using iterators. 

Wenyi is more focusing on the RNN part.
