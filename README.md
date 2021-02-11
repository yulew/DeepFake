
https://www.kaggle.com/c/deepfake-detection-challenge
deepfake-detection-challenge
Detecting whether a video has been deepfaked.         
          
#
deepfacke.ipynb -- co-authors: Yule Wang and Wenyi Wang 
deepfake_batch.py -- author: Yule Wang

#
We implemented package facenet-pytorch to extract face embeddings of the frames in the videos (500 GB training dataset). Then we trained a convolutional LSTM model for temporal sequence analysis of the frames on Google Cloud Platform (GCP).

To be more specific, we targeted on extract frames from videos and randomizing these frames. Then, we obtain a dataset of frames. 
The original plan is to extract face embeddings from these random frames (along with some important features, such as relative positions between eyes, nose etc.) and save them as metadata in the storage. 

What is the problem? The dataset is too huge. You cannot load all of them into memory. Secondly, save the embeddings as metadata in the storage costs extra huge stroage space. The below is the idea that how to solve the problem.
#

In deepfake_batch.py, I am focusing on the whole (generator) pipeline framework: Loading the huge training dataset 500GB into batches using method of iterator.  
Loading  a batch of videos and extract the frames from the videos. Then I randomized these frames and then directly extract face embeddings from these frames. And then feed the batch of emdeddings into the neural network (NN). Then save the trained NN and start another batch of videos and repeat and load the previous trained NN. In this way, we don't need to save them into stroage.

