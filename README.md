
https://www.kaggle.com/c/deepfake-detection-challenge
deepfake-detection-challenge
Detecting whether a video has been deepfaked.         
          
#
deepfacke.ipynb -- co-authors: Yule Wang and Wenyi Wang 
deepfake_batch.py -- author: Yule Wang

#
We implemented the package facenet-pytorch to extract face embeddings of the frames in the (500 GB training dataset). Then we trained a convolutional LSTM model for temporal sequence analysis of the frames on Google Cloud Platform (GCP).

To be more specific, we targeted extracting frames from videos and randomizing these frames. We then obtained a dataset of frames. 
The original plan was to extract face embeddings from these random frames (along with some important features, such as relative positions between the eyes, the nose etc.) and output them as metadata to the storage. 

What is the problem? The dataset is too huge. You cannot load them all into memory. Secondly, saving the embeddings as metadata in the storage takes up a huge amount of storage space. The below describe how to solve the problem.
#

In deepfake_batch.py, I am focusing on the whole (generator) pipeline framework -- splitting the huge training dataset 500GB into batches using an iterator method.  
Firstly, the program only loads a batch of videos and extracts the frames from these videos. I then randomized these frames and then directly extracted face embeddings from these frames. I then fed the batch of embeddings into the neural network (NN) without saving these embeddings. Lastly I saved the trained NN. On the next batch of videos, the system repeats the previous process, based on the previous trained NN. This way, we don't need to cost extra storage and the memory issues can be solved.

