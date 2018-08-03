Using msml as loss function and some units of google net as nn.
Accuracy approaching 95% and 85% in lfw(split lfw into two part: train_dataset and val_dataset), and test_dataset is collected from my friends's and mine, and on test_dataset, the accuracy approach 80%.

Because of laking GPU for training process, so i will not continue training the model.
The reason why i record the code is that reminding me of some trick that i learned from facenet[1] and msml[2]

Requirement:  
  1. python3.6
  2. Tensorflow 1.8.0

My device: Mac OS

Reference:  
  [1] FaceNet: A Unified Embedding for Face Recognition and Clustering  
  [2] Margin Sample Mining Loss: A Deep Learning Based Method for Person Re-identification
