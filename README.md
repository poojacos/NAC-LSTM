# NAC-LSTM
## Fast Neural Accumulator (NAC) based Badminton Video Action Classification

In  the  last  decade,  as  the  available  sports  multimedia  has grown,  the  technology  for  analysis  of  content-based  sportsvideo has followed. Thereafter, due to high commercial potential and wide viewership, it has become paramount to develop apotent representation of sports video content.
Current methods of extracting spatio-temporal features work very well however they run at the cost of high computational  requirement  which  affects  its  real time  application and deployment in broadcasted events especially in games with rapid motions such as badminton. In this work we aim to address the this issue. 

Two  novel  end-to-end trained NAC based frameworks for action classificationin video analysis have been proposed in this work.  The proposed models need not be trained on GPUs.  They achieve high classification accuracy in strikingly minimal training and testing time.  Forthe purpose of comparison three deep learning based methods viz. Denoising FCN Autoencoder, Temporal convolutional net-work and CNN-LSTM (LRCN) for stroke classification havebeen implemented.  These models were required to be trainedon  GPUs.   The  proposed  models  perform  better,  in  terms  of classification accuracy, than the comparing methods, except the LRCN. However as far as the computation time, our models always exhibit lower training and testing time, even when they are runon CPU while the comparing methods on GPU. In other words, if  all  are  run  on  either  GPUs  or  CPUs,  the  contrast  in  time difference (i.e the superiority of the proposed models) will be more prominent.   Experiments have been performed using 5-fold cross validation for several test-train splits varying from 10% to 50% to verify the effectiveness of the proposed model.

<img src="https://github.com/poojacos/NAC-LSTM/blob/master/Performance-nac.png" width="800">
Check the project wiki for the dataset.
