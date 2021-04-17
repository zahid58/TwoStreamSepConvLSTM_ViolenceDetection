<h1 align="center">
<p>Efficient Two-Stream Network for Violence Detection Using Separable Convolutional LSTM
</h1>
<h6 align="center">
<p> <a href="https://github.com/Zedd1558">Zahidul Islam</a>, Mohammad Rukonuzzaman, Raiyan Ahmed, Md. Hasanul Kabir, Moshiur Farazi
</h3>
<!---
<p align="center">
 <img alt="cover" src="https://github.com/Zedd1558/traffic-sign-recognition-tutorial-code/blob/master/documentation/overview.jpg" height="60%" width="60%">
</p>
-->

### Dataset preparation
To get RWF2000 dataset,
1. go to https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
2. sign their agreement sheet to get the download link from them. 
3. prepare the downloaded dataset like the following folder structure, 
```
📦project_directory
  ┣ 📂RWF-2000
    ┣ 📂train
      ┣ 📂fight
      ┣ 📂nonFight
    ┣ 📂test
      ┣ 📂fight
      ┣ 📂nonFight
```
4. When running *train.py* for the first time, pass the argument *--preprocessData*, this will uniformly sample 32 frames from each video, remove black borders and save them as *.npy* files. During the next times no need to pass the argument *--preprocessData*, as you already have converted the videos into *.npy* files during the first time.

Hockey and Movies dataset can be downloaded from this link - 
https://drive.google.com/file/d/1-4yHiSzAzOz9L0EEbw58e-soZnlFEVpP/view?usp=sharing
Here we have already splitted the dataset and converted into .npy files. So, no need to pass *--preprocessData* while using Hockey and Movies dataset from the link above.

### How to run
#### train
To train models go to project directory and run *train.py* like below, 
```
python train.py --dataset rwf2000 --vidLen 32 --batchSize 4 --numEpochs 150 --mode both --preprocessData --lstmType sepconv --savePath FOLDER_TO_SAVE_MODELS
```
The training curves and history will be saved in */results* and updated after every epoch. 
#### evaluate
To evaluate an already trained model, use *evaluate.py* like below,
```
python evaluate.py --dataset rwf2000 --vidLen 32 --batchSize 4 --mode both --lstmType sepconv --fusionType M --weightsPath PATH_TO_SAVED_MODEL
```
this will save the results in *test_results.csv*.

### Required libraries
Python 3.7, Tensorflow 2.3.1, OpenCV 4.1.2, Numpy, Matplotlib, sci-kit learn
```
pip install -r requirements.txt
```

<!---
### Implementation
#### Localization 
To find out the regions containing traffic signs we used a
well known machine learning technique called Haar Cascade
Classifier. 
We used <a href="https://amin-ahmadi.com/cascade-trainer-gui/">this GUI tool</a> to train our cascade classifier using 500 positive images (samples) i.e. images of traffic signs from GTRSB dataset and 500 negative samples i.e. images of random objects. The features learned are contained in the output *cascade.xml* which is used by *OpenCV* to find out the Region of Interests (ROI) that might contain traffic sign.
<p align="center">
  <img src="https://github.com/Zedd1558/traffic-sign-recognition-tutorial-code/blob/master/documentation/detect.png" width="400" /></p>
#### Recognition
The ROIs are cropped and passed to a CNN implemented on tensorflow. We  used publicly available dataset German Traffic Sign Recognition Benchmark to train our model. GTSRB dataset  is  a  multi-category  classification  competition  held  at IJCNN  2011.  The  dataset  is  composed  of  50,000  images  in total and 43 classes. The model is trained end to end using Adam optimizer with a initial learning rate of *0.0001* and a learning rate decay of *0.0001/(numberof epoch×0.5)*. The model was trained for 50 epochs with mini batch size of 64.
<p align="center">
  <img src="https://github.com/Zedd1558/traffic-sign-recognition-tutorial-code/blob/master/documentation/data.png" width="200" />
    <img src="https://github.com/Zedd1558/traffic-sign-recognition-tutorial-code/blob/master/documentation/accuracy.png" width="200" />
    <img src="https://github.com/Zedd1558/traffic-sign-recognition-tutorial-code/blob/master/documentation/loss(1).png" width="200" />
</p>
Detailed documentation of the project can be found <a href="https://github.com/Zedd1558/Traffic-Sign-Localization-and-Recognition/blob/master/Report_on_the_Project.pdf">here</a>.
-->
