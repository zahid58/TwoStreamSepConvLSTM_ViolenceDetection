<h1 align="center">
<p><a href="https://arxiv.org/abs/2102.10590">Efficient Two-Stream Network for Violence Detection Using Separable Convolutional LSTM</a>
</h1>
<h6 align="center">
<p> <a href="https://github.com/Zedd1558">Zahidul Islam</a>, Mohammad Rukonuzzaman, Raiyan Ahmed, Md. Hasanul Kabir, Moshiur Farazi
</h3>
<!---
<p align="center">
 <img alt="cover" src="https://github.com/Zedd1558/traffic-sign-recognition-tutorial-code/blob/master/documentation/overview.jpg" height="60%" width="60%">
</p>
-->

This repository contains the codes for our [[PAPER]](https://arxiv.org/abs/2102.10590) on violence detection titled *Efficient Two-Stream Network for Violence Detection Using Separable Convolutional LSTM* which is accepted to be presented at Int'l Joint Conference on Neural Networks (IJCNN) 2021. 

### Dataset preparation
To get RWF2000 dataset,
1. go to github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
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

Hockey and Movies dataset can be downloaded from these links - 

[Hockey_Dataset](https://www.kaggle.com/datasets/yassershrief/hockey-fight-vidoes)

[Movies_Dataset](https://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635)

Then, preprocess the datasets in the same way as rwf2000 dataset.

### How to run
#### train
To train models go to project directory and run *train.py* like below, 
```
python train.py --dataset rwf2000 --vidLen 32 --batchSize 4 --numEpochs 150 --mode both --preprocessData --lstmType sepconv --savePath FOLDER_TO_SAVE_MODELS
```
The training curves and history will be saved in *./results* and updated after every epoch. 

#### evaluate
To evaluate an already trained model, use *evaluate.py* like below,

```
python evaluate.py --dataset rwf2000 --vidLen 32 --batchSize 4 --mode both --lstmType sepconv --fusionType M --weightsPath PATH_TO_SAVED_MODEL
```
this will save the results in *test_results.csv*.

#### run evaluate.py on trained_models
The trained models weigths are available in the drive folder [trained_models](https://drive.google.com/drive/folders/1igx-plktW069IgXyWg3H78AKuTg-jCza?usp=sharing). Copy the model you want to use into your project directory like shown below. Then you can evaluate the trained_model like below.

![trained_model_evaluate](https://github.com/Zedd1558/TwoStreamSepConvLSTM_ViolenceDetection/blob/master/imgs/3.png)

```
python evaluate.py --dataset rwf2000 --vidLen 32 --batchSize 4 --mode both --lstmType sepconv --fusionType M --weightsPath "/content/violenceDetection/model/rwf2000_model"
```

#### loading trained_models weights inside script
The trained models weigths are available in the drive folder [trained_models](https://drive.google.com/drive/folders/1igx-plktW069IgXyWg3H78AKuTg-jCza?usp=sharing). Copy the entire folder and its contents into the project directory. Then you can use the trained models like shown below.
``` python
path = "./trained_models/rwf2000_model/sepconvlstm-M/model/rwf2000_model"     
# path = "./trained_models/movies/sepconvlstm-A/model/movies_model"   
model =  models.getProposedModelM(...) # build the model
model.load_weights(path) # load the weights
```
The folder also contains training history, training curves and test results. 

### Required libraries
Python 3.7, Tensorflow 2.3.1, OpenCV 4.1.2, Numpy, Matplotlib, sci-kit learn
```
pip install -r requirements.txt
```

### Bibtex
If you do use ideas from the paper in your work please cite as below:
```
@misc{islam2021efficient,
      title={Efficient Two-Stream Network for Violence Detection Using Separable Convolutional LSTM}, 
      author={Zahidul Islam and Mohammad Rukonuzzaman and Raiyan Ahmed and Md. Hasanul Kabir and Moshiur Farazi},
      year={2021},
      eprint={2102.10590},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!---
### Implementation
<p align="center">
  <img src="link" width="200" />
    <img src="link" width="200" />
</p>
<a href="link">here</a>.
-->
