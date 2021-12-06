# fastspeech

## How to train the model


### Preparations
```shell
!git clone https://github.com/xPoSx/fastspeech.git
!pip install -r fastspeech/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

!wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
!tar -xjf LJSpeech-1.1.tar.bz2
!mv LJSpeech-1.1 fastspeech/

!git clone https://github.com/NVIDIA/waveglow.git
!mv waveglow fastspeech/
```

### Train
```shell
!cd fastspeech && python3 train.py
```