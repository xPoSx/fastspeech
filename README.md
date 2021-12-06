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

### Weigths
`fastspeech`: https://drive.google.com/file/d/1vUNaNPTKqN46cUjAlskzcbJnH0MBUW4L/view?usp=sharing

`fastspeech_new`: https://drive.google.com/file/d/1-HyvXNfD_xmNWHZD0teWNnQketfa9Y7y/view?usp=sharing

`fastspeech_16`: https://drive.google.com/file/d/1D4y7MBepbMvVLSPOqvlgAshKp_RNDUS9/view?usp=sharing