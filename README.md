# Few-shot sound recognition using attentional similarity

Pytorch implementation of [Learning to match transient sound events using attentional similarity for few-shot sound recognition] ([paper](https://arxiv.org/abs/1812.01269))

## Citation
If you use this code in your research, please cite our paper.

    @inproceedings{chou2019learning,
        title={Learning to match transient sound events using attentional similarity for few-shot sound recognition},
        author={Szu-Yu Chou and Kai-Hsiang Cheng and Jyh-Shing Roger Jang and Yi-Hsuan Yang},
        booktitle = {Proceedings of the International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
        year={2019}
    }

## Requirements

- Python 2.7
- PyTorch 0.4.0
- LibROSA 0.5.0
- Cuda-9.0

## Getting Started
### 1. Mel-spec data of (noise) ESC50 ([link](https://drive.google.com/open?id=1dWiqIc8xTBN4wYPwiYObegWVM7J5oLkm))
We provide the mel-spectrogram data, which extracted from wave files with default parameters showed in ```main.py```. Once the data acquired, please unzip ```data.zip``` and have ```data``` under ```attentional-similarity``` folder.

**```data``` contains following files :**
- ```ESC_sep.npy```: mel-spec of ESC50
- ```ESC_noise_sep.npy```: mel-spec of noise ESC50
- ```ESC_tag.npy```: class index of *ESC_sep* and *ESC_noise_sep* entries
- ```ESC_tag2idx.npy```: matching of class names and class indices

### 2. Train
#### 2.1 Specify cuda device
It's an optional choice to modify ```CUDA_VISIBLE_DEVICES``` in ```Trainer.py``` to specify the gpu device that the model is going to run on. The default setting is ```0```

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#### 2.2 Start training
Run ```main.py``` script and specify the dataset name

- ESC50

        $ python main.py --dn ESC_50

- noise ESC50
        
        $ python main.py --dn ESC_noise_50

### 3. Test with trained model
#### 3.1 Specify cuda device
It's an optional choice to modify ```CUDA_VISIBLE_DEVICES``` in ```Tester.py``` to specify the gpu device that the model is going to run on. The default setting is ```0```

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#### 3.2 Specify trained model path
Set model path ```pmp``` in ```main_test.py``` to the path that store the model you are going to test. 

#### 3.3 Start testing
Run ```main_test.py``` script and specify the dataset name

- ESC50

        $ python main_test.py --dn ESC_50

- noise ESC50
        
        $ python main_test.py --dn ESC_noise_50


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

