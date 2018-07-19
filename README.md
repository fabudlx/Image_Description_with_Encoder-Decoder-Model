# Autoamated Image Description with Encoder-Decoder-Model

Automated Image Description System that uses classic encoder (CNN+LSTM) decoder (stacked LSTM) model to generate appropriate descriptions for images

### Study

This implementation is part of a Independent Study, done at the University of Hamburg, under the Department of Language Technology. The architecture is inspired by https://github.com/anuragmishracse/caption_generator! Thanks goes to Anurag Mishra for that.

### Prerequisites

clone https://github.com/fabudlx/Image2SequenceFiles.git next to this project and go to the data and w2vModel folder. Follow the instructions to get the MS COCO 2017 dataset and the fasttext word embedding model.

### Dataset and pretrained models

MS COCO 2017 was used for training and validation

English fasttest word embedding was used to embedd the sentences

Pretrained VGG16 was used to embedd the images into image vectors


### Model

The architecture of the model:

![alt text](https://github.com/fabudlx/Image2Sequence/blob/master/arch.png)

### Training and Validation

Execute Img2SeqMain.py with the arguments: 

no. of training epochs (10 by default, 0 for only validation) 

no. of validation sentences (5000 by default, 0 for only training) 

w2v_model (0 default = wiki-news-300d-1M-subword.vec, 1 =crawl-300d-2M.vec) 

size of data partition (how many images are loaded in memory at once, 20000 by default)

e.g.

```
python Img2SeqMain.py 50 500 0 10000
```

will execute 50 training epochs with 10.000 images at a time, meaning 10.000 images are loaded, trained for 50 epochs, then the next 10.000 images are loaded

```
python Img2SeqMain.py 0 5000 0 10000
```

will only evaluate the model with 5000 validation images (in this case line 133 has to be used and changed to the model to be tested)





## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details
