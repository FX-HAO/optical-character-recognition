# Optical Character Recognition

Machine learning OCR with Convolutional Recurrent Neural Network (CRNN).
The input shape is (32, 400), so if your dataset has different shapes,
you need to resize the inputs or modify the architecture.

## How does it work

### generate dataset

```
python3 generate_dateset.py
```

### Train the model

There're many ways to train the model, you can just run `python3 crnn-ctc.py` on your machine,
but it probably has poor performance if you do not have a high-performance GPU (or TPU) and [CUDA](https://www.tensorflow.org/install/gpu) support.
Or you can train your model in the cloud (e.g. AWS), but it's absolutely not free.
The way I recommend is [Colab](https://colab.research.google.com/), it provides you great 
GPUs (and TPUs) and is totally free.

You can take a look at [ocr.ipynb](./ocr.ipynb).

### Load pre-trained model

Download `best_model.hdf5` and put it into the current directory.

A pre-trained model here: [best_model.hdf5](https://drive.google.com/open?id=1QlE4qsSB2hARdHv3yWxoPirDHp9m6V7K)

### Make a prediction
```
python3 generate_dataset.py 1 | xargs python3 predict.py
```
