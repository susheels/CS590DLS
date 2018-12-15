# CS590DLS




This repository contains the code used for CS590 DLS project. This project utilizes inst2vec embeddings for source code vulnerability detection. 


## Code

### Requirements

For training ```inst2vec``` embeddings:
* GNU / Linux or Mac OS
* Python (3.6.5)
  * tensorflow (1.7.0) or preferably: tensorflow-gpu (1.7.0)
  * networkx (2.1)
  * scipy (1.1.0)
  * absl-py (0.2.2)
  * jinja2 (2.10)
  * bokeh (0.12.16)
  * umap (0.1.1)
  * sklearn (0.0)
  * wget (3.2)

Additionally, for training vulnerability detection model:
* GNU / Linux or Mac OS
* Python (3.6.5)
  * labm8 (0.1.2)
  * keras (2.2.0) 

### Running the code

#### 1. Download Dataset 

One can download the dataset automatically by running 
```shell
$ python data_download.py 
```


#### 2. `inst2vec` embeddings

One can download the pre-trained embeddings from https://drive.google.com/open?id=1Kmd6AVZQKvfhhmfdCMikzJLQri_J17u6. Place the pretrained files in a folder named pretrained. 


#### 3. Training Vulnerability Detection Model

Train:
```shell
$ python train.py 
```

