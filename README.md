## DL_pipeline (Internal Team Tool)

* This is a binary classifier pipeline adapted from the RDDL GitHub repo (https://github.com/cobisLab/RDDL)

* For citation in thesis writing, please refer to the following work:<br> 
Tzu-Hsien Yang*, Zhan-Yi Liao, Yu-Huai Yu, and Min Hsia, "RDDL: a systematic ensemble pipeline tool that streamlines balancing training schemes to reduce the effects of data imbalance in rare-disease-related deep-learning applications".

----------------

### File Structure


```
DL_pipeline/
├── Hemo/
│   ├── hyperparameters/     # Setting parameters for training
│   ├── outputs/             # Generated results and saved models
│   ├── prob/                # Output probability of testset sample
│   │ 
│   ├── log.txt              # training logs
│   ├── test1.csv            # test set
│   ├── training_1.csv       # training dataset 1 out of 5
│   ├── training_2.csv       # training dataset 2 out of 5       
│   ├── training_3.csv       # training dataset 3 out of 5
│   ├── training_4.csv       # training dataset 4 out of 5
│   ├── training_5.csv       # training dataset 5 out of 5
│   │ 
│   └── USER_model.py        # Custom model architecture definition
|
├── input_data/
│   └── 8D49Lzs/             # Encoded array as input data
|
├── training_func/           # Utility functions for the training pipeline
|
├── run_train.py             # Main entry point for starting training
├── requirements_gpu.txt     # Dependencies for GPU-enabled environments
└── requirements.txt         # Standard CPU-only dependencies
```


---

### Step 1: Environment Preparation

Suggested running environments: Python 3.8.13

```
conda create -n "dl38" python=3.8.13
```

If you intend to use GPU during the model training, please use the following file to install the packages:

```
pip install -r requirements_gpu.txt
```

Otherwise, 

```
pip install -r requirements.txt
```

---

### Step 2: Data Splitting

In typical 5-fold cross validation, we need 5 training set (training_1.csv ~ training_5.csv) and 1 test set (test1.csv)

---

### Step 3: Allocate Data Encoded array

---

### Step 4: Select Balancing Methods

* "OS1-1": Oversample (positive:negative = 1:1) 
* "OS1-2": Oversample (positive:negative = 1:2)
* "US1-1": Undersample (positive:negative = 1:1)
* "US1-2": Undersample (positive:negative = 1:2)
* "FL-gamma-1": Focal loss (gamma = 1)
* "FL-gamma-2": Focal loss (gamma = 2)
* "SW": Sample weight
* "CW": Class weight
* "BB": Balance batch training
* "MFE": Mean false error

note: Before running the balancing methods, users should set the hyperparameters in the json format. 
These  json files should be placed in the USER_hyperparameters folder:



---

### Step 5: Build Model Architecture

* design your model architecture in USER_model.py

---

### Step 6: Deep Learning Training

* execute run_train.py

---


* for more information, please visit RDDL (https://github.com/cobisLab/RDDL)

---