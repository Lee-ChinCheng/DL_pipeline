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
│   ├── log.txt              # training logs
│   ├── test1.csv            # test dataset
│   ├── training_1.csv       # training dataset 1 out of 5
│   ├──        
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