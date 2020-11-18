TransNets
===
> The code implementation for the paper：  
Rose Catherine, William Cohen. "TransNets: Learning to Transform for Recommendation." (2017).

# Environments
  + python 3.8
  + pytorch 1.60

# Dataset
  You need to prepare the following documents:  
  1. dataset(`/data/music/Digital_Music_5.json` or `/data/music/Digital_Music_5.json.gz`)  
   Download from http://jmcauley.ucsd.edu/data/amazon (Choose Digital Music)

# settings

1. config.py  
   You can see many settings of hyper parameters for this project.

2. The default model is **TransNets** without extension.  
    If you want to run **TransNets-EXT**, 
    Setting `extend_model=True` when initialized a Source Net in main.py.  
    And you must setting `user_count` and `item_count` in `config.py` 
    which you can get their values from output of `python preprocess.py`
    
    + main.py
    ```
    SourceNet(config, word_emb, extend_model=True)
    ```
    + config.py
    ```
    class Config:
        # others...
        user_count = 5541  # Parameter of TransNet-EXT. Got from output of running preprocess.py
        item_count = 3568
        # others...
   ```


# Running

Preprocess origin dataset in json format to train.csv,valid.csv and test.csv.  
**Rewrite some necessary settings** in this file before running it. 
```
python preprocess.py
```

Train and evaluate the model:
```
python main.py
```
