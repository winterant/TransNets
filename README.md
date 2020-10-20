TransNets
===
> The code implementation for the paper：  
Rose Catherine, William Cohen. "TransNets: Learning to Transform for Recommendation." (2017).

# Environments
  + python 3.8
  + pytorch 1.60

# Dataset
  You need to prepare the following documents:  
  1. dataset(`/data/music/Digital_Music_5.json`)  
   Download from http://jmcauley.ucsd.edu/data/amazon (Choose Digital Music)

# settings

1. config.py  
   You can see many settings of hyper parameters for this project.

2. preprocess.py. Running it you will get: 
   + number of users/items which you can fill in config.py 
   + file **train.csv, valid.csv and test.csv**
   ```
   python preprocess.py
   ```
3. The default model is **TransNets** without extension.  
   If you want to run **TransNets-EXT**, 
   appointing that `extend_model=True` when initialized a Source Net in main.py,
   and then you must fill in config.py with correct values of `user_count` and `item_count`
   which you can see in the output of preprocess.py.
   ```
   SourceNet(config, word_emb, extend_model=True)
   ```


# Running

Train and evaluate the model:
```
python main.py
```
