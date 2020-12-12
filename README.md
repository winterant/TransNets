TransNets
===
> The code implementation for the paper：  
Rose Catherine, William Cohen. "TransNets: Learning to Transform for Recommendation." (2017).

# Environments
  + python 3.8
  + pytorch 1.70

# Dataset

`data/Digital_Music_5.json.gz`

Download from http://jmcauley.ucsd.edu/data/amazon (Choose Digital Music)

Preprocess origin dataset in json format to **train.csv,valid.csv and test.csv.**
```
python preprocess.py
```

# Running

Train and evaluate the model

+ **TransNet**

```
python main.py
```

+ **TransNet-Ext**

```
python main.py --extension True --user_count 5541 --item_count 3568
```
You must setting `user_count` and `item_count` 
which you can get their values from output of `python preprocess.py`
