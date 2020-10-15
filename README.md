TransNets
===
The code implementation for the paper：  
Rose Catherine, William Cohen. "TransNets: Learning to Transform for Recommendation." (2017).

# Environments
  + python 3.8
  + pytorch 1.60

# Dataset
  You need to prepare the following documents:  
  1. dataset(`/data/music/Digital_Music_5.json`)  
   Download from http://jmcauley.ucsd.edu/data/amazon (Choose Digital Music)

# Running
  Data preprocessing, it will generate train.csv, valid.csv and test.csv:
  ```
  python preprocess.py
  ```
  Train and evaluate the model:
  ```
  python main.py
  ```
