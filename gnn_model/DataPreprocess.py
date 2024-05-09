import pickle
import numpy as np
import pandas as pd

# Open the .pkl file in binary mode
with open('./data/pol_new_data_time.pkl', 'rb') as f:
    # Load the object stored in the file
    data = pickle.load(f)
data
