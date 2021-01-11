# AUDIENCE SEGMENTATION

# FILE NAME: main.py

# DEVELOPED BY: Vigneshwar Ravichandar, Moulishankar M R

# TOPICS: Regression, Machine Learning, TensorFlow

# IMPORTING REQUIRED MODULES
import os
import argparse

# DISABLING TENSORFLOW DEBUGGING INFORMATION
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
print("TensorFlow Debugging Information is hidden.")

# FUNCTION TO CONVERT STR INPUT TO BOOL
def strBool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expected a Boolean Value.')

