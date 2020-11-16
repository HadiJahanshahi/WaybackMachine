import networkx as nx
import random
from tqdm import tqdm 
import time
import matplotlib.pyplot as plt
from collections import OrderedDict
import statistics
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import plotly.graph_objects as go
import plotly.express as px
import ast
import json
import pickle
import copy
#import datetime

def isNaN(num):
    return num != num
def isnotNaN(inp):
    return inp == inp
def convert_string_to_array(string_):
    """ 
    it converts "[1,2,3]" to [1,2,3]
    """
    array_string = ''.join(string_.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))
def string_to_time(string, format_ = '%Y-%m-%dT%H:%M:%SZ'):
    """ 
    * Default format is '%Y-%m-%dT%H:%M:%SZ'.
    
    string_to_time(string, format_ = '%Y-%m-%dT%H:%M:%SZ')
    """
    return datetime.strptime(string, format_)
    
def mean_list (x):
    return np.mean(list(x))
# %matplotlib inline