import pandas as pd
from sklearn import tree
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split

student_data = pd.read_csv(open('data/student/student-mat.csv', 'r'), sep=';')
portugese_data = pd.read_csv(open('data/student/student-por.csv', 'r'), sep=';')

student_data.merge(portugese_data, on=[
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"
])
