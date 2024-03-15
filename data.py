#Importing the modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Dataset loading
path="/heart.csv"
heart_df = pd.read_csv(path)
heart_df.head()

#Creation of subset dataset for having disease and not having disease
chol_having = heart_df.loc[heart_df['target'] == 1, 'chol']
chol_having
chol_nohaving = heart_df.loc[heart_df['target'] == 0, 'chol']
chol_nohaving

#Sample dataframe development
havingsample = chol_having_disease.sample(n = 50)

#Compute the mean and std
sample_mean  = havingsample.mean()
sample_std  = havingsample.std()

#Evaluating the confidence interval for 95% probability
sample_mean - 2 * sample_std / np.sqrt(havingsample.size), sample_mean + 2 * sample_std / np.sqrt(havingsample.size)

#Evaluating the confidence interval for 68% probability
sample_mean -  sample_std / np.sqrt(havingsample.size), sample_mean + sample_std / np.sqrt(havingsample.size)

#Normalization of the values to the same scale
def normalize(x):
    normarray = (x - np.mean(x)) / np.std(x)
    return normarray
norm_having= normalize(chol_having)

#Compute the mean and std for normalized data
sample_mean  = norm_having.mean()
sample_std  = norm_having.std()

