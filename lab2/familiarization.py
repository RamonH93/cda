import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# set pandas output width
desired_width = 320
pd.set_option('display.width', desired_width)

# read datasets
df_trn1 = pd.read_csv('BATADAL_dataset03.csv', nrows=8762)
df_trn2 = pd.read_csv('BATADAL_dataset04.csv', nrows=100)
df_test = pd.read_csv('BATADAL_test_dataset.csv', nrows=100)

# clean datasets
df_trn1['DATETIME'] = pd.to_datetime(df_trn1['DATETIME'], dayfirst=True)  # convert timestamps
print df_trn1['DATETIME']
df_trn2['DATETIME'] = pd.to_datetime(df_trn2['DATETIME'], dayfirst=True)  # convert timestamps
df_trn2.rename(columns=lambda x: x.lstrip(), inplace=True) # remove whitespaces in headers
df_test['DATETIME'] = pd.to_datetime(df_test['DATETIME'], dayfirst=True)  # convert timestamps

print 'df_trn1: ' + str(list(df_trn1))
print 'df_trn2: ' + str(list(df_trn2))
print 'df_test: ' + str(list(df_test))
print ''

x = df_trn1['DATETIME']
y = df_trn1['L_T1']

plt.plot(x, y)
plt.show()
