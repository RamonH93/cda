import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils

df_trn1, _, _ = utils.import_datasets()

print 'df_trn1: ' + str(list(df_trn1))
print ''

x = df_trn1['DATETIME']
y = df_trn1['L_T1']
plt.plot(x, y)
plt.show()
