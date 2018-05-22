import utils
import patsy # required for statsmodels
from statsmodels.tsa.arima_model import ARIMA

df_trn1, df_trn2, df_test = utils.import_datasets()

model = ARIMA(df_trn1['L_T1'], order=(5,1,0))
model_fit = model.fit()
print model_fit.summary()
