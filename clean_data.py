import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
from datetime import timedelta
from sklearn.preprocessing import StandardScaler

#set pandas output width
desired_width = 320
pd.set_option('display.width', desired_width)

df = pd.read_csv('data_for_student_case.csv')

#change date strings to datetime objects
df['bookingdate'] = pd.to_datetime(df['bookingdate'])
df['creationdate'] = pd.to_datetime(df['creationdate'])

#converting to whole euros & scale with StandardScaler
currency_dict = {'MXN': 0.00044, 'SEK': 0.0011, 'AUD': 0.0067, 'GBP': 0.0128, 'NZD': 0.0061}
euro = map(lambda x,y: currency_dict[y]*x, df['amount'],df['currencycode'])
df['amount'] = euro
df['amount'] = StandardScaler().fit_transform(df['amount'].values.reshape(-1, 1))

#fill Nan values with 'none'
df = df.fillna('none')

#Categorize String values
issuercountrycode_category = pd.Categorical(df['issuercountrycode'])
txvariantcode_category = pd.Categorical(df['txvariantcode'])
currencycode_category = pd.Categorical(df['currencycode'])
shoppercountrycode_category = pd.Categorical(df['shoppercountrycode'])
shopperinteraction_category = pd.Categorical(df['shopperinteraction'])
cardverificationcodesupplied_category = pd.Categorical(df['cardverificationcodesupplied'])
accountcode_category = pd.Categorical(df['accountcode'])
mail_id_category = pd.Categorical(df['mail_id'])
ip_id_category = pd.Categorical(df['ip_id'])
card_id_category = pd.Categorical(df['card_id'])

df['issuercountrycode'] = issuercountrycode_category.codes
df['txvariantcode'] = txvariantcode_category.codes
df['currencycode'] = currencycode_category.codes
df['shoppercountrycode'] = shoppercountrycode_category.codes
df['shopperinteraction'] = shopperinteraction_category.codes
df['cardverificationcodesupplied'] = cardverificationcodesupplied_category.codes
df['accountcode'] = accountcode_category.codes
df['mail_id'] = mail_id_category.codes
df['ip_id'] = ip_id_category.codes
df['card_id'] = card_id_category.codes
df = df[df.simple_journal != 'Refused']
journal = map(lambda x:1 if str(x) == 'Chargeback' else 0 if str(x) == 'Settled' else 0, df['simple_journal'])
df['simple_journal'] = journal
df['cvcresponsecode'] = map(lambda x:3 if x > 2 else x+0, df['cvcresponsecode']) #0 = Unknown, 1=Match, 2=No Match, 3=Not checked

#extra columns for bonus task
df['bintotalamount'] = df.groupby('bin')['amount'].transform('sum') #total amount of all transaction of a bin
df['binnumberofcountries'] = df.groupby('bin')['shoppercountrycode'].transform(pd.Series.nunique) #number of different country ip addresses used by a bin
df['hour'] = df['creationdate'].dt.hour #helper column
df['min'] = df['creationdate'].dt.minute #helper column
df['minuteamount'] = df.groupby(['hour','min'])['mail_id'].transform('count') #number of transactions in the minute of a day, total of all days
del df['hour']
del df['min']

df.to_csv('clean2.csv', index=False)
