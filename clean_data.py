import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy

#set pandas output width
desired_width = 320
pd.set_option('display.width', desired_width)

df = pd.read_csv('data_for_student_case.csv')

#change date strings to datetime objects
df['bookingdate'] = pd.to_datetime(df['bookingdate'])
df['creationdate'] = pd.to_datetime(df['creationdate'])

#converting to whole euros
currency_dict = {'MXN': 0.00044, 'SEK': 0.0011, 'AUD': 0.0067, 'GBP': 0.0128, 'NZD': 0.0061}
euro = map(lambda x,y: currency_dict[y]*x, df['amount'],df['currencycode'])
df['amount'] = euro


#df['weekday'] = df['creationdate'].dt.day


df = df.fillna('none')



#weekday_category = pd.Categorical(df['weekday'])
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

#weekday_dict = dict(set(zip(weekday_category, weekday_category.codes)))
issuercountrycode_dict = dict(set(zip(issuercountrycode_category, issuercountrycode_category.codes)))
txvariantcode_dict = dict(set(zip(txvariantcode_category, txvariantcode_category.codes)))
currencycode_dict = dict(set(zip(currencycode_category, currencycode_category.codes)))
shoppercountrycode_dict = dict(set(zip(shoppercountrycode_category, shoppercountrycode_category.codes)))
shopperinteraction_dict = dict(set(zip(shopperinteraction_category, shopperinteraction_category.codes)))
cardverificationcodesupplied_dict = dict(set(zip(cardverificationcodesupplied_category, cardverificationcodesupplied_category.codes)))
accountcode_dict = dict(set(zip(accountcode_category, accountcode_category.codes)))
mail_id_dict = dict(set(zip(mail_id_category, mail_id_category.codes)))
ip_id_dict = dict(set(zip(ip_id_category, ip_id_category.codes)))
card_id_dict = dict(set(zip(card_id_category, card_id_category.codes)))

#df['weekday'] = weekday_category.codes
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

#bonus
df['bintotalamount'] = df.groupby('bin')['amount'].transform('sum')
df['binnumberofcountries'] = df.groupby('bin')['shoppercountrycode'].transform(pd.Series.nunique)
df['binnumberofcards'] = df.groupby('bin')['card_id'].transform(pd.Series.nunique)
#df['occur4'] = df.groupby('mail_id')['mail_id'].transform('count')


print df.head

print '\nshape of data'
print df.shape
print '\ntypes of index'
print df.dtypes
print '\ndescribe (only for float data)'
print df.describe()
print list(df)

df.to_csv('clean2.csv', index=False)
