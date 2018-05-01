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

df = df.fillna('none')

print list(df)
print df.issuercountrycode.unique()
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
journal = map(lambda x:1 if str(x) == 'Chargeback' else 0 if str(x) == 'Settled' else 0, df['simple_journal'])
df['simple_journal'] = journal
df['cvcresponsecode'] = map(lambda x:3 if x > 2 else x+0, df['cvcresponsecode']) #0 = Unknown, 1=Match, 2=No Match, 3=Not checked

print '\nshape of data'
print df.shape
print '\ntypes of index'
print df.dtypes
print '\ndescribe (only for float data)'
print df.describe()




d_view = [ (v,k) for k,v in txvariantcode_dict.iteritems() ]
d_view.sort() # natively sort tuples by first element
for v,k in d_view:
    print "\"%s\"," % (k),

#df.to_csv('clean.csv', index=False)


#df1 = df_clean.ix[(df_clean['label_int']==1) | (df_clean['label_int']==0)]#237036 instances

#print df1.head()
#
# df_input = (df1[['issuercountrycode', 'cardtype', 'issuer_id', 'currencycode', 'shoppercountrycode', 'shoppingtype', 'cvcsupply', 'cvcresponse_int', 'merchant_id', 'euro', 'label_int']])
# df_input[['issuer_id','label_int']] = df_input[['issuer_id','label_int']].astype(int)
# print df_input.dtypes
# x = df_input[df_input.columns[0:-1]].as_matrix()
# y = df_input[df_input.columns[-1]].as_matrix()
#
#
# unique, counts = numpy.unique(y, return_counts=True)
# print dict(zip(unique, counts))
#
# import matplotlib.pyplot as plt
# plt.rc("font", size=14)
# from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import train_test_split
# import seaborn as sns
# sns.set(style="white")
# sns.set(style="whitegrid", color_codes=True)
#
# sns.heatmap(df_input.corr())
# plt.show()