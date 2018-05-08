import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
sns.set(style="white", color_codes=True)

#set pandas output width
desired_width = 320
pd.set_option('display.width', desired_width)

df = pd.read_csv('clean.csv')

df['bookingdate'] = pd.to_datetime(df['bookingdate'])
df['creationdate'] = pd.to_datetime(df['creationdate'])

print list(df)

del df['bookingdate']
del df['creationdate']
del df['amount']
del df['card_id']
del df['ip_id']
del df['txid']
del df['mail_id']
del df['bin']

df1 = df[df['simple_journal'] == 1]
df0 = df[df['simple_journal'] == 0]


# columns = ['issuercountrycode', 'txvariantcode','cardverificationcodesupplied', 'currencycode', 'cvcresponsecode', 'shoppercountrycode', 'shopperinteraction']
# df1.hist(density=True)
# df0.hist(density=True)
#
# plt.show()

# sns.boxplot(x="simple_journal", y="amount", data=df)
# sns.FacetGrid(df, hue="simple_journal", size=6) \
# .map(sns.kdeplot, "cardverificationcodesupplied") \
#    .add_legend()
# plt.show()

# f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
# sns.distplot(df['currencycode'], kde=False, color="b", ax=axes[0, 0], norm_hist=True)


text_values = ["cirrus", "electron", "mc", "mccredit", "mcdebit", "visa", "visabusiness", "visaclassic", "visacorporate", "visadebit", "visagold", "visaplatinum", "visapurchasing", "visasignature", "vpay"]
x_values = np.arange(0, len(text_values) + 1, 1)
plt.hist(df1['txvariantcode'], bins=x_values, alpha=0.8, density=True, color="blue", label="chargeback")
plt.hist(df0['txvariantcode'], bins=x_values, alpha=0.8, density=True, color="orange", label="no chargeback")
plt.xticks(x_values+0.5, text_values, rotation=90)
plt.xlabel("txvariantcode", fontsize=16)
plt.ylabel("frequency", fontsize=16)
plt.legend(loc='upper right')
plt.show()


text_values = ["AD", "AE", "AI", "AL", "AM", "AO", "AR", "AT", "AU", "AZ", "BA", "BB", "BD", "BE", "BF", "BG", "BH", "BJ", "BM", "BN", "BO", "BR", "BS", "BW", "BY", "BZ", "CA", "CD", "CG", "CH", "CK", "CL", "CM", "CN", "CO", "CR", "CV", "CW", "CY", "CZ", "DE", "DK", "DO", "DZ", "EC", "EE", "EG", "ES", "FI", "FJ", "FR", "GB", "GE", "GH", "GI", "GR", "GT", "HK", "HN", "HR", "HU", "ID", "IE", "IL", "IN", "IQ", "IS", "IT", "JM", "JO", "JP", "KE", "KH", "KR", "KW", "KY", "KZ", "LA", "LB", "LK", "LT", "LU", "LV", "LY", "MD", "ME", "MK", "MN", "MO", "MT", "MU", "MV", "MX", "MY", "NG", "NL", "NO", "NZ", "OM", "PA", "PE", "PH", "PK", "PL", "PR", "PT", "PY", "QA", "RO", "RS", "RU", "SA", "SB", "SE", "SG", "SI", "SK", "SL", "SV", "TH", "TJ", "TM", "TR", "TT", "TW", "TZ", "UA", "US", "UY", "VE", "VG", "VN", "ZA", "ZM", "ZW", "ZZ", "none"]
x_values = np.arange(0, len(text_values) + 1, 1)
plt.hist(df1['issuercountrycode'], bins=x_values, alpha=0.8, density=True, color="blue", label="chargeback")
plt.hist(df0['issuercountrycode'], bins=x_values, alpha=0.8, density=True, color="orange", label="no chargeback")
plt.xticks(np.array([8, 51, 92, 113]), text_values, rotation=90)
plt.xlabel("issuercountrycode", fontsize=16)
plt.ylabel("frequency", fontsize=16)
plt.legend(loc='upper right')
plt.show()

text_values = ["APACAccount", "MexicoAccount", "SwedenAccount", "UKAccount"]
x_values = np.arange(0, len(text_values) + 1, 1)
plt.hist(df1['accountcode'], bins=x_values, alpha=0.8, density=True, color="blue", label="chargeback")
plt.hist(df0['accountcode'], bins=x_values, alpha=0.8, density=True, color="orange", label="no chargeback")
plt.xticks(x_values+0.5, text_values, rotation=90)
plt.xlabel("accountcode", fontsize=16)
plt.ylabel("frequency", fontsize=16)
plt.legend(loc='upper right')
plt.show()


text_values = ["unknown", "match", "no code", "not checked"]
x_values = np.arange(0, len(text_values) + 1, 1)
plt.hist(df1['cvcresponsecode'], bins=x_values, alpha=0.8, density=True, color="blue", label="chargeback")
plt.hist(df0['cvcresponsecode'], bins=x_values, alpha=0.8, density=True, color="orange", label="no chargeback")
plt.xticks(x_values+0.5, text_values, rotation=90)
plt.xlabel("cvcresponsecode", fontsize=16)
plt.ylabel("frequency", fontsize=16)
plt.legend(loc='upper right')
plt.show()

print list(df)

names = ['issuercountrycode', 'txvariantcode', 'currencycode', 'shoppercountrycode', 'shopperinteraction', 'simple_journal', 'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode']
data = df
correlations = data.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1, cmap=plt.cm.nipy_spectral)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.xticks(rotation=70)
plt.show()
