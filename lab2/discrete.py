import utils
import numpy as np
from saxpy.znorm import znorm
from saxpy.alphabet import cuts_for_asize
from saxpy.sax import sax_via_window
from saxpy.paa import paa
from saxpy.sax import sax_via_window
from saxpy.sax import ts_to_string
import matplotlib.pyplot as plt
from collections import Counter

if __name__ == "__main__":
    df_trn1, df_trn2, df_test = utils.import_datasets()
    df_trn2 = df_trn2[df_trn2['ATT_FLAG'] != 0]
    #col = 'F_PU5'
    #col = 'L_T2'
    col = 'P_J422'
    n = 5 #max n for ngram
    s = 5 #number of letters

    #normalize the data
    df_trn1[col] = znorm(df_trn1[col])
    df_trn2[col] = znorm(df_trn2[col])
    df_test[col] = znorm(df_test[col])


    #to plot the data:
    x = df_test['DATETIME']
    y = df_test[col]
    plt.plot(x, y)
    plt.show()
    #y = list(ts_to_string(df_trn2[col], cuts_for_asize(5)))
    df_test[col] = list(ts_to_string(df_test[col], cuts_for_asize(s)))
    y = [ord(x) for x in df_test[col]]
    x = df_test['DATETIME']
    plt.step(x, y)
    plt.show()


    def find_ngrams(input_list, n):
        grams = []
        for i in range(2,n+1):
            grams.append(zip(*[input_list[j:] for j in range(i)]))
        return grams


    def find_occurances(input_list): #number of occurances per letter sequence
        occurances = []
        for i in range(len(input_list)):
            occurances.append(Counter(input_list[i]))
        return occurances

    def find_probabilities(input_list):
        probabilities = []
        for i in range(len(input_list)):
            som = sum(input_list[i].itervalues())
            dict = {}
            for key, value in input_list[i].iteritems():
                dict.update({key:(float(value)/float(som))})
            probabilities.append(dict)
        return probabilities

    df_trn1[col] = list(ts_to_string(df_trn1[col], cuts_for_asize(s)))
    df_trn2[col] = list(ts_to_string(df_trn2[col], cuts_for_asize(s)))
    df_test[col] = list(ts_to_string(df_test[col], cuts_for_asize(s)))

    ngrams = find_ngrams(df_trn1[col] , n)
    ngrams2 = find_ngrams(df_trn2[col], n)
    ngrams3 = find_ngrams(df_trn2[col], n)
    combinedngrams = ngrams + ngrams2


    occurances =  find_occurances(combinedngrams)
    probas = find_probabilities(occurances)

    counter = n-1
    for r in ngrams3[3]:
        #if probas[3].get(r) < 0.0001:
        if probas[3].get(r) < 0.0000000001:
            print df_test['DATETIME'][counter] #print datatime if below treshold
        counter += 1

