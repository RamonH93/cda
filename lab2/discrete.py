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

    # to plot the data:
    # x = df_trn2['DATETIME']
    # y = df_trn2['L_T7']
    # plt.plot(x, y)
    # plt.show()
    # y = list(ts_to_string(df_trn2['L_T7'], cuts_for_asize(5)))
    # x = df_trn2['DATETIME']
    # plt.step(x, y)
    # plt.show()

    #normalize the data
    df_trn1['L_T7'] = znorm(df_trn1['L_T7'])
    df_trn2['L_T7'] = znorm(df_trn2['L_T7'])

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


    n = 5 #max n for ngram
    s = 4 #number of letters

    df_trn1['L_T7'] = list(ts_to_string(df_trn1['L_T7'], cuts_for_asize(s)))
    df_trn2['L_T7'] = list(ts_to_string(df_trn2['L_T7'], cuts_for_asize(s)))

    ngrams = find_ngrams(df_trn1['L_T7'] , n)
    ngrams2 = find_ngrams(df_trn2['L_T7'], n)
    occurances =  find_occurances(ngrams)
    probas = find_probabilities(occurances)

    counter = n-1
    for r in ngrams2[3]:
        if probas[3].get(r) < 0.0001:
            print df_trn2['DATETIME'][counter] #print datatime if below treshold
        counter += 1

