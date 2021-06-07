import sys, os

import scipy
from labMTsimple.storyLab import *
import codecs  ## handle utf8

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.stats import stats


def read_docs(doc_path, users):
    docs = []
    for user in users:
        index = 0
        while index < 2000:
            user_file = doc_path + user + '.' + str(index) + '.txt'
            # user_file = doc_path + user + '.' + str(1600 + index) + '.txt'
            try:
                f = codecs.open(user_file, "r", "utf8")
                doc = f.readline()
                f.close()
                docs.append(doc)
            except:
                break
            index += 1
    return docs

def reoder(statistic_result, categories):
    i = 0
    while i < len(categories)/2 - 1:
        j = i + 1
        while j < len(categories)/2:
            if np.mean(statistic_result[i]) > np.mean(statistic_result[j]):
                tmp = statistic_result[i]
                statistic_result[i] = statistic_result[j]
                statistic_result[j] = tmp

                tmp = categories[i]
                categories[i] = categories[j]
                categories[j] = tmp
            j += 1
        i += 1
    i = int(len(categories) / 2)
    print(i)
    while i < len(categories)-1:
        j = i + 1
        while j < len(categories):
            if np.mean(statistic_result[i]) > np.mean(statistic_result[j]):
                tmp = statistic_result[i]
                statistic_result[i] = statistic_result[j]
                statistic_result[j] = tmp

                tmp = categories[i]
                categories[i] = categories[j]
                categories[j] = tmp
            j += 1
        i += 1

if __name__ == '__main__':
    lang = 'english'
    labMT, labMTvector, labMTwordList = emotionFileReader(stopval=0.0, lang=lang, returnVector=True)

    categories = ["fever", "itch", "cough", "swelling", "neck_pain", "delirium", "pneumonia", "headache", "apnea", "generic_drugs", "tremor", "anxiety", "common_cold", "nausea", "flu", "asthma", "conjunctivitis", "abdominal_pain", "arthritis", "pale_skin", "bleeding", "diarrhea"]
    # categories = ['wine', 'coffee', 'banana', 'espresso', 'croissant', 'salmon', 'quinoa', 'brie', 'macaroon',
    #               'chicken_nuggets', 'ham', 'french_fries', 'chicken_wings', 'sausage', 'biscuit', 'collards',
    #               'bbq_sauce', 'fried_chicken']
    # categories = ['golf', 'hiking', 'lacrosse', 'racquetball', 'yoga',
    #               'basketball', 'coaching', 'dancing', 'football', 'hunting']
    statistic_result = []
    for file_name in categories:
        saturdays = read_docs('tweet2doc/', [file_name])
        valence = []
        for saturday in saturdays:
            ## compute valence score and return frequency vector for generating wordshift
            saturdayValence, saturdayFvec = emotion(saturday, labMT, shift=True, happsList=labMTvector)
            ## but we didn't apply a lens yet, so stop the vectors first
            saturdayStoppedVec = stopper(saturdayFvec, labMTvector, labMTwordList, stopVal=1.0)
            ## compute valence score
            saturdayValence = emotionV(saturdayStoppedVec, labMTvector)
            valence.append(saturdayValence)
        while len(valence) < 2000:
           valence.append(np.mean(valence))
        print(len(valence))
        statistic_result.append(valence)
        # plt.hist(valence, bins='auto', color='#0504aa',
        #                     alpha=0.7, rwidth=0.85)
        #
        # x_major_locator = MultipleLocator(0.5)
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(x_major_locator)
        # plt.xlim(0, 9)
        # plt.xlabel("happiness score")
        # plt.ylabel("frequency")
        # plt.title(file_name + '(' + str(len(saturdays)) + ')|mean:' + str(np.mean(valence)) + '|std:' + str(np.std(valence)))
        # # plt.title(file_name + '(' + str(len(saturdays)) + ')|mode:' + str(stats.mode(valence)[0][0]) + '|std:' + str(np.std(valence)))
        # plt.show()
    reoder(statistic_result, categories)
    plt.boxplot(statistic_result, showfliers=False)
    ax = plt.gca()
    ax.set_xticklabels(categories, rotation=60, fontsize='small')
    plt.show()

    i = 0
    obesity_negative = []
    while i < len(categories)/2:
       obesity_negative += statistic_result[i]
       i += 1


    obesity_positive = []
    i = int(len(categories)/2)
    while i < len(categories):
       obesity_positive += statistic_result[i]
       i += 1

    print(obesity_negative)
    print(obesity_positive)
    w_value, p_value = scipy.stats.wilcoxon(obesity_negative, obesity_positive, correction=True)
    print('w = ' + str(w_value) + ', p = ' + str(p_value))
    t_value, p_value = stats.ttest_ind(obesity_negative, obesity_positive)
    print('t = ' + str(t_value) + ', p = ' + str(p_value))



