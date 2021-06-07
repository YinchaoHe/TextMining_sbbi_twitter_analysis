import codecs

import json
import math
import os
import shutil

import preprocessor as p
import re
import numpy as np
import seaborn
from kneed import KneeLocator
from scipy.spatial.distance import cdist
from sklearn.decomposition import NMF, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import pickle
from mpl_toolkits.mplot3d import Axes3D

def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 2, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

def tweets_processing(doc):
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.SMILEY, p.OPT.NUMBER)
    cleaned_doc = p.clean(doc)
    myre = re.compile(u'('
                      u'\ud83c[\udf00-\udfff]|'
                      u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
                      u'[\u2600-\u26FF\u2700-\u27BF])+',
                      re.UNICODE)
    myre.sub('', cleaned_doc)
    stopwords = ['\\udc47', '\\u2019', '\\u2026', '\\ud83c', '\\ud83d', '\\ufe0f', '\\udc99', '\\ude17', '\\n', '\\ude36', '\\ud83e', 'amp', '\\udf55', '\\udc40']
    for word in stopwords:
        if word in cleaned_doc:
            cleaned_doc = cleaned_doc.replace(word, "")

    return cleaned_doc

def tweets2doc(users_path, doc_path):
    users = os.listdir(users_path)

    try:
        os.mkdir(doc_path)
    except:
        pass

    index = 0
    for user in users:
        # if index > 1600:
        #     print(index)
        #     break
        index += 1
        user_file = users_path + user
        print(user_file)
        f = codecs.open(user_file, "r", "utf8")
        user_infos = f.readlines()
        i = 0
        for user_info in user_infos:
            try:
                user_text_blo = str(user_info).split('\"text": \"')[1]
            except:
                continue
            user_text = user_text_blo.split('\",')[0]
            cleaned_text = tweets_processing(user_text)
            if len(cleaned_text) < 1:
                continue
            with open(doc_path + user.replace('json', str(i)+'.txt'), 'w') as d:
                d.write(cleaned_text)
            d.close()
            i += 1
        f.close()

def read_docs(doc_path, users):
    docs = []
    for user in users:
        index = 0
        while index < 150:
            user_file = doc_path + user + '.' + str(index) + '.txt'
            print(user_file)
            # user_file = doc_path + user + '.' + str(1600 + index) + '.txt'
            try:
                with open(user_file, 'r') as f:
                    doc = f.readline()
                f.close()
                docs.append(doc)
            except:
                break
            index += 1
    return docs

    # for user in users:
    #     if 'obe'
    #     if index < 1600:
    #         index += 1
    #         continue
    #
    #     user_file = doc_path + user
    #     with open(user_file, 'r') as f:
    #         doc = f.readline()
    #     f.close()
    #     docs.append(doc)
    # return docs

def top_features_extraction(tfidf_vectorizer, nmf, n_top_w):
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    n_top_words = n_top_w
    topics_top_features = []
    for topic_idx, topic in enumerate(nmf.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [tfidf_feature_names[i] for i in top_features_ind]
        topics_top_features.append(top_features)

    plot_top_words(nmf, tfidf_feature_names, n_top_words,
                   'Topics in NMF model (generalized Kullback-Leibler divergence)')

    return topics_top_features

def feature_matrix(topics_top_features, docs, C, doc_number):
    multi_topics_feature = []
    multi_topics_mean_vector = []
    for features_1_topic in topics_top_features:
        features_inOne = []
        appearance = 0
        feature_number = len(features_1_topic)
        for top_feature in features_1_topic:
            feature_row = []

            for doc in docs:
                feature_count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(top_feature), doc))
                feature_row.append(feature_count)
                appearance += feature_count
            features_inOne.append(feature_row)
        #mean = appearance / (doc_number * feature_number)
        mean = appearance / doc_number
        features_inOne_matrix = np.array(features_inOne)
        docs_degree = np.sum(features_inOne_matrix, axis = 0)

        index_c = 0
        remove_c = []
        while index_c < len(docs_degree):
            if docs_degree[index_c] < mean:
                remove_c.append(index_c)
                i = 0
                while i < feature_number:
                    features_inOne_matrix[i][index_c] = 0
                    i += 1
            index_c += 1

        features_inOne_cleaned_matrix = np.delete(features_inOne_matrix, remove_c, 1)
        support_nu = np.size(features_inOne_cleaned_matrix, 1)
        print(features_1_topic[0] + '\'s support number: ' + str(support_nu))
        if support_nu >= C:
            mean_vector = np.mean(features_inOne_matrix, 1)
            multi_topics_mean_vector.append(mean_vector)
            multi_topics_feature.append(features_inOne_matrix)
    return multi_topics_feature, multi_topics_mean_vector

def cluster_creator(multi_topics_feature, multi_topics_mean_vector, n_clusters, n_top_w):
    # Cluster on features_inOne_cleaned_matrix which need to be reversed row is document column is feature, centrio is multi_topics_mean_vector [merge all keywords]
    topic_number = len(multi_topics_mean_vector)
    index = 0
    cluster_centrias = []
    while index < topic_number:
        front = 0
        centria = list(multi_topics_mean_vector[index])
        while front < index:
            front_list = [0] * len(multi_topics_mean_vector[front])
            centria = front_list + centria
            front += 1

        back = index + 1
        while back < topic_number:
            back_list = [0] * len(multi_topics_mean_vector[back])
            centria  += back_list
            back += 1
        cluster_centrias.append(centria)
        index += 1

    max_dimension = max([i.shape[1] for i in multi_topics_feature])
    minx_dimension = min([i.shape[1] for i in multi_topics_feature])
    print('max_dimension' + str(max_dimension))
    print('minx_dimension' + str(minx_dimension))

    document_matrix = np.concatenate((multi_topics_feature[:]), axis=0).T
    print(document_matrix)

    cluster_centrias = np.array(cluster_centrias)

    model = KMeans(n_clusters = n_clusters, init=cluster_centrias).fit(document_matrix)
    label = model.predict(document_matrix)
    with open("tweets_model.pkl", 'wb') as file:
        pickle.dump(model, file)
    return label, document_matrix, model.cluster_centers_

def plot_painter(label, document_matrix, categories):

    u_labels = np.unique(label)
    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=3))])
    df = pipeline.fit_transform(document_matrix)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    i = 0
    categories = ["fever", "swelling", "pneumonia",  "bleeding", "fever_swelling", "fever_pneumonia", "fever_bleeding", "swelling_pneumonia", "swelling_bleeding", "pneumonia_bleeding", "itch", "cough",     "neck_pain", "delirium", "headache", "apnea", "generic_drugs", "tremor", "anxiety", "common_cold", "nausea", "flu", "asthma", "conjunctivitis", "abdominal_pain", "arthritis", "pale_skin",  "diarrhea"]
    # categories = ['wine', 'coffee', 'banana', 'espresso', 'croissant', 'salmon', 'quinoa', 'brie', 'macaroon',
    #                'chicken_nuggets', 'ham', 'french_fries', 'chicken_wings', 'sausage', 'biscuit', 'collards', 'bbq_sauce', 'fried_chicken']
    # categories = ['golf', 'hiking', 'lacrosse', 'racquetball', 'yoga',
    #               'basketball', 'coaching', 'dancing', 'football', 'hunting']
    for i in u_labels:
        print(str(i) + ': ' + categories[i])
        x = df[label == i, 0]
        print(len(x))
        y = df[label == i, 1]
        z = df[label == i, 2]
        ax.scatter(xs = x, ys = y, zs = z, label=categories[i])

    ax.legend(loc="best")
    plt.show()



def topic_extractor(categories, n_top_w):
    topics_top_features = []
    for category in categories:
        docs = read_docs('tweet2doc/', [category])

        n_features = 8
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=0.01,
                                           max_features=n_features,
                                           stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(docs)

        n_components = 1
        nmf = NMF(n_components=n_components, random_state=1,
                  beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
                  l1_ratio=.5).fit(tfidf)

        n_top_w = n_top_w
        topic_top_features = top_features_extraction(tfidf_vectorizer, nmf, n_top_w)
        topics_top_features += topic_top_features

    with open('topics.json', 'w') as topic_f:
        topic = {"topics": topics_top_features}
        json.dump(topic, topic_f)
    topic_f.close()
    return topics_top_features

def min_distance(centers):
    ps_distance = []
    for p1 in centers:
        p1_distance = []
        for p2 in centers:
            i = 0
            distance = 0
            while i < len(p1):
                distance = distance + (p2[i] - p1[i])**2
                i += 1
            distance = round(math.sqrt(distance), 2)
            if distance == 0:
                distance = 10
            p1_distance.append(distance)
        ps_distance.append(p1_distance)
    print(ps_distance)

    # with open("new_file.csv", "w+") as my_csv:
    #     csvWriter = csv.writer(my_csv, delimiter=',')
    #     csvWriter.writerows(ps_distance)
    #
    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=3))])
    df = pipeline.fit_transform(ps_distance)

    # show Hierarchically-clustered Heatmap
    seaborn.clustermap(centers, z_score=0, cmap="vlag")
    plt.show()


    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    i = 0
    categories = ["fever", "swelling", "pneumonia",  "bleeding", "fever_swelling", "fever_pneumonia", "fever_bleeding", "swelling_pneumonia", "swelling_bleeding", "pneumonia_bleeding", "itch", "cough",     "neck_pain", "delirium", "headache", "apnea", "generic_drugs", "tremor", "anxiety", "common_cold", "nausea", "flu", "asthma", "conjunctivitis", "abdominal_pain", "arthritis", "pale_skin",  "diarrhea"]
    # categories = ['wine', 'coffee', 'banana', 'espresso', 'croissant', 'salmon', 'quinoa', 'brie', 'macaroon',
    #               'chicken_nuggets', 'ham', 'french_fries', 'chicken_wings', 'sausage', 'biscuit', 'collards', 'bbq_sauce', 'fried_chicken']
    # categories = ['golf', 'hiking', 'lacrosse', 'racquetball', 'yoga',
    #               'basketball', 'coaching', 'dancing', 'football', 'hunting']
    while i < len(ps_distance):
        x = df[i][0]
        y = df[i][1]
        z = df[i][2]
        ax.scatter(xs = x, ys = y, zs = z, label=categories[i])
        i += 1
    ax.legend(loc="best")
    plt.show()

    distortions = []
    K = range(1, 7)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(centers)
        kmeanModel.fit(centers)
        distortions.append(sum(np.min(cdist(centers, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / centers.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

    kn = KneeLocator(
        K,
        distortions,
        curve='convex',
        direction='decreasing',
        interp_method='interp1d',
    )
    print("kn.knee" + str(kn.knee))
    # result = {0:0}
    # i = 0
    # min_dis = min(ps_distance[0])
    # j = ps_distance[0].index(min_dis)
    # ps_distance[j][0] = 10
    # result[j] = min_dis
    # while i < len(centers):
    #     j_pre = j
    #     min_dis = min(ps_distance[j])
    #     j = ps_distance[j].index(min_dis)
    #     flag = True
    #     while flag:
    #         if len(result) >= 21:
    #             break
    #         if j in result.keys():
    #             ps_distance[j_pre][j] = 10
    #             min_dis = min(ps_distance[j_pre])
    #             j = ps_distance[j_pre].index(min_dis)
    #         else:
    #             break
    #
    #     ps_distance[j][j_pre] = 10
    #     result[j] = min_dis
    #     i += 1
    #     print(result)


def init_model():
    # categories = ['wine', 'coffee', 'banana', 'espresso', 'croissant', 'salmon', 'quinoa', 'brie', 'macaroon',
    #                'chicken_nuggets', 'ham', 'french_fries', 'chicken_wings', 'sausage', 'biscuit', 'collards', 'bbq_sauce', 'fried_chicken']
    # categories = ['golf', 'hiking', 'lacrosse', 'racquetball', 'yoga',
    #               'basketball', 'coaching', 'dancing', 'football', 'hunting']
    categories = ["fever", "swelling", "pneumonia",  "bleeding", "fever_swelling", "fever_pneumonia", "fever_bleeding", "swelling_pneumonia", "swelling_bleeding", "pneumonia_bleeding", "itch", "cough",     "neck_pain", "delirium", "headache", "apnea", "generic_drugs", "tremor", "anxiety", "common_cold", "nausea", "flu", "asthma", "conjunctivitis", "abdominal_pain", "arthritis", "pale_skin",  "diarrhea"]
    n_top_w = 10
    topics_top_features = topic_extractor(categories, n_top_w)
    docs = read_docs('tweet2doc/', categories)
    doc_number = len(docs)
    # C = len(docs) / n_components
    C = 2
    print("C is: " + str(C))
    multi_topics_feature, multi_topics_mean_vector = feature_matrix(topics_top_features, docs, C, doc_number)
    print('number of topics: ' + str(len(multi_topics_feature)))

    label, clusters, centers = cluster_creator(multi_topics_feature, multi_topics_mean_vector, len(multi_topics_feature), n_top_w )
    index = 0
    # while index < doc_number:
    #     print('label: ' + str(label[index]))
    #     print(docs[index])
    #     index += 1

    plot_painter(label, clusters, categories)
    min_distance(centers)

def tweet_feature(topics_top_features, docs):
    multi_topics_feature = []
    for features_1_topic in topics_top_features:
        print(features_1_topic)
        features_inOne = []
        appearance = 0
        feature_number = len(features_1_topic)
        for top_feature in features_1_topic:
            feature_row = []
            for doc in docs:
                #### problem 1: weight,rank
                feature_count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(top_feature), doc))
                feature_row.append(feature_count)
                appearance += feature_count
            features_inOne.append(feature_row)
        features_inOne_matrix = np.array(features_inOne)
        multi_topics_feature.append(features_inOne_matrix)
    return multi_topics_feature

def user_type(label, document_matrix):
    dic = {
        "obesity_positive": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "obesity_negative": [11, 12, 13, 14, 15, 16, 17]
    }
    u_labels = np.unique(label)

    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=2))])

    df = pipeline.fit_transform(document_matrix)
    obesity_positive = 0
    obesity_negative = 0
    ###score
    for i in u_labels:
        x = df[label == i, 0]
        if i in dic['obesity_positive']:
            obesity_positive += len(x)
        else:
            obesity_negative += len(x)

    if obesity_negative > obesity_positive:
        print(obesity_negative)
        print('obesity_negative')
    else:
        print(obesity_positive)
        print('obesity_positive')


def training():
    # shutil.rmtree('tweet2doc/')
    # users_paths = ['obesity/']
    # for users_path in users_paths:
    #     tweets2doc(users_path, doc_path = 'tweet2doc/')
    init_model()

def valid_model():
    with open("tweets_model.pkl", 'rb') as file:
        model = pickle.load(file)
    file.close()
    with open('topics.json') as file:
        data = json.load(file)
    file.close()
    topics_top_features = data['topics']

    docs = read_docs('tweet2doc_predict/', ['Billayee_', 'AmazingWomen'])


    user_feature = tweet_feature(topics_top_features, docs)
    document_matrix = np.concatenate((user_feature[:]), axis=0).T
    label = model.predict(document_matrix)
    print(label)
    exit()
    user_type(label, document_matrix)
    #plot_painter(label, document_matrix)

def testing():
    # shutil.rmtree('tweet2doc_predict/')
    tweets2doc(users_path='obesity/', doc_path='tweet2doc_predict/')
    valid_model()

if __name__ == '__main__':
    training()
    #testing()