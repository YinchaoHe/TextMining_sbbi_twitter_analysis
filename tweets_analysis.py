import os
import shutil

import preprocessor as p
import re
import numpy as np
from sklearn.decomposition import NMF, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(3, 4, figsize=(30, 15), sharex=True)
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
    stopwords = ['\\udc47', '\\u2019', '\\u2026', '\\ud83c', '\\ud83d', '\\ufe0f', '\\udc99', '\\ude17', '\\n', '\\ude36', '\\ud83e', 'amp', '\\udf55']
    for word in stopwords:
        if word in cleaned_doc:
            cleaned_doc = cleaned_doc.replace(word, "")
    return cleaned_doc

def tweets2doc(users_path):
    users = os.listdir(users_path)
    doc_path = 'tweet2doc/'
    try:
        os.mkdir(doc_path)
    except:
        pass

    index = 0
    for user in users:
        if index > 20:
            break
        index += 1
        user_file = users_path + user
        print(user_file)
        doc = ""
        with open(user_file, 'r') as f:
            user_infos = f.readlines()
            for user_info in user_infos:
                user_text_blo = str(user_info).split('\"text": \"')[1]
                user_text = user_text_blo.split('\",')[0]
                cleaned_text = tweets_processing(user_text)
                doc = doc + cleaned_text
        f.close()

        with open(doc_path + user.replace('json', 'txt'), 'w') as f:
            f.write(doc)
        f.close()

def read_docs():
    doc_path = 'tweet2doc/'

    users = os.listdir(doc_path)
    docs = []
    index = 0
    for user in users:
        # if index < 202:
        #     index += 1
        # else:
        #     break
        user_file = doc_path + user
        with open(user_file, 'r') as f:
            doc = f.readline()
        f.close()
        docs.append(doc)
    return docs


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
        mean = appearance / (doc_number * feature_number)
        features_inOne_matrix = np.array(features_inOne)
        docs_degree = np.sum(features_inOne_matrix, axis = 0)

        index_c = 0
        remove_c = []
        while index_c < len(docs_degree):
            if docs_degree[index_c] < mean:
                remove_c.append(index_c)
            index_c += 1

        features_inOne_cleaned_matrix = np.delete(features_inOne_matrix, remove_c, 1)
        support_nu = np.size(features_inOne_cleaned_matrix, 1)
        print(support_nu)
        if support_nu >= C:
            mean_vector = np.mean(features_inOne_cleaned_matrix, 1)
            multi_topics_mean_vector.append(mean_vector)
            # print('the mean for the topic is: ' + str(mean))
            # print(features_inOne_cleaned_matrix)
            # print(features_inOne_matrix)
            # print("-" * 10)
            multi_topics_feature.append(features_inOne_cleaned_matrix)

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

    max_dimension = max([ i.shape[1] for i in multi_topics_feature])
    minx_dimension = min([i.shape[1] for i in multi_topics_feature])
    print('max_dimension' + str(max_dimension))
    print('minx_dimension' + str(minx_dimension))
    extends = np.zeros(n_top_w)

    index_topic = 0
    while index_topic < len(multi_topics_feature):
        while multi_topics_feature[index_topic].shape[1] < max_dimension:
            multi_topics_feature[index_topic] = np.c_[multi_topics_feature[index_topic], extends]
        index_topic += 1
    document_matrix = np.concatenate((multi_topics_feature[:]), axis=0).T

    cluster_centrias = np.array(cluster_centrias)
    label = KMeans(n_clusters = n_clusters, init=cluster_centrias).fit_predict(document_matrix)
    return label, document_matrix

def plot_painter(label, document_matrix):
    u_labels = np.unique(label)

    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=2))])

    df = pipeline.fit_transform(document_matrix)
    for i in u_labels:
        x = df[label == i, 0]
        print(len(x))
        y = df[label == i, 1]
        plt.scatter(x, y, label=i)
    plt.legend()
    plt.show()

def main():
    docs = read_docs()

    n_features = 8
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=0.1,
                                       max_features=n_features,
                                       stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(docs)
    n_components = 4
    nmf = NMF(n_components=n_components, random_state=1,
              beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
              l1_ratio=.5).fit(tfidf)

    n_top_w = 6
    topics_top_features = top_features_extraction(tfidf_vectorizer, nmf, n_top_w)

    doc_number = len(docs)
    C = len(docs) / n_components
    C = 6
    print("C is: " + str(C))
    multi_topics_feature, multi_topics_mean_vector = feature_matrix(topics_top_features, docs, C, doc_number)
    print('number of topics:' + str(len(multi_topics_feature)))

    label, clusters = cluster_creator(multi_topics_feature, multi_topics_mean_vector, len(multi_topics_feature), n_top_w )
    plot_painter(label, clusters)

if __name__ == '__main__':
    shutil.rmtree('tweet2doc/')
    users_paths = ['obesity_user_group/', 'sports_user_group/']
    for users_path in users_paths:
        tweets2doc(users_path)
    main()