from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation

nltk.download('wordnet')

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space'
]
groups = fetch_20newsgroups(subset='all', categories=categories)
labels = groups.target
label_names = groups.target_names


# data preprocessing
def is_letter(word):
    for char in word:
        if not char.isalpha():
            return False
    return True


lemmatizer = WordNetLemmatizer()
data_cleaned = []
for doc in groups.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split()
                           if is_letter(word))
    data_cleaned.append(doc_cleaned)

# conver to count vector
count_vectorizer = CountVectorizer(stop_words='english', max_features=None, max_df=0.5,
                                   min_df=2)  # max frwquency is 50% and 2% is minimum. Document frequency is
# measured by fraction of documents that contain the word
data = count_vectorizer.fit_transform(data_cleaned)

# Now we cluster
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(data)
clusters = kmeans.labels_
print(Counter(clusters))
# Output: Counter({3: 3363, 0: 12, 1: 9, 2: 3}) Cluster 3 has 3363 samples, not a good representation

# We will use TFidVectorizer instead of count vectorizer for better results
tfid_vectorizer = TfidfVectorizer(stop_words='english', max_features=None, max_df=0.5, min_df=2)
data = tfid_vectorizer.fit_transform(data_cleaned)
kmeans.fit(data)
clusters = kmeans.labels_
print(Counter(clusters))
# Counter({2: 1467, 0: 802, 3: 586, 1: 532}) A more balanced result

# Topic Modeling using NMF -- can use count vectorizer or tfid
num_topics = 20
nmf = NMF(n_components=num_topics, random_state=42)
data = count_vectorizer.fit_transform(data_cleaned)
nmf.fit(data)
# Displaying top 10 terms for each topic
terms = count_vectorizer.get_feature_names()
for topic_idx, topic in enumerate(nmf.components_):
    print("Topic  {}:".format(topic_idx))
    print(" ".join([terms[i] for i in topic.argsort()[-10:]]))

# Topic Modeling using LDA -- can only use count vectorizer
num_topics = 20
lda = LatentDirichletAllocation(n_components=num_topics, learning_method='batch', random_state=42)
data = count_vectorizer.fit_transform(data_cleaned)
lda.fit(data)
# Displaying top 10 terms for each topic
terms = count_vectorizer.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
    print("Topic  {}:".format(topic_idx))
    print(" ".join([terms[i] for i in topic.argsort()[-10:]]))

