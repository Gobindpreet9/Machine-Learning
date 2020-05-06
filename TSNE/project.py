# Goal - Cover fundamentals of NLP - tokenizing, stemming etc. and plot newsgroup data using t-SNE
import numpy as np
from sklearn.datasets import fetch_20newsgroups

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem import WordNetLemmatizer  # Downloaded below
from sklearn.manifold import TSNE

nltk.download('wordnet')

groups = fetch_20newsgroups()
groups.keys()
print(groups['target_names'])  # get name of categories
np.unique(groups.target)  # returns sorted unique elements of array
# print(groups.data[0])  to print 1st article
categories_3 = ['talk.religion.misc', 'comp.graphics', 'sci.space']
groups_3 = fetch_20newsgroups(categories=categories_3)

# To display distribution of classes

sns.distplot(groups.target)
plt.show()  # uniform means topic distribution is not biased

# Using Bag of Words(Bow) model - counting the occurrences of
# certain words to define topic of given data sample.
# We will make matrix with each row an article and each column represents a word token
# value of each element will be the count of the word.

# To use only top 500 most frequent words use max_features parameter
# # and don't count built in stop words for english that can viewed- from sklearn.feature_extraction
# # import stop_words; stop_words.ENGLISH_STOP_WORDS

count_vector = CountVectorizer(max_features=500, stop_words='english')


# To get rid of numbers from data

def is_letter(word):
    for char in word:
        if not char.isalpha():
            return False
    return True


# Stemming words to get rid of duplicates eg. trying and try -> tri
# Stemming can be meaning less but is consistent, can use lemmatizing instead but it takes more time

lemmatizer = WordNetLemmatizer()
data_cleaned = []
for doc in groups_3.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split()
                           if is_letter(word))
    data_cleaned.append(doc_cleaned)

data_count = count_vector.fit_transform(data_cleaned)  # creates a sparse matrix
count_vector.get_feature_names()
data_count.toarray()  # converts sparse matrix to dense matrix

# Using TSNE to reduce dimensionality
# Part 1 - TSNE Demo
categories_3 = ['talk.religion.misc', 'comp.graphics', 'sci.space']
groups_3 = fetch_20newsgroups(categories=categories_3)
tsne_model = TSNE(n_components=2, perplexity=40, random_state=42, learning_rate=500)
data_tsne = tsne_model.fit_transform(data_count.toarray())  # only accepts dense matrix
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=groups_3.target)
plt.show()  # Observation: Data points from three topics are in different colors, data points
# from same topic are close while from different topics are far away.
# We can say count vectors do great job in maintaining disparity, to check similarity
# we shall plot points from similar categories. It does well in that regard as well.
