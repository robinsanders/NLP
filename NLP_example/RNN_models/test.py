from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np 

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(["This This the the the", "cat cat"])
print(x)

clf = OneVsRestClassifier(LogisticRegression())

le = LabelEncoder()
le.fit(['a', 'b', 'b', 'a', 'c', 'd', 'd'])
print(le.fit_transform(['a', 'd', 'b', 'c', 'd', 'a']))

# def pre_process_char_tokenize_wocka(cls, data, min_length, max_length):

#     #Takes in data and tokenizes into characters

#     body_list = [] 

#     #Building a list of unique tokens (vocabulary)
#     vocab = []

#     for i in range(len(data)):

#         #We keep just the alphabets, numbers and important punctuations. We remove all escape sequences and unwanted punctuations

#         str = data[i]['body'].strip()

#         #Removing unwanted characters. We replace ,?! by a space to not lose data
#         str = re.sub('[^0-9a-zA-Z,.!?\'\" ]+', '', str)

#         tmp_list = list(str)

#         #Only appending if the length is nominal
#         if len(tmp_list) >= min_length and len(tmp_list) <= max_length:
#             body_list.append(tmp_list)
#             vocab = list(set(vocab + tmp_list))

#     vocab.sort()

#     return body_list, vocab