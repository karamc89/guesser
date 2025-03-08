import pandas as pd
import numpy as np
import re
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from gensim.models import KeyedVectors
word2vec = KeyedVectors.load_word2vec_format("../GoogleNews-vectors-negative300.bin", binary=True)



nltk.download('all')

data_dir = "../Resume.csv"
df = pd.read_csv(data_dir)

stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = str(text).lower() # convert everything to lowercase
    text = re.sub(r'\d+', '', text) # remove numbers
    text = re.sub(r'[^\w\s]', '', text) # remove special characters
    text = re.sub(r'\s+', ' ', text).strip() # get rid of extra spaces
    words = word_tokenize(text)

    # remove stopwords
    filtered_words = []
    for word in words:
        if word not in stop_words:
            filtered_words.append(word)
    words = filtered_words

    return words # combine the words and seperate with a whitespace


def embed(words, word2vec, max_len = 100):
    dimensions = word2vec.vector_size
    embedded_resumes = [word2vec[word] for word in words if word in word2vec.key_to_index]

    # pad or truncate according to need
    if len(embedded_resumes) < max_len:
        padding = [np.zeros(dimensions)] * (max_len - len(embedded_resumes))
        embedded_resumes.extend(padding)
    else:
        embedded_resumes = embedded_resumes[:max_len]

    return np.array(embedded_resumes)



df['Cleaned_Resume'] = df['Resume_str'].apply(clean_text)
df['Embedded_Resume'] = df['Cleaned_Resume'].apply(lambda x: embed(x, word2vec))

original_resume = df.loc[5, 'Resume_str']
cleaned_resume = df.loc[5, 'Cleaned_Resume']
embedded_resume = df.loc[5, 'Embedded_Resume']
category = df.loc[5, 'Category']

print(f"Category: {category}\n")
print("Original Resume:")
print(original_resume)
print("\nCleaned Resume:")
print(cleaned_resume)
print('\nEmbedded_Resume\n')
print(embedded_resume)


