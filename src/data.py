# Imports
import pandas as pd
import numpy as np
import re
import ssl
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from collections import Counter

ssl._create_default_https_context = ssl._create_unverified_context
word2vec = KeyedVectors.load_word2vec_format("../GoogleNews-vectors-negative300.bin", binary=True)

#download stopwords
nltk.download('all')
data_dir = "../Resume.csv"
df = pd.read_csv(data_dir)
stop_words = set(stopwords.words('english'))

#check for class imbalance
category_counts = df['Category'].value_counts()

#convert string labels into integer classes
df['Category'], unique_categories = pd.factorize(df['Category'])

#clean the text
def clean_text(text):
    text = str(text).lower() # convert everything to lowercase
    
    #replace common abbreviations
    text = re.sub(r'\b(vs|etc|ie|eg)\b', '', text)
    
    #remove email addresses
    text = re.sub(r'\S+@\S+', ' email ', text)
    
    #extract skills and keywords 
    skills = ['python', 'java', 'c++', 'javascript', 'html', 'css', 'sql', 'django', 
              'react', 'angular', 'node', 'aws', 'azure', 'cloud', 'machine learning',
              'data science', 'analyst', 'developer', 'engineer', 'manager', 'director',
              'leadership', 'project management', 'agile', 'scrum', 'waterfall',
              'marketing', 'sales', 'finance', 'accounting', 'hr', 'human resources']
    
    #replace digits with special token
    text = re.sub(r'\d+', ' num ', text)
    
    #keep important punctuation 
    text = re.sub(r'[^\w\s\.]', ' ', text)
    
    #handle special characters and normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = word_tokenize(text)

    # remove stopwords while keeping important words
    filtered_words = []
    for word in words:
        if word not in stop_words or any(skill in word for skill in skills):
            filtered_words.append(word)
    
    return filtered_words

#embedding function
def embed(words, word2vec, max_len=150):
    dimensions = word2vec.vector_size
    
    #use weighted averaging
    embedded_resumes = []
    for word in words:
        if word in word2vec.key_to_index:
            embedded_resumes.append(word2vec[word])
        else:
            #try to break down words not in vocab
            subwords = [w for w in re.split(r'(\W+)', word) if w.strip()]
            vectors = [word2vec[w] for w in subwords if w in word2vec.key_to_index]
            
            if vectors:
                #avg the vectors
                avg_vector = np.mean(vectors, axis=0)
                embedded_resumes.append(avg_vector)
            else:
                #if no subwords use small rand noise
                embedded_resumes.append(np.random.normal(0, 0.01, dimensions))
    
    #handle sequence length
    if len(embedded_resumes) < max_len:
        padding = [np.zeros(dimensions)] * (max_len - len(embedded_resumes))
        embedded_resumes.extend(padding)
    else:
        #keep important parts
        half_len = max_len // 2
        embedded_resumes = embedded_resumes[:half_len] + embedded_resumes[-half_len:]
    
    return np.array(embedded_resumes)

#embedding vocab
def check_vocab_coverage(df):
    all_words = []
    for text in df['Cleaned_Resume']:
        all_words.extend(text)
    
    unique_words = set(all_words)
    words_in_vocab = sum(1 for word in unique_words if word in word2vec.key_to_index)
    
    coverage = words_in_vocab / len(unique_words) if unique_words else 0
    
    
    # DELETE I THINK
    # Find most common OOV words
    oov_words = [word for word in unique_words if word not in word2vec.key_to_index]
    oov_counter = Counter([word for word in all_words if word in oov_words])
    print("Most common out-of-vocabulary words:")
    for word, count in oov_counter.most_common(10):
        print(f"  {word}: {count}")

#Clean and embedding
df['Cleaned_Resume'] = df['Resume_str'].apply(clean_text)
check_vocab_coverage(df)
df['Embedded_Resume'] = df['Cleaned_Resume'].apply(lambda x: embed(x, word2vec))


# Reserve 10% of the dataset to be untouched before testing
# DO NOT MODIFY random_state variable
df_trainval, eval_data_set = train_test_split(df, test_size = 0.1, random_state = 22, shuffle = True)

'''
# Print the results
original_resume = df.loc[5, 'Resume_str']
cleaned_resume = df.loc[5, 'Cleaned_Resume']
embedded_resume = df.loc[5, 'Embedded_Resume']
category = df.loc[5, 'Category']

category1 = df.loc[400, 'Category']
original_resume1 = df.loc[400, 'Resume_str']

print(f"Category: {category}\n")
print("Original Resume:")
print(original_resume)
print("\nCleaned Resume:")
print(cleaned_resume)
print('\nEmbedded_Resume\n')
print(embedded_resume)

print(f"Category: {category1}\n")
print("Original Resume1:")
print(original_resume1)

'''