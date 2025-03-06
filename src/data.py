import pandas as pd
import re
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize




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

    return ' '.join(words) # combine the words and seperate with a whitespace

df['Cleaned_Resume'] = df['Resume_str'].apply(clean_text)

original_resume = df.loc[5, 'Resume_str']
cleaned_resume = df.loc[5, 'Cleaned_Resume']
category = df.loc[5, 'Category']

print(f"Category: {category}\n")
print("Original Resume:")
print(original_resume)
print("\nCleaned Resume:")
print(cleaned_resume)

