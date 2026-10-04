# import pandas as pd
# import numpy as np
# import re
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity


# file_path = r"C:\Users\megha\Desktop\nsclc_ga_svm\nsclc_chatbot_dataset_60.csv"
# df = pd.read_csv(file_path)


# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z0-9 ]', '', text)                                                                                                                            
#     return text


# df['Question'] = df['Question'].apply(preprocess_text)


# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(df['Question'])

# def get_response(user_input, threshold=0.2):
#     user_input = preprocess_text(user_input)
#     user_tfidf = vectorizer.transform([user_input])
#     similarities = cosine_similarity(user_tfidf, tfidf_matrix)

#     best_match_idx = np.argmax(similarities)
#     best_score = similarities[0, best_match_idx]

#     if best_score < threshold:
#         return "I'm sorry, I didn't understand. Can you rephrase your question?"

#     return df.iloc[best_match_idx]['Answer']


# print("NSCLC Chatbot: Hi! Ask me a medical question. (type 'exit' to quit)")
# while True:
#     user_query = input("You: ")
#     if user_query.lower() in ['exit', 'quit']:
#         print("NSCLC Chatbot: Goodbye!")
#         break
#     response = get_response(user_query)
#     print("NSCLC Chatbot:", response)








import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

file_path = r"C:\Users\megha\Desktop\nsclc_ga_svm\nsclc_chatbot_dataset_60.csv"
df = pd.read_csv(file_path)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

df['Question'] = df['Question'].apply(preprocess_text)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Question'])

def get_response(user_input, threshold=0.2):
    user_input = preprocess_text(user_input)
    user_tfidf = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)

    best_match_idx = np.argmax(similarities)
    best_score = similarities[0, best_match_idx]

    if best_score < threshold:
        return "I'm sorry, I didn't understand. Can you rephrase your question?"

    return df.iloc[best_match_idx]['Answer']