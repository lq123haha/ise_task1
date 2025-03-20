########## 1. Import required libraries ##########

import pandas as pd
import numpy as np
import re
import math
import random
from sklearn import set_config

# Text and feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec  # Added for Word2Vec

# Evaluation and tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)

# Classifier
from sklearn.svm import SVC  # Changed to SVM

# Text cleaning & stopwords
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# perform cross-validation and evaluate a model's performance
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Setting a random seed helps improve the consistency of results
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

########## 2. Define text preprocessing methods ##########

def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)


def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# Stopwords
NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...']  # You can customize this list as needed
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list


def remove_stopwords(text):
    """Remove stopwords from the text."""
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])


def clean_str(string):
    """
    Clean text by removing non-alphanumeric characters,
    and convert it to lowercase.
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


########## 3. Download & read data ##########
import os
import subprocess

# Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
project = 'pytorch'
path = f'./data/{project}.csv'

pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999)  # Shuffle

# Merge Title and Body into a single column; if Body is NaN, use Title only
pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
    axis=1
)

# Keep only necessary columns: id, Number, sentiment, text (merged Title+Body)
pd_tplusb = pd_all.rename(columns={
    "Unnamed: 0": "id",
    "class": "sentiment",
    "Title+Body": "text"
})
pd_tplusb.to_csv('./output/Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])

########## 4. Configure parameters & Start training ##########

# ========== Key Configurations ==========

# 1) Data file to read
datafile = './output/Title+Body.csv'

# 2) Number of repeated experiments
REPEAT = 10

# 3) Output CSV file name
out_csv_name = f'./results/{project}_average_SVM_Feature_Fusion.csv'  # Updated file name

# ========== Read and clean data ==========
data = pd.read_csv(datafile).fillna('')
text_col = 'text'

# Keep a copy for referencing original data if needed
original_data = data.copy()

# Text cleaning
data[text_col] = data[text_col].apply(remove_html)
data[text_col] = data[text_col].apply(remove_emoji)
data[text_col] = data[text_col].apply(remove_stopwords)
data[text_col] = data[text_col].apply(clean_str)

# ========== Word2Vec + TF-IDF Feature Engineering ==========

# Define a function to get TF-IDF weighted Word2Vec embeddings
def get_weighted_word2vec_embedding(text, model, tfidf_vectorizer, tfidf_features, index):
    words = text.split()
    word_vectors = []
    weights = []
    for word in words:
        if word in model.wv:
            word_vectors.append(model.wv[word])
            # Get TF-IDF weight for the word
            if word in tfidf_vectorizer.vocabulary_:
                weights.append(tfidf_features[index, tfidf_vectorizer.vocabulary_[word]])
            else:
                weights.append(0)  # If word not in TF-IDF vocabulary, assign weight 0
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)  # Return zero vector if no words are found
    weights_sum = np.sum(weights)
    if weights_sum == 0:
        return np.mean(word_vectors, axis=0)  # Fallback to simple average if weights sum to zero
    return np.average(word_vectors, axis=0, weights=weights)

# Define a grid of Word2Vec hyperparameters
word2vec_params = {
    'vector_size': [50, 100, 200],  # Dimensionality of word vectors
    'window': [3, 5, 7],  # Context window size
    'min_count': [1, 2, 5]  # Minimum word frequency
}

# ========== Hyperparameter grid for SVM ==========
params = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear', 'rbf'],  # Kernel type
    'gamma': ['scale', 'auto']  # Kernel coefficient for 'rbf'
}

# Lists to store metrics across repeated runs
accuracies = []
precisions = []
recalls = []
f1_scores = []
auc_values = []

# Used to store detailed results of each experiment
detailed_results = []

for repeated_time in range(REPEAT):
    # --- 4.1 Split into train/test ---
    indices = np.arange(data.shape[0])
    train_index, test_index = train_test_split(
        indices, test_size=0.2, random_state=repeated_time
    )

    train_text = data[text_col].iloc[train_index]
    test_text = data[text_col].iloc[test_index]

    y_train = data['sentiment'].iloc[train_index]
    y_test = data['sentiment'].iloc[test_index]

    # --- 4.2 Grid search for Word2Vec parameters ---
    best_word2vec_score = -1
    best_word2vec_params = None
    best_word2vec_embeddings = None

    # Compute TF-IDF features
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    tfidf_features_train = tfidf.fit_transform(train_text).toarray()

    for vector_size in word2vec_params['vector_size']:
        for window in word2vec_params['window']:
            for min_count in word2vec_params['min_count']:
                # Train Word2Vec model
                sentences = [text.split() for text in data[text_col]]
                word2vec_model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4,seed=42)

                # Compute weighted Word2Vec embeddings
                word2vec_embeddings = np.array([
                    get_weighted_word2vec_embedding(text, word2vec_model, tfidf, tfidf_features_train, i)
                    for i, text in enumerate(train_text)
                ])

                # Train SVM model with current Word2Vec embeddings
                clf = SVC(probability=True,random_state=42)
                clf.fit(word2vec_embeddings, y_train)

                # Evaluate on training set (cross-validation)
                score = np.mean(cross_val_score(clf, word2vec_embeddings, y_train, cv=5, scoring='roc_auc'))

                # Update best parameters
                if score > best_word2vec_score:
                    best_word2vec_score = score
                    best_word2vec_params = {'vector_size': vector_size, 'window': window, 'min_count': min_count}
                    best_word2vec_model = word2vec_model
                    best_word2vec_embeddings = word2vec_embeddings

    print(f"Best Word2Vec parameters: {best_word2vec_params}")

    # --- 4.3 TF-IDF features for test set ---
    tfidf_features_test = tfidf.transform(test_text).toarray()

    # --- 4.4 Compute weighted Word2Vec embeddings for test set ---
    word2vec_embeddings_test = np.array([
        get_weighted_word2vec_embedding(text, best_word2vec_model, tfidf, tfidf_features_test, i)
        for i, text in enumerate(test_text)
    ])

    # --- 4.5 Combine Word2Vec and TF-IDF features ---
    X_train = np.hstack((best_word2vec_embeddings, tfidf_features_train))
    X_test = np.hstack((word2vec_embeddings_test, tfidf_features_test))

    # --- 4.6 SVM model & GridSearch ---
    clf = SVC(probability=True,class_weight='balanced',random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        clf,
        params,
        cv=kf,  # 5-fold CV (can be changed)
        scoring='roc_auc',  # Using roc_auc as the metric for selection
    )
    grid.fit(X_train, y_train)

    # Retrieve the best model
    best_clf = grid.best_estimator_
    best_clf.fit(X_train, y_train)

    # --- 4.7 Make predictions & evaluate ---
    y_pred = best_clf.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    # Precision (macro)
    prec = precision_score(y_test, y_pred, average='macro')
    precisions.append(prec)

    # Recall (macro)
    rec = recall_score(y_test, y_pred, average='macro')
    recalls.append(rec)

    # F1 Score (macro)
    f1 = f1_score(y_test, y_pred, average='macro')
    f1_scores.append(f1)

    # AUC
    y_pred_proba = best_clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=1)
    auc_val = auc(fpr, tpr)
    auc_values.append(auc_val)

# --- 4.8 Aggregate results ---
final_accuracy = np.mean(accuracies)
final_precision = np.mean(precisions)
final_recall = np.mean(recalls)
final_f1 = np.mean(f1_scores)
final_auc = np.mean(auc_values)

print("=== SVM + Word2Vec + TF-IDF (Weighted) Results ===")
print(f"Number of repeats:     {REPEAT}")
print(f"Average Accuracy:      {final_accuracy:.4f}")
print(f"Average Precision:     {final_precision:.4f}")
print(f"Average Recall:        {final_recall:.4f}")
print(f"Average F1 score:      {final_f1:.4f}")
print(f"Average AUC:           {final_auc:.4f}")

# save detailed results for each experiement
detailed_results = {
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls,
    'F1': f1_scores,
    'AUC': auc_values
}

# Convert Format
results_df = pd.DataFrame(detailed_results)

# Save the detailed information to CSV
results_df.to_csv(f'./results/{project}_SVM_Feature_Fusion_detailed_results.csv', index=False)
print(f"\nDetailed results have been saved to: ./results/{project}_SVM_Feature_Fusion_detailed_results.csv")


# Save final results to CSV (append mode)
try:
    # Attempt to check if the file already has a header
    existing_data = pd.read_csv(out_csv_name, nrows=1)
    header_needed = False
except:
    header_needed = True

df_log = pd.DataFrame(
    {
        'repeated_times': [REPEAT],
        'Accuracy': [final_accuracy],
        'Precision': [final_precision],
        'Recall': [final_recall],
        'F1': [final_f1],
        'AUC': [final_auc],
        'CV_list(AUC)': [str(auc_values)]
    }
)

df_log.to_csv(out_csv_name, mode='a', header=header_needed, index=False)

print(f"\nResults have been saved to: {out_csv_name}")