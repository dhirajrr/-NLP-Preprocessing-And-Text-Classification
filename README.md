# NLP Emotion Classification Project

This project involves building a machine learning model to classify emotions from text using natural language processing (NLP) techniques. The dataset consists of various textual data, and the goal is to preprocess the text, apply feature extraction, and train classifiers to predict emotional labels.

## Project Overview

In this project, the following steps were carried out:

1. **Data Preprocessing:**
   - Text data is cleaned by lowercasing, tokenization, removing punctuation and stop words, and applying stemming and lemmatization.
   
2. **Feature Extraction:**
   - Both **TF-IDF** (Term Frequency-Inverse Document Frequency) and **Count Vectorizer** techniques were used to convert text data into numerical features.
   
3. **Model Training:**
   - Two machine learning models were trained: **Logistic Regression** and **Multinomial Naive Bayes**.
   
4. **Model Evaluation:**
   - The performance of the models is evaluated using accuracy, classification report, confusion matrix, and ROC curve.

5. **Visualizations:**
   - Several visualizations were created to understand the data, evaluate model performance, and compare the models.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- NLTK
- WordCloud

You can install the required dependencies using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud
