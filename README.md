![alt text](image.png)



📰 Natural Language Processing Project (Project 2)

Presentation here: [Link to slides]()

📌  Intro

For this project, our goal was to detect if a news title could be fake or real. We applied Natural Language Processing (NLP) techniques to classify short headlines based on their likelihood of being authentic or misleading.

🧠 Project Overview
This binary classification task is based on a dataset that contains news headlines and a corresponding label:

label:

0 → Fake news

1 → Real news

title: The headline of the article

🧪 Objective
Build an NLP classifier that can accurately predict whether a headline is fake or real. 

🧹 Preprocessing Steps

✅ Text cleaning: Lowercasing, removing punctuation and special characters

✅ Tokenization & Lemmatization 

✅ Stopword Removal

✅ Train/Test Split: 80% train / 20% test

✅ TF-IDF vectorization for traditional ML models

✅ BERT Tokenization for transformer-based models

✅ Logistic Regression (and others..... )


##### on this I need to check all the model or we can do it together:

🔧 Modeling Approaches
🔹 TF-IDF + Logistic Regression: Accuracy ~94%

🔹 BERT + Logistic Regression: Accuracy ~97%

🔹 Multinomial_Naive_Bayes 0.9999


📊 Evaluation Metrics


Accuracy

F1 Score

Confusion Matrix

ROC-AUC Curve

Model	Accuracy	F1 Score
TF-IDF + Logistic Regression	0.94	0.93
BERT + Logistic Regression	0.97	0.96

0.96

🔮 Predictions
We used our best model (Naive Bayes) to generate predictions on testing_data_lowercase_nolabels.csv, replacing all 2s with either 0 or 1.

🧾 Project Structure

🛠️ Requirements
Package	Version

scikit-learn	1.x
pandas	1.x
nltk	latest
matplotlib	latest

📈 Conclusion
While transformer-based models like BERT demonstrated strong performance, we found that traditional models (e.g., TF-IDF + Logistic Regression) were more suitable for this task. Given the short length of the headlines and the structured nature of the dataset, classical approaches achieved high accuracy (~94%) with significantly faster training and inference times.

For this reason, we preferred the traditional model for deployment — it offers a lightweight, efficient, and interpretable solution that still delivers excellent results in real-time through our web app.



