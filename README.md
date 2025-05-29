📰 Fake News Classification using Machine Learning Models
👥 Team Members
Baggiyam Shanmugam

Julien

Piero

Presentation: [Link to slides]

📌 Introduction
This project explores the effectiveness of various machine learning models to classify fake vs real news articles. From traditional ML approaches to transformer-based models like BERT, we evaluate multiple strategies to identify which techniques perform best on this task.

📂 Dataset Summary
Training Samples: 34,151 news articles (lowercased, preprocessed)

Test Samples: 9,983 articles

Classes: Fake (0), Real (1)

Class Distribution: 51.5% fake, 48.5% real

Average Text Length: ~11.7 words per article

🧠 Models Implemented
Model	Approach	Strengths
Logistic Regression	Custom preprocessing + CountVectorizer + Logistic Regression	Simple, fast, and effective baseline
Multinomial Naive Bayes	Bag-of-Words (BoW) with CountVectorizer	Lightweight and interpretable
Random Forest	Ensemble of decision trees on vectorized text	Feature importance insights, risk of overfitting on sparse data
Linear Support Vector Classifier (Linear SVC)	Linear classifier for high-dimensional text space	Strong margin-based separation
XGBoost	Gradient boosting framework	Handles complex patterns with regularization
BERT (Simple and Fine-tuned)	Transformer-based feature extraction and fine-tuning	State-of-the-art performance with contextual language understanding

📊 Model Performance Summary
Model	Accuracy	Precision (macro)	Recall (macro)	F1-score (macro)
Logistic Regression	94.7%	0.95	0.95	0.95
Linear SVC	92.3%	0.92	0.92	0.92
Random Forest	89.8%	0.90	0.90	0.90
Multinomial Naive Bayes	92.8%	0.93	0.93	0.93
BERT (with embeddings)	95.5%	0.96	0.96	0.96


🔎 Key Takeaways
Logistic Regression remains a robust baseline model with 94.7% accuracy.

Naive Bayes offers strong performance and fast execution, suitable for quick benchmarks.

Linear SVC effectively handles high-dimensional text features.

Random Forest shows lower performance due to overfitting on sparse data.

XGBoost and BERT models provide improvements, with BERT achieving the best overall accuracy and balanced metrics thanks to its deep contextual understanding.

🧰 Technologies Used
Python: Core programming language

scikit-learn: ML models, pipelines, and evaluation

nltk/spacy: Text preprocessing

joblib: Model persistence

matplotlib/seaborn: Visualization

📁 Repository Structure
Copy
Edit
├── logistic_regression_model.ipynb
├── Multinomial_Naive_Bayes.ipynb
├── XGBoost.ipynb
├── model_comparison.ipynb
├── Bert_Model
│   ├── bert_simple.ipynb
│   └── bert_finetuned.ipynb
├── preprocess.py
├── dataset/
│   ├── training_data_lowercase.csv
│   ├── testing_data_lowercase_nolabels.csv
│   └── validation_predictions.csv
└── README.md

📍 Conclusion
This project offers a comprehensive comparison between traditional and modern NLP modeling approaches for fake news classification. While BERT achieves the highest accuracy and provides deep contextual understanding, it comes with significantly higher computational costs and slower inference times. For practical applications where speed and resource efficiency are critical, Logistic Regression emerges as a strong alternative, delivering excellent accuracy close to BERT's performance but with much faster processing and simpler implementation. Thus, depending on the deployment constraints, one might prefer Logistic Regression or another traditional model as a reliable and efficient choice.