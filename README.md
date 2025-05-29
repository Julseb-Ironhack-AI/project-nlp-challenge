![alt text](image.png)



📰 Fake News Classification using Machine Learning Models

:silhouettes: Team Members
* Baggiyam Shanmugam
* Julien
* Piero

Presentation here: [Link to slides]()

📌  Intro

This project explores the effectiveness of different machine learning models to classify fake vs real news articles. From traditional ML approaches to transformer-based models like BERT, we evaluate multiple strategies to identify which techniques work best for the task.

## :open_file_folder: Dataset Summary
* **Training Samples**: 34,151 news articles (lowercased, preprocessed)
* **Test Samples**: 9,983 articles
* **Classes**: Fake (0), Real (1)
* **Class Distribution**: 51.5% fake, 48.5% real
* **Average Text Length**: \~11.7 words per article
---
## :brain: Models Implemented
### 1. Logistic Regression (`logistic_regression_model.ipynb`)
* **Pipeline**: Custom preprocessing (punctuation removal, tokenization, stopwords, stemming) + CountVectorizer
* **Model**: Logistic Regression
* **Strengths**: Simple, fast, and very effective baseline
### 2. Multinomial Naive Bayes (`naive_bayes_model.ipynb`)
* **Approach**: Bag-of-Words (BoW) with CountVectorizer
* **Strengths**: Extremely lightweight and interpretable
### 3. Random Forest (`random_forest_model.ipynb`)
* **Approach**: Ensemble model using decision trees on vectorized text
* **Strengths**: Good for feature importance, but may overfit on sparse data
### 4. Linear Support Vector Classifier (Linear SVC) (`svm_model.ipynb`)
* **Approach**: Linear classifier for high-dimensional space
* **Strengths**: Strong margin-based separation
### 5. XGBoost (`xgboost_model.ipynb`)
* **Approach**: Gradient boosting framework
* **Strengths**: Handles complex patterns with regularization control
### 6. BERT (Simple and Fine-tuned) (`bert_simple.ipynb`, `bert_finetuned.ipynb`)
* **Approach**: Transformer-based feature extraction and fine-tuning
* **Strengths**: Best-in-class performance with pre-trained language understanding
---
## :bar_chart: Model Performance Summary
| Model                         | Accuracy  | Precision (macro) | Recall (macro) | F1-score (macro) |
| ----------------------------- | --------- | ----------------- | -------------- | ---------------- |
| **Logistic Regression**       | **94.7%** | **0.95**          | **0.95**       | **0.95**         |
| Linear SVC                    | 92.3%     | 0.92              | 0.92           | 0.92             |
| Random Forest                 | 89.8%     | 0.90              | 0.90           | 0.90             |
| Multinomial Naive Bayes (BoW) | 92.8%     | 0.93              | 0.93           | 0.93             |
---
## :magnifying_glass: Key Takeaways
* **Logistic Regression** proved to be a robust baseline model with 94.7% accuracy.
* **Naive Bayes** remains strong despite simplicity, making it great for quick benchmarks.
* **Linear SVC** offers competitive accuracy and is effective in high-dimensional feature space.
* **Random Forest** is less effective for sparse text features due to overfitting potential.
* **XGBoost and BERT** models (evaluated separately) showed improvements with more compute but diminishing returns.
---
## :brain: Technologies Used
* **Python**: Core programming language
* **scikit-learn**: ML models, pipelines, evaluation
* **nltk/spacy**: Text preprocessing
* **joblib**: Model persistence
* **matplotlib/seaborn**: Visualization
---
```
---
## :file_folder: Repository Structure
```
├── [logistic_regression_model.ipynb]
├── [Multinomial_Naive_Bayes.ipynb]
├── [XGBoost.ipynb]
├── [model_comparison.ipynb]
├── [Bert_Model]
├── bert_simple.ipynb
├── bert_finetuned.ipynb
├── preprocess.py
├── dataset/
│   ├── training_data_lowercase.csv
│   ├── testing_data_lowercase_nolabels.csv
│   └── validation_predictions.csv
└── README.md
```
---
## :drawing_pin: Conclusion
This project provides a comprehensive view of traditional vs. modern NLP modeling approaches. While Logestic Regression offers the highest accuracy, traditional models like Logistic Regression and Naive Bayes still offer excellent performance at a fraction of the compute cost.



