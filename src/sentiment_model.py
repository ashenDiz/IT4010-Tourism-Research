import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from gensim.models import Word2Vec

class Word2VecVectorizer:
    """Computes document-level embeddings by averaging word vectors."""
    def __init__(self, vector_size=100, window=5, min_count=2):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

    def fit(self, texts):
        tokenized = [str(t).split() for t in texts]
        self.model = Word2Vec(sentences=tokenized, vector_size=self.vector_size, 
                              window=self.window, min_count=self.min_count, workers=4, seed=42)
        return self

    def transform(self, texts):
        vectors = []
        for text in texts:
            words = [w for w in str(text).split() if w in self.model.wv]
            if words:
                vec = np.mean([self.model.wv[w] for w in words], axis=0)
            else:
                vec = np.zeros(self.vector_size)
            vectors.append(vec)
        return np.array(vectors)

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

def run_vader_baseline(df):
    print("\n" + "="*50)
    print("1. Running VADER Lexicon Baseline Evaluation")
    print("="*50)
    analyzer = SentimentIntensityAnalyzer()
    
    def get_vader_label(text):
        if not isinstance(text, str) or not text.strip():
            return 'Neutral'
        scores = analyzer.polarity_scores(text)
        comp = scores['compound']
        if comp >= 0.05:
            return 'Positive'
        elif comp <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    df['Vader_Pred'] = df['Clean_Text'].apply(get_vader_label)
    y_true = df['Sentiment_Class']
    y_pred = df['Vader_Pred']

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"VADER Results -> Accuracy: {acc:.4f} | Macro F1: {f1_macro:.4f} | Precision: {prec_macro:.4f} | Recall: {rec_macro:.4f}")
    
    return {
        'Feature': 'Lexicon-Rule',
        'Model': 'VADER Baseline',
        'Accuracy': round(acc, 4),
        'Precision': round(prec_macro, 4),
        'Recall': round(rec_macro, 4),
        'Macro_F1': round(f1_macro, 4)
    }

def conduct_error_analysis(y_true, y_pred, test_df, labels):
    print("\n" + "="*50)
    print("3. Conducting Error Analysis on Misclassified Reviews")
    print("="*50)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_dict = {}
    for i, true_l in enumerate(labels):
        cm_dict[true_l] = {pred_l: int(cm[i][j]) for j, pred_l in enumerate(labels)}

    errors_df = test_df[y_true.values != y_pred].copy()
    errors_df['True_Label'] = y_true.values[y_true.values != y_pred]
    errors_df['Predicted_Label'] = y_pred[y_true.values != y_pred]

    # Analyze patterns
    short_review_errors = errors_df[errors_df['Clean_Text'].astype(str).apply(lambda x: len(x.split()) < 15)]
    mixed_or_neutral_errors = errors_df[errors_df['True_Label'] == 'Neutral']
    neg_misclassified_as_pos = errors_df[(errors_df['True_Label'] == 'Negative') & (errors_df['Predicted_Label'] == 'Positive')]

    analysis = {
        'confusion_matrix': cm_dict,
        'total_misclassified': len(errors_df),
        'misclassification_rate': round(len(errors_df) / len(test_df), 4),
        'short_review_errors_count': len(short_review_errors),
        'neutral_borderline_errors_count': len(mixed_or_neutral_errors),
        'critical_negative_missed_as_positive': len(neg_misclassified_as_pos),
        'sample_failure_cases': []
    }

    # Extract sample failure patterns for qualitative insight
    for _, row in errors_df.head(5).iterrows():
        sample_case = {
            'location': str(row.get('Location_Name', 'Unknown')),
            'true_label': str(row['True_Label']),
            'predicted_label': str(row['Predicted_Label']),
            'snippet': str(row.get('Text', ''))[:150] + "..."
        }
        analysis['sample_failure_cases'].append(sample_case)

    print(f"Total misclassifications: {len(errors_df)} / {len(test_df)} ({analysis['misclassification_rate']*100:.1f}%)")
    print(f"- Neutral/Mixed borderline errors: {len(mixed_or_neutral_errors)}")
    print(f"- Critical negative reviews predicted as positive: {len(neg_misclassified_as_pos)}")
    print(f"- Short reviews (<15 words) error count: {len(short_review_errors)}")

    with open('data/processed/sentiment_error_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)

    return analysis

def train_and_benchmark(df):
    print("\n" + "="*50)
    print("2. Benchmarking Supervised ML & Feature Representations")
    print("="*50)
    
    df = df.dropna(subset=['Clean_Text']).reset_index(drop=True)
    labels = ['Negative', 'Neutral', 'Positive']
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Sentiment_Class'])
    y_train = train_df['Sentiment_Class']
    y_test = test_df['Sentiment_Class']

    benchmark_records = []

    # -------------------------------------------------------------
    # 2A. TF-IDF Feature Extraction + SMOTE
    # -------------------------------------------------------------
    print("\n--- Extracting TF-IDF Features (Unigrams + Bigrams) ---")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['Clean_Text'])
    X_test_tfidf = tfidf_vectorizer.transform(test_df['Clean_Text'])

    print("Balancing TF-IDF training classes with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_tfidf_sm, y_train_tfidf_sm = smote.fit_resample(X_train_tfidf, y_train)

    # -------------------------------------------------------------
    # 2B. Word2Vec Dense Embeddings Feature Extraction
    # -------------------------------------------------------------
    print("\n--- Training Domain-Specific Word2Vec Embeddings ---")
    w2v_vectorizer = Word2VecVectorizer(vector_size=100, window=5, min_count=2)
    X_train_w2v = w2v_vectorizer.fit_transform(train_df['Clean_Text'])
    X_test_w2v = w2v_vectorizer.transform(test_df['Clean_Text'])

    print("Balancing Word2Vec training classes with SMOTE...")
    X_train_w2v_sm, y_train_w2v_sm = smote.fit_resample(X_train_w2v, y_train)

    # -------------------------------------------------------------
    # Evaluate Models on Both Representations
    # -------------------------------------------------------------
    tfidf_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(max_iter=2000, dual=False, random_state=42))
    }

    w2v_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(max_iter=2000, dual=False, random_state=42))
    }

    best_overall_model = None
    best_overall_vec = None
    best_overall_f1 = 0
    best_predictions = None

    # Evaluate TF-IDF models
    print("\nEvaluating TF-IDF Models:")
    for name, model in tfidf_models.items():
        model.fit(X_train_tfidf_sm, y_train_tfidf_sm)
        preds = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')
        prec = precision_score(y_test, preds, average='macro', zero_division=0)
        rec = recall_score(y_test, preds, average='macro', zero_division=0)

        print(f"  [TF-IDF + {name:19s}] Accuracy: {acc:.4f} | Macro F1: {f1:.4f} | Recall: {rec:.4f}")
        benchmark_records.append({
            'Feature': 'TF-IDF (1,2)',
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'Macro_F1': round(f1, 4)
        })

        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_overall_model = (f"TF-IDF + {name}", model)
            best_overall_vec = tfidf_vectorizer
            best_predictions = preds

    # Evaluate Word2Vec models
    print("\nEvaluating Word2Vec Dense Embedding Models:")
    for name, model in w2v_models.items():
        model.fit(X_train_w2v_sm, y_train_w2v_sm)
        preds = model.predict(X_test_w2v)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')
        prec = precision_score(y_test, preds, average='macro', zero_division=0)
        rec = recall_score(y_test, preds, average='macro', zero_division=0)

        print(f"  [Word2Vec + {name:15s}] Accuracy: {acc:.4f} | Macro F1: {f1:.4f} | Recall: {rec:.4f}")
        benchmark_records.append({
            'Feature': 'Word2Vec (100d)',
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'Macro_F1': round(f1, 4)
        })

        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_overall_model = (f"Word2Vec + {name}", model)
            best_overall_vec = w2v_vectorizer
            best_predictions = preds

    # Perform Error Analysis on best model
    conduct_error_analysis(y_test, best_predictions, test_df, labels)

    # Save artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_overall_model[1], 'models/best_sentiment_model.pkl')
    joblib.dump(best_overall_vec, 'models/tfidf_vectorizer.pkl')

    return benchmark_records, best_overall_model

def predict_sentiment_with_confidence(model, vectorizer, text, threshold=0.60):
    """
    Inference helper: predicts sentiment class along with confidence probability
    and a flag indicating if prediction confidence is low.
    """
    if isinstance(vectorizer, TfidfVectorizer):
        vec = vectorizer.transform([text])
    else:
        vec = vectorizer.transform([text])
        
    probs = model.predict_proba(vec)[0]
    classes = model.classes_
    best_idx = np.argmax(probs)
    pred_class = classes[best_idx]
    confidence = float(probs[best_idx])
    
    is_low_confidence = confidence < threshold
    return {
        'sentiment': pred_class,
        'confidence': round(confidence * 100, 2),
        'is_low_confidence': is_low_confidence,
        'probabilities': {cls: round(float(p) * 100, 2) for cls, p in zip(classes, probs)}
    }

def main():
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    
    data_path = os.path.join('data', 'processed', 'Cleaned_Reviews.csv')
    if not os.path.exists(data_path):
        print("Cleaned data not found. Running data cleaning...")
        from data_cleaning import main as clean_main
        clean_main()
        
    df = pd.read_csv(data_path)
    
    # 1. Baseline
    vader_res = run_vader_baseline(df)
    
    # 2. Benchmark ML Models & Embeddings
    benchmark_records, best_model = train_and_benchmark(df)
    
    # Prepend VADER baseline
    benchmark_records.insert(0, vader_res)
    
    # Save Benchmark Results Table
    benchmark_df = pd.DataFrame(benchmark_records)
    benchmark_csv_path = 'models/sentiment_benchmark_results.csv'
    benchmark_df.to_csv(benchmark_csv_path, index=False)
    
    print("\n" + "="*50)
    print("FINAL BENCHMARK COMPARISON TABLE:")
    print("="*50)
    print(benchmark_df.to_string(index=False))
    print(f"\nBenchmark saved successfully to: {benchmark_csv_path}")
    print(f"Selected Best Model: {best_model[0]}")

if __name__ == '__main__':
    main()