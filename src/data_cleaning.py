import os
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize NLTK utilities
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase text
    text = text.lower()
    # Remove HTML tags and special characters (keep only letters and spaces)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove stopwords & lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

def map_sentiment(rating):
    # Mapping strategy aligned with project assessment form
    if rating <= 2:
        return 'Negative'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Positive'

def main():
    # Resolve relative paths
    raw_path = os.path.join('data', 'raw', 'Reviews_SL.csv')
    processed_path = os.path.join('data', 'processed', 'Cleaned_Reviews.csv')
    
    print(f"Loading raw dataset from {raw_path}...")
    df = pd.read_csv(raw_path, encoding='latin-1')
    print(f"Loaded {len(df)} records successfully.")
    
    print("Preprocessing text and engineering sentiment classes...")
    df['Clean_Text'] = df['Text'].apply(clean_text)
    df['Sentiment_Class'] = df['Rating'].apply(map_sentiment)
    
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Processed dataset saved successfully to {processed_path}")
    print("\nClass distribution:")
    print(df['Sentiment_Class'].value_counts())

if __name__ == '__main__':
    main()