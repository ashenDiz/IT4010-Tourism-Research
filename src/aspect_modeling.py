import os
import re
import json
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS

# Ensure necessary NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# -------------------------------------------------------------
# 1. Predefined Domain Aspect Categories
# -------------------------------------------------------------
ASPECTS_DICT = {
    'cost': ['price', 'cost', 'expensive', 'cheap', 'budget', 'value', 'money', 'overpriced', 'ticket', 'fee', 'charge', 'rate', 'affordable'],
    'cleanliness': ['clean', 'dirty', 'litter', 'rubbish', 'polluted', 'garbage', 'smell', 'hygiene', 'waste', 'neat', 'tidy', 'spotless'],
    'crowd': ['crowd', 'crowded', 'busy', 'quiet', 'peaceful', 'deserted', 'overcrowded', 'packed', 'tourists', 'noisy', 'calm', 'serene'],
    'staff': ['staff', 'friendly', 'rude', 'service', 'helpful', 'guide', 'unhelpful', 'management', 'polite', 'welcoming', 'hospitality'],
    'safety': ['safe', 'unsafe', 'danger', 'dangerous', 'current', 'drowning', 'harass', 'security', 'scam', 'guard', 'warning', 'careful', 'secure'],
    'accessibility': ['transport', 'tuk', 'road', 'far', 'access', 'train', 'bus', 'walk', 'reach', 'climb', 'stairs', 'path', 'drive']
}

def extract_sentence_level_aspects(df):
    """
    Extracts aspects at the sentence level and computes polarity specifically
    for the sentence discussing the aspect, fixing the document-level flaw.
    """
    print("\n" + "="*50)
    print("Extracting Sentence-Level Aspect Mentions and Polarity")
    print("="*50)
    
    vader = SentimentIntensityAnalyzer()
    records = []
    
    total_reviews = len(df)
    print(f"Processing {total_reviews:,} reviews...")

    for idx, row in df.iterrows():
        raw_text = str(row.get('Text', ''))
        destination = row['Location_Name']
        category = row.get('Location_Type', 'Unknown')
        
        if not raw_text.strip():
            continue
            
        # Segment review into sentences
        try:
            sentences = nltk.sent_tokenize(raw_text)
        except Exception:
            sentences = [s.strip() for s in re.split(r'[.!?]+', raw_text) if s.strip()]

        for sent in sentences:
            sent_clean = sent.lower()
            sent_score = vader.polarity_scores(sent)['compound']
            
            # Map compound score to sentiment label
            if sent_score >= 0.05:
                sent_class = 'Positive'
            elif sent_score <= -0.05:
                sent_class = 'Negative'
            else:
                sent_class = 'Neutral'

            for aspect, keywords in ASPECTS_DICT.items():
                if any(re.search(r'\b' + re.escape(kw) + r'\b', sent_clean) for kw in keywords):
                    records.append({
                        'Location_Name': destination,
                        'Location_Type': category,
                        'Aspect': aspect,
                        'Sentence': sent.strip(),
                        'Sentiment_Score': sent_score,
                        'Sentiment_Class': sent_class
                    })

    aspect_df = pd.DataFrame(records)
    print(f"Total sentence-level aspect mentions extracted: {len(aspect_df):,}")
    return aspect_df

def run_lda_topic_modeling(df, num_topics=6, passes=10):
    """
    Unsupervised LDA topic modeling to discover latent discussion topics
    and complement the rule-based aspect extraction.
    """
    print("\n" + "="*50)
    print(f"Training Unsupervised LDA Topic Model ({num_topics} Topics)")
    print("="*50)

    custom_stopwords = STOPWORDS.union({
        'sri', 'lanka', 'place', 'visit', 'good', 'great', 'nice', 'time', 'day',
        'one', 'really', 'see', 'also', 'went', 'go', 'view', 'much', 'beautiful',
        'experience', 'trip', 'temple', 'beach', 'park'
    })

    tokenized_docs = []
    for text in df['Clean_Text'].dropna():
        tokens = [w for w in str(text).split() if len(w) > 3 and w not in custom_stopwords]
        if tokens:
            tokenized_docs.append(tokens)

    # Build dictionary & corpus
    dictionary = corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    print(f"Corpus prepared: {len(corpus)} documents, {len(dictionary)} unique tokens.")
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, 
                         random_state=42, passes=passes, alpha='auto')

    topic_summary = {}
    print("\nDiscovered LDA Latent Topics:")
    for t_id in range(num_topics):
        top_words = [w for w, _ in lda_model.show_topic(t_id, topn=8)]
        topic_summary[f"Topic_{t_id+1}"] = {
            "keywords": top_words,
            "interpretation": infer_topic_theme(top_words)
        }
        print(f"  Topic {t_id+1} ({topic_summary[f'Topic_{t_id+1}']['interpretation']}): {', '.join(top_words)}")

    # Save LDA assets
    os.makedirs('models', exist_ok=True)
    lda_model.save('models/lda_aspect_model.gensim')
    with open('models/lda_topics.json', 'w', encoding='utf-8') as f:
        json.dump(topic_summary, f, indent=2)

    return lda_model, topic_summary

def infer_topic_theme(words):
    """Heuristic mapping of discovered LDA keywords to tourism themes."""
    word_str = " ".join(words)
    if any(k in word_str for k in ['train', 'road', 'walk', 'climb', 'step', 'tuk']):
        return "Accessibility & Journey"
    elif any(k in word_str for k in ['elephant', 'animal', 'safari', 'wild', 'bird']):
        return "Wildlife & Safari Experience"
    elif any(k in word_str for k in ['ticket', 'price', 'money', 'cost', 'rupee', 'entry']):
        return "Cost, Tickets & Value"
    elif any(k in word_str for k in ['hotel', 'staff', 'room', 'food', 'restaurant', 'service']):
        return "Hospitality, Service & Dining"
    elif any(k in word_str for k in ['history', 'ancient', 'buddha', 'king', 'rock', 'culture']):
        return "Cultural Heritage & History"
    elif any(k in word_str for k in ['water', 'sea', 'wave', 'sand', 'sunset', 'swim']):
        return "Beach, Waves & Relaxation"
    else:
        return "General Destination Environment"

def aggregate_aspect_scores(aspect_df):
    """
    Aggregates sentence-level scores per destination and aspect.
    """
    print("\nAggregating destination-level aspect sentiment matrix...")
    grouped = aspect_df.groupby(['Location_Name', 'Aspect']).agg(
        Average_Score=('Sentiment_Score', 'mean'),
        Mention_Count=('Sentiment_Score', 'count'),
        Positive_Count=('Sentiment_Class', lambda x: (x == 'Positive').sum()),
        Negative_Count=('Sentiment_Class', lambda x: (x == 'Negative').sum()),
        Neutral_Count=('Sentiment_Class', lambda x: (x == 'Neutral').sum())
    ).reset_index()

    grouped['Positive_Ratio'] = (grouped['Positive_Count'] / grouped['Mention_Count']).round(3)
    grouped['Negative_Ratio'] = (grouped['Negative_Count'] / grouped['Mention_Count']).round(3)
    grouped['Average_Score'] = grouped['Average_Score'].round(3)

    # Filter out combinations with fewer than 3 mentions to avoid noisy outliers
    grouped = grouped[grouped['Mention_Count'] >= 3].reset_index(drop=True)
    return grouped

def main():
    data_path = os.path.join('data', 'processed', 'Cleaned_Reviews.csv')
    df = pd.read_csv(data_path)

    # 1. Extract sentence-level aspect mentions & polarity
    aspect_mentions_df = extract_sentence_level_aspects(df)
    
    # Save detailed sentence mentions for validation & qualitative drill-down
    aspect_mentions_path = 'data/processed/Aspect_Sentence_Mentions.csv'
    aspect_mentions_df.to_csv(aspect_mentions_path, index=False)
    print(f"Sentence mentions saved to {aspect_mentions_path}")

    # 2. Aggregate per destination
    dest_aspect_matrix = aggregate_aspect_scores(aspect_mentions_df)
    matrix_path = 'data/processed/Destination_Aspect_Scores.csv'
    dest_aspect_matrix.to_csv(matrix_path, index=False)
    print(f"Destination Aspect Scores matrix saved to {matrix_path}")

    # 3. Unsupervised LDA Topic Modeling
    run_lda_topic_modeling(df, num_topics=6)

    print("\nPreview of Destination-by-Aspect Matrix:")
    print(dest_aspect_matrix.head(10).to_string(index=False))

if __name__ == '__main__':
    main()