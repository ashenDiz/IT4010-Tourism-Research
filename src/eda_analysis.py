import os
import json
import pandas as pd
import numpy as np

def run_eda(raw_path='data/raw/Reviews_SL.csv', output_dir='data/processed'):
    print("="*60)
    print("SLIIT IT4010: Exploratory Data Analysis (Sentiment & Reviews)")
    print("="*60)
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")
        
    df = pd.read_csv(raw_path, encoding='latin-1')
    total_reviews = len(df)
    unique_destinations = df['Location_Name'].nunique()
    unique_categories = df['Location_Type'].nunique() if 'Location_Type' in df.columns else 0
    
    print(f"Total reviews loaded: {total_reviews:,}")
    print(f"Unique destinations: {unique_destinations}")
    print(f"Unique destination categories: {unique_categories}")
    
    # 1. Rating Distribution
    rating_counts = df['Rating'].value_counts().sort_index().to_dict()
    print("\nRating Distribution (1-5 stars):")
    for r, count in rating_counts.items():
        pct = (count / total_reviews) * 100
        print(f"  [{r} Star]: {count:,} ({pct:.1f}%)")
        
    # 2. Sentiment Class Mapping (Aligned with Research Proposal)
    # 1-2 stars -> Negative, 3 stars -> Neutral, 4-5 stars -> Positive
    def map_sentiment(rating):
        if rating <= 2:
            return 'Negative'
        elif rating == 3:
            return 'Neutral'
        else:
            return 'Positive'
            
    df['Sentiment_Class'] = df['Rating'].apply(map_sentiment)
    sentiment_dist = df['Sentiment_Class'].value_counts().to_dict()
    
    print("\nSentiment Class Distribution (Demonstrating Class Imbalance):")
    for s_class, count in sentiment_dist.items():
        pct = (count / total_reviews) * 100
        print(f"  {s_class:8s}: {count:,} ({pct:.1f}%)")
        
    imbalance_ratio = sentiment_dist.get('Positive', 1) / max(sentiment_dist.get('Negative', 1), 1)
    print(f"  Positive-to-Negative Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    # 3. Top and Bottom Destinations by Review Volume
    dest_counts = df['Location_Name'].value_counts()
    top_10_dest = dest_counts.head(10).to_dict()
    bottom_5_dest = dest_counts.tail(5).to_dict()
    
    # 4. Text Length Analysis
    df['Word_Count'] = df['Text'].astype(str).apply(lambda x: len(x.split()))
    df['Char_Count'] = df['Text'].astype(str).apply(len)
    
    length_stats = {
        'mean_word_count': float(round(df['Word_Count'].mean(), 1)),
        'median_word_count': float(round(df['Word_Count'].median(), 1)),
        'min_word_count': int(df['Word_Count'].min()),
        'max_word_count': int(df['Word_Count'].max()),
        'short_reviews_under_15_words': int((df['Word_Count'] < 15).sum())
    }
    
    print("\nText Length Statistics:")
    print(f"  Average word count: {length_stats['mean_word_count']} words")
    print(f"  Median word count:  {length_stats['median_word_count']} words")
    print(f"  Short reviews (<15 words): {length_stats['short_reviews_under_15_words']:,} reviews")
    
    # 5. Compile and export EDA summary
    eda_summary = {
        'total_reviews': total_reviews,
        'unique_destinations': unique_destinations,
        'unique_categories': unique_categories,
        'rating_distribution': rating_counts,
        'sentiment_distribution': sentiment_dist,
        'imbalance_ratio': round(imbalance_ratio, 2),
        'top_10_destinations': top_10_dest,
        'bottom_5_destinations': bottom_5_dest,
        'text_length_stats': length_stats
    }
    
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, 'eda_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(eda_summary, f, indent=2)
        
    print(f"\nEDA summary successfully exported to: {summary_path}")
    return eda_summary

if __name__ == '__main__':
    run_eda()
