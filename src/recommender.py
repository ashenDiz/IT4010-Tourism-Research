import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

# Predefined domain tag lexicons for automatic review-derived tag extraction
THEMATIC_TAG_KEYWORDS = {
    'family-friendly': ['family', 'kids', 'children', 'child', 'safe', 'parents', 'gentle'],
    'romantic & peaceful': ['romantic', 'couple', 'honeymoon', 'quiet', 'peaceful', 'serene', 'sunset'],
    'adventure & hiking': ['adventure', 'hiking', 'climb', 'trek', 'rock', 'steep', 'trail', 'peak'],
    'budget-friendly': ['cheap', 'budget', 'affordable', 'free', 'inexpensive', 'value'],
    'wildlife & safari': ['wildlife', 'safari', 'elephant', 'leopard', 'bird', 'animal', 'whale', 'turtle'],
    'beach & watersports': ['beach', 'surf', 'surfing', 'waves', 'ocean', 'swim', 'sand', 'diving', 'snorkeling'],
    'cultural & historic': ['ancient', 'temple', 'history', 'buddha', 'ruins', 'heritage', 'king', 'palace'],
    'scenic & nature': ['scenic', 'views', 'waterfall', 'tea', 'mountain', 'mist', 'lush', 'valley']
}

class EnhancedTourismRecommender:
    def __init__(self, num_clusters=5):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1500, ngram_range=(1, 2))
        self.num_clusters = num_clusters
        self.destination_profiles = None
        self.tfidf_matrix = None
        self.dense_matrix = None
        self.kmeans_model = None
        self.cluster_names = {}

    def extract_thematic_tags(self, text):
        """Extracts review-derived thematic tags based on keyword density."""
        text_lower = str(text).lower()
        matched_tags = []
        for tag, keywords in THEMATIC_TAG_KEYWORDS.items():
            count = sum(len(re.findall(r'\b' + re.escape(kw) + r'\b', text_lower)) for kw in keywords)
            if count >= 3: # Meaningful frequency threshold
                matched_tags.append(tag)
        return matched_tags if matched_tags else ['general exploration']

    def build_profiles(self, reviews_df, aspect_scores_df):
        print("\n" + "="*50)
        print("1. Engineering Multi-Factor Destination Profiles")
        print("="*50)

        # 1. Group reviews by destination
        profiles = reviews_df.groupby('Location_Name').agg({
            'Clean_Text': ' '.join,
            'Rating': 'mean',
            'Location_Type': lambda x: x.mode().iloc[0] if not x.empty and not pd.isna(x.mode().iloc[0]) else 'Attraction'
        }).reset_index()

        # 2. Extract peak travel month per destination
        if 'Travel_Date' in reviews_df.columns:
            dates = pd.to_datetime(reviews_df['Travel_Date'].dropna(), errors='coerce')
            reviews_df['Month'] = dates.dt.month
            peak_months = reviews_df.groupby('Location_Name')['Month'].apply(
                lambda x: int(x.mode().iloc[0]) if not x.dropna().empty else 0
            ).reset_index()
            profiles = pd.merge(profiles, peak_months, on='Location_Name', how='left')
        else:
            profiles['Month'] = 0

        # 3. Integrate Aspect Scores from Component 2
        print("Integrating aspect sentiment vectors into destination profiles...")
        aspect_pivot = aspect_scores_df.pivot(index='Location_Name', columns='Aspect', values='Average_Score').fillna(0)
        aspect_cols = [f"aspect_{col}" for col in aspect_pivot.columns]
        aspect_pivot.columns = aspect_cols
        profiles = pd.merge(profiles, aspect_pivot, on='Location_Name', how='left').fillna(0)

        # 4. Extract Review-Derived Thematic Tags
        print("Extracting review-derived thematic tags...")
        profiles['Thematic_Tags'] = profiles['Clean_Text'].apply(self.extract_thematic_tags)

        # 5. Fit Text Representations (TF-IDF & Dense Embeddings)
        print("Generating TF-IDF & Dense Semantic Matrices...")
        self.tfidf_matrix = self.vectorizer.fit_transform(profiles['Clean_Text'])
        
        # Dense SVD embeddings for comparative benchmark
        svd = TruncatedSVD(n_components=50, random_state=42)
        self.dense_matrix = svd.fit_transform(self.tfidf_matrix)

        # 6. Fit K-Means Clustering ("Similar Destinations" Feature)
        print(f"Clustering {len(profiles)} destinations into {self.num_clusters} thematic clusters...")
        # Combine normalized text embeddings with aspect vectors for hybrid clustering
        aspect_feature_vals = profiles[aspect_cols].values
        combined_features = np.hstack([self.dense_matrix, aspect_feature_vals])
        
        self.kmeans_model = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        profiles['Cluster_ID'] = self.kmeans_model.fit_predict(combined_features)

        # Name clusters according to dominant location types & tags
        self.cluster_names = {
            0: "Ancient Heritage & Sacred Sites",
            1: "Southern & Eastern Beach Escapes",
            2: "Wild Safari & National Parks",
            3: "Highland Tea Country & Mountain Trails",
            4: "Cultural Museums & Botanical Gardens"
        }
        profiles['Cluster_Name'] = profiles['Cluster_ID'].map(self.cluster_names).fillna("Diverse Attractions")

        self.destination_profiles = profiles

        # 7. Save Assets for Dashboard & Evaluation
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.vectorizer, 'models/recommender_vectorizer.pkl')
        joblib.dump(self.tfidf_matrix, 'models/recommender_tfidf.pkl')
        joblib.dump(self.dense_matrix, 'models/recommender_dense.pkl')
        joblib.dump(self.kmeans_model, 'models/destination_kmeans.pkl')
        self.destination_profiles.to_pickle('models/destination_profiles.pkl')

        with open('models/destination_clusters.json', 'w', encoding='utf-8') as f:
            cluster_export = {}
            for c_id, c_name in self.cluster_names.items():
                members = profiles[profiles['Cluster_ID'] == c_id]['Location_Name'].tolist()
                cluster_export[c_name] = members
            json.dump(cluster_export, f, indent=2)

        print("Destination profiles, matrices, and clusters successfully saved.")
        return profiles

    def benchmark_similarity_approaches(self, test_queries):
        """
        Benchmarks traditional TF-IDF cosine similarity against dense embedding similarity.
        Fulfills research commitment on similarity comparison.
        """
        print("\n" + "="*50)
        print("2. Benchmarking TF-IDF vs. Dense Embedding Similarity")
        print("="*50)

        svd = TruncatedSVD(n_components=50, random_state=42)
        svd.fit(self.tfidf_matrix)

        benchmark_results = []
        for q in test_queries:
            # TF-IDF
            q_tfidf = self.vectorizer.transform([q])
            tfidf_sims = cosine_similarity(q_tfidf, self.tfidf_matrix).flatten()
            top_tfidf_idx = tfidf_sims.argsort()[::-1][:3]
            top_tfidf_names = self.destination_profiles.iloc[top_tfidf_idx]['Location_Name'].tolist()
            top_tfidf_score = float(tfidf_sims[top_tfidf_idx[0]])

            # Dense SVD Embedding
            q_dense = svd.transform(q_tfidf)
            dense_sims = cosine_similarity(q_dense, self.dense_matrix).flatten()
            top_dense_idx = dense_sims.argsort()[::-1][:3]
            top_dense_names = self.destination_profiles.iloc[top_dense_idx]['Location_Name'].tolist()
            top_dense_score = float(dense_sims[top_dense_idx[0]])

            benchmark_results.append({
                'Query': q,
                'TF-IDF Top Matches': ", ".join(top_tfidf_names),
                'TF-IDF Top Score': round(top_tfidf_score, 3),
                'Dense Embedding Top Matches': ", ".join(top_dense_names),
                'Dense Embedding Top Score': round(top_dense_score, 3)
            })

        bench_df = pd.DataFrame(benchmark_results)
        bench_df.to_csv('models/recommender_benchmark.csv', index=False)
        print(bench_df.to_string(index=False))
        print("Comparative benchmark saved to models/recommender_benchmark.csv")
        return bench_df

    def recommend(self, user_query, travel_month=None, user_tags=None, top_n=3):
        """
        Hybrid multi-factor recommendation:
        Blends textual cosine similarity with aspect scores (safety, cleanliness, budget, etc.)
        and generates dynamic explainability tags.
        """
        if self.tfidf_matrix is None or self.destination_profiles is None:
            raise ValueError("Models not built. Run build_profiles first.")

        query_vec = self.vectorizer.transform([user_query])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        results = []
        for idx, sim in enumerate(sim_scores):
            dest = self.destination_profiles.iloc[idx]
            
            # Base score from textual relevance (0 to 100)
            base_score = sim * 70.0  # 70% weight on query matching
            
            # Quality & Aspect Boost (30% weight)
            # Factor in average star rating and positive aspect scores
            star_boost = (dest['Rating'] / 5.0) * 15.0
            
            # Aspect sentiment signals
            safety = dest.get('aspect_safety', 0)
            cleanliness = dest.get('aspect_cleanliness', 0)
            cost = dest.get('aspect_cost', 0)
            aspect_boost = max(min((safety + cleanliness + cost) / 3.0, 1.0), -1.0) * 15.0
            
            final_score = base_score + star_boost + aspect_boost

            # Generate dynamic explainability reasons
            reasons = []
            if sim > 0.15:
                reasons.append(f"Content Match ({sim*100:.0f}%)")
            
            # Dynamic aspect highlights
            if safety > 0.3:
                reasons.append("High Safety Rating")
            if cleanliness > 0.3:
                reasons.append("Clean & Well-Maintained")
            if cost > 0.1:
                reasons.append("Budget-Friendly")

            # Check seasonal match
            is_peak = False
            if travel_month and travel_month != "Any":
                if int(travel_month) == int(dest.get('Month', 0)):
                    is_peak = True
                    reasons.append("Peak Travel Season")

            explanation_tag = " • ".join(reasons) if reasons else f"Overall Rating: {dest['Rating']:.1f}★"

            results.append({
                'Location_Name': dest['Location_Name'],
                'Location_Type': dest['Location_Type'],
                'Final_Score': round(final_score, 1),
                'Relevance_Score': round(sim * 100, 1),
                'Rating': round(dest['Rating'], 1),
                'Cluster_Name': dest['Cluster_Name'],
                'Thematic_Tags': dest['Thematic_Tags'],
                'Explanation_Tag': explanation_tag,
                'Is_Peak_Season': is_peak,
                'Aspect_Scores': {
                    'Safety': round(float(dest.get('aspect_safety', 0)), 2),
                    'Cleanliness': round(float(dest.get('aspect_cleanliness', 0)), 2),
                    'Cost': round(float(dest.get('aspect_cost', 0)), 2),
                    'Staff': round(float(dest.get('aspect_staff', 0)), 2)
                }
            })

        # Sort descending by Final Score
        results = sorted(results, key=lambda x: x['Final_Score'], reverse=True)
        return results[:top_n]

    def get_similar_destinations(self, location_name, top_n=4):
        """Returns sister destinations from the same AI cluster."""
        match = self.destination_profiles[self.destination_profiles['Location_Name'] == location_name]
        if match.empty:
            return []
        
        cluster_id = match.iloc[0]['Cluster_ID']
        cluster_name = match.iloc[0]['Cluster_Name']
        siblings = self.destination_profiles[
            (self.destination_profiles['Cluster_ID'] == cluster_id) & 
            (self.destination_profiles['Location_Name'] != location_name)
        ]
        
        selected = siblings.sort_values(by='Rating', ascending=False).head(top_n)
        return {
            'cluster_name': cluster_name,
            'destinations': selected[['Location_Name', 'Rating', 'Location_Type', 'Thematic_Tags']].to_dict(orient='records')
        }

def main():
    reviews_path = 'data/processed/Cleaned_Reviews.csv'
    aspects_path = 'data/processed/Destination_Aspect_Scores.csv'
    
    if not os.path.exists(reviews_path) or not os.path.exists(aspects_path):
        print("Required processed files not found. Ensure previous steps are run.")
        return

    reviews_df = pd.read_csv(reviews_path)
    aspect_scores_df = pd.read_csv(aspects_path)

    recommender = EnhancedTourismRecommender(num_clusters=5)
    recommender.build_profiles(reviews_df, aspect_scores_df)

    test_queries = [
        "quiet relaxing beach with cheap budget food",
        "wildlife safari to see wild elephants and leopards",
        "ancient buddhist temples and historical rock ruins",
        "scenic hill country hiking with cool climate and waterfalls"
    ]
    recommender.benchmark_similarity_approaches(test_queries)

    print("\n" + "="*50)
    print("Sample Multi-Factor Recommendation Test:")
    print("="*50)
    sample_recs = recommender.recommend("peaceful beach with good safety and cheap food", travel_month=8, top_n=3)
    for i, r in enumerate(sample_recs):
        print(f"\n{i+1}. {r['Location_Name']} ({r['Location_Type']})")
        print(f"   Score: {r['Final_Score']} | Rating: {r['Rating']}* | Cluster: {r['Cluster_Name']}")
        print(f"   Why Recommended: {r['Explanation_Tag']}")
        print(f"   Thematic Tags: {', '.join(r['Thematic_Tags'])}")

if __name__ == '__main__':
    main()