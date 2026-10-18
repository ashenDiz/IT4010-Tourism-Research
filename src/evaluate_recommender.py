import os
import json
import joblib
import pandas as pd
import numpy as np
from recommender import EnhancedTourismRecommender

def evaluate_recommendations():
    print("="*60)
    print("SLIIT IT4010: Recommendation System Quantitative Evaluation")
    print("="*60)

    reviews_path = 'data/processed/Cleaned_Reviews.csv'
    aspects_path = 'data/processed/Destination_Aspect_Scores.csv'
    
    reviews_df = pd.read_csv(reviews_path)
    aspects_df = pd.read_csv(aspects_path)

    recommender = EnhancedTourismRecommender()
    recommender.build_profiles(reviews_df, aspects_df)

    # 5 archetypal tourist personas reflecting real travel patterns in Sri Lanka
    personas = [
        {
            'persona': 'Wildlife Enthusiast',
            'query': 'wild safari to see wild elephants and leopards in nature',
            'relevant_destinations': [
                'Yala National Park', 'Wilpattu National Park', 'Udawalawe National Park',
                'Minneriya National Park', 'Kaudulla National Park', 'Bundala National Park',
                'Sinharaja Forest Reserve', 'Pinnawala Elephant Orphanage', 'Kumana National Park'
            ]
        },
        {
            'persona': 'Beach & Surf Vacationer',
            'query': 'golden sandy beach with surfing waves ocean sunset and swimming',
            'relevant_destinations': [
                'Arugam Bay', 'Mirissa Beach', 'Unawatuna Beach', 'Bentota Beach',
                'Hikkaduwa Beach', 'Nilaveli Beach', 'Tangalle Beach', 'Welighama', 'Hiriketiya'
            ]
        },
        {
            'persona': 'Cultural & Heritage Seeker',
            'query': 'ancient unesco world heritage ruins buddhist temples and rock fortress',
            'relevant_destinations': [
                'Sigiriya Ancient Rock Fortress', 'Ancient City of Polonnaruwa', 
                'Sacred City of Anuradhapura', 'Dambulla Cave Temple', 
                'Temple of the Sacred Tooth Relic', 'Galle Fort', 'Mihintale'
            ]
        },
        {
            'persona': 'Scenic Hiking & Tea Country',
            'query': 'cool mountain climate scenic hiking trails tea plantations and waterfalls',
            'relevant_destinations': [
                'Ella Rock', 'Little Adams Peak', 'Horton Plains National Park', 
                'Worlds End', 'Pidurutalagala', 'Nuwara Eliya', 'Liptons Seat', 
                'Bakers Falls', 'Bambarakanda Falls', 'Ambewela Farms',
                'Bluefield Tea Gardens', 'Handunugoda Tea Estate', 'Ceylon Tea Museum', 'Tea'
            ]
        },
        {
            'persona': 'Family Safe & Relaxation',
            'query': 'peaceful clean environment safe for children with friendly staff',
            'relevant_destinations': [
                'Royal Botanical Gardens Peradeniya', 'Hakgala Botanical Garden',
                'Victoria Park of Nuwara Eliya', 'Viharamahadevi Park', 'Bentota Beach'
            ]
        }
    ]

    precisions_at_3 = []
    precisions_at_5 = []
    reciprocal_ranks = []
    detailed_results = []

    print(f"\nEvaluating system across {len(personas)} representative tourist personas...")

    for p in personas:
        recs = recommender.recommend(p['query'], top_n=5)
        rec_names = [r['Location_Name'] for r in recs]
        
        # Check relevance: exact or partial substring match
        def is_relevant(name, target_list):
            return any(target.lower() in name.lower() or name.lower() in target.lower() for target in target_list)

        hits = [1 if is_relevant(name, p['relevant_destinations']) else 0 for name in rec_names]
        
        # Precision@3 and Precision@5
        p3 = sum(hits[:3]) / 3.0
        p5 = sum(hits[:5]) / 5.0
        precisions_at_3.append(p3)
        precisions_at_5.append(p5)

        # Reciprocal Rank (1 / rank of first relevant match)
        rr = 0.0
        for rank_idx, hit in enumerate(hits):
            if hit == 1:
                rr = 1.0 / (rank_idx + 1)
                break
        reciprocal_ranks.append(rr)

        print(f"\nPersona: {p['persona']}")
        print(f"  Query: '{p['query']}'")
        print(f"  Top-3 Recommended: {', '.join(rec_names[:3])}")
        print(f"  Precision@3: {p3:.2f} | Precision@5: {p5:.2f} | RR: {rr:.2f}")

        detailed_results.append({
            'persona': p['persona'],
            'query': p['query'],
            'top_recommendations': rec_names,
            'precision_at_3': round(p3, 2),
            'precision_at_5': round(p5, 2),
            'reciprocal_rank': round(rr, 2)
        })

    mean_p3 = np.mean(precisions_at_3)
    mean_p5 = np.mean(precisions_at_5)
    mrr = np.mean(reciprocal_ranks)

    summary_evaluation = {
        'mean_precision_at_3': round(float(mean_p3), 3),
        'mean_precision_at_5': round(float(mean_p5), 3),
        'mean_reciprocal_rank_mrr': round(float(mrr), 3),
        'number_of_evaluated_personas': len(personas),
        'persona_evaluations': detailed_results
    }

    os.makedirs('data/processed', exist_ok=True)
    eval_path = 'data/processed/recommender_evaluation.json'
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(summary_evaluation, f, indent=2)

    print("\n" + "="*60)
    print("RECOMMENDER SYSTEM EVALUATION SUMMARY:")
    print("="*60)
    print(f"Mean Precision@3 : {mean_p3 * 100:.1f}%")
    print(f"Mean Precision@5 : {mean_p5 * 100:.1f}%")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.3f}")
    print(f"Evaluation results exported to: {eval_path}")

if __name__ == '__main__':
    evaluate_recommendations()
