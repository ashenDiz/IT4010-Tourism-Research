import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Key aspect keywords to compute monthly aspect sentiment scores
ASPECT_WORDS = {
    'safety': ['safe', 'unsafe', 'danger', 'security', 'guard', 'warning'],
    'cost': ['price', 'cost', 'expensive', 'cheap', 'budget', 'ticket', 'fee'],
    'crowd': ['crowd', 'crowded', 'busy', 'quiet', 'peaceful', 'packed'],
    'cleanliness': ['clean', 'dirty', 'rubbish', 'garbage', 'hygiene', 'smell']
}

def prepare_multi_feature_timeseries():
    print("="*60)
    print("1. Engineering Advanced Time-Series Features & Aspect Regressors")
    print("="*60)

    # 1. Load data
    reviews_df = pd.read_csv('data/processed/Cleaned_Reviews.csv')
    arrivals_df = pd.read_csv('data/raw/sl_tourism_country_wise_final.csv')

    reviews_df['Travel_Date'] = pd.to_datetime(reviews_df['Travel_Date'], errors='coerce')
    arrivals_df['Date'] = pd.to_datetime(arrivals_df['Date'], errors='coerce')

    # 2. National Monthly Arrivals
    monthly_arrivals = arrivals_df.groupby(arrivals_df['Date'].dt.to_period('M'))['Arrivals'].sum().reset_index()
    monthly_arrivals['Date'] = monthly_arrivals['Date'].dt.to_timestamp()

    # 3. Monthly Overall Sentiment Ratio
    reviews_valid = reviews_df.dropna(subset=['Travel_Date']).copy()
    reviews_valid['Period'] = reviews_valid['Travel_Date'].dt.to_period('M')

    monthly_sentiment = reviews_valid.groupby(['Period', 'Sentiment_Class']).size().unstack(fill_value=0).reset_index()
    monthly_sentiment['Date'] = monthly_sentiment['Period'].dt.to_timestamp()
    
    total_reviews = monthly_sentiment.sum(axis=1, numeric_only=True)
    pos_count = monthly_sentiment['Positive'] if 'Positive' in monthly_sentiment.columns else 0
    monthly_sentiment['Sentiment_Score'] = (pos_count / total_reviews).fillna(0.75)

    # 4. Aspect-Specific Monthly Sentiment Regressors
    print("Computing monthly aspect sentiment indicators...")
    monthly_aspect_scores = []
    sentiment_map = {'Positive': 1.0, 'Neutral': 0.0, 'Negative': -1.0}
    reviews_valid['Num_Score'] = reviews_valid['Sentiment_Class'].map(sentiment_map).fillna(0.0)

    for period, group in reviews_valid.groupby('Period'):
        record = {'Period': period}
        text_corpus = group['Clean_Text'].astype(str)
        
        for asp_name, kw_list in ASPECT_WORDS.items():
            pattern = '|'.join(kw_list)
            mask = text_corpus.str.contains(pattern, case=False, regex=True)
            matched = group[mask]
            if len(matched) > 0:
                record[f"{asp_name.capitalize()}_Sentiment"] = round(float(matched['Num_Score'].mean()), 3)
            else:
                record[f"{asp_name.capitalize()}_Sentiment"] = 0.0
        monthly_aspect_scores.append(record)

    aspect_df = pd.DataFrame(monthly_aspect_scores)
    aspect_df['Date'] = aspect_df['Period'].dt.to_timestamp()

    # 5. Merge Datasets
    merged = pd.merge(monthly_arrivals, monthly_sentiment[['Date', 'Sentiment_Score']], on='Date', how='inner')
    merged = pd.merge(merged, aspect_df.drop(columns=['Period']), on='Date', how='left').fillna(0.0)
    merged = merged.sort_values('Date').reset_index(drop=True)

    # 6. Time-Series Feature Engineering
    # Lag Features (t-1, t-3, t-6)
    merged['Arrivals_Lag1'] = merged['Arrivals'].shift(1)
    merged['Arrivals_Lag3'] = merged['Arrivals'].shift(3)
    merged['Arrivals_Lag6'] = merged['Arrivals'].shift(6)

    # Rolling Sentiment (3-month moving average)
    merged['Sentiment_Rolling3'] = merged['Sentiment_Score'].rolling(window=3, min_periods=1).mean()

    # Calendar & Sri Lankan Seasonal Features
    merged['Month'] = merged['Date'].dt.month
    merged['Is_Peak_Season'] = merged['Month'].isin([12, 1, 2, 3]).astype(int)
    merged['Is_Monsoon_Season'] = merged['Month'].isin([5, 6, 10, 11]).astype(int)
    merged['Is_Festival_Month'] = merged['Month'].isin([4, 7, 8]).astype(int)

    # Drop rows where lag-6 is NaN
    merged = merged.dropna(subset=['Arrivals_Lag6']).reset_index(drop=True)
    print(f"Time-series aligned: {len(merged)} monthly records with {merged.shape[1]} features.")
    return merged, reviews_valid

def evaluate_models_and_ablation(df):
    print("\n" + "="*60)
    print("2. Model Benchmarking & Sentiment Ablation Testing")
    print("="*60)

    # Split for holdout evaluation: train on historical data, test on last 6 months
    train = df.iloc[:-6].copy()
    test = df.iloc[-6:].copy()
    actuals = test['Arrivals'].values

    # -------------------------------------------------------------
    # Model 1A: Baseline Prophet (WITHOUT Sentiment)
    # -------------------------------------------------------------
    df_prophet_base = train[['Date', 'Arrivals']].rename(columns={'Date': 'ds', 'Arrivals': 'y'})
    m_p_base = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.05)
    m_p_base.fit(df_prophet_base)
    future_p_base = m_p_base.make_future_dataframe(periods=6, freq='MS')
    fc_p_base = m_p_base.predict(future_p_base)
    preds_p_base = fc_p_base['yhat'].iloc[-6:].values

    # -------------------------------------------------------------
    # Model 1B: Sentiment-Enhanced Prophet (WITH Sentiment Regressor)
    # -------------------------------------------------------------
    df_prophet_enh = train[['Date', 'Arrivals', 'Sentiment_Score']].rename(columns={'Date': 'ds', 'Arrivals': 'y'})
    m_p_enh = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.05)
    m_p_enh.add_regressor('Sentiment_Score')
    m_p_enh.fit(df_prophet_enh)
    future_p_enh = m_p_enh.make_future_dataframe(periods=6, freq='MS')
    future_p_enh['Sentiment_Score'] = df['Sentiment_Score'].values
    fc_p_enh = m_p_enh.predict(future_p_enh)
    preds_p_enh = fc_p_enh['yhat'].iloc[-6:].values

    # -------------------------------------------------------------
    # Model 2: Machine Learning Models (Random Forest & XGBoost)
    # -------------------------------------------------------------
    features_with_sentiment = [
        'Arrivals_Lag1', 'Arrivals_Lag3', 'Arrivals_Lag6',
        'Month', 'Is_Peak_Season', 'Is_Monsoon_Season', 'Is_Festival_Month',
        'Sentiment_Score', 'Sentiment_Rolling3',
        'Safety_Sentiment', 'Cost_Sentiment', 'Crowd_Sentiment', 'Cleanliness_Sentiment'
    ]
    features_no_sentiment = [
        'Arrivals_Lag1', 'Arrivals_Lag3', 'Arrivals_Lag6',
        'Month', 'Is_Peak_Season', 'Is_Monsoon_Season', 'Is_Festival_Month'
    ]

    target = 'Arrivals'

    # Random Forest without sentiment
    rf_base = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_base.fit(train[features_no_sentiment], train[target])
    preds_rf_base = rf_base.predict(test[features_no_sentiment])

    # Random Forest with sentiment
    rf_enh = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_enh.fit(train[features_with_sentiment], train[target])
    preds_rf_enh = rf_enh.predict(test[features_with_sentiment])

    # XGBoost without sentiment
    xgb_base = XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    xgb_base.fit(train[features_no_sentiment], train[target])
    preds_xgb_base = xgb_base.predict(test[features_no_sentiment])

    # XGBoost with sentiment
    xgb_enh = XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    xgb_enh.fit(train[features_with_sentiment], train[target])
    preds_xgb_enh = xgb_enh.predict(test[features_with_sentiment])

    def calc_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        return round(mae, 0), round(rmse, 0), round(mape, 2)

    results = []
    models_dict = {
        "Prophet (Baseline, No Sentiment)": (preds_p_base, False),
        "Prophet (Enhanced + Sentiment)": (preds_p_enh, True),
        "Random Forest (No Sentiment)": (preds_rf_base, False),
        "Random Forest (+ Sentiment Lags)": (preds_rf_enh, True),
        "XGBoost (No Sentiment)": (preds_xgb_base, False),
        "XGBoost (+ Sentiment Lags & Aspects)": (preds_xgb_enh, True)
    }

    for name, (preds, has_sent) in models_dict.items():
        mae, rmse, mape = calc_metrics(actuals, preds)
        results.append({
            'Model': name,
            'Sentiment_Included': "Yes" if has_sent else "No",
            'MAE': mae,
            'RMSE': rmse,
            'MAPE (%)': mape
        })

    benchmark_df = pd.DataFrame(results)
    benchmark_df.to_csv('models/forecaster_benchmark.csv', index=False)
    print(benchmark_df.to_string(index=False))

    # Ablation summary
    p_imp = ((benchmark_df.loc[0, 'MAPE (%)'] - benchmark_df.loc[1, 'MAPE (%)']) / benchmark_df.loc[0, 'MAPE (%)']) * 100
    rf_imp = ((benchmark_df.loc[2, 'MAPE (%)'] - benchmark_df.loc[3, 'MAPE (%)']) / benchmark_df.loc[2, 'MAPE (%)']) * 100
    xgb_imp = ((benchmark_df.loc[4, 'MAPE (%)'] - benchmark_df.loc[5, 'MAPE (%)']) / benchmark_df.loc[4, 'MAPE (%)']) * 100

    print("\n" + "="*50)
    print("Ablation Study: Accuracy Improvement from Sentiment Integration:")
    print(f"  Prophet Improvement       : {p_imp:+.2f}%")
    print(f"  Random Forest Improvement : {rf_imp:+.2f}%")
    print(f"  XGBoost Improvement       : {xgb_imp:+.2f}%")

    # -------------------------------------------------------------
    # 3. Aspect Feature Importance Analysis
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("3. Analyzing Aspect Sentiment Feature Importance")
    print("="*60)
    importances = xgb_enh.feature_importances_
    feat_imp = {}
    for feat, imp in zip(features_with_sentiment, importances):
        feat_imp[feat] = round(float(imp) * 100, 2)

    feat_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))
    with open('models/aspect_feature_importance.json', 'w', encoding='utf-8') as f:
        json.dump(feat_imp, f, indent=2)

    print("Top Feature Importances (% Contribution):")
    for k, v in list(feat_imp.items())[:8]:
        print(f"  - {k:25s}: {v:.2f}%")

    # Save best models
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb_enh, 'models/xgboost_forecaster.pkl')
    joblib.dump(rf_enh, 'models/random_forest_forecaster.pkl')

    # Add forecasts to test df
    test['Forecast_XGBoost'] = preds_xgb_enh
    test['Residual'] = test['Arrivals'] - test['Forecast_XGBoost']
    test['Percentage_Divergence'] = (np.abs(test['Residual']) / test['Arrivals'] * 100).round(2)

    return df, benchmark_df, feat_imp, test

def generate_segmented_and_anomalies(df, reviews_valid, test_eval):
    print("\n" + "="*60)
    print("4. Generating Segmented Category Demand & Anomaly Flags")
    print("="*60)

    # 1. Segmented Category Demand
    categories = ['Beaches', 'Religious Sites', 'Nature & Wildlife Areas', 'Historic Sites']
    category_ts = []

    for period, group in reviews_valid.groupby('Period'):
        row = {'Date': period.to_timestamp()}
        for cat in categories:
            cat_group = group[group['Location_Type'] == cat]
            row[f"{cat}_Review_Volume"] = len(cat_group)
            if len(cat_group) > 0:
                pos_pct = (cat_group['Sentiment_Class'] == 'Positive').mean()
                row[f"{cat}_Sentiment"] = round(float(pos_pct), 3)
            else:
                row[f"{cat}_Sentiment"] = 0.5
        category_ts.append(row)

    cat_df = pd.DataFrame(category_ts).sort_values('Date').reset_index(drop=True)
    cat_df = pd.merge(cat_df, df[['Date', 'Arrivals']], on='Date', how='inner')
    
    cat_csv_path = 'data/processed/TimeSeries_Segmented.csv'
    cat_df.to_csv(cat_csv_path, index=False)
    print(f"Segmented time-series saved to: {cat_csv_path}")

    # 2. Anomaly Flags
    anomalies = []
    # Historical domain shocks documented in Sri Lankan tourism literature
    shock_annotations = {
        '2019-05-01': "Post-Easter Sunday Terror Incident (Significant Arrival Slump)",
        '2020-04-01': "Global COVID-19 Travel Lockdowns (Complete Demand Collapse)",
        '2022-05-01': "Macroeconomic & Fuel Crisis Shocks (Severe Disruption)",
        '2023-01-01': "Post-Crisis Winter Peak Recovery Surge"
    }

    for idx, row in df.iterrows():
        d_str = str(row['Date'])[:10]
        if d_str in shock_annotations:
            anomalies.append({
                'date': d_str,
                'type': 'Macroeconomic / External Shock',
                'description': shock_annotations[d_str],
                'arrivals': int(row['Arrivals']),
                'sentiment': round(float(row['Sentiment_Score']), 3)
            })

    # Flag recent test forecast divergences > 20%
    for _, t_row in test_eval.iterrows():
        if t_row['Percentage_Divergence'] > 20.0:
            anomalies.append({
                'date': str(t_row['Date'])[:10],
                'type': 'Model Divergence Anomaly',
                'description': f"Actual ({t_row['Arrivals']:,}) vs Forecast ({t_row['Forecast_XGBoost']:,.0f}) diverged by {t_row['Percentage_Divergence']}%",
                'arrivals': int(t_row['Arrivals']),
                'divergence_pct': float(t_row['Percentage_Divergence'])
            })

    anomalies_path = 'data/processed/arrival_anomalies.json'
    with open(anomalies_path, 'w', encoding='utf-8') as f:
        json.dump(anomalies, f, indent=2)
    print(f"Detected {len(anomalies)} anomaly/divergence events saved to: {anomalies_path}")

def main():
    df, reviews_valid = prepare_multi_feature_timeseries()
    df, benchmark_df, feat_imp, test_eval = evaluate_models_and_ablation(df)
    generate_segmented_and_anomalies(df, reviews_valid, test_eval)

    # Export merged master time-series
    merged_path = 'data/processed/TimeSeries_Merged.csv'
    df.to_csv(merged_path, index=False)
    print(f"\nFinal merged time-series saved to {merged_path}")
    print("Forecasting pipeline completed successfully!")

if __name__ == '__main__':
    main()