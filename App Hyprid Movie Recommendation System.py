import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict

# Surprise Library Imports
from surprise import SVD, Dataset, Reader, accuracy

#1. DATA INGESTION & PREPROCESSING
st.set_page_config(page_title="Advanced Hybrid Recommender", layout="wide")
st.title("Hybrid Movie Recommendation System")

@st.cache_data
def load_and_preprocess_data():
    movies_path = r"movies.csv"
    ratings_path = r"ratings.csv"
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    
    # Preprocessing genres
    movies['genres'] = movies['genres'].replace('(no genres listed)', 'Unknown').fillna('Unknown')
    movies['genres_space'] = movies['genres'].str.replace('|', ' ', regex=False)
    
    return movies, ratings

movies, ratings = load_and_preprocess_data()

# --- 2. EVALUATION & SPLIT (80/20) ---
train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

# --- 3. CONTENT-BASED FILTERING (TF-IDF) ---
@st.cache_resource
def build_content_model(movies_df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres_space'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = build_content_model(movies)

# --- 4. COLLABORATIVE FILTERING (Surprise SVD) ---
@st.cache_resource
def build_svd_model(train_data):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(train_data[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    svd = SVD(n_factors=50, random_state=42)
    svd.fit(trainset)
    return svd

svd_model = build_svd_model(train_df)

# --- 5. HYBRID LOGIC & PREDICTION FUNCTION ---
def predict_rating(user_id, movie_id, alpha=0.5):
    # 1. CF Prediction
    cf_score = svd_model.predict(user_id, movie_id).est
    
    # 2. CBF Prediction
    user_ratings = train_df[train_df['userId'] == user_id]
    if user_ratings.empty:
        return cf_score
    
    try:
        m_idx = movies[movies['movieId'] == movie_id].index[0]
        top_m_ids = user_ratings.sort_values(by='rating', ascending=False).head(3)['movieId'].values
        top_m_indices = movies[movies['movieId'].isin(top_m_ids)].index.tolist()
        cbf_score_raw = np.mean(cosine_sim[m_idx, top_m_indices])
        cbf_score = 0.5 + (cbf_score_raw * 4.5)
    except:
        cbf_score = 3.0
        
    return (alpha * cf_score) + ((1 - alpha) * cbf_score)

def get_hybrid_recs(user_id, alpha=0.5, top_n=10):
    all_movie_ids = movies['movieId'].unique()
    user_ratings = train_df[train_df['userId'] == user_id]
    if user_ratings.empty: return None
    
    seen_ids = user_ratings['movieId'].values
    unseen_ids = [m for m in all_movie_ids if m not in seen_ids]
    
    predictions = []
    for mid in unseen_ids:
        score = predict_rating(user_id, mid, alpha)
        predictions.append((mid, score))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_predictions = predictions[:top_n]
    
    top_m_ids = [p[0] for p in top_predictions]
    top_scores = [p[1] for p in top_predictions]
    
    res = movies[movies['movieId'].isin(top_m_ids)].copy()
    res['predicted_rating'] = top_scores
    return res.sort_values(by='predicted_rating', ascending=False)

# --- 6. EVALUATION METRICS ---
def precision_recall_at_k(predictions, k=10, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (est, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, true_r) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    avg_precision = np.mean(list(precisions.values()))
    avg_recall = np.mean(list(recalls.values()))
    
    # Calculate F1-Score
    if (avg_precision + avg_recall) > 0:
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        f1 = 0
        
    return avg_precision, avg_recall, f1

def evaluate_models(alpha=0.5):
    reader = Reader(rating_scale=(0.5, 5.0))
    test_data_surprise = Dataset.load_from_df(test_df[['userId', 'movieId', 'rating']], reader).build_full_trainset().build_testset()
    
    # 1. CF Metrics
    cf_preds = svd_model.test(test_data_surprise)
    rmse_cf = accuracy.rmse(cf_preds, verbose=False)
    mae_cf = accuracy.mae(cf_preds, verbose=False)
    prec_cf, rec_cf, f1_cf = precision_recall_at_k(cf_preds)
    
    # 2. Hybrid Metrics
    hybrid_preds = []
    for uid, mid, true_r in test_data_surprise:
        est = predict_rating(uid, mid, alpha)
        hybrid_preds.append((uid, mid, true_r, est, None))
        
    rmse_hy = np.sqrt(mean_squared_error([x[2] for x in hybrid_preds], [x[3] for x in hybrid_preds]))
    mae_hy = mean_absolute_error([x[2] for x in hybrid_preds], [x[3] for x in hybrid_preds])
    prec_hy, rec_hy, f1_hy = precision_recall_at_k(hybrid_preds)
    
    return {
        "CF": [rmse_cf, mae_cf, prec_cf, rec_cf, f1_cf],
        "Hybrid": [rmse_hy, mae_hy, prec_hy, rec_hy, f1_hy]
    }

# --- 7. UI ---
tab1, tab2 = st.tabs(["Recommendations", "Comparison Metrics"])

with tab1:
    u_id = st.number_input("User ID", 1, 610, 1)
    alpha_val = st.slider("Weight (Alpha)", 0.0, 1.0, 0.5)
    if st.button("Get Recommendations"):
        res = get_hybrid_recs(u_id, alpha=alpha_val)
        if res is not None:
            st.dataframe(res[['title', 'genres', 'predicted_rating']])

with tab2:
    if st.button("Calculate Performance"):
        metrics = evaluate_models(alpha_val)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Collaborative (SVD)")
            st.metric("RMSE", round(metrics["CF"][0], 4))
            st.metric("MAE", round(metrics["CF"][1], 4))
            st.metric("Precision", round(metrics["CF"][2], 4))
            st.metric("Recall", round(metrics["CF"][3], 4))
            st.metric("F1-Score", round(metrics["CF"][4], 4))
            
        with col2:
            st.subheader(f"Hybrid (Alpha={alpha_val})")
            st.metric("RMSE", round(metrics["Hybrid"][0], 4))
            st.metric("MAE", round(metrics["Hybrid"][1], 4))
            st.metric("Precision", round(metrics["Hybrid"][2], 4))
            st.metric("Recall", round(metrics["Hybrid"][3], 4))
            st.metric("F1-Score", round(metrics["Hybrid"][4], 4))
        
        st.divider()
        
