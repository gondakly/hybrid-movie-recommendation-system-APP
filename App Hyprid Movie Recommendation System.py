import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise import accuracy
from collections import defaultdict

#page config
st.set_page_config(page_title="Movie Recommendation Engine", layout="wide")

@st.cache_resource
def train_models_from_scratch():
    # 1. Load Cleaned Datasets
    movies_df = pd.read_csv('D:/2305388_Abdullah Mohamed Elgondakly/Cleaned movies.csv')
    ratings_df = pd.read_csv('D:/2305388_Abdullah Mohamed Elgondakly/Cleaned ratings.csv')
    
    # 2. Build Content-Based Model (TF-IDF & Cosine Similarity)
    # Assuming genres or tags are used for similarity
    tfidf = TfidfVectorizer(stop_words='english')
    # Fill NaN genres if any
    movies_df['genres'] = movies_df['genres'].fillna('')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Create mappings
    id_map = dict(zip(movies_df['movieId'], list(range(len(movies_df)))))
    id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))
    
    # 3. Build Collaborative Model (SVD)
    reader = Reader(rating_scale=(0.5, 5.0))
    # Using the full dataset for training to ensure the app has maximum knowledge
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    svd_model = SVD()
    svd_model.fit(trainset)
    
    # 4. Precompute User Ratings Map
    user_ratings_map = ratings_df.groupby('userId').apply(
        lambda x: dict(zip(x['movieId'], x['rating']))
    ).to_dict()
    
    return movies_df, ratings_df, cosine_sim, id_map, id_to_title, svd_model, user_ratings_map

# Execute Training
with st.spinner("Initializing Recommendation Engines (Training Models)..."):
    try:
        movies_df, ratings_df, cosine_sim, id_map, id_to_title, svd_model, user_ratings_map = train_models_from_scratch()
    except FileNotFoundError:
        st.error("Error: Cleaned movies.csv or Cleaned ratings.csv not found in directory.")
        st.stop()

# --- PREDICTION LOGIC ---

def get_cb_prediction(uid, iid):
    #Content-Based Logic using the live-trained Cosine Matrix
    if uid not in user_ratings_map or iid not in id_map:
        return ratings_df['rating'].mean()
    target_idx = id_map[iid]
    sim_scores = cosine_sim[target_idx]
    weighted_sum, sim_sum = 0, 0
    for rated_mid, rating in user_ratings_map[uid].items():
        if rated_mid in id_map:
            sim = sim_scores[id_map[rated_mid]]
            weighted_sum += (sim * rating)
            sim_sum += sim
    return weighted_sum / sim_sum if sim_sum > 0 else ratings_df['rating'].mean()

def calculate_metrics(predictions, threshold=3.5):
    """Metric Logic for Evaluation Tab"""
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    precisions, recalls = [], []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:10])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:10])
        precisions.append(n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0)
        recalls.append(n_rel_and_rec_k / n_rel if n_rel != 0 else 0)
    avg_p = np.mean(precisions)
    avg_r = np.mean(recalls)
    f1 = 2*(avg_p*avg_r)/(avg_p+avg_r) if (avg_p+avg_r) > 0 else 0
    return {"RMSE": rmse, "MAE": mae, "Precision": avg_p, "Recall": avg_r, "F1-Score": f1}

#
st.title("Multi-model Hybrid Movie Recommender")

tabs = st.tabs(["Content-Based", "Collaborative", "Hybrid System", "Model Evaluation"])

# TAB 1: CONTENT-BASED
with tabs[0]:
    st.header("Content-Based Filtering")
    movie_name = st.selectbox("Search Movie Similarity:", movies_df['title'].values)
    if st.button("Find Similar Movies"):
        m_id = movies_df[movies_df['title'] == movie_name]['movieId'].values[0]
        idx = id_map[m_id]
        sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:6]
        for i, score in sim_scores:
            rec_id = [k for k, v in id_map.items() if v == i][0]
            st.write(f"{id_to_title[rec_id]} (Similarity: {score:.2f})")

# TAB 2: COLLABORATIVE
with tabs[1]:
    st.header("Collaborative Filtering (SVD)")
    col_u, col_m = st.columns(2)
    u_input = col_u.number_input("User ID", min_value=1, step=1)
    m_input = col_m.number_input("Movie ID", min_value=1, step=1)
    if st.button("Predict User Rating"):
        prediction = svd_model.predict(u_input, m_input)
        st.metric("Estimated Rating", f"{prediction.est:.2f} / 5.0")

# TAB 3: HYBRID SYSTEM
with tabs[2]:
    st.header("Hybrid Engine")
    h_u = st.number_input("Target User ID", min_value=1, step=1, key="h_u")
    h_m = st.number_input("Target Movie ID", min_value=1, step=1, key="h_m")
    alpha = st.slider("Alpha (SVD Weight)", 0.0, 1.0, 0.7)
    
    if st.button("Get Hybrid Score"):
        cf_est = svd_model.predict(h_u, h_m).est
        cb_est = get_cb_prediction(h_u, h_m)
        final_score = (alpha * cf_est) + ((1 - alpha) * cb_est)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Collaborative", f"{cf_est:.2f}")
        c2.metric("Content", f"{cb_est:.2f}")
        c3.metric("Final Hybrid", f"{final_score:.2f}")

# TAB 4: EVALUATION
with tabs[3]:
    st.header("System Evaluation Metrics")
    st.info("Metrics are calculated using a random test sample of the data.")
    if st.button("Generate Performance Report"):
        with st.spinner("Calculating across test sample..."):
            test_sample = ratings_df.sample(500, random_state=42)
            svd_preds, cb_preds, hybrid_preds = [], [], []

            for _, row in test_sample.iterrows():
                u, i, r = int(row['userId']), int(row['movieId']), row['rating']
                cf_e = svd_model.predict(u, i).est
                cb_e = get_cb_prediction(u, i)
                hy_e = (0.7 * cf_e) + (0.3 * cb_e)
                
                svd_preds.append((u, i, r, cf_e, None))
                cb_preds.append((u, i, r, cb_e, None))
                hybrid_preds.append((u, i, r, hy_e, None))

            eval_results = {
                "SVD": calculate_metrics(svd_preds),
                "Content": calculate_metrics(cb_preds),
                "Hybrid": calculate_metrics(hybrid_preds)
            }
            st.table(pd.DataFrame(eval_results).T)
