import streamlit as st
import pandas as pd
import numpy as np
import pickle
from surprise import dump, accuracy
from collections import defaultdict

st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")

@st.cache_resource
def load_assets():
    # Load Datasets
    movies_df = pd.read_csv('Cleaned movies.csv')
    ratings_df = pd.read_csv('Cleaned ratings.csv')
    
    # Load Models
    with open('tfidf_model.pkl', 'rb') as f:
        tfidf_data = pickle.load(f)
    with open('cosine_similarity_model.pkl', 'rb') as f:
        cosine_data = pickle.load(f)
    _, svd_model = dump.load('collaborative_model.surprise')
    
    # Precompute User Ratings Map for Content/Hybrid Logic
    user_ratings_map = ratings_df.groupby('userId').apply(
        lambda x: dict(zip(x['movieId'], x['rating']))
    ).to_dict()
    
    return movies_df, ratings_df, cosine_data, svd_model, user_ratings_map

movies_df, ratings_df, cosine_data, svd_model, user_ratings_map = load_assets()
cosine_sim = cosine_data['matrix']
id_map = cosine_data['id_map']
id_to_title = cosine_data['id_to_title']

def get_cb_prediction(uid, iid):
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
st.title("Multi-Model Movie Recommendation System")

tab1, tab2, tab3, tab4 = st.tabs(["Content-Based", "Collaborative", "Hybrid System", "Model Evaluation"])

with tab1:
    st.header("Content-Based Filtering (TF-IDF + Cosine)")
    movie_choice = st.selectbox("Select a Movie you liked:", movies_df['title'].values, key="cb_movie")
    if st.button("Get Content Recommendations"):
        m_id = movies_df[movies_df['title'] == movie_choice]['movieId'].values[0]
        idx = id_map[m_id]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        
        st.write("Top 5 similar movies based on genres:")
        for i, score in sim_scores:
            st.success(f"{id_to_title[idx_to_id := list(id_map.keys())[i]]}")

with tab2:
    st.header("Collaborative Filtering (SVD)")
    u_id = st.number_input("Enter User ID:", min_value=1, value=1, key="cf_user")
    m_id_cf = st.number_input("Enter Movie ID to predict rating:", min_value=1, value=1, key="cf_movie")
    if st.button("Predict User Preference"):
        pred = svd_model.predict(u_id, m_id_cf)
        st.metric("Predicted Rating", f"{pred.est:.2f} / 5.0")

with tab3:
    st.header("Hybrid Recommendation System")
    h_uid = st.number_input("Enter User ID:", min_value=1, value=1, key="h_user")
    h_mid = st.number_input("Enter Movie ID:", min_value=1, value=1, key="h_movie")
    alpha = st.slider("Weight for Collaborative Model (Alpha)", 0.0, 1.0, 0.7)
    
    if st.button("Calculate Hybrid Score"):
        cf_score = svd_model.predict(h_uid, h_mid).est
        cb_score = get_cb_prediction(h_uid, h_mid)
        hybrid_score = (alpha * cf_score) + ((1 - alpha) * cb_score)
        
        col1, col2, col3 = st.columns(3)
        col1.write(f"SVD Score: {cf_score:.2f}")
        col2.write(f"Content Score: {cb_score:.2f}")
        col3.write(f"**Final Hybrid: {hybrid_score:.2f}**")

with tab4:
    st.header("Model Evaluation Performance")
    if st.button("Run Full Evaluation (Sampled)"):
        with st.spinner("Calculating metrics..."):
            test_data = ratings_df.sample(500, random_state=42)
            svd_p, cb_p, hy_p = [], [], []

            for _, row in test_data.iterrows():
                u, i, r = int(row['userId']), int(row['movieId']), row['rating']
                # CF
                cf_est = svd_model.predict(u, i).est
                svd_p.append((u, i, r, cf_est, None))
                # CB
                cb_est = get_cb_prediction(u, i)
                cb_p.append((u, i, r, cb_est, None))
                # Hybrid
                hy_est = (0.7 * cf_est) + (0.3 * cb_est)
                hy_p.append((u, i, r, hy_est, None))

            report = {
                "Collaborative (SVD)": calculate_metrics(svd_p),
                "Content-Based": calculate_metrics(cb_p),
                "Hybrid Engine": calculate_metrics(hy_p)
            }
            st.table(pd.DataFrame(report).T)
