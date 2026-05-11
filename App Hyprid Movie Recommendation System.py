import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise import accuracy
from collections import defaultdict

# Page config
st.set_page_config(page_title="Movie Recommendation Engine", layout="wide")

@st.cache_resource
def train_models_from_scratch():
    #Load Cleaned Datasets
    movies_df = pd.read_csv('D:/2305388_Abdullah Mohamed Elgondakly/Cleaned movies.csv')
    ratings_df = pd.read_csv('D:/2305388_Abdullah Mohamed Elgondakly/Cleaned ratings.csv')
    
    #Build Content-Based Model
    tfidf = TfidfVectorizer(stop_words='english')
    movies_df['genres'] = movies_df['genres'].fillna('')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    id_map = dict(zip(movies_df['movieId'], list(range(len(movies_df)))))
    id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))
    
    #Build Collaborative Model (SVD)
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    svd_model = SVD()
    svd_model.fit(trainset)
    
    #Precompute User Ratings Map
    user_ratings_map = ratings_df.groupby('userId').apply(
        lambda x: dict(zip(x['movieId'], x['rating']))
    ).to_dict()
    
    return movies_df, ratings_df, cosine_sim, id_map, id_to_title, svd_model, user_ratings_map

#Execute Training
with st.spinner("Initializing Recommendation Engines..."):
    try:
        movies_df, ratings_df, cosine_sim, id_map, id_to_title, svd_model, user_ratings_map = train_models_from_scratch()
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()
        
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

def get_top_recommendations(uid, model_type="SVD", alpha=0.7, n=10):
    #Generic function to get top N movies not seen by user
    #Get movies user has already seen
    seen_movies = set(user_ratings_map.get(uid, {}).keys())
    all_movie_ids = movies_df['movieId'].unique()
    unseen_movies = [m for m in all_movie_ids if m not in seen_movies]
    
    predictions = []
    for m_id in unseen_movies:
        if model_type == "SVD":
            score = svd_model.predict(uid, m_id).est
        else: # Hybrid
            cf_e = svd_model.predict(uid, m_id).est
            cb_e = get_cb_prediction(uid, m_id)
            score = (alpha * cf_e) + ((1 - alpha) * cb_e)
        predictions.append((m_id, score))
    
    #Sort and take top N
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]
    
    #Format for display
    res = []
    for m_id, score in top_n:
        res.append({"Title": id_to_title.get(m_id, "Unknown"), "Predicted Rating": round(score, 2)})
    return pd.DataFrame(res)

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

#UI INTERFACE
st.title("Multi-model Hybrid Movie Recommender")

tabs = st.tabs(["Content-Based", "Collaborative", "Hybrid System", "Model Evaluation"])

#TAB 1: CONTENT-BASED
with tabs[0]:
    st.header("Content-Based Filtering")
    movie_name = st.selectbox("Search Movie Similarity:", movies_df['title'].values)
    if st.button("Find Similar Movies"):
        m_id = movies_df[movies_df['title'] == movie_name]['movieId'].values[0]
        idx = id_map[m_id]
        sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:11]
        for i, score in sim_scores:
            rec_id = [k for k, v in id_map.items() if v == i][0]
            st.write(f"{id_to_title[rec_id]} (Match: {score*100:.1f}%)")

#TAB 2: COLLABORATIVE
with tabs[1]:
    st.header("Collaborative Filtering (SVD)")
    u_input = st.number_input("Enter User ID to get Top Picks:", min_value=1, step=1, value=1)
    
    if st.button("Generate Recommendations for User"):
        with st.spinner("Analyzing user behavior..."):
            recs = get_top_recommendations(u_input, model_type="SVD")
            st.subheader(f"Top 10 Picks for User {u_input}")
            st.table(recs)

#TAB 3: HYBRID SYSTEM
with tabs[2]:
    st.header("Hybrid Engine (SVD + Content)")
    h_u = st.number_input("Enter User ID:", min_value=1, step=1, key="h_u_input", value=1)
    alpha = st.slider("Alpha (Weight given to SVD)", 0.0, 1.0, 0.7)
    
    if st.button("Get Hybrid Recommendations"):
        with st.spinner("Blending Collaborative & Content signals..."):
            recs = get_top_recommendations(h_u, model_type="Hybrid", alpha=alpha)
            st.subheader(f"Hybrid Recommended List (User {h_u})")
            st.dataframe(recs, use_container_width=True)

#TAB 4: EVALUATION
with tabs[3]:
    st.header("System Evaluation Metrics")
    if st.button("Generate Performance Report"):
        with st.spinner("Calculating metrics..."):
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
                "SVD (Collaborative)": calculate_metrics(svd_preds),
                "Content-Based": calculate_metrics(cb_preds),
                "Hybrid (Balanced)": calculate_metrics(hybrid_preds)
            }
            st.table(pd.DataFrame(eval_results).T)
