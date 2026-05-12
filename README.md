# 🎬 CineRex - AI Movie Recommendation System

CineRex is a content-based movie recommendation system built using Machine Learning and deployed with Streamlit. It suggests movies based on similarity of plot, genres, and metadata.

---

## 📌 Features
- AI-powered movie recommendations
- Content-based filtering using TF-IDF
- Cosine similarity for matching movies
- Interactive Streamlit web app
- Netflix-style UI with hover effects
- Clickable posters linking to IMDb pages

---

## 🧠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn (TF-IDF, Cosine Similarity)
- Streamlit
- HTML/CSS (UI styling)

---

## 📊 Dataset
- Movie metadata dataset (titles, overview, genres, ratings, poster links)
- Preprocessed combined_features column used for vectorization

---

## ⚙️ How it Works
1. Text features (overview, genres) are combined
2. TF-IDF converts text into vectors
3. Cosine similarity calculates movie similarity
4. Top 5 most similar movies are recommended

---

## 📸 UI Preview
<img width="1365" height="672" alt="image" src="https://github.com/user-attachments/assets/d45d984d-ba0b-4940-a83d-fb8084a03a30" />
<img width="1361" height="672" alt="image" src="https://github.com/user-attachments/assets/17da041d-07e9-4d71-affe-00d0eb723ee4" />


---

## ▶️ Run Locally

```bash
git clone https://github.com/yourusername/cinerex.git
cd cinerex
pip install -r requirements.txt
streamlit run app/app.py
