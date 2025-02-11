import streamlit as st
import requests
import faiss
import numpy as np
import re
import pandas as pd
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns



# --- API Details ---
API_URL = "https://twitter-x-api.p.rapidapi.com/api/search/top"

HEADERS = {
    "x-rapidapi-key": "3095133f3cmsh2fedc9b8233d5c1p16606cjsn5f93a72b44cd",  # Replace with your API Key
    "x-rapidapi-host": "twitter-x-api.p.rapidapi.com"
}

# --- Function to Fetch Tweets ---
def fetch_tweets(keyword, count=20):
    params = {"keyword": keyword, "count": str(count)}
    response = requests.get(API_URL, headers=HEADERS, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API Error: {response.status_code}, {response.text}")
        return {}

# --- Function to Clean Text ---
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|\#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower().strip()

# --- Function to Get Sentiment ---
def get_sentiment(text):
    analysis = TextBlob(text)
    return "positive" if analysis.sentiment.polarity > 0 else "negative" if analysis.sentiment.polarity < 0 else "neutral"

# --- Streamlit UI ---
st.title("🐦 Twitter FAISS Search")
keyword = st.text_input("🔍 Enter a keyword to search tweets")

if st.button("📥 Fetch Tweets"):
    tweets_data = fetch_tweets(keyword)
    tweet_texts = []
    processed_tweets = []

    if "data" in tweets_data:
        for tweet in tweets_data["data"]:
            text = clean_text(tweet.get("full_text", tweet.get("text", "")))
            if text:
                sentiment = get_sentiment(text)
                tweet_texts.append(text)
                processed_tweets.append({"Tweet": text, "Sentiment": sentiment})

    if tweet_texts:
        st.success("✅ Tweets processed successfully!")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = np.array(model.encode(tweet_texts))

        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        faiss.write_index(index, "tweets_faiss.index")

        # Store in session state
        st.session_state["processed_tweets"] = processed_tweets
        st.session_state["faiss_index"] = index
        st.session_state["sentence_model"] = model

        # Display tweets in a **tabular format**
        df = pd.DataFrame(processed_tweets)
        st.dataframe(df)

        # --- Sentiment Distribution Bar Chart ---
        st.subheader("📊 Sentiment Distribution")
        sentiment_counts = df["Sentiment"].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax)
        ax.set_ylabel("Number of Tweets")
        ax.set_xlabel("Sentiment")
        st.pyplot(fig)

        # --- Word Cloud ---
        st.subheader("☁️ Word Cloud of Tweets")
        all_text = " ".join(tweet_texts)
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    else:
        st.error("❌ No valid tweets found.")

# --- Search Section ---
query = st.text_input("🔍 Enter a query to find similar tweets")

if st.button("🔎 Search Similar Tweets"):
    if "faiss_index" in st.session_state and "sentence_model" in st.session_state:
        query_embedding = st.session_state["sentence_model"].encode([query]).astype("float32")
        D, I = st.session_state["faiss_index"].search(query_embedding, k=5)
        
        # Prepare results
        similar_tweets = []
        for i in I[0]:
            if 0 <= i < len(st.session_state["processed_tweets"]):
                tweet = st.session_state["processed_tweets"][i]
                similar_tweets.append(tweet)

        # Show results in **tabular format**
        if similar_tweets:
            st.subheader("📌 Top Similar Tweets:")
            df_similar = pd.DataFrame(similar_tweets)
            st.dataframe(df_similar)

            # --- Sentiment Distribution Bar Chart ---
            st.subheader("📊 Sentiment Distribution")
            sentiment_counts = df_similar["Sentiment"].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax)
            ax.set_ylabel("Number of Tweets")
            ax.set_xlabel("Sentiment")
            st.pyplot(fig)

            # --- Word Cloud ---
            st.subheader("☁️ Word Cloud of Similar Tweets")
            all_text = " ".join(df_similar["Tweet"])
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.error("❌ No similar tweets found.")
    else:
        st.error("❌ Please fetch tweets first.")
