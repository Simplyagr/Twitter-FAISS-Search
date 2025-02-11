# Twitter-FAISS-Search

🐦 Twitter FAISS Search
A Streamlit-based web application that allows users to:

- Fetch real-time tweets using the Twitter API.
- Perform sentiment analysis (positive, negative, neutral).
- Store tweet embeddings using FAISS for efficient similarity search.
- Visualize sentiment distribution and generate word clouds.

🚀 Features

✔ Fetch tweets using **Twitter-X API**  
✔ Preprocess & clean tweets  
✔ Perform **sentiment analysis** using `TextBlob`  
✔ Vectorize tweets using `SentenceTransformer`  
✔ Store embeddings in **FAISS** for efficient similarity search  
✔ Search for **similar tweets**  
✔ **Visualizations**: Sentiment bar chart & Word cloud  

# How to Run?

1. Install required dependencies: ```pip install -r requirements.txt```

2. Set up your key from RapidAPI

3. Run the command: ```streamlit run sentimentor_app.py``` 
