import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration (Makes the UI look clean and professional)
st.set_page_config(
    page_title="Intelligent AI FAQ Chatbot", 
    page_icon="🤖", 
    layout="centered"
)

# 2. Load the Semantic Model (Cached so it only loads once on startup)
@st.cache_resource
def load_model():
    # A lightweight, fast, and highly accurate model for text embeddings
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# 3. Knowledge Base (Your predefined FAQs and Answers)
faq_questions = [
    "How do I get my certificate?",
    "What tracks or domains are available for internships?",
    "What is AI development?",
    "How long does the internship program last?",
    "Who can apply for these programs?"
]

faq_answers = [
    "You can download your completion certificate from your profile dashboard once you submit your final project evaluation successfully.",
    "We offer multiple domain tracks including Web Development, Data Analytics, Artificial Intelligence, and Machine Learning.",
    "AI development involves building intelligent systems using technologies like Python, Machine Learning, and NLP to automate tasks.",
    "Our standard virtual internship duration is usually 4 weeks, with flexible schedules to manage your academic commitments.",
    "Any student pursuing technical degrees like B.Tech, BCA, MCA, or tech enthusiasts looking to build hands-on skills can apply."
]

# Pre-compute embeddings for our knowledge base FAQs
faq_embeddings = model.encode(faq_questions)

# 4. User Interface Structure
st.title("🤖 Intelligent AI FAQ Chatbot")
st.write("Ask any question below about our platform, internships, or AI development!")

# User input text box
user_query = st.text_input("Type your question here...", placeholder="e.g., How do I get my certificate?")

# 5. Matching & Fallback Logic
if user_query.strip():
    # Add a visual loader while the model processes the query
    with st.spinner("Searching for the best answer..."):
        # Generate embedding for the user's input
        query_embedding = model.encode([user_query])
        
        # Calculate cosine similarity against all knowledge base questions
        similarities = cosine_similarity(query_embedding, faq_embeddings)[0]
        
        # Identify the highest score and its corresponding index
        best_match_idx = np.argmax(similarities)
        highest_score = similarities[best_match_idx]
    
    st.write("---")
    
    # Check if the match meets our confidence threshold
    if highest_score > 0.40:
        st.subheader("💡 Answer:")
        st.info(faq_answers[best_match_idx])
    else:
        st.subheader("😟 Answer:")
        st.warning("I'm sorry, I couldn't find an exact match for that question in my knowledge base. Please try phrasing it differently!")