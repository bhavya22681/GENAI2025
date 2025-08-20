import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv(
    "E:/BANKING POLICY QA BOT/ChatBot-flask/data/archive2/Dataset_Banking_chatbot.csv",
    encoding="latin1"
)

# Prepare data
queries = df["Query"].astype(str).tolist()
responses = df["Response"].astype(str).tolist()

# Vectorize queries
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(queries)

def get_response(user_input):
    """Return chatbot response for user input"""
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X)
    
    best_match_idx = similarities.argmax()
    best_score = similarities[0, best_match_idx]

    if best_score < 0.3:  # confidence threshold
        return "Sorry, I’m not sure I understood that. Could you rephrase?"
    
    return responses[best_match_idx]


# ✅ Example conversation loop
if __name__ == "__main__":
    print("🤖 Banking Chatbot (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye! 👋")
            break
        answer = get_response(user_input)
        print("Chatbot:", answer)


