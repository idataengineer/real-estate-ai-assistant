import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# Sample real estate documents (we'll start simple)
real_estate_docs = [
    "Property: 123 Main St, Austin TX. 3 bed, 2 bath. Price: $450,000. Built 2018. Excellent schools nearby.",
    "Property: 456 Oak Ave, Austin TX. 4 bed, 3 bath. Price: $620,000. Built 2020. Swimming pool, large yard.",
    "Market Report: Austin TX average home price increased 8% in 2024. High demand area with tech job growth.",
    "Investment tip: Properties near downtown Austin appreciate 10-12% annually due to tech company expansion."
]

def simple_search(query, docs):
    """Simple search - find docs that contain query words"""
    query_words = query.lower().split()
    relevant_docs = []
    
    for doc in docs:
        doc_lower = doc.lower()
        if any(word in doc_lower for word in query_words):
            relevant_docs.append(doc)
    
    return relevant_docs

def ask_question(question):
    # Find relevant documents
    relevant_docs = simple_search(question, real_estate_docs)
    
    # Create context from relevant docs
    context = "\n".join(relevant_docs)
    
    # Ask AI with context
    prompt = f"""Based on this real estate information:
{context}

Question: {question}

Please answer based only on the provided information."""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    
    return response.choices[0].message.content

# Test it
if __name__ == "__main__":
    test_questions = [
        "What's the average price of homes in Austin?",
        "Tell me about 123 Main St",
        "What properties have swimming pools?"
    ]
    
    for question in test_questions:
        print(f"\n🏠 Question: {question}")
        answer = ask_question(question)
        print(f"🤖 Answer: {answer}")