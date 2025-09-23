import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize components
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# Initialize embedding model (runs locally, no API cost)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="real_estate")

# More detailed real estate documents
detailed_docs = [
    "Property listing: 123 Main Street, Austin TX 78701. This beautiful 3-bedroom, 2-bathroom home sits on 0.25 acres in the heart of downtown Austin. Built in 2018, it features modern appliances, hardwood floors, and granite countertops. The home is priced at $450,000 and is located in the highly-rated Austin ISD school district. Walking distance to restaurants and entertainment.",
    
    "Property listing: 456 Oak Avenue, Austin TX 78704. Stunning 4-bedroom, 3-bathroom family home built in 2020. Features include a resort-style swimming pool, spacious backyard perfect for entertaining, upgraded kitchen with stainless steel appliances, and a two-car garage. Listed at $620,000 in the desirable South Austin neighborhood with easy access to downtown.",
    
    "Austin Real Estate Market Report Q4 2024: The Austin metropolitan area continues to show strong growth with median home prices reaching $485,000, representing an 8% increase year-over-year. The tech industry boom has driven demand, particularly in central Austin neighborhoods. Properties are selling on average within 25 days of listing. First-time homebuyer activity remains strong despite rising interest rates.",
    
    "Investment Analysis: Austin's real estate market presents excellent opportunities for long-term appreciation. Properties within 10 miles of downtown have shown consistent 10-12% annual appreciation over the past 5 years. The influx of major tech companies including Apple, Google, and Tesla has created sustained demand. Rental properties in university areas yield 6-8% returns.",
    
    "Neighborhood Guide: Austin TX 78701 (Downtown) - Urban living at its finest. Walk score of 95. Average home price $520,000. Known for high-rise condos, converted lofts, and historic homes. Excellent restaurants, nightlife, and cultural attractions. Public transportation available.",
    
    "Neighborhood Guide: Austin TX 78704 (South Austin) - Family-friendly area with tree-lined streets. Average home price $465,000. Known for local businesses, food trucks, and community parks. Highly rated schools and safe neighborhoods. 15-minute drive to downtown."
]

def setup_vector_database():
    """Add documents to ChromaDB with embeddings"""
    for i, doc in enumerate(detailed_docs):
        # Create embedding
        embedding = embedding_model.encode(doc).tolist()
        
        # Add to ChromaDB
        collection.add(
            documents=[doc],
            embeddings=[embedding],
            ids=[f"doc_{i}"]
        )
    print(f"✅ Added {len(detailed_docs)} documents to vector database")

def semantic_search(query, n_results=3):
    """Search using semantic similarity"""
    query_embedding = embedding_model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    return results['documents'][0]

def rag_query(question):
    """RAG: Retrieve relevant docs + Generate answer"""
    # Retrieve relevant documents
    relevant_docs = semantic_search(question, n_results=2)
    
    # Create context
    context = "\n\n".join(relevant_docs)
    
    # Generate answer with context
    prompt = f"""You are a knowledgeable real estate assistant. Based on the following information, answer the user's question accurately and helpfully.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    # Setup
    setup_vector_database()
    
    # Test questions
    questions = [
        "What's the average home price in Austin?",
        "Tell me about neighborhoods in Austin",
        "What investment opportunities are available?",
        "Which property has a swimming pool?",
        "What's the walk score for downtown Austin?"
    ]
    
    for question in questions:
        print(f"\n🏠 Question: {question}")
        answer = rag_query(question)
        print(f"🤖 Answer: {answer}")