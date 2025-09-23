import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Set up DeepSeek client (using OpenAI package)
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# Test the connection
def test_api():
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello! Can you help with real estate?"}],
            max_tokens=50
        )
        print("✅ API connection successful!")
        print("Response:", response.choices[0].message.content)
        return True
    except Exception as e:
        print("❌ API connection failed:", e)
        return False

if __name__ == "__main__":
    test_api()