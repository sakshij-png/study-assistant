from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq()

print("Available Groq models:")
try:
    models = client.models.list()
    for m in models.data:
        print(f" - {m.id}")
except Exception as e:
    print(f"Failed to list models: {type(e).__name__}: {e}")

print("\nTesting chat completion with llama-3.3-70b-versatile:")
try:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print("WORKS")
except Exception as e:
    print(f"{type(e).__name__}: {e}")
