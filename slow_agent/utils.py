
from google import genai
import os
from tenacity import retry, stop_after_attempt, wait_random_exponential

# instantiate a single client with your API key (or read from ENV)
genai_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(10))
def completion_with_backoff(*, model, messages, n=1, temperature=0, top_p=1):
    return genai_client.models.generate_content(
        model=model,
        contents=messages
    )