import os
import re
from fastapi import FastAPI
from openai import OpenAI
from bs4 import BeautifulSoup
from pydantic import BaseModel
from metaphor_python import Metaphor
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

search_client = Metaphor(api_key=os.environ.get("METAPHOR_API_KEY"))
client = OpenAI(api_key=os.environ.get("TOGETHER_API_KEY"),
  base_url='https://api.together.xyz',
)

class QueryInput(BaseModel):
    query: str
    num_results: int = 5
    model: str = "openchat/openchat-3.5-1210"

def search(query: str, num_results: int):
    response = search_client.search(
                query,
                num_results=num_results,
                use_autoprompt=False
            )
    content = response.get_contents().contents
    clean_content = ''
    for i in content:
        cleaned_str = f"{i.title}[{i.url}]\ncontent: "
        soup = BeautifulSoup(i.extract, 'html.parser')
        cleaned_str += soup.get_text().strip()
        # replace multiple newlines with single newline using regex
        cleaned_str = re.sub(r'\n+', '\n', cleaned_str)
        clean_content += cleaned_str + '\n\n'

    return clean_content

@app.post("/")
def get_response(query: QueryInput):
    user_query = query.query
    num_results = query.num_results
    content = search(user_query, num_results)
    chat_completion = client.chat.completions.create(
        messages=[
            {
            "role": "system",
            "content": "You are an AI assistant. Based on the user query and search results, you have to generate crisp and concise answers. Its should be relevant to the user query. Please give a reference to the source such as links if only in the search results.",
            },
            {
            "role": "user",
            "content": f"Here is the content: {content}\nUser Query: {user_query}",
            }
        ],
        model=query.model,
        max_tokens=1024
    )
    return chat_completion.choices[0].message.content