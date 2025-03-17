from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key= "<OPENROUTER_API_KEY>", #API key
)



def query_openrouter(prompt):
    completion = client.chat.completions.create(
    model="google/gemma-3-27b-it:free",
    messages=[
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt
            },
        ]
        }
    ]
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content


if __name__ == "__main__":
    keyword_prompt = """
    Hi, can you please generate a keyword set with 4 keyword subsets: subjects, predicates, objects, and emotions,
    satisfying the following criteria?
    - The emotion set should have 3 different words (positive, negative, neutral)
    - The remaining sets should have 16 high probability words
    - Each set should containg the words and their corresponding sampling probabilites
    The theme can be cooking.
    """
    keywords_response = query_openrouter(keyword_prompt)
