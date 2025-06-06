from pathlib import Path
from dotenv import dotenv_values
import openai

def main():
    env_path = Path(__file__).resolve().parents[1] / '.devcontainer' / '.env'
    env = dotenv_values(env_path)
    api_key = env.get('OPENAI_API_KEY')
    if not api_key:
        print('OPENAI_API_KEY not found in', env_path)
        return
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': 'Hello, world!'}]
    )
    print(response.choices[0].message.content)

if __name__ == '__main__':
    main()