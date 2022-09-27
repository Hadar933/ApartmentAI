import openai

with open('openai_secret_key.txt') as f:
    openai_secret_key = f.readlines()[0]
openai.api_key = openai_secret_key


def invoke_model(prompt):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    actual_return = response.choices[0].text
    print(actual_return)
    return response, actual_return


if __name__ == '__main__':
    invoke_model("What is love?")