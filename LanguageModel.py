import openai

with open('openai_secret_key.txt') as f:
    openai_secret_key = f.readlines()[0]
openai.api_key = openai_secret_key


def invoke_model(prompt, model="text-davinci-002", temperature=0.7, max_tokens=256, top_p=1, frequency_penalty=0,
                 presence_penalty=0):
    """
    takes the given prompt and returns the output provided from open-ai's model.
    :param prompt: some text (with additional few shot)
    :return: the entire response with the textual model's return value
    """
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    actual_return = response.choices[0].text
    if response['usage']['prompt_tokens'] > max_tokens:
        print("<Warning>: exceeding max tokens allowed")
    return response, actual_return


def merge_text_with_two_shot(txt):
    """
    takes the prompt we wish to provide out model with add concatenates it
    with two-shot example in the right format.
    """
    with open('two_shot_example.txt') as f:
        two_shot_example = f.readlines()
    prompt = f"{''.join(two_shot_example)} {txt}\nOutput: "
    return prompt


if __name__ == '__main__':
    txt = "Elimelech Street in Ramat Gan, 3.5-room apartment of 80 sqm," \
          " on the ground floor surrounded by greenery. NIS 5,800, housing committee 100, property tax 640." \
          " contact 0502666935"
    prompt = merge_text_with_two_shot(txt)
    res, ret = invoke_model(prompt)
    print(ret)

