import os
import openai
import settings

# Load your API key from an environment variable or secret management service
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
except:
    print("Please set your OPENAI_API_KEY environment variable to your OpenAI API key.")
    exit()

def get_chatgpt_prompt(transcript, 
                       max_tokens = 70,   # 77 is the max for SD prompts so don't generate prompts that are too long for SD
                       chatgpt_mode = "chat-completion",
                       #chatgpt_mode = "text-completion",
                       verbose = False):

    chatgpt_prompt = settings.task_description + '\n\"... ' + transcript + ' ...\"'

    if chatgpt_mode == 'text-completion': # use text completion
        response = openai.Completion.create(model="text-curie-001", prompt=chatgpt_prompt, temperature=0.7, max_tokens=max_tokens)
        #response = openai.Completion.create(model="text-davinci-003", prompt=chatgpt_prompt, temperature=0.7, max_tokens=max_tokens)
        prompt = response.choices[0].text

    elif chatgpt_mode == 'chat-completion': # use chat completion
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system", "content": settings.system_description},
                    {"role": "user", "content": chatgpt_prompt},
                    #{"role": "assistant", "content": "..."},
                    #{"role": "user", "content": "..."}
                ], 
            max_tokens=max_tokens,
        )
        prompt = response.choices[0].message.content

    if verbose: # pretty print the full response json:
        print(response)

    return prompt

if __name__ == "__main__":
    transcript = "It's hard to identify the main problems with the jungle in Congo. In a lot of cases the elephants are already protected by the government, but illegal poachers are still killing them. In some severe cases the elephants are being pushed close to extinction."
    print(get_chatgpt_prompt(transcript, verbose = 1))