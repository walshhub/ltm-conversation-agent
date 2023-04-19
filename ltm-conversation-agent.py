import pickle
import os
from termcolor import colored
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory

from dotenv import load_dotenv
load_dotenv()

llm=ChatOpenAI(temperature=0)

filename = input("Enter a filename for the conversation memory: ")
if not filename.endswith(".pickle"):
    filename += ".pickle"

if os.path.exists(filename):
    with open(filename, "rb") as f:
       memory = pickle.load(f)
else:
  memory = ConversationSummaryBufferMemory(llm=llm, return_messages=True, max_token_limit=1600)

conversation = ConversationChain(llm=llm, memory=memory)

print(colored("Enter your initial prompt: (Press enter to exit):", "green"))
user_input = input("> ")

while user_input:
    output = conversation.run(input=user_input)
    print(colored("Agent:", "blue"), output)
    user_input = input(colored("> ", "green"))

print(colored("Goodbye!", "magenta"))

with open(filename, 'wb') as f:
  pickle.dump(conversation.memory, f)
