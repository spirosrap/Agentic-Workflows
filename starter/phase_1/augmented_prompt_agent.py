# TODO: 1 - Import the AugmentedPromptAgent class
from workflow_agents.base_agents import AugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

# TODO: 2 - Instantiate an object of AugmentedPromptAgent with the required parameters
augmented_agent = AugmentedPromptAgent(openai_api_key, persona)

# TODO: 3 - Send the 'prompt' to the agent and store the response in a variable named 'augmented_agent_response'
augmented_agent_response = augmented_agent.respond(prompt)

# Print the agent's response
print(augmented_agent_response)

# TODO: 4 - Add a comment explaining:
# - What knowledge the agent likely used to answer the prompt.
# - How the system prompt specifying the persona affected the agent's response.

# Knowledge used: The agent used its general knowledge about world geography and capitals,
# specifically its knowledge about France and its capital city Paris. This knowledge comes
# from the LLM's training data which includes geographical information.

# Persona effect: The system prompt specified that the agent should act as a college professor
# and start responses with "Dear students,". This affected the response by:
# 1. Making the response more formal and educational in tone
# 2. Adding the "Dear students," greeting at the beginning
# 3. Potentially providing more detailed or explanatory information than a direct answer would
