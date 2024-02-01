EXTRACTION_PROMPT = """
Given this transcript from the communication between user and an agent, your task is to extract following information:
###JSON Structure
{}
Make sure your response is in ENGLISH.
"""

SUMMARY_JSON_STRUCTURE = {"summary": "Summary of the conversation goes here"}

SUMMARIZATION_PROMPT = """
Given this transcript from the communication between user and an agent your task is to summarize the conversation.
"""

completion_json_format = {"answer": "A simple Yes or No based on if you should cut the phone or not"}

CHECK_FOR_COMPLETION_PROMPT = """
You are an helpful AI assistant that's having a conversation with customer. 
Based on the given transcript, should you cut the call?\n\n 
RULES: 
1. If user is not interested in talking, or is annoying or something, we need to cut the phone. 
2. You are also provided with original prompt to make your decision if we need to cut the phone or not.  

### JSON Structure
{}

""".format(completion_json_format)

EXTRACTION_PROMPT_GENERATION_PROMPT = """
I've asked user to explain in English what data would they like to extract from the conversation. A user will write in points and your task is to form a JSON by converting every point into a respective key value pair.
Always use SNAKE_CASE with lower case characters as JSON Keys

### Example input
1. user intent - intent for the user to come back on app. Example cold, lukewarm, warm, hot.
2. user pulse - Whether the user beleives India will win the world cup or not. Example Austrailia will win the cup, yields no, Rohit Sharma will finally get a world cup medal yields yes 

### Example Output
{
"user_intent": "Classify user's intent to come back to app into cold, warm, lukewarm and hot",
"user_pulse": "Classify user's opinion on who will win the worldcup as "Yes" if user thinks India will win the world cup. Or "No" if user thinks India will not win the worldcup.
}

### Rules
{}
"""

CONVERSATION_SUMMARY_PROMPT = """
Your job is to create the persona of users on based of previous messages in a conversation between an AI persona and a human to maintain a persona of user from assistant's perspective.
Messages sent by the AI are marked with the 'assistant' role.
Messages the user sends are in the 'user' role.
Gather the persona of user like their name, likes dislikes, tonality of their conversation, theme of the conversation or any anything else a human would notice.
Keep your persona summary less than 150 words, do NOT exceed this word limit.
Only output the persona, do NOT include anything else in your output.
If there were any proper nouns, or number or date or time involved explicitly maintain it.
"""