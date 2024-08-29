from datetime import datetime


EXTRACTION_PROMPT = """
Given this transcript from the communication between user and an agent, your task is to extract following information:

###JSON Structure
{}
- Make sure your response is in ENGLISH. 
- If required data doesn't exist or if the transcript is empty, PLEASE USE NULL, 0, or "DOESN'T EXIST" as the values. DO NOT USE RANDOM ARBRITARY DATA.
"""

SUMMARY_JSON_STRUCTURE = {"summary": "Summary of the conversation goes here"}

SUMMARIZATION_PROMPT = """
Given this transcript from the communication between user and an agent your task is to summarize the conversation.
"""

completion_json_format = {"answer": "A simple Yes or No based on if you should cut the phone or not"}

CHECK_FOR_COMPLETION_PROMPT = """
You are an helpful AI assistant that's having a conversation with customer on a phone call. 
Based on the given transcript, should you cut the call?\n\n 
RULES: 
1. If user is not interested in talking, or is annoyed or is angry we might need to cut the phone. 
2. You are also provided with original prompt use the content of original prompt to make your decision. For example if the purpose of the phone call is done and we have all the required content we need to cut the call.

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

FILLER_PROMPT = "Please, do not start your response with fillers like Got it, Noted.\nAbstain from using any greetings like hey, hello at the start of your conversation"

DATE_PROMPT = "### Date\n Today\'s Date is {}"

FUNCTION_CALL_PROMPT = "We did made a function calling for user. We hit the function : {} and send a {} request and it returned us the response as given below: {} \n\n . Understand the above response and convey this response in a context to user. ### Important\n1. If there was an issue with the API call, kindly respond with - Hey, I'm not able to use the system right now, can you please try later? \n2. IF YOU CALLED THE FUNCTION BEFORE, PLEASE DO NOT CALL THE SAME FUNCTION AGAIN!"