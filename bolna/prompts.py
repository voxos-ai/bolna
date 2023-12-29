EXTRACTION_PROMPT = """
Given this transcript from the communication between user and an agent, your task is to extract following information:
###JSON Structure
{}
Make sure your response is in ENGLISH.
"""

SUMMARY_JSON_STRUCTURE = {"summary": "Summary of the conversation goes here"}

SUMMARIZATION_PROMPT = """
Given this transcript from the communication between user and an agent your task is to summarize the conversation.

Always respond in given json format

###JSON Structure
{}
""".format(SUMMARY_JSON_STRUCTURE)

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