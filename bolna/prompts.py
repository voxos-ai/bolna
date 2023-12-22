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
