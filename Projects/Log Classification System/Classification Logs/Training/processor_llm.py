from dotenv import load_dotenv
import os

from groq import Groq

load_dotenv(r"C:\Users\Smoke Nandos\Desktop\Computer Science Uni\Machine Learning\Projects\Log Classification System\Classification Logs\Training\.env")

groq = Groq()


def classify_with_llm(log_msg):

    prompt = f''' Classify the log message into one of these categories:
    (1) Workflow Error, (2) Deprecation Warning.
    If you can't figure out a category, return "Unclassified".
    Only return the category name. No preamble.
    Log message: {log_msg}'''

    chat_completion = groq.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ])
    return (chat_completion.choices[0].message.content)

if __name__ == "__main__":

    print(classify_with_llm(
        "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."
    ))

    print(classify_with_llm(
        "ahsahsahhadshsahsa"
    ))

    print(classify_with_llm(
        "The 'BulkEmailSender' feature is no longer supported. Use 'EmailCampaignManager' for improved functionality."
    ))