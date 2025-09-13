import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

# Initialize the OpenAI LLM
load_dotenv()
model = ChatOpenAI(
    model="gpt-5-nano",  # or "gpt-4o" for better quality
    temperature=0,
    max_tokens=256,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

def get_summaries(metadata_list: list[str]) -> list[str]:
    # Build all prompts
    prompts = []
    for metadata in metadata_list:
        system_prompt = f"""
You are an expert email summarizer. Generate ONE professional paragraph summarizing the email.

REQUIREMENTS:
- The summary MUST include all of these fields exactly once: Email_id, Thread_id, From, To, CC, Source, Date, Labels, Subject.
- Write as a single natural paragraph (NOT bullet points).
- Reproduce all field values exactly as provided. Do NOT shorten, paraphrase, or alter names, email addresses, dates, or labels.
- For "To" and "CC", you MUST explicitly list EVERY recipient including their email address exactly as given. Do NOT use "others", "etc.", "and so on", or similar.
- If a field is empty or 'None', explicitly write "None".
- Use proper grammar and sentence structure, but keep every detail intact.
- The "To" field must contain only the To recipients. The "CC" field must contain only the CC recipients.

Email Details:
{metadata}

Examples:
1. Email abc123, part of thread xyz456, was sent by John Doe (john@example.com). The To recipients are jane@example.com and mark@example.com. The CC's are boss@example.com and hr@example.com. It originated from Gmail on 2024-08-20. The email carries the labels Important and Work. The subject of the email is 'Project Update'.
2. Email eml987654, part of thread thr123456, was sent by Alice Johnson (alice.johnson@example.com). The To recipients are bob.smith@example.com, carol.white@example.com, and dave.miller@example.com. The CC's are erin.green@example.com, frank.brown@example.com, and george.king@example.com. It originated from Microsoft Outlook on 2024-09-15. The email carries the labels High Priority, Client Communication, and Follow-Up. The subject of the email is 'Quarterly Financial Review and Next Steps'.
3. Email eml543210, part of thread thr678910, was sent by Michael Roberts (michael.roberts@example.com). The To recipients are natalie.lee@example.com, oliver.james@example.com, sophia.clark@example.com, and thomas.walker@example.com. The CC's are victoria.hall@example.com, william.young@example.com, and xavier.turner@example.com. It originated from Gmail on 2024-08-25. The email carries the labels Confidential, Internal Review, and Strategy Planning. The subject of the email is 'Upcoming Product Launch Strategy and Marketing Alignment'.

Summary:
"""
        prompts.append(system_prompt)

    # Batch call using generate
    responses = llm.generate(
        [[SystemMessage(content=prompt)] for prompt in prompts]
    )

    # Extract summaries
    summaries = []
    for res in responses.generations:
        text_output = res[0].text.strip()
        if "Summary:" in text_output:
            summary = text_output.split("Summary:")[-1].strip()
        else:
            summary = text_output
        summaries.append(summary)

    return summaries


# ✅ Example Usage
if __name__ == "__main__":
    metadata_samples = [
        """Email_id: eml987654
Thread_id: thr123456
From: Alice Johnson (alice.johnson@example.com)
To: bob.smith@example.com, carol.white@example.com, dave.miller@example.com
CC: erin.green@example.com, frank.brown@example.com, george.king@example.com
Source: Microsoft Outlook
Date: 2024-09-15
Labels: High Priority, Client Communication, Follow-Up
Subject: Quarterly Financial Review and Next Steps""",
        """Email_id: eml543210
Thread_id: thr678910
From: Michael Roberts (michael.roberts@example.com)
To: natalie.lee@example.com, oliver.james@example.com, sophia.clark@example.com, thomas.walker@example.com
CC: victoria.hall@example.com, william.young@example.com, xavier.turner@example.com
Source: Gmail
Date: 2024-08-25
Labels: Confidential, Internal Review, Strategy Planning
Subject: Upcoming Product Launch Strategy and Marketing Alignment"""
    ]

    results = get_summaries(metadata_samples)
    for r in results:
        print(r)
