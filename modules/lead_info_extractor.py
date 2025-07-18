from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def get_company_leads(company_name: str):
    prompt = f"""
You are a lead research assistant. Your task is to find one relevant lead for the company: {company_name}

Please return exactly **one lead** for each of the 3 tiers defined below.
Please mke sure you search the web well for the linkedIn urls before its very important. Only return 'N/A' if not found.

Tier 1 (Head of Company): CEO, President, Founder, Managing Director  
Tier 2 (Head of Sales): Chief Commercial Officer (CCO), Head of Sales, Vice President of Sales, Sales Manager, Director of Sales  
Tier 3 (Head of Marketing): Chief Marketing Officer (CMO), Head of Marketing, VP of Marketing, Marketing Director

Return a list of exactly 3 dictionaries in the following format:

{{
  "name": "<Full Name>",
  "role": "<Role>",
  "company": "{company_name}",
  "tier": "<Tier Number>",
  "linkedin_url": "<LinkedIn Profile URL or 'N/A'>"
}}

Instructions:
- Do not return more than 3 leads.
- Return only one lead per tier (Tier 1 to Tier 3).
- If a lead is not found for a tier, do not return anything for that tier.
- Return only the JSON array. No explanation, no intro, no extra text.
"""

    response = client.responses.create(
        model="gpt-4.1",
        tools=[{
            "type": "web_search_preview",
            "search_context_size": "high",
        }],
        input=prompt
    )

    return response.output_text
