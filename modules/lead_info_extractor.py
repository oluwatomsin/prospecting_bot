from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import json
import re


load_dotenv()
model_name = "gemini-2.5-flash-preview-05-20"
temperature = 0.1

llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_retries=2)


# Initialize Tavily Search Tool
tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
)

agent = create_react_agent(llm, [tavily_search_tool])


def get_company_leads(company_name: str):
    prompt = f"""
        You are a lead research assistant. Your task is to find relevant lead for the company: {company_name}
        
        Please make sure you search the web well for the linkedIn urls because its very important. Only return 'N/A' if not found.
        
        Tier 1 (Head of Company): CEO, President, Founder, Managing Director  
        Tier 2 (Head of Sales): Chief Commercial Officer (CCO), Head of Sales, Vice President of Sales, Sales Manager, Director of Sales  
        Tier 3 (Head of Marketing): Chief Marketing Officer (CMO), Head of Marketing, VP of Marketing, Marketing Director
        
        Return a list of dictionaries in the following format:
        
        {{
          "name": "<Full Name>",
          "role": "<Role>",
          "company": "{company_name}",
          "tier": "<Tier Number>",
          "linkedin_url": "<LinkedIn Profile URL or 'N/A'>"
        }}
        
        Instructions:
        - You can return multiple leads per tier (Tier 1 to Tier 3).
        - If no lead is not found for a tier, do not return anything for that tier.
        - Return only the JSON array. No explanation, no intro, no extra text.
        """
    # The agent is a graph, and we need to invoke it with the input
    inputs = {"messages": [("user", prompt)]}
    response = agent.invoke(inputs)

    try:
        raw_output = response.get("messages", [])[-1].content

        # Use regex to extract the JSON string from within the markdown code block
        match = re.search(r"```json\n(.*)\n```", raw_output, re.DOTALL)
        if match:
            json_string = match.group(1)
            return json.loads(json_string)
        else:
            # Fallback if no markdown block is found (less common for models instructed to use it)
            return json.loads(raw_output)

    except (IndexError, json.JSONDecodeError, AttributeError) as e:
        print(f"Error processing agent response: {e}")
        print(f"Raw agent response: {response}")
        return []
