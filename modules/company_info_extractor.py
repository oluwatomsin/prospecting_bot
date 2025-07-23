from rich import print
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



def get_company_info(company_name: str) -> dict:
    """
    Get the LinkedIn and website URLs for a given company name.
    :param company_name: Name of the company to search for.
    :return: Dictionary containing the website and LinkedIn URL.
    """
    prompt = f"""
    Search the web and get me the official linkedIn url and website url for company: {company_name}. You response should be 
    in this format: \n\n 
    
    {{
          "company": "{company_name}",
          "linkedin": "<LinkedIn URL or 'N/A>",
          "website": "<Website Url or 'N/A'>"
        }}
    
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


# Example Usage:
if __name__ == "__main__":
    company = "Superbo"
    leads = get_company_info(company)
    print(json.dumps(leads, indent=2))

