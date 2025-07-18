import yaml
from rich import print
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import json
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from ddgs import DDGS
import json


load_dotenv()

model_name = "gemini-2.5-flash-preview-05-20"
temperature = 0.1

llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_retries=2)


@tool
def get_company_links(company: str) -> dict:
    """Returns the website and LinkedIn URL of a given company using DuckDuckGo"""
    website = None
    linkedin = None

    with DDGS() as ddgs:
        query = f"{company} site:linkedin.com OR site:{company}.com"
        results = ddgs.text(query, max_results=15)

        for result in results:
            url = result.get("href", "").lower()
            if not linkedin and "linkedin.com/company" in url:
                linkedin = url
            elif not website and company.lower() in url and not "linkedin" in url:
                website = url

            if website and linkedin:
                break

    return {
        "company": company,
        "website": website,
        "linkedin": linkedin
    }


# Create the agent
agent = initialize_agent(
    tools=[get_company_links],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)


def get_company_info(company_name: str) -> dict:
    """
    Get the LinkedIn and website URLs for a given company name.
    :param company_name: Name of the company to search for.
    :return: Dictionary containing the website and LinkedIn URL.
    """
    response = agent.invoke(f"Find the official LinkedIn and website for {company_name}. Return only the URLs in JSON format.")

    # Extract the output string
    output = response.get("output") if isinstance(response, dict) else response

    # Remove code block markers if present
    if isinstance(output, str):
        content = output.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
    else:
        content = output

    try:
        data = json.loads(content)
        # If the result is nested (company_name as key), extract the inner dict
        if isinstance(data, dict) and company_name in data:
            return data[company_name]
        # If the result is already the desired dict
        if isinstance(data, dict) and "website" in data and "linkedin" in data:
            return data
        # Fallback: return empty
        return {"website": None, "linkedin": None}
    except Exception as e:
        print(f"Error decoding JSON response: {e}")
        return {"website": None, "linkedin": None}
