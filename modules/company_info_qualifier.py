import yaml
from rich import print
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import json
from dotenv import load_dotenv
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.schema import CompanyQualifier
import asyncio


load_dotenv()


# Load YAML from file
with open("configs/company_requirements.yml", "r") as file:
    data = yaml.safe_load(file)

company_classification_instructions = data["prompt_v2"]["company_requirements"]
instruction_json = json.dumps(company_classification_instructions, indent=2)
print(instruction_json)
model_name = "gemini-2.5-flash-preview-05-20"
temperature = 0.1


llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_retries=2)


# Set up the parser for JobQualifier
parser = PydanticOutputParser(pydantic_object=CompanyQualifier)

prompt = PromptTemplate(
    template="""
You are an AI agent. Use the following JSON configuration to analyze company website and LinkedIn.

Instructions:
{instruction_json}

Here is the company info:
{query}

{format_instructions}

When providing your answer, explicitly reference each requirement from the Instructions. Summarize the overall reason for the classification.
""",
    input_variables=["query"],
    partial_variables={
        "instruction_json": instruction_json,
        "format_instructions": parser.get_format_instructions(),
    },
)



async def company_classifier_function(company_linked_info: str, company_website_info: str) -> str:
    """
    This function used AI to classify companies into 2 distinct groups for prospecting.
    :param company_linked_info:
    :param company_website_info:
    :return: content the return called which is of type CompanyQualifier
    """

    company_info = f"company linkedIn info: {company_linked_info}\n\ncompany website info: {company_website_info}"


    prompt_str = prompt.format(query=company_info)


    output = llm.invoke([{"role": "user", "content": prompt_str}])

    # Extract the content string from the AIMessage object
    content = output.content if hasattr(output, "content") else output

    # Remove code block markers if present
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        parsed = parser.parse(content)
        print("[bold green]Parsed output as CompanyQualifier:[/bold green]")
        return parsed.model_dump_json(indent=2)
    except Exception as e:
        print("[yellow]Could not parse output as CompanyQualifier. Raw output:[/yellow]")
        print("[red]Error:[/red]", e)
        return content
