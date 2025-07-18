import yaml
from rich import print
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import json
from dotenv import load_dotenv
from modules.schema import JobQualifier


load_dotenv()



# Load YAML from file
with open("configs/job_qualification.yml", "r") as file:
    data = yaml.safe_load(file)

job_post_analysis = data["prompt_v2"]["job_post_analysis"]
instruction_json = json.dumps(job_post_analysis, indent=2)
model_name = "gemini-2.5-flash-preview-05-20"
temperature = 0.1


llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_retries=2)


# Set up the parser for JobQualifier
parser = PydanticOutputParser(pydantic_object=JobQualifier)

prompt = PromptTemplate(
    template="""You are an AI agent. Use the following JSON configuration to analyze job posts.\n{instruction_json}\n\nHere is the job post:\n{query}\n\n{format_instructions}\n""",
    input_variables=["query"],
    partial_variables={
        "instruction_json": instruction_json,
        "format_instructions": parser.get_format_instructions(),
    },
)



async def job_classifier_function(job_post: str) -> str:
    """
    This function used AI to classify job post into 3 distinct groups for prospecting.
    :param job_post:
    :return: content the return called which is of type JobQualifier
    """
    prompt_str = prompt.format(query=job_post)


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
        print("[bold green]Parsed output as JobQualifier:[/bold green]")
        return parsed.model_dump_json(indent=2)
    except Exception as e:
        print("[yellow]Could not parse output as JobQualifier. Raw output:[/yellow]")
        print("[red]Error:[/red]", e)
        return content
