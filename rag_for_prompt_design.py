# imports for lang chain tools
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import TavilySearchAPIRetriever
load_dotenv()
class prompt(BaseModel):
    system: str = Field(description="system instructions generated")
    user: str = Field(description="user instructions generated")



# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=prompt)
# web search tool 


retriever = TavilySearchAPIRetriever(k=3)

sys_rephrase_prompt = """You will be given a writing task that the user wants to perform. Suggest 1 web search query to find best practices that a writer should follow to accomplish that writing task. """
sys_inst_prompt =  """You will be given a writing task that the user wants to perform. You will also be given best practices from a web search done by the user to follow to help the user finish the task. Using the given search results, generate coherent, enumerated and specific instructions that a writer can follow to accomplish the task. """
sys_pt = """You will be given a writing task that the user wants to perform through a third party AI writer. You will also be given enumerated and specific guidelines to follow for the task.
     \n ## instructions: {instructions}
    Generate a prompt for a AI writer assistant using the instructions so that the user is able to accomplish the task through the AI writer assistant. 
    First repeat the instructions. 
    These are ONLY writing tasks and the AI writer can only generate text to accomplish the task. Stick to instructions that address writing guidelines only.
    
    ## On Prompt Format 
    Use the json format:
                    {{
                    "system": <These are system level instructions that delineate the persona that the AI writer must adopt to generate the desired output. For example, for a task to grade work, the system prompt can be "You are a teacher grading the work of ...". These also include guidelines for the AI writer to follow and any specific format to follow. Rephrase the above instructions to perform the writing task. Be detailed in your instructions and only use the given instructions above. Generate atleast 8 instructions.>
                    "user" : {task} 
                    }}
    The user will use the above json output as instructions for a AI writer assistant that will perform the writing task for the user. The AI writer only understand input in the above "system" and "user" format"""
    
# rephrase query tool
def make_pipeline(task):
    rephrase_prompt = ChatPromptTemplate.from_messages(
                    [
                    ("system", sys_rephrase_prompt),
                    ("user","task: {task}")
                ]
                )
    print(rephrase_prompt)
    AI_writer_rephrase = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens = 50)
    # write instructions 
    instruction_prompt = ChatPromptTemplate.from_messages(
                    [
                    ("system",sys_inst_prompt),
                    ("user","task: {task}\n web search results: {best_practices}")
                ]
                )

    # ask for approval from user checkbox style 

    AI_writer_instruction = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens = 500)
    # write into prompt 
    meta_prompt_template = ChatPromptTemplate.from_messages(
                    [
                    ("system", sys_pt),
                    ("user","task: {task}")
                ]
                )
    AI_writer_meta = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens = 3000)
    main_prompt = ChatPromptTemplate.from_messages(
                    [
                    ("system", """{system}\n  Generate output in the format:
                     {{
                     "output": <the actual writing that the user asked to generate.>,
                     "reasoning":<explain how you followed the given instructions to generate the output>}}
                     """),
                    ("user","{task}")
                ]
                )
    AI_writer_main = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens = 4000)
    pipeline =   rephrase_prompt | AI_writer_rephrase | StrOutputParser() | {"best_practices":retriever , "task": RunnablePassthrough()} | instruction_prompt | AI_writer_instruction  | StrOutputParser() |  {"instructions":RunnablePassthrough() , "task": RunnablePassthrough()}| meta_prompt_template | AI_writer_meta | parser | {"system":RunnablePassthrough(), "instructions":RunnablePassthrough() , "user":RunnablePassthrough() , "task": RunnablePassthrough()} | main_prompt | AI_writer_main | StrOutputParser()
    return pipeline






tasks = ["I want to send an email for birthday invitation", "I want to write a performance review feedback for manager", "I want to cold message a senior executive for a coffee chat and introduce myself", "write a paper abstract", "provide feedback to a team member on an issue, team member has certain characteristics"]
for task in tasks[1:3]:
    pipeline = make_pipeline(task)

    response = (pipeline.invoke({"task": task}))
    print(">>>>>",response)
    