import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

#Step 1: Set up LLM (BioMistral with HF)
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="BioMistral/BioMistral-7B"

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,              #keeping it low enough for medical purposes
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm

#Step 2: Connect LLM with FAISS and create QA chain
DB_FAISS_PATH="vectorstore"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything outside of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# load database:
DB_FAISS_PATH="vectorstore"
embedding_model=HuggingFaceEmbeddings(model_name="") #not sure which model (if any at all) was used in the database creation
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# QA Chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",   #concatenates all retrieved ans into a single prompt and ends everything to the LLM at once
    retriever=db.as_retriever(search_kwargs={'k':5}),  #top 5 results
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# activate chain w single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])

