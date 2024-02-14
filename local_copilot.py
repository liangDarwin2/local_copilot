import gradio as gr
from ollama import Client
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI

client = Client(host='http://localhost:11434')
openai_client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
# MODEL = "phi:chat"
MODEL = "qwen:1.8b-chat"
MODE = "openai"
# MODEL = "mistral:7b-instruct-q3_K_S"

def read_config_file(file="config.txt"):
    '''
    read config file and get value
    '''
    return_dict = {}
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "PDF_PATH" in line:
                PDF_PATH = line.split('=')[1].strip()
                return_dict["PDF_PATH"] = PDF_PATH
            
            if "EMBEDDING_MODEL_NAME" in line:
                EMBEDDING_MODEL_NAME = line.split('=')[1].strip()
                return_dict["EMBEDDING_MODEL_NAME"] = EMBEDDING_MODEL_NAME
            
            if "PLUGIN" in line:
                PLUGIN = line.split('=')[1].strip()
                return_dict["PLUGIN"] = PLUGIN
    return return_dict

# PDF_READER
def process_pdf_and_rag_query(file, question, EMBEDDING_MODEL_NAME):
    '''
    process pdf file use langchain
    '''
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()

    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "mps"},
    )

    faiss_index = FAISS.from_documents(pages, embeddings)
    docs = faiss_index.similarity_search(question, k=2)
    context = ""
    for doc in docs:
        context += doc.page_content[:500] + "\n"
    
    _rag_query_text = """
    Here are the set of contexts:

    {context}

    And here is the user question:
    {question}

    Let's think step by step.
    """
    post_question = _rag_query_text.format(context=context, question=question)
    return post_question

# IELTS
def IELTS_Writing_Assistant(question):
    system_prompt = """
    This GPT, named 'IELTS Writing Assistant', is designed to help users with IELTS writing task 2. You should aim for at least 250 words for Task 2.

    For Writing Task 2, Your task is to write an essay responding to the given statement and question. Discuss both these views and give your own opinion. Provide reasons for your answer and include any relevant examples from your own knowledge or experience. Write at least 250 words.

    Guidelines:

    Introduction: Briefly introduce the topic and state your thesis, outlining the main points you will discuss.
    Body Paragraph 1: Discuss the viewpoint that social media platforms have negative impacts. Provide specific reasons and examples.
    Body Paragraph 2: Discuss the viewpoint that social media platforms offer significant benefits. Again, support your discussion with reasons and examples.
    Conclusion: Summarize the main points discussed in both body paragraphs and clearly state your own opinion, justifying it with the arguments presented.
    
    Remember to:

    You must strictly follow the above writing guidelines.
    Use a formal tone throughout the essay.
    Organize your essay clearly and logically.
    Make sure your essay is well-structured, with distinct paragraphs for the introduction, body, and conclusion.
    Check your work for grammatical accuracy and coherence. DO NOT quote any reference in your essay.
    Write at least 250 words.

    Here is the writing task:
    {question}

    Complete your essay in the following.
    """
    prompt = system_prompt.format(question=question)
    return prompt

def code_interpreter(question):
    system_prompt = """
    You are a coding expert that can help users solve their tasks with Python code. 
    Write some Python code to solve the following question, in the end of the code, you should use print instead of return to output the result:
    {question}
    Remember, you don't have to execute the code, just write the code to solve the question.
    """
    return system_prompt.format(question=question)

def predict(message, history):
    history_ollama_format = []
    for human, ai in history:
        history_ollama_format.append({"role": "user", "content": human})
        history_ollama_format.append({"role": "assistant", "content": ai})
    
    ## YOUR CODE HERE
    return_dict = read_config_file()
    print(return_dict)
    if return_dict["PLUGIN"] == 'PDF_READER':
        post_question = process_pdf_and_rag_query(return_dict["PDF_PATH"], message, EMBEDDING_MODEL_NAME=return_dict["EMBEDDING_MODEL_NAME"])
        history_ollama_format.append({"role": "user", "content": post_question})
        print(post_question)
    elif return_dict["PLUGIN"] == 'IELTS':
        post_question = IELTS_Writing_Assistant(message)
        history_ollama_format.append({"role": "user", "content": post_question})
        print(post_question)
    elif return_dict["PLUGIN"] == 'CODE':
        post_question = code_interpreter(message)
        history_ollama_format.append({"role": "user", "content": post_question})
        print(post_question)
    else:
        history_ollama_format.append({"role": "user", "content": message})
    ## END OF YOUR CODE
    
    if MODE == "ollama":
        stream = client.chat(MODEL, messages=history_ollama_format, stream=True)
        gpt_response = []
        for chunk in stream:
            gpt_response.append(chunk['message']['content'])
            yield ''.join(gpt_response)
    elif MODE == "openai":
        gpt_response = []
        completion = openai_client.chat.completions.create(
            model="local-model", # this field is currently unused
            messages=history_ollama_format,
            temperature=0.7,
            stream=True,
        )
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                gpt_response.append(chunk.choices[0].delta.content)
                yield ''.join(gpt_response)
    else:
        raise ValueError("MODE must be either 'ollama' or 'openai'")

def test():
    return_dict = read_config_file()
    print(return_dict)
    result = process_pdf_and_rag_query(return_dict["PDF_PATH"], "Summarize the article.",EMBEDDING_MODEL_NAME=return_dict["EMBEDDING_MODEL_NAME"])
    print(result)


if __name__ == "__main__":
    # test()
    with gr.Blocks() as demo:
        gr.Markdown("""<p align="center"><img src="https://www.thesoftwarereport.com/wp-content/uploads/2023/10/Copilot.jpg" style="height: 80px"/><p>""")
        gr.HTML("""<h1 align="center">Local Copilot</h1>""")
        gr.ChatInterface(predict)
    demo.launch()