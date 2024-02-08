import gradio as gr
from ollama import Client
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from googlesearch import search
import requests
from bs4 import BeautifulSoup

client = Client(host='http://localhost:11434')
# MODEL = "phi:chat"
# MODEL = "qwen:1.8b-chat"
MODEL = "mistral:7b-instruct-q3_K_S"

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

# WEB_PILOT
def web_pilot(query,topk=3):
    '''
    search on the web with Google and get the topk results
    '''
    system_prompt = """
    Given a set of information retrieved from an online search, your task is to analyze, interpret, and synthesize this data to answer the query provided. You are expected to extract relevant details from the information at hand, ensuring your response is informed, precise, and clear. Approach the data critically, identifying key points and themes that directly address the question. Summarize your findings in a comprehensive yet concise manner, maintaining a neutral and informative tone throughout. Your goal is to provide a well-rounded answer that reflects a deep understanding of the topic, based on the evidence provided."

    Information Provided: {context}

    Steps:
    1. Review the provided information carefully, focusing on the most relevant aspects related to the query.
    2. Extract and synthesize key facts, arguments, and insights from the search results.
    3. Organize your answer logically, directly addressing the specific question or topic.
    4. If possible, highlight any consensus or notable perspectives revealed by the search results.
    5. Conclude with a clear, direct answer or summary that encapsulates your understanding of the topic, based on the provided information.

    And answer the following question:
    {query}

    Let's think step by step.
    """

    urls = []
    # 搜索并获取前5条结果的URL
    for i, url in enumerate(search(query, num_results=topk)):
        urls.append(url)
    urls = list(set(urls))

    text = ""
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # 使用BeautifulSoup抓取和处理每个页面的内容
        # 这里的处理方式会根据页面的结构而有所不同
        text += soup.get_text()
    
    # 使用splitlines()方法将文本分割成行
    lines = text.splitlines()

    # 使用列表推导式去除空行
    non_empty_lines = [line for line in lines if line.strip()]

    # 将处理后的行合并回一个字符串
    clean_text = "\n".join(non_empty_lines)

    prompt = system_prompt.format(query=query, context=clean_text)
    return prompt

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
    elif return_dict["PLUGIN"] == 'WEB_PILOT':
        post_question = web_pilot(message)
        history_ollama_format.append({"role": "user", "content": post_question})
        print(post_question)
    else:
        history_ollama_format.append({"role": "user", "content": message})
    ## END OF YOUR CODE

    gpt_response = client.chat(MODEL, messages=history_ollama_format)
    print("GPT RESPONSE: ", gpt_response)
    return gpt_response['message']['content']

def test():
    return_dict = read_config_file()
    print(return_dict)
    result = process_pdf_and_rag_query(return_dict["PDF_PATH"], "Summarize the article.",EMBEDDING_MODEL_NAME=return_dict["EMBEDDING_MODEL_NAME"])
    print(result)


if __name__ == "__main__":
    # test()
    gr.ChatInterface(predict).launch()