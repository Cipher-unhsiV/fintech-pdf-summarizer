"""
import os
import requests
import torch
import gradio as gr
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.embeddings import LangchainEmbedding

def download_pdf_from_url(url, save_path="/content/Data/input.pdf"):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"PDF downloaded and saved to {save_path}")
    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")

def mod(pdf_url):
    if not os.path.exists("/Data/"):                         # /content/Data --> /Data/
        os.makedirs("/Data/")                                # /content/Data --> /Data/
    download_pdf_from_url(pdf_url)                           # /content/Data --> /Data/
    documents = SimpleDirectoryReader("/Data/").load_data()
    system_prompt = '''You are an expert share market document summarizer specializing in creating concise, comprehensive summaries tailored for professional audiences. Your goal is to summarize pdf which may also include tabular columns, as
accurately as possible based on the instructions and context provided.'''
    query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")
    from huggingface_hub import login
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not found. Please set it in your Space settings.")
    login(token=hf_token)
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=750,
        generate_kwargs={"temperature": 0.5, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        device_map="auto",
        model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}
    )
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )
    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embed_model
    )
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine()
    response = query_engine.query('''You are an expert share market document summarizer specializing in creating concise, comprehensive summaries tailored for professional audiences. Your task is to analyze the given document and generate a structured summary in approximately 500 words. Ensure the summary:
Captures all key points, including data, insights, and observations.
Clearly outlines the context, such as the purpose of the document and relevant background information.
Summarizes tabular data and numerical figures effectively, while retaining accuracy and relevance.
Highlights significant trends, comparisons, or impacts mentioned in the document.
Uses formal and precise language suitable for a corporate or academic audience.
The output should be well-organized with clear headings or bullet points where applicable. Avoid omitting any critical information, and focus on maintaining a balance between brevity and detail.''')
    return str(response.response)

def func(url):
    return mod(url)

iface = gr.Interface(
        fn=func,
        inputs="text",
        outputs=gr.Textbox(
          label="Output Summary",
          placeholder="The summary will appear here . . .",
          lines=10, 
          interactive=False),
        examples=[['https://cdn-sn.samco.in/ec90fa5b637541d3c86fdb86f45d920c.pdf'],
                  ['https://cdn-sn.samco.in/7c8616b72b4aa639c0eda9f44285ab1d.pdf'],
                  ['https://cdn-sn.samco.in/a4b95bc0bdb8361459a8b41bfc0ff317.pdf']],
        flagging_options=["Useful", "Mediocre 50-50", "Not Useful"],
        description="Flag it for every response and classify it according to what you feel!"
    )

iface.launch(share=True, debug=True)
"""






import os
import requests
import torch
import gradio as gr
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.embeddings import LangchainEmbedding
import fitz  # PyMuPDF

# Function to process the PDF directly from URL
def process_pdf_from_url(pdf_url):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        pdf_data = response.content
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text")  # Extract text from each page
        return text
    else:
        print(f"Failed to retrieve PDF. Status code: {response.status_code}")
        return ""

def mod(pdf_url):
    # Process the PDF directly from URL
    document_text = process_pdf_from_url(pdf_url)
    if not document_text:
        return "Failed to process the PDF."
    
    documents = [document_text]  # Just using the text directly
    
    system_prompt = """You are an expert share market document summarizer specializing in creating concise, comprehensive summaries tailored for professional audiences. Your goal is to summarize pdf which may also include tabular columns, as accurately as possible based on the instructions and context provided."""
    query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

    # Hugging Face Token
    from huggingface_hub import login
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not found. Please set it in your Space settings.")
    login(token=hf_token)

    # Define the LLM and embeddings models
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=750,
        generate_kwargs={"temperature": 0.5, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        device_map="auto",
        model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}
    )

    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )

    # Create service context and index
    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embed_model
    )

    # Indexing the document
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine()

    # Query to generate summary
    response = query_engine.query("""You are an expert share market document summarizer specializing in creating concise, comprehensive summaries tailored for professional audiences. Your task is to analyze the given document and generate a structured summary in approximately 500 words. Ensure the summary:
    - Captures all key points, including data, insights, and observations.
    - Clearly outlines the context, such as the purpose of the document and relevant background information.
    - Summarizes tabular data and numerical figures effectively, while retaining accuracy and relevance.
    - Highlights significant trends, comparisons, or impacts mentioned in the document.
    - Uses formal and precise language suitable for a corporate or academic audience.
    The output should be well-organized with clear headings or bullet points where applicable. Avoid omitting any critical information, and focus on maintaining a balance between brevity and detail.""")

    return str(response.response)

# Gradio Interface
def func(url):
    return mod(url)

iface = gr.Interface(
    fn=func,
    inputs="text",
    outputs=gr.Textbox(
        label="Output Summary",
        placeholder="The summary will appear here . . .",
        lines=10,
        interactive=False
    ),
    examples=[
        ['https://cdn-sn.samco.in/ec90fa5b637541d3c86fdb86f45d920c.pdf'],
        ['https://cdn-sn.samco.in/7c8616b72b4aa639c0eda9f44285ab1d.pdf'],
        ['https://cdn-sn.samco.in/a4b95bc0bdb8361459a8b41bfc0ff317.pdf']
    ],
    flagging_options=["Useful", "Mediocre 50-50", "Not Useful"],
    description="Flag it for every response and classify it according to what you feel!"
)

iface.launch(share=True, debug=True)
