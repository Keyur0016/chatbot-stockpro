import requests
from bs4 import BeautifulSoup
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Pinecone

# Embedding model
embeddings = OpenAIEmbeddings(openai_api_key="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")

# Index name
index_name = "langchain-chatbot"

# Pinecone index
pinecone.init(
    api_key="be71ab51-a1d4-453b-8498-cd67dd5911b7",
    environment="gcp-starter"
)
index = pinecone.Index(index_name)

# Get Website page

pages_list = [
    "https://stage.livestockpro.app/docs/1.0/dashboard",
    "https://stage.livestockpro.app/docs/1.0/paddock",
    "https://stage.livestockpro.app/docs/1.0/todolist"
]

for website_url in pages_list:

    # Website page response
    response = requests.get(website_url)

    # Soup
    soup = BeautifulSoup(response.content, "html.parser")

    # Target division

    target_div = soup.find('div', class_='documentation')

    if target_div:
        element_text = target_div.get_text(separator="\n")
        element_text = element_text.replace("\n", "")

        # Split data
        header_to_split = []
        markdown_spliter = MarkdownHeaderTextSplitter(headers_to_split_on=header_to_split)
        chunk_value = markdown_spliter.split_text(element_text)

        print("Chunk value information ======>")
        print(chunk_value)

        Pinecone.from_documents(chunk_value, embeddings, index_name=index_name)
