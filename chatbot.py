import pinecone
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

# Give title
st.title("Live StockPro Chatbot")

# Chat history
chat_history = []

# Setup openai api key
openai_api_key = "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"

# Setup openai model
openai_model = "gpt-4-vision-preview"

# Pinecone setup
index_name = "langchain-chatbot"
pinecone.init(
    api_key="be71ab51-a1d4-453b-8498-cd67dd5911b7",
    environment="gcp-starter"
)
index = pinecone.Index(index_name)

# Embedding model
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

vectorstore = Pinecone.from_existing_index(index_name, embedding=embedding)


def generate_response(question):
    """ Generate user question response information """

    prompt = "Please, provide valid answer of question from give context information"

    query_result = embedding.embed_query(question)
    pinecone_data = index.query(
        vector=query_result,
        top_k=1,
        include_metadata=True
    )

    # Prompt passing context information

    context = ""
    for item in pinecone_data['matches']:
        context = context + item['metadata']['text']

    # Prompt templates
    template = """Question: {question}.  
    Provide answer of question from provided context and answer must be accurate and meaningful.Not include answer keywords in response """
    template = template + f"Context information =====> {context}"

    prompt = PromptTemplate(template=template, input_variables=['question'])

    llm = OpenAI(openai_api_key=openai_api_key)
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    answer = llm_chain.run(question)
    return answer


def main():
    # Setup OpenAI model
    model = ChatOpenAI(model=openai_model, temperature="0.8", openai_api_key=openai_api_key)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    st.session_state.qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(search_type="mmr"),
        memory=memory
    )

    st.chat_message("ai").write("How can i help you?")

    if "message" not in st.session_state:
        st.session_state.message = []

    for message in st.session_state.message:
        if message['role'] == "human":
            st.chat_message(message['role']).write(message['content'])
        else:
            st.markdown(message['content'])

    if query := st.chat_input("You message"):
        st.session_state.message.append({"role": "human", "content": query})
        st.chat_message("human").write(query)

        with st.spinner("Generating response"):
            response = generate_response(query)
            st.session_state.message.append(({"role": "ai", "content": response}))
            st.chat_message('ai').write(response)


if __name__ == '__main__':
    main()
