import pinecone
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
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
openai_api_key = st.secrets['OPENAI_API']

# Setup openai model
openai_model = "gpt-3.5-turbo-16k"

# Pinecone setup
index_name = "langchain-chatbot"

pinecone.init(
    api_key= st.secrets['PINECONE_API'],
    environment="gcp-starter"
)
index = pinecone.Index(index_name)

# Embedding model
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

vectorstore = Pinecone.from_existing_index(index_name, embedding=embedding)

def generate_response(question):
    """ Generate user question response information """

    # Filter data from Pinecone
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
    Provide a detailed and accurate response based on the context provided. Avoid including answer keywords in your response."""
    template = template + f"\nContext Information: {context}"

    prompt = PromptTemplate(template=template, input_variables=['question'])

    # Call OpenAI LLMChain
    llm = OpenAI(openai_api_key=openai_api_key)
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    answer = llm_chain.run(question)
    return answer


def main():
    # Setup OpenAI model
    model = ChatOpenAI(model=openai_model, temperature="0.8", openai_api_key=openai_api_key, max_tokens=3000)

    # Setup Conversational buffer memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Setup Conversational buffer chain
    st.session_state.qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(search_type="mmr"),
        memory=memory
    )

    # Setup initial message
    st.chat_message("ai").write("How can i help you?")

    # Load data from session storage
    if "message" not in st.session_state:
        st.session_state.message = []

    # Show session data
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
