#importing packages
from pyexpat import model

from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os #to tackle apis

#importing stuff for langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

#creating a func that gets text from pdf
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

#creating a func that gets chunks from the text
def get_text_chunks(text,model_name):
    if model_name=="Google AI":
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=text_splitter.split_text(text)
    return chunks

#embedding the chunks and storing them in a vector store
def get_vector_store(text_chunks, model_name, api_key=None):

    print("Number of chunks:", len(text_chunks))

    if len(text_chunks) > 0:
        print("First chunk preview:")
        print(text_chunks[0][:200])

    if model_name == "Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2",
            google_api_key=api_key
        )

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")
    return vector_store

#creating a func that creates a conversational chain using langchain
def get_conv_chain(api_key):

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using only the provided context.

        Context:
        {context}

        Question:
        {input}

        If the answer is not present in the context,
        say: "Answer is not available in the context."

        Answer:
        """
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    return document_chain
    

# creating a func to take user input
def get_user_input(user_question,model_name,api_key,pdf_docs,conversation_history):
    if api_key is None or pdf_docs is None:
        st.warning("Please upload any pdf and provide api key:")
        return
    if "vector_store" not in st.session_state:
        st.warning("Please click 'Submit & Process' first.")
        return

    vector_store = st.session_state.vector_store
    user_question_output=""
    response_output=""
    if model_name=="Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)

        retriever = new_db.as_retriever()
        document_chain = get_conv_chain(api_key)
        retrieval_chain = create_retrieval_chain(retriever,document_chain)

        response = retrieval_chain.invoke({"input": user_question})
        user_question_output=user_question
        response_output = response["answer"]
        pdf_names=[pdf.name for pdf in pdf_docs] if pdf_docs else []
        conversation_history.append((user_question_output,response_output,model_name,datetime.now().strftime('%Y-%m-%d %H-%M-%S'),",".join(pdf_names)))

        #creating the chat design
        st.markdown(
        f"""
        <style>
            .chat-message {{
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
            }}
            .chat-message.user {{
                background-color: #2b313e;
            }}
            .chat-message.bot {{
                background-color: #475063;
            }}
            .chat-message .avatar {{
                width: 20%;
            }}
            .chat-message .avatar img {{
                max-width: 78px;
                max-height: 78px;
                border-radius: 50%;
                object-fit: cover;
            }}
            .chat-message .message {{
                width: 80%;
                padding: 0 1.5rem;
                color: #fff;
            }}
            .chat-message .info {{
                font-size: 0.8rem;
                margin-top: 0.5rem;
                color: #ccc;
            }}
        </style>
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
            </div>    
            <div class="message">{user_question_output}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
            </div>
            <div class="message">{response_output}</div>
            </div>
            
        """,
        unsafe_allow_html=True
    )
    if len(conversation_history)==1:
        conversation_history=[]
    elif len(conversation_history)>1:
        last_item=conversation_history[-1]
        conversation_history.remove(last_item)
    for question,answer,model_name,timestamp,pdf_name in reversed(conversation_history):
        st.markdown(
            f"""
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                </div>    
                <div class="message">{question}</div>
            </div>
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
                </div>
                <div class="message">{answer}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    if len(st.session_state.conversation_history) > 0:
        df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])

        # df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV file</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.markdown("To download the conversation, click the Download button on the left side at the bottom of the conversation.")
    st.balloons()

#main function to run the app
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs (v1) :books:")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    linkedin_profile_link = "https://www.linkedin.com/in/raxita/"
    github_profile_link = "https://github.com/raxxitaa"

    st.sidebar.markdown(
        f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin_profile_link}) "
        f"[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github_profile_link})"
    )



    model_name = st.sidebar.radio("Select the Model:", ( "Google AI"))

    api_key = None

    if model_name == "Google AI":
        api_key = st.sidebar.text_input("Enter your Google API Key:")
        st.sidebar.markdown("Click [here](https://ai.google.dev/) to get an API key.")
        
        if not api_key:
            st.sidebar.warning("Please enter your Google API Key to proceed.")
            return

   
    with st.sidebar:
        st.title("Menu:")
        
        col1, col2 = st.columns(2)
        
        reset_button = col2.button("Reset")
        clear_button = col1.button("Rerun")

        if reset_button:
            st.session_state.conversation_history = []  # Clear conversation history
            st.session_state.user_question = None  # Clear user question input 
            
            
            api_key = None  # Reset Google API key
            pdf_docs = None  # Reset PDF document
            
        else:
            if clear_button:
                if 'user_question' in st.session_state:
                    st.warning("The previous query will be discarded.")
                    st.session_state.user_question = ""  # Temizle
                    if len(st.session_state.conversation_history) > 0:
                        st.session_state.conversation_history.pop()  # Son sorguyu kaldır
                else:
                    st.warning("The question in the input will be queried again.")




        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):

                    text = get_pdf_text(pdf_docs)
                    print("Text length:", len(text))

                    chunks = get_text_chunks(text, model_name)
                    print("Chunks generated:", len(chunks))

                    vector_store = get_vector_store(
                        chunks,
                        model_name,
                        api_key
                    )

                    st.session_state.vector_store = vector_store

                st.success("PDF processed successfully!")

            else:
                st.warning("Please upload PDF files before processing.")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        get_user_input(user_question, model_name, api_key, pdf_docs, st.session_state.conversation_history)
        st.session_state.user_question = ""  # Clear user question input 

if __name__ == "__main__":
    main()