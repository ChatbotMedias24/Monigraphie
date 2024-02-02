import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from streamlit_chat import message  # Importez la fonction message
import toml
import docx2txt
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
if 'previous_question' not in st.session_state:
    st.session_state.previous_question = []
st.markdown(
    """
    <style>

        .user-message {
            text-align: left;
            background-color: #E8F0FF;
            padding: 8px;
            border-radius: 15px 15px 15px 0;
            margin: 4px 0;
            margin-left: 10px;
            margin-right: -40px;
            color:black;
        }

        .assistant-message {
            text-align: left;
            background-color: #F0F0F0;
            padding: 8px;
            border-radius: 15px 15px 15px 0;
            margin: 4px 0;
            margin-left: -10px;
            margin-right: 10px;
            color:black;
        }

        .message-container {
            display: flex;
            align-items: center;
        }

        .message-avatar {
            font-size: 25px;
            margin-right: 20px;
            flex-shrink: 0; /* Empêcher l'avatar de rétrécir */
            display: inline-block;
            vertical-align: middle;
        }

        .message-content {
            flex-grow: 1; /* Permettre au message de prendre tout l'espace disponible */
            display: inline-block; /* Ajout de cette propriété */
}
        .message-container.user {
            justify-content: flex-end; /* Aligner à gauche pour l'utilisateur */
        }

        .message-container.assistant {
            justify-content: flex-start; /* Aligner à droite pour l'assistant */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar contents
textcontainer = st.container()
with textcontainer:
    logo_path = "medi.png"
    logoo_path = "mono.png"
    st.sidebar.image(logo_path,width=150)
    st.sidebar.image(logoo_path,width=150)
   
    
st.sidebar.subheader("Suggestions:")
questions = [
        "Donnez-moi un résumé du rapport ",
        "Qu'est-ce qu'un datacenter ?",
        "Comment le gouvernement marocaine soutient-il les projets de transformation digitale dans le pays ?",
        "Quelles sont les prévisions de croissance du marché de datacenters au Maroc jusqu'en 2026 ?",
        "Quels défis le Maroc doit-il relever pour devenir un hub technologique africain dans le domaine des Datacenters ?"
    ]    
 
load_dotenv(st.secrets["OPENAI_API_KEY"])
conversation_history = StreamlitChatMessageHistory()

def main():
    conversation_history = StreamlitChatMessageHistory()  # Créez l'instance pour l'historique
    st.header("Rapport Monographie sectorielle : Les Datacenters  💬")
    # upload a PDF file
    docx = 'monographie.docx'
 
    # st.write(pdf)
    if docx is not None:
        text = docx2txt.process(docx)
         # Get the first page as an image
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        # st.write(chunks)
 
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open("aaa.pkl", "wb") as f:
            pickle.dump(VectorStore, f)
 
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        #st.markdown("**Posez vos questions ci-dessous:**")
        st.markdown('<p style="margin-bottom: 0;"><h7><b>Posez vos questions ci-dessous:</b></h7></p>', unsafe_allow_html=True)

        query_input = st.text_input("")
        selected_questions = st.sidebar.radio("****Choisir :****",questions)
        if query_input and query_input not in st.session_state.previous_question:
            query = query_input
            st.session_state.previous_question.append(query_input)
        elif selected_questions:
            query = selected_questions
        else:
            query=""
    
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI(model="gpt-3.5-turbo-instruct")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                if "Donnez-moi un résumé du rapport" in query:
                    response = "Le rapport Monographie sectorielle détaille le secteur des Datacenters au Maroc et dans le monde, en mettant en lumière le marché en pleine croissance, les technologies clés, et les stratégies de développement à l'échelle nationale et internationale. Il aborde l'essor du marché marocain des Datacenters, soulignant son potentiel de croissance grâce à la stratégie nationale du numérique et l'ambition du Maroc de devenir un hub technologique africain. Le document explore également les récents développements du secteur à l'échelle mondiale, y compris les tendances, les technologies de pointe, et l'importance des Datacenters dans la transformation digitale. Des analyses spécifiques sur l'infrastructure, les services offerts, et les défis tels que la sécurité des données et l'efficacité énergétique sont inclus, ainsi que des perspectives sur les opportunités d'investissement et les implications pour le futur du secteur."

                conversation_history.add_user_message(query)
                conversation_history.add_ai_message(response)  # Utilisez add_ai_message
        
            formatted_messages = []

            for msg in conversation_history.messages:
                role = "user" if msg.type == "human" else "assistant"
                avatar = "🧑" if role == "user" else "🤖"
                css_class = "user-message" if role == "user" else "assistant-message"
        
                message_div = f'<div class="{css_class}">{msg.content}</div>'
                avatar_div = f'<div class="avatar">{avatar}</div>'
        
                if role == "user":
                    formatted_message = f'<div class="message-container user"><div class="message-avatar">{avatar_div}</div><div class="message-content">{message_div}</div></div>'
                else:
                    formatted_message = f'<div class="message-container assistant"><div class="message-content">{message_div}</div><div class="message-avatar">{avatar_div}</div></div>'
        
                formatted_messages.append(formatted_message)

            messages_html = "\n".join(formatted_messages)
            st.markdown(messages_html, unsafe_allow_html=True)

            # Affichage des avatars à l'extérieur de la div des messages
            

if __name__ == '__main__':
    main()
