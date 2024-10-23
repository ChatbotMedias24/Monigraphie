import streamlit as st
import openai
import streamlit as st
from dotenv import load_dotenv
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os
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
import docx2txt
from dotenv import load_dotenv
if 'previous_question' not in st.session_state:
    st.session_state.previous_question = []

# Chargement de l'API Key depuis les variables d'environnement
load_dotenv(st.secrets["OPENAI_API_KEY"])

# Configuration de l'historique de la conversation
if 'previous_questions' not in st.session_state:
    st.session_state.previous_questions = []

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
        input[type="text"] {
            background-color: #E0E0E0;
        }

        /* Style for placeholder text with bold font */
        input::placeholder {
            color: #555555; /* Gris foncé */
            font-weight: bold; /* Mettre en gras */
        }

        /* Ajouter de l'espace en blanc sous le champ de saisie */
        .input-space {
            height: 20px;
            background-color: white;
        }
    
    </style>
    """,
    unsafe_allow_html=True
)
# Sidebar contents
textcontainer = st.container()
with textcontainer:
    logo_path = "medi.png"
    logoo_path = "NOTEPRESENTATION.png"
    st.sidebar.image(logo_path,width=150)
   
    
st.sidebar.subheader("Suggestions:")
questions = [
        "Donnez-moi un résumé du rapport ",
        "Qu'est-ce qu'un datacenter ?",
        "Comment le gouvernement marocaine soutient-il les projets de transformation digitale dans le pays ?",
        "Quelles sont les prévisions de croissance du marché de datacenters au Maroc jusqu'en 2026 ?",
        "Quels défis le Maroc doit-il relever pour devenir un hub technologique africain dans le domaine des Datacenters ?"  ]    
load_dotenv(st.secrets["OPENAI_API_KEY"])
# Initialisation de l'historique de la conversation dans `st.session_state`
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = StreamlitChatMessageHistory()
def main():
    conversation_history = StreamlitChatMessageHistory()  # Créez l'instance pour l'historique

    st.header("Rapport Monographie sectorielle : Les Datacenters 💬")
    
    # Load the document
    docx = 'monographie.docx'
    
    if docx is not None:
        # Lire le texte du document
        text = docx2txt.process(docx)

        # Afficher toujours la barre de saisie
        st.markdown('<div class="input-space"></div>', unsafe_allow_html=True)
        selected_questions = st.sidebar.radio("****Choisir :****", questions)
        # Afficher toujours la barre de saisie
        query_input = st.text_input("", key="text_input_query", placeholder="Posez votre question ici...", help="Posez votre question ici...")
        st.markdown('<div class="input-space"></div>', unsafe_allow_html=True)

        if query_input and query_input not in st.session_state.previous_question:
            query = query_input
            st.session_state.previous_question.append(query_input)
        elif selected_questions:
            query = selected_questions
        else:
            query = ""

        if query :
            st.session_state.conversation_history.add_user_message(query)  # Ajouter à l'historique
            if "Donnez-moi un résumé du rapport" in query:
                summary = "Le rapport Monographie sectorielle détaille le secteur des Datacenters au Maroc et dans le monde, en mettant en lumière le marché en pleine croissance, les technologies clés, et les stratégies de développement à l'échelle nationale et internationale. Il aborde l'essor du marché marocain des Datacenters, soulignant son potentiel de croissance grâce à la stratégie nationale du numérique et l'ambition du Maroc de devenir un hub technologique africain. Le document explore également les récents développements du secteur à l'échelle mondiale, y compris les tendances, les technologies de pointe, et l'importance des Datacenters dans la transformation digitale. Des analyses spécifiques sur l'infrastructure, les services offerts, et les défis tels que la sécurité des données et l'efficacité énergétique sont inclus, ainsi que des perspectives sur les opportunités d'investissement et les implications pour le futur du secteur."
                st.session_state.conversation_history.add_ai_message(summary) 


        
            else:
                messages = [
                {
                    "role": "user",
                    "content": (
                        f"{query}. En tenant compte du texte suivant essayer de ne pas dire le texte ne contient pas les informations si ne trouve pas à partir de texte répondre d'aprés votre connaissance stp et ne dire pas stp le texte incomplet ou incompréhensible essayer de formulé une bon réponse sans critiquer le texte par exemple ne pas dire texte fragmenter ou quelque chose comme ça répondre directement stp parceque je vais l'afficher o lecteur: {text} "
                    )
                }
            ]

            # Appeler l'API OpenAI pour obtenir le résumé
                response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages
            )

            # Récupérer le contenu de la réponse

                summary = response['choices'][0]['message']['content']
            
                # Votre logique pour traiter les réponses
            #conversation_history.add_user_message(query)
            #conversation_history.add_ai_message(response)
                st.session_state.conversation_history.add_ai_message(summary)  # Ajouter à l'historique
            
            # Afficher la question et le résumé de l'assistant
            #conversation_history.add_user_message(query)
            #conversation_history.add_ai_message(summary)

            # Format et afficher les messages comme précédemment
            formatted_messages = []
            previous_role = None 
            if st.session_state.conversation_history.messages: # Variable pour stocker le rôle du message précédent
                    for msg in conversation_history.messages:
                        role = "user" if msg.type == "human" else "assistant"
                        avatar = "🧑" if role == "user" else "🤖"
                        css_class = "user-message" if role == "user" else "assistant-message"

                        if role == "user" and previous_role == "assistant":
                            message_div = f'<div class="{css_class}" style="margin-top: 25px;">{msg.content}</div>'
                        else:
                            message_div = f'<div class="{css_class}">{msg.content}</div>'

                        avatar_div = f'<div class="avatar">{avatar}</div>'
                
                        if role == "user":
                            formatted_message = f'<div class="message-container user"><div class="message-avatar">{avatar_div}</div><div class="message-content">{message_div}</div></div>'
                        else:
                            formatted_message = f'<div class="message-container assistant"><div class="message-content">{message_div}</div><div class="message-avatar">{avatar_div}</div></div>'
                
                        formatted_messages.append(formatted_message)
                        previous_role = role  # Mettre à jour le rôle du message précédent

                    messages_html = "\n".join(formatted_messages)
                    st.markdown(messages_html, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
