import os
import streamlit as st
# from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from st_copy_to_clipboard import st_copy_to_clipboard
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_community import BigQueryVectorStore
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PipelinePromptTemplate, PromptTemplate
from difflib import HtmlDiff
import streamlit.components.v1 as components
import os
from openai import OpenAI




st.set_page_config(page_title='Konbini Correction Tool', page_icon=None, layout="wide")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model="gpt-5.2", api_key=os.environ["OPENAI_API_KEY"])
llm_output = ""
PROJECT_ID = '188768948707'
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "secrets/streamlit-service-account.json"

@st.cache_resource
def rag_initiate():
    # Initiate Embedding Model
    embedding_model = VertexAIEmbeddings(
        model_name="text-multilingual-embedding-002", project=PROJECT_ID
    )

    #Vector Store des captions
    bq_store_captions = BigQueryVectorStore(
        project_id=PROJECT_ID,
        location='us-central1',
        dataset_name='rag_dataset',
        table_name='instagram_captions',
        embedding=embedding_model,
        distance_type='COSINE'
    )

    return embedding_model,bq_store_captions

embedding_model,bq_store_captions = rag_initiate()

# Prompt Template pour le Validateur (LLM2)
validator_prompt_template = (
    "<ROLE>\n"
    "Tu es un assistant linguistique expert chargé de valider la correction apportée à un texte. "
    "Ton rôle est de détecter les erreurs linguistiques résiduelles et de t'assurer que le style rédactionnel est intact."
    "\n</ROLE>"

    "\n\n"

    "<TASK>\n"
    "Ta mission est de : "
    "\n1. Relire attentivement le texte corrigé entre les balises <TEXT TO VALIDATE>. "
    "\n2. Vérifier qu'il ne reste aucune :"
    "\n   - faute d'orthographe,"
    "\n   - faute de syntaxe,"
    "\n   - faute d'accord grammatical ou verbal."
    "\n3. T'assurer que le texte respecte les règles suivantes :"
    "\n   - Le style rédactionnel initial doit être conservé."
    "\n   - Les usernames ou mentions commençant par '@' ne doivent pas être modifiés."
    "\n   - Aucune mise en forme supplémentaire ne doit être ajoutée."
    "\n4. Si tu détectes des erreurs, corrige-les immédiatement en respectant le style rédactionnel et les consignes."
    "\n</TASK>"

    "\n\n"

    "<RULES>\n"
    "Voici les règles à suivre strictement :"
    "\n1. Ne dis RIEN D'AUTRE que la correction du texte. Le résultat final doit pouvoir être copié-collé directement sans aucun changement supplémentaire. "
    "\n2. Si aucune correction n'est nécessaire, renvoie exactement le texte tel qu'il est."
    "\n3. Si tu détectes des erreurs subtiles, comme un accord incorrect entre le sujet et l'attribut, corrige-les immédiatement."
    "\n4. Assure-toi que tous les noms et adjectifs sont accordés correctement, en particulier dans les expressions complexes."
    "\n5. Ne réécris pas le texte inutilement. Concentre-toi uniquement sur les fautes résiduelles."
    "\n6. Ne modifie pas le sens, le ton ou le style du texte sauf si c'est absolument nécessaire pour corriger une erreur."
    "\n7. Ne fais pas de suggestions ou de commentaires, renvoie uniquement le texte corrigé."
    "\n8. Très important : **Préserve les sauts de lignes**"
    "\n9. Préservation des éléments techniques :"
    "\n   - Usernames : Ne modifie pas les usernames qui sont suivis de '@'. Ceux-ci doivent rester inchangés."
    "\n   - Heures : Ne modifie pas les heures ou leur mise en forme."
    "\n   - HTML : Si du HTML est présent, ne le modifie surtout pas. Toutes les balises HTML doivent être respectées et laissées telles quelles."
    "\n</RULES>"

    "\n\n"

    "<TEXT TO VALIDATE>\n"
    "{input}"
    "\n</TEXT TO VALIDATE>"
)


validator_prompt = PromptTemplate.from_template(validator_prompt_template)

def highlight_differences(original, corrected):
    diff = HtmlDiff(wrapcolumn=45)
    diff_html = diff.make_file(original.splitlines(), corrected.splitlines())
    ""
    return diff_html

# Fonction pour valider les corrections avec le validateur
def validate(input):
    print('Validating the output...')
    messages = validator_prompt.invoke({"input": input})
    response = llm.invoke(messages)
    return response



system_prompt_template = (
    "<ROLE>\n"
    "Tu es un assistant expert en correction linguistique, spécialisé dans la préservation des styles rédactionnels des captions de posts Instagram. "
    "\nTu sais parfaitement allier correction grammaticale et respect de l'identité rédactionnelle de l'auteur."
    "\n</ROLE>"

    "\n\n"

    "<TASK>\n"
    "Ta mission est de : "
    "\n- Corriger toutes les fautes d'orthographe, de syntaxe, et de grammaire présentes dans la caption qui est entre les balises <TEXT TO CORRECT>. "
    "\n- Corrige"
    "\n- Prêter une attention particulière aux accords de genre, de nombre, et de temps."
    "\n- Respecter intégralement le style rédactionnel, les tournures de phrases, et le ton de l'auteur du texte. Ton objectif est de conserver l'intention de l'auteur tout en apportant les corrections nécessaires."
    "\n- Inspire-toi des exemples de captions fournis entre les balises <EXAMPLES> pour ajuster ton écriture en respectant les particularités du style rédactionnel tout en corrigeant les erreurs mais également corriger potentiellement les noms des formats"
    "\n- T’inspirer également des noms propres et des noms de formats du média (par exemple, Vidéo Club, Small Talk) pour adapter ta correction au contexte stylistique ou thématique."
    "\n</TASK>"

    "\n\n"
    
    "<RULES>\n"
    "Voici les règles à suivre strictement :"
    "\n1. **Correction sans modification excessive :** Ne dis RIEN D'AUTRE que la correction du texte. Le résultat final doit pouvoir être copié-collé directement dans Instagram sans aucun changement supplémentaire. "
    "\n2. **Aucune mise en forme supplémentaire :** Ne rajoute pas de mise en forme, car il s'agit de captions destinées à Instagram. Tous les éléments HTML présents doivent rester intacts. Ne fais aucune modification de balises."
    "\n3. Très important : **Préserve les sauts de lignes**"
    "\n4. **Préservation des éléments techniques :**"
    "\n  - **Usernames :** Ne modifie pas les usernames qui sont suivis de '@'. Ceux-ci doivent rester inchangés."
    "\n  - **Heures :** Ne modifie pas les heures ou leur mise en forme."
    "\n5. **Pas d'ajouts externes :** Ne rajoute pas d'informations qui ne sont pas présentes dans le texte de base. Ne fais pas de suggestions ou d'ajouts non demandés."
    "\n6. **Cohérence stylistique :** Si une correction est nécessaire, veille à ce qu'elle respecte le ton et le style rédactionnel de la caption. Aucune modification de style n'est permise sauf si elle est strictement liée à la correction de fautes."
    "\n</RULES>"

    "\n\n"

    "<TEXT TO CORRECT>\n"
    "{input}"
    "\n</TEXT TO CORRECT>"

    "\n\n"

    "<EXAMPLES>\n"
    "{context}"
    "\n</EXAMPLES>"
)

full_prompt = PromptTemplate.from_template(system_prompt_template)


def retrieve(input):
    print('Retrieving similar documents...')
    query_vector = embedding_model.embed_query(input)
    retrieved_docs = bq_store_captions.similarity_search_by_vector(query_vector, k=20)

    return retrieved_docs

def generate(input, context):
    print('Generating the output...')
    docs_content = "\n\n----------------------\n\n".join(doc.page_content for doc in context)
    messages = full_prompt.invoke({"input": input, "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response}


st.title("[Test] Konbini Captions Correction")

def caption_correction(input):
    global llm_output

    # Création de la barre de progression
    progress = st.progress(0)
    status_message = st.empty()

    # Étape 1: Retrieve
    status_message.text("Retrieve similar captions...")
    context = retrieve(input)
    progress.progress(33)  # Mise à jour de la barre de progression

    # Étape 2: Generate
    status_message.text("""Retrieve similar captions ✅
                        Generating the correction...
                        """)
    response = generate(input, context)
    progress.progress(66)  # Mise à jour de la barre de progression

    # Étape 3: Validate
    status_message.text("""Retrieve similar captions ✅
                        Generating the correction ✅
                        Validating the correction...
                        """)
    validation = validate(response['answer'].content)
    progress.progress(100)  # Mise à jour de la barre de progression

    # st.info(response['answer'].content)
    llm_output = validation.content
    status_message.empty()  # Supprimer le message après que tout soit terminé

    st.code(body=llm_output, language=None, wrap_lines=True)


with st.form("myform"):
    input = st.text_area("Enter caption", "")
    st.write(f"You wrote {len(input)} characters.")

    submitted = st.form_submit_button("Submit")
    
with st.container(border=True):
    if submitted:
        caption_correction(input)
        # st_copy_to_clipboard(llm_output)
        # st.info(llm_output)
        diff_html = highlight_differences(input, llm_output)

        st.write("🚨 **Pour copier la correction, utilisez le bouton en haut à droite de la zone de texte ci-dessus (qui apparaît lorsque vous la survolez avec la souris).**")

        with st.expander("Voir les différences"):
            components.html(diff_html, height=400, scrolling=True)