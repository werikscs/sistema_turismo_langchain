import os
from dotenv import load_dotenv

# Componentes do LangChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

# Componentes do Router
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

# --- 1. FUNÇÕES DE CONFIGURAÇÃO E INICIALIZAÇÃO ---

def setup_components():
    """
    Carrega variáveis de ambiente e inicializa os componentes principais (LLM, Embeddings, Pinecone).
    """
    print("--- Inicializando componentes... ---")
    load_dotenv()
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not all([groq_api_key, pinecone_api_key, index_name]):
        raise ValueError("Erro: Verifique as variáveis de ambiente.")

    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0, api_key=groq_api_key)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    
    print("--- Componentes inicializados com sucesso! ---")
    return llm, vectorstore

def create_chains_and_router(llm):
    """
    Cria as cadeias de destino, a cadeia padrão, e a cadeia de roteamento.
    """
    prompt_infos = {
        # ... (as outras cadeias continuam iguais) ...
        "info_local": {
            "template": "Você é um assistente de turismo. Responda à pergunta do usuário de forma clara, usando APENAS as informações do contexto. Se a resposta não estiver no contexto, diga 'Não encontrei essa informação.'.\nContexto: {context}\nPergunta: {input}\nResposta:",
            "description": "Ideal para responder perguntas sobre um local específico, como horário de funcionamento, localização ou detalhes de um ponto turístico."
        },
        "logistica": {
            "template": "Você é um especialista em logística de viagens. Responda à pergunta do usuário sobre transporte de forma direta, usando as informações do contexto.\nContexto: {context}\nPergunta: {input}\nInstruções de Transporte:",
            "description": "Use para responder perguntas sobre transporte, como chegar a um lugar, aeroportos e locomoção na cidade."
        },
        "roteiro_viagem": {
            "template": "Você é um agente de viagens experiente. Crie um roteiro de viagem personalizado com base na solicitação do usuário. Use o contexto para sugerir pontos turísticos.\nContexto: {context}\nSolicitação: {input}\nRoteiro Detalhado:",
            "description": "Perfeito para quando o usuário pede um roteiro ou itinerário de viagem para uma cidade por um ou mais dias."
        },
        "traducao": {
            "template": "Você é um guia de idiomas para viajantes. Responda à solicitação do usuário fornecendo frases úteis e suas traduções.\nSolicitação do Usuário: {input}\nFrases e Traduções:",
            "description": "Útil para fornecer traduções ou frases comuns em um idioma para viajantes."
        },
        # ***** NOVA CADEIA PADRÃO *****
        "conversa_geral": {
            "template": "Você é um assistente de viagens amigável. Responda à pergunta do usuário de forma conversacional.\nSe o usuário perguntar o que você faz, explique que você pode criar roteiros de viagem, dar informações sobre locais, ajudar com transporte e traduções.\n\nPergunta: {input}\nResposta:",
            "description": "Use para conversas gerais, saudações ou quando nenhuma outra opção se encaixa."
        }
    }

    destination_chains = {}
    for name, info in prompt_infos.items():
        input_vars = ['input', 'context'] if 'context' in info['template'] else ['input']
        prompt = PromptTemplate(template=info['template'], input_variables=input_vars)
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain
    
    # Define a cadeia 'conversa_geral' como a padrão
    default_chain = destination_chains['conversa_geral']

    destinations = [f"{name}: {info['description']}" for name, info in prompt_infos.items()]
    destinations_str = "\n".join(destinations)

    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(template=router_template, input_variables=["input"], output_parser=RouterOutputParser())

    # Adiciona o 'default_chain' ao criar o roteador
    router_chain = LLMRouterChain.from_llm(llm, router_prompt, default_chain=default_chain, verbose=False)

    return router_chain, destination_chains

# --- 2. FUNÇÃO DE PROCESSAMENTO DA CONSULTA ---

def process_query(query, vectorstore, router_chain, destination_chains):
    """
    Processa a consulta do usuário: busca contexto, roteia e executa a cadeia correta.
    """
    docs = vectorstore.similarity_search(query, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    # O roteador agora sempre retornará um destino, mesmo que seja a cadeia padrão
    route = router_chain.invoke({"input": query})
    destination_name = route['destination']
    
    print(f"DEBUG: Rota escolhida -> {destination_name}")

    if destination_name in destination_chains:
        destination_chain = destination_chains[destination_name]
        inputs = {"input": query}
        if "context" in destination_chain.prompt.input_variables:
            inputs["context"] = context
        
        result = destination_chain.invoke(inputs)
        return result['text']
    else:
        # Este 'else' agora é menos provável de ser atingido, mas é uma boa segurança
        return "Desculpe, ocorreu um erro ao processar sua pergunta."

# --- 3. BLOCO DE EXECUÇÃO PRINCIPAL ---

if __name__ == "__main__":
    llm, vectorstore = setup_components()
    router_chain, destination_chains = create_chains_and_router(llm)

    print("\n--- Assistente de Viagens ---")
    print("Olá! Como posso te ajudar a planejar sua viagem? (digite 'sair' para encerrar)")

    while True:
        try:
            query = input("\nVocê: ")
            if query.lower() in ["sair", "exit", "quit"]:
                print("Até a próxima!")
                break
            
            response = process_query(query, vectorstore, router_chain, destination_chains)
            print(f"\nAssistente: {response}")
        except Exception as e:
            print(f"\nOcorreu um erro: {e}")