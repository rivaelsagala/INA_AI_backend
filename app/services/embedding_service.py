import os
from typing import List, Dict, Any, Union
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from supabase import create_client, Client
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

supabase: Client = create_client(os.getenv("SUPABASE_URL", ""), os.getenv("SUPABASE_KEY", ""))

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-large"),
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.maiarouter.ai/v1")
)

def store_medical_documents_to_supabase(documents: List[Union[Dict[str, Any], Document]]):
    """
    Store medical documents ke Supabase vector database.
    """
    try:
        langchain_docs = []
        for doc in documents:
            if isinstance(doc, dict):
                langchain_docs.append(Document(
                    page_content=doc['page_content'],
                    metadata=doc.get('metadata', {})
                ))
            elif isinstance(doc, Document):
                langchain_docs.append(doc)
        
        logger.info(f"Menyimpan {len(langchain_docs)} documents ke Supabase...")
        
        vector_store = SupabaseVectorStore.from_documents(
            langchain_docs,
            embeddings,
            client=supabase,
            table_name=os.getenv("SUPABASE_TABLE_NAME", "documents"),
            query_name="match_documents" 
        )
        
        logger.info(f"Berhasil menyimpan {len(langchain_docs)} documents ke vector database")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error menyimpan ke Supabase: {str(e)}")
        raise

def get_medical_data_statistics() -> Dict[str, Any]:
    """
    Mendapatkan statistik dari medical data di vector database.
    """
    try:
        response = supabase.table(os.getenv("SUPABASE_TABLE_NAME", "documents")).select("*", count="exact").execute()
        total_docs = response.count if hasattr(response, 'count') else 0
        
        medical_docs = supabase.table(os.getenv("SUPABASE_TABLE_NAME", "documents"))\
            .select("metadata")\
            .eq("metadata->>type", "medical_knowledge")\
            .execute()
        
        medical_count = len(medical_docs.data) if medical_docs.data else 0
        
        penyakit_list = []
        if medical_docs.data:
            for doc in medical_docs.data:
                metadata = doc.get('metadata', {})
                if isinstance(metadata, dict) and 'nama_penyakit' in metadata:
                    penyakit_list.append(metadata['nama_penyakit'])
        
        stats = {
            'total_documents': total_docs,
            'medical_documents': medical_count,
            'penyakit_count': len(set(penyakit_list)),
            'penyakit_list': sorted(set(penyakit_list))
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error mendapatkan statistik: {str(e)}")
        return {
            'total_documents': 0,
            'medical_documents': 0,
            'penyakit_count': 0,
            'penyakit_list': []
        }

def store_chunks_to_supabase(chunks):
    """
    DEPRECATED: Gunakan store_medical_documents_to_supabase untuk medical chatbot.
    """
    logger.warning("store_chunks_to_supabase is deprecated. Use store_medical_documents_to_supabase instead.")
    vector_store = SupabaseVectorStore.from_documents(
        chunks,
        embeddings,
        client=supabase,
        table_name=os.getenv("SUPABASE_TABLE_NAME", "documents"),
        query_name="match_documents" 
    )
    return vector_store