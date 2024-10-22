from pymongo import MongoClient
from haystack.document_stores import BaseDocumentStore
from haystack import Document


class MongoDBDocumentStore(BaseDocumentStore):
    def __init__(self, url="mongodb://localhost:27017", database="haystack", collection="documents"):
        self.client = MongoClient(url)
        self.db = self.client[database]
        self.collection = self.db[collection]

    def write_documents(self, documents):
        # Convertir documentos de Haystack a formato dict para MongoDB
        docs_to_insert = [doc.to_dict() for doc in documents]
        self.collection.insert_many(docs_to_insert)

    def get_all_documents(self):
        # Obtener todos los documentos de la colección
        docs_from_db = self.collection.find({})
        return [Document.from_dict(doc) for doc in docs_from_db]

    def query(self, query_text):
        # Realizar una búsqueda básica en MongoDB
        results = self.collection.find({"content": {"$regex": query_text, "$options": "i"}})
        return [Document.from_dict(result) for result in results]
    
# Crear una instancia de tu MongoDBDocumentStore
mongo_store = MongoDBDocumentStore()

# Escribir documentos en MongoDB
documents = [
    Document(content="Python es un lenguaje de programación popular."),
    Document(content="MongoDB es una base de datos NoSQL flexible."),
]
mongo_store.write_documents(documents)

# Recuperar todos los documentos
#all_docs = mongo_store.get_all_documents()
#print(all_docs)

# Buscar documentos que coincidan con una consulta
#query_results = mongo_store.query(query_text="Python")
#print(query_results)