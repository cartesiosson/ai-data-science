from haystack.nodes import BM25Retriever
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import FARMReader

# Usar MongoDB como document store en tu pipeline
retriever = BM25Retriever(document_store=mongo_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
pipeline = ExtractiveQAPipeline(reader, retriever)

# Ejecutar una consulta
query = "¿Qué es MongoDB?"
result = pipeline.run(query=query)
print(result)