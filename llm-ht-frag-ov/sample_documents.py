import os
import logging
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.rankers import TransformersSimilarityRanker
from fastrag.generators.openvino import OpenVINOGenerator

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s') 

document_collection = [{'id': '11457596',
  'text': 'Quest", the "Ultima" series, "EverQuest", the "Warcraft" series, and the "Elder Scrolls" series of games as well as video games set in Middle-earth itself. Research also suggests that some consumers of fantasy games derive their motivation from trying to create an epic fantasy narrative which is influenced by "The Lord of the Rings". In 1965, songwriter Donald Swann, who was best known for his collaboration with Michael Flanders as Flanders & Swann, set six poems from "The Lord of the Rings" and one from "The Adventures of Tom Bombadil" ("Errantry") to music. When Swann met with Tolkien to play the',
  'title': 'The Lord of the Rings'},
 {'id': '11457582',
  'text': 'helped "The Lord of the Rings" become immensely popular in the United States in the 1960s. The book has remained so ever since, ranking as one of the most popular works of fiction of the twentieth century, judged by both sales and reader surveys. In the 2003 "Big Read" survey conducted in Britain by the BBC, "The Lord of the Rings" was found to be the "Nation\'s best-loved book". In similar 2004 polls both Germany and Australia also found "The Lord of the Rings" to be their favourite book. In a 1999 poll of Amazon.com customers, "The Lord of the',
  'title': 'The Lord of the Rings'},
 {'id': '11457540',
  'text': 'of Tolkien\'s works is such that the use of the words "Tolkienian" and "Tolkienesque" has been recorded in the "Oxford English Dictionary". The enduring popularity of "The Lord of the Rings" has led to numerous references in popular culture, the founding of many societies by fans of Tolkien\'s works, and the publication of many books about Tolkien and his works. "The Lord of the Rings" has inspired, and continues to inspire, artwork, music, films and television, video games, board games, and subsequent literature. Award-winning adaptations of "The Lord of the Rings" have been made for radio, theatre, and film. In',
  'title': 'The Lord of the Rings'},
 {'id': '11457587',
  'text': 'has been read as fitting the model of Joseph Campbell\'s "monomyth". "The Lord of the Rings" has been adapted for film, radio and stage. The book has been adapted for radio four times. In 1955 and 1956, the BBC broadcast "The Lord of the Rings", a 13-part radio adaptation of the story. In the 1960s radio station WBAI produced a short radio adaptation. A 1979 dramatization of "The Lord of the Rings" was broadcast in the United States and subsequently issued on tape and CD. In 1981, the BBC broadcast "The Lord of the Rings", a new dramatization in 26',
  'title': 'The Lord of the Rings'},
 {'id': '11457592',
  'text': '"The Lord of the Rings", was released on the internet in May 2009 and has been covered in major media. "Born of Hope", written by Paula DiSante, directed by Kate Madison, and released in December 2009, is a fan film based upon the appendices of "The Lord of the Rings". In November 2017, Amazon acquired the global television rights to "The Lord of the Rings", committing to a multi-season television series. The series will not be a direct adaptation of the books, but will instead introduce new stories that are set before "The Fellowship of the Ring". Amazon said the',
  'title': 'The Lord of the Rings'},
 {'id': '7733817',
  'text': 'The Lord of the Rings Online The Lord of the Rings Online: Shadows of Angmar is a massive multiplayer online role-playing game (MMORPG) for Microsoft Windows and OS X set in a fantasy universe based upon J. R. R. Tolkien\'s Middle-earth writings, taking place during the time period of "The Lord of the Rings". It launched in North America, Australia, Japan, and Europe in 2007. Originally subscription-based, it is free-to-play, with a paid VIP subscription available that provides players various perks.  The game\'s environment is based on "The Lord of the Rings" and "The Hobbit". However, Turbine does not',
  'title': 'The Lord of the Rings Online'},
 {'id': '22198847',
  'text': 'of "The Lord of the Rings", including Ian McKellen, Andy Serkis, Hugo Weaving, Elijah Wood, Ian Holm, Christopher Lee, Cate Blanchett and Orlando Bloom who reprised their roles. Although the "Hobbit" films were even more commercially successful than "The Lord of the Rings", they received mixed reviews from critics. Numerous video games were released to supplement the film series. They include: "," Pinball, "", "", , "", "", "", "", "The Lord of the Rings Online", "", "", "", "Lego The Lord of the Rings", "Guardians of Middle-earth", "", and "".',
  'title': 'The Lord of the Rings (film series)'},
 {'id': '24071573',
  'text': 'Lord of the Rings (musical) The Lord of the Rings is the most prominent of several theatre adaptations of J. R. R. Tolkien\'s epic high fantasy novel of the same name, with music by A. R. Rahman, Christopher Nightingale and the band Värttinä, and book and lyrics by Matthew Warchus and Shaun McKenna. Set in the world of Middle-earth, "The Lord of the Rings" tells the tale of a humble hobbit who is asked to play the hero and undertake a treacherous mission to destroy an evil, magic ring without being seduced by its power. The show was first performed',
  'title': 'Lord of the Rings (musical)'},
 {'id': '11457536',
  'text': 'The Lord of the Rings The Lord of the Rings is an epic high fantasy novel written by English author and scholar J. R. R. Tolkien. The story began as a sequel to Tolkien\'s 1937 fantasy novel "The Hobbit", but eventually developed into a much larger work. Written in stages between 1937 and 1949, "The Lord of the Rings" is one of the best-selling novels ever written, with over 150 million copies sold. The title of the novel refers to the story\'s main antagonist, the Dark Lord Sauron, who had in an earlier age created the One Ring to rule',
  'title': 'The Lord of the Rings'},
 {'id': '13304003',
  'text': "The Lord of the Rings (disambiguation) The Lord of the Rings is a fantasy novel by J. R. R. Tolkien. The title refers to Sauron, the story's main antagonist. The Lord of the Rings may also refer to:",
  'title': 'The Lord of the Rings (disambiguation)'}]

store = InMemoryDocumentStore()

documents = [Document(id=item["id"], content=item["text"], meta={"title": item["title"]}) for item in document_collection]
total_documents=store.write_documents(documents)

logging.debug(f" Total number of documents: {total_documents}")

retriever = InMemoryBM25Retriever(document_store=store)
ranker = TransformersSimilarityRanker()

prompt_template = """
Given these documents, answer the question.
Documents:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}
Question: {{query}}
Answer:
"""
openvino_compressed_model_path = "Gunulhona/openvino-llama-3.1-8B_int8"

generator = OpenVINOGenerator(
    model="Gunulhona/openvino-llama-3.1-8B_int8",
    compressed_model_dir=openvino_compressed_model_path,
    device_openvino="AUTO:GPU.1,CPU",
    task="text-generation",
    generation_kwargs={
        "max_new_tokens": 100,
    }
)

pipe = Pipeline()

pipe.add_component("retriever", retriever)
pipe.add_component("ranker", ranker)
pipe.add_component("prompt_builder", PromptBuilder(template=prompt_template))
pipe.add_component("llm", generator)

pipe.connect("retriever.documents", "ranker.documents")
pipe.connect("ranker", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")
print("Pipeline created")

query = "Who is the main villan in Lord of the Rings?"
answer_result = pipe.run({
    "prompt_builder": {
        "query": query
    },
    "retriever": {
        "query": query
    },
    "ranker": {
        "query": query,
        "top_k": 1
    }
})

logging.debug(answer_result["llm"]["replies"][0])