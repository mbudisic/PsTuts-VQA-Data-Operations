from typing import List
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms

from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

from pathlib import Path
import loader


def create_golden_dataset(
    docs: List[Document],
    testset_size,
    group_name: str = "",
    filename: str = "",
    generator_llm=ChatOpenAI(model="gpt-4.1"),
    embedding_model: Embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    ),
    kg: KnowledgeGraph | None = None,
):
    """
    Creates a synthetic golden dataset for RAG evaluation using the RAGAS framework.

    This function generates a knowledge graph from provided documents (if not supplied)
    and then creates synthetic questions with corresponding answers based on the content.
    The generated dataset is tailored for Photoshop tutorials with specific personas.

    Args:
        docs (List[Document]): List of LangChain Document objects containing content to build the knowledge graph.
        testset_size (int): Number of question-answer pairs to generate.
        group_name (str, optional): Name identifier for the dataset group, used in filename generation. Defaults to "".
        filename (str, optional): Custom filename for the knowledge graph. If provided, overrides group_name-based naming.
                                 Defaults to "".
        generator_llm (LLM, optional): LangChain LLM to use for generation tasks.
                                      Defaults to ChatOpenAI(model="gpt-4.1").
        embedding_model (Embeddings, optional): LangChain embedding model for semantic operations.
                                              Defaults to OpenAIEmbeddings(model="text-embedding-3-small").
        kg (KnowledgeGraph, optional): Pre-existing knowledge graph. If provided, skips knowledge graph creation.
                                      Defaults to None.

    Returns:
        ragas.testset.Testset: Generated synthetic dataset with questions and answers based on input documents.

    Note:
        The function creates a knowledge graph file in the current directory if one does not exist.
        It uses predefined Photoshop-related personas for generating contextually relevant questions.
    """

    wrapped_generator_llm = LangchainLLMWrapper(generator_llm)
    wrapped_embedding_model = LangchainEmbeddingsWrapper(embedding_model)

    root = Path(".")
    kg_filename = Path(
        filename
        if filename
        else f"kg_{group_name}.json" if group_name else "kg.json"
    )
    kg_path = root.joinpath(kg_filename)
    print(kg_path)

    if not kg:
        kg = KnowledgeGraph()
        for doc in docs:
            kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": doc.page_content,
                        "document_metadata": doc.metadata,
                    },
                )
            )
        print(f"Initial size {str(kg)}")
        transforms = default_transforms(
            documents=docs,
            llm=wrapped_generator_llm,
            embedding_model=wrapped_embedding_model,
        )
        apply_transforms(kg, transforms)
        print(f"After transformations size {str(kg)}")

    personas = [
        Persona(
            name="Beginner Photoshop User",
            role_description=(
                "Beginner Photoshop user, learning to complete "
                "simple tasks, use the tools in Photoshop "
                "and navigate the graphical user interface"
            ),
        ),
        Persona(
            name="Photoshop trainer",
            role_description=(
                "Experienced trainer in Photoshop. Looking to develop"
                "step-by-step guides for Photoshop beginners"
            ),
        ),
    ]

    generator = TestsetGenerator(
        llm=wrapped_generator_llm,
        embedding_model=wrapped_embedding_model,
        persona_list=personas,
        knowledge_graph=kg,
    )

    print(generator)

    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=wrapped_generator_llm), 0.5),
        (MultiHopAbstractQuerySynthesizer(llm=wrapped_generator_llm), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=wrapped_generator_llm), 0.25),
    ]

    #

    testset = generator.generate(
        testset_size=testset_size,
        batch_size=8,
        num_personas=len(personas),
        query_distribution=query_distribution,
    )

    return testset


# In[ ]:
