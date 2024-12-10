from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.base.response.schema import Response
from llama_index.core import StorageContext, load_index_from_storage
from utils import get_index_version

from typing import List


class GraphRagRetriever(BaseRetriever):
    """Custom retriever that performs both Vector search and Knowledge Graph search"""

    def __init__(
        self,
        vector_retriever,
        pg_retriever,
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._pg_retriever = pg_retriever
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle, db_mode: str = "OR") -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        if db_mode not in ("AND", "OR", "VECTOR_ONLY", "PG_ONLY"):
            raise ValueError(f"Invalid db mode: {db_mode}.")

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        pg_nodes = self._pg_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        pg_ids = {n.node.node_id for n in pg_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in pg_nodes})

        if db_mode == "AND":
            retrieve_ids = vector_ids.intersection(pg_ids)
        elif db_mode == "OR":
            retrieve_ids = vector_ids.union(pg_ids)
        elif db_mode == "VECTOR_ONLY":
            retrieve_ids = vector_ids
        elif db_mode == "PG_ONLY":
            retrieve_ids = pg_ids
        else:
            retrieve_ids = vector_ids.union(pg_ids)
        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes

class CustomQueryEngine(RetrieverQueryEngine):

    def __init__(self,
                 retriever,
                 response_synthesizer,
                 query_mode = 'ALL',
                 job_list = ["Engineer", "Analyst", "Manager"]):
        """
        Constructor to initialize the retriever and the job list.

        Args:
            retriever: The retriever instance to use.
            response_synthesizer: The response synthesizer instance to use.
            query_mode (str): The mode to choose query input for the retriever. Options are 'ALL', 'PROMPT_ONLY', and 'SPECIFIC_ONLY'.
            job_list (list of str): A list of job titles to search for in the nodes.
        """
        super().__init__(retriever=retriever, response_synthesizer=response_synthesizer)  # Initialize the parent class
        self.query_mode = query_mode
        self.job_list = job_list
    
    def query(self,
              prompt,
              specific_queries=[]):
        """
        Query the retriever with the prompt and specific queries.

        Args:
            prompt (str): The prompt to query the retriever.
            specific_queries (list of str): A list of specific queries to query the retriever.
        """
        # Retrieve nodes based on the query mode
        nodes = []        
        if self.query_mode == 'ALL' or self.query_mode == 'PROMPT_ONLY':
            nodes = self._retriever.retrieve(prompt)
        if self.query_mode == 'ALL' or self.query_mode == 'SPECIFIC_ONLY':
            for specific_query in specific_queries:
                nodes += self._retriever.retrieve(specific_query)
        
        matched_jobs = set()

        # Check for jobs in each node
        for job in self.job_list:
            for node in nodes:
                if job in node.get_content():
                    matched_jobs.add(job)
                    break  # Avoid duplicate checks for the same node

        # Add matched jobs to the query context
        job_context = f"Jobs allowed for assignment: {', '.join(matched_jobs)}\n\n"
        node_context = "\n".join(node.get_content() for node in nodes)
        full_query_context = prompt + "\n" + job_context + "\n" + node_context
        print("Full Query Context:")
        print(full_query_context)
        # how llamaindex would normally do it
        # https://github.com/run-llama/llama_index/blob/3c5ec51ddc0bdec0e3a39c27a52cfdb451d1eadd/llama-index-core/llama_index/core/query_engine/retriever_query_engine.py#L173
        # Pass the modified query to the LLM
        return super().query(full_query_context)
    
    def _postprocess_retrieved_nodes(self, nodes, query_bundle):
        # Considering using this as a place for node manipulation as well.
        for node in nodes:
            # Can do something here to manipulate the nodes
            node.text = node.get_content()
        return nodes


"""
Consider also doing a lot of the work in the response synthesizer
"""

class JobMatchingResponseSynthesizer(BaseSynthesizer):
    def synthesize(self, query_bundle, nodes, additional_context=None):
        """
        Custom synthesis of the response by including matched jobs from nodes.

        Args:
            query_bundle: Query object containing the query string.
            nodes: Retrieved nodes from the retriever.
            additional_context: Any additional context to include in the response.

        Returns:
            Response: The synthesized response object.
        """
        # Your list of jobs
        job_list = ["Engineer", "Analyst", "Manager"]
        matched_jobs = set()

        # Search for jobs in the retrieved nodes
        for job in job_list:
            for node in nodes:
                if job in node.get_content():
                    matched_jobs.add(job)
                    break  # Stop searching further nodes for this job

        # Prepare the custom context to include matched jobs
        job_context = f"Jobs found: {', '.join(matched_jobs)}\n\n"

        # Combine the additional context, job context, and retrieved nodes
        synthesized_content = job_context
        for node in nodes:
            synthesized_content += node.get_content() + "\n"

        # Return the response
        return Response(
            query=query_bundle.query_str,
            context=synthesized_content,
            nodes=nodes
        )

def load_cached_indexes(pg_store_dir, vector_store_dir, pg_version="latest", vector_version="latest"):
    print("Loading cached Property Graph Index...")
    pg_store_dir = get_index_version(pg_store_dir, version=pg_version)
    pg_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=pg_store_dir))

    print("Loading cached Vector Index...")
    vector_store_dir = get_index_version(vector_store_dir, version=vector_version)
    storage_context = StorageContext.from_defaults(persist_dir=vector_store_dir)
    vector_index = load_index_from_storage(storage_context)
    return pg_index, vector_index, pg_store_dir, vector_store_dir
