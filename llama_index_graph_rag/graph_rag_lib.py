from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import CustomQueryEngine


from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.base.response.schema import Response

from typing import List




class GraphRagRetriever(BaseRetriever):
    """Custom retriever that performs both Vector search and Knowledge Graph search"""

    def __init__(
        self,
        vector_retriever,
        pg_retriever,
        db_mode="OR",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._pg_retriever = pg_retriever
        self._db_mode = db_mode
        if self._db_mode not in ("AND", "OR", "VECTOR_ONLY", "PG_ONLY"):
            raise ValueError(f"Invalid db mode: {self._db_mode}.")
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes from both Vector and Property Graph retrievers.
        
        Args:
            query_bundle (QueryBundle): The query bundle to retrieve nodes.
            db_mode (str): The mode to retrieve nodes. Options are 'AND', 'OR', 'VECTOR_ONLY', and 'PG_ONLY'.
        
        Returns:
            List[NodeWithScore]: The list of nodes with scores.
        """

        

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        pg_nodes = self._pg_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        pg_ids = {n.node.node_id for n in pg_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in pg_nodes})

        if self._db_mode == "AND":
            retrieve_ids = vector_ids.intersection(pg_ids)
        elif self._db_mode == "OR":
            retrieve_ids = vector_ids.union(pg_ids)
        elif self._db_mode == "VECTOR_ONLY":
            retrieve_ids = vector_ids
        elif self._db_mode == "PG_ONLY":
            retrieve_ids = pg_ids
        else:
            retrieve_ids = vector_ids.union(pg_ids)
        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes

class PlainQueryEngine(CustomQueryEngine):
    
    def custom_query(self, query):
        """
        Query the retriever with the prompt and specific queries.

        Args:
            query (str): The prompt to query the llm.
        """
        # TODO(adrianmarkelov): Re sort nodes so that vector and pg nodes are separated
        # TODO(adrianmarkelov): for job search don't search the Graph Nodes!!!!!!!!!!!!!!!!!

        # Add matched jobs to the query context
        print("Full Query Context:")
        print(query)
        # how llamaindex would normally do it
        # https://github.com/run-llama/llama_index/blob/3c5ec51ddc0bdec0e3a39c27a52cfdb451d1eadd/llama-index-core/llama_index/core/query_engine/retriever_query_engine.py#L173
        # Pass the modified query to the LLM
        return self.llm.complete(query)


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
