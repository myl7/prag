import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from MPCDenseRetrievalExactSearch import MPCDenseRetrievalExactSearch

import logging
import pathlib, os, numpy as np
import random, pickle

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


def benchmark_retriever(retriever, corpus, queries, qrels):
    start_time = time.time()
    results = retriever.retrieve(corpus, queries)
    end_time = time.time()
    try:
        # print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
        #### Evaluate your retrieval using NDCG@k, MAP@K ...
        # logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
        recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")
        hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")
        # print("Performance of DenseRetrievalExactSearch: {recall}, {precision}, {ndcg}, {map}, {mrr}, {recall_cap}, {hole}".format(recall=recall, precision=precision, ndcg=ndcg, map=_map, mrr=mrr, recall_cap=recall_cap, hole=hole))
        print(
            "Time taken: {:.2f} Recall@1: {}, Recall@5: {}".format(
                end_time - start_time,
                recall["Recall@1"],
                recall.get("Recall@5", np.NaN),
            )
        )
    except:
        print("Time taken: {:.2f}".format(end_time - start_time))
        return end_time - start_time, None, None, None, None, None, None, results
    return (
        end_time - start_time,
        recall,
        precision,
        ndcg,
        mrr,
        recall_cap,
        hole,
        results,
    )


# def setup(reduce_corpus_size: bool = True, sample_size: int = 500, proportion: float = 0.1):
def setup():
    """This is a good one-time function to run to start embedding the corpus and queries and then save them to file for later use."""
    from LocalDenseRetrievalExactSearch import DenseRetrievalExactSearch

    # Generate synthetic dataset instead of downloading
    num_docs = int(os.getenv("NUM_DOCS"))
    suffix = f"{num_docs}"

    num_queries = 1

    corpus = {
        str(i): {"title": f"Doc {i}", "text": f"This is document {i}"} for i in range(num_docs)
    }
    queries = {str(i): f"Query {i}" for i in range(num_queries)}
    qrels = {
        str(i): {str(x): 1 for x in range(num_docs) if x % num_queries == i}
        for i in range(num_queries)
    }  # Some random qrels

    print("Corpus size: {} on {} queries".format(len(corpus), len(queries)))
    # It's good to save these for reproducibility
    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    pickle.dump([corpus, qrels, queries], open(f"datasets/corpus_fiqa_{suffix}.pkl", "wb"))

    #### Dense Retrieval using SBERT (Sentence-BERT) ####
    print("Beginning embedding")
    # embedding_model = models.SentenceBERT("msmarco-distilbert-base-v3", device="cuda")
    embedding_model = None  # Use None as we use random embeddings

    model = DenseRetrievalExactSearch(
        embedding_model,
        batch_size=256,
        corpus_chunk_size=512 * 2**6,
        k_values=[1, 3, 5, 50],
    )

    model.preembed_queries(queries, save_path=f"datasets/query_embeddings_fiqa_{suffix}.pt")
    model.preemebed_corpus(corpus, save_path=f"datasets/corpus_embeddings_fiqa_{suffix}.pt")

    # model.load_preembeddings(
    #     f"datasets/corpus_embeddings_fiqa_{suffix}.pt", f"datasets/query_embeddings_fiqa_{suffix}.pt"
    # )

    print("Finished embedding, testing out retrieval")
    # Now we benchmark normal dense retrieval
    # retriever = EvaluateRetrieval(model, score_function="dot_score", k_values=[1, 3, 5])
    # results = retriever.retrieve(corpus, queries)

    # timetaken, recall, precision, ndcg, mrr, recall_cap, hole, results = benchmark_retriever(retriever, corpus, queries, qrels)

    # top_k = 16
    # query_id = 1824
    # query_id = list(queries.keys())[0]

    # query_id, ranking_scores = random.choice(list(results.items()))
    # ranking_scores = results[str(query_id)]
    # scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    # print("Query : %s\n" % queries[str(query_id)])

    # for rank in range(top_k):
    #     if rank < len(scores_sorted):
    #         doc_id = scores_sorted[rank][0]
    #         # Format: Rank x: ID [Title] Body
    #         print(
    #             "Rank %d: %s [%s] - %s\n"
    #             % (
    #                 rank + 1,
    #                 doc_id,
    #                 corpus[doc_id].get("title"),
    #                 corpus[doc_id].get("text"),
    #             )
    #         )


num_docs = os.getenv("NUM_DOCS")
top_k_env = os.getenv("TOP_K")
suffix = f"{num_docs}"

if os.path.exists(f"datasets/corpus_fiqa_{suffix}.pkl"):
    print("Loading existing dataset and embeddings")
else:
    print("No existing dataset found, generating now")
    setup()

corpus, qrels, queries = pickle.load(open(f"datasets/corpus_fiqa_{suffix}.pkl", "rb"))

# Now we benchmark MPC dense retrieval
print("Building BEIR model and loading pre-embeddings.")
model = MPCDenseRetrievalExactSearch(None, corpus_chunk_size=512 * 6)

# Load in premade embeddings
model.load_preembeddings(
    f"datasets/corpus_embeddings_fiqa_{suffix}.pt", f"datasets/query_embeddings_fiqa_{suffix}.pt"
)
print("Loaded embeddings")

# model._search(corpus, queries, top_k=5, score_function="dot_score");
# model._search_mulit_threaded(corpus, queries, top_k=5, score_function="cos_sim");


# print("Running basic retrieval")
# retriever = EvaluateRetrieval(model, score_function="dot_score",  k_values=[1,3,5,10])
# timetaken, recall, *the_rest =benchmark_retriever(retriever, corpus, queries, qrels)
# pickle.dump([timetaken, recall, *the_rest], open("beir_results_dot_score.pkl", "wb"))
# print(timetaken, recall)

# print("Running MPC distance with basic top-k")
# retriever = EvaluateRetrieval(model, score_function="mpc_dot_vanilla_topk",  k_values=[1,3,5, 10])
# timetaken, recall, *the_rest =benchmark_retriever(retriever, corpus, queries, qrels)
# pickle.dump([timetaken, recall, *the_rest], open("beir_results_dot_score.pkl", "wb"))

k = int(os.getenv("TOP_K"))

print("Running MPC distance with MPC top-k")
retriever = EvaluateRetrieval(model, score_function="mpc_dot_topk", k_values=[k])
timetaken, recall, *the_rest = benchmark_retriever(retriever, corpus, queries, qrels)
print(timetaken, recall)
# pickle.dump([timetaken, recall, *the_rest], open("beir_results_mpc_dot_topk.pkl", "wb"))

# print("Now we loop through and benchmark everything, saving the results to beir_results.pkl")
# results = {}
# for score_function in ["cos_sim", "dot_score", "mpc_dot_vanilla_topk", "mpc_cos_vanilla_topk", "mpc_cos2_vanilla_topk", "mpc_eucld_vanilla_topk", "mpc_dot_topk", "mpc_cos_topk", "mpc_cos2_topk", "mpc_eucld_topk"]:
#     retriever = EvaluateRetrieval(model, score_function="cos_sim",  k_values=[1,3,5,10])
#     timetaken, recall, *the_rest = benchmark_retriever(retriever, corpus, queries, qrels)
#     results[score_function] = [timetaken, recall]

#     pickle.dump(results, open("beir_results.pkl", "wb"))


# from beir.retrieval.search.dense import DenseRetrievalExactSearch
