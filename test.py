from sentence_transformers import SentenceTransformer, util
import torch
import json

embedder = SentenceTransformer('all-MiniLM-L6-v2')

with open("./database.json", 'r') as f:
  contents = json.load(f)
  corpus = list(contents.keys())

# Corpus with example sentences
# corpus = ['A man is eating food.',
#           'A man is eating a piece of bread.',
#           'The girl is carrying a baby.',
#           'A man is riding a horse.',
#           'A woman is playing violin.',
#           'Two men pushed carts through the woods.',
#           'A man is riding a white horse on an enclosed ground.',
#           'A monkey is playing drums.',
#           'A cheetah is running behind its prey.'
#           ]
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Query sentences:
queries = ['Does geo-tagging photos improve local rankings', 'Who is whitespark']


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
print("\n\n======================\n")
print("Pre-defined Questions : \n")
for c in corpus:
   print(c, "\n")

top_k = min(1, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n")
    print("Converted query from speech with OpenAI Whisper : ", query)
    print("\nTop 1 most similar sentences in questions : ")

    top_score = -1
    top_score_question = ''
    for score, idx in zip(top_results[0], top_results[1]):
        if top_score < score:
           top_score_question = corpus[idx]
        print(corpus[idx], "(Score: {:.4f})".format(score))

    print(contents[top_score_question])
    """
    # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
    hits = hits[0]      #Get the hits for the first query
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
    """