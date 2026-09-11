"""
Grounding retrieval: given a new customer message, find historical
(customer_message -> brand_reply) pairs for this brand whose customer
message is most similar. Those precedent replies are what the reply
generator is told to ground its draft in.

Design choice: TF-IDF cosine similarity rather than a neural embedding
index. This is intentional — see DECISION_LOG.md: it's free, has zero
external dependency/latency, is fully inspectable (you can show *why* a
precedent was retrieved), and for short, jargon-heavy support tweets it's a
surprisingly strong baseline. The eval harness reports retrieval quality
directly so this choice is falsifiable, not just asserted.
"""
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import THREADS_PARQUET


@dataclass
class RetrievalHit:
    customer_message: str
    brand_reply: str
    similarity: float
    customer_tweet_id: str = ""


def _id_str(value) -> str:
    """Normalize tweet IDs so int 1054087 matches parquet string '1054087'."""
    if value is None:
        return ""
    return str(value).strip()


class PrecedentIndex:
    def __init__(self, pairs_df: pd.DataFrame):
        self.df = pairs_df.reset_index(drop=True)
        if "customer_tweet_id" in self.df.columns:
            self.df["customer_tweet_id"] = self.df["customer_tweet_id"].map(_id_str)
        self.vectorizer = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), stop_words="english"
        )
        self._matrix = self.vectorizer.fit_transform(self.df["customer_message"].fillna(""))

    @classmethod
    def load(cls) -> "PrecedentIndex":
        if not THREADS_PARQUET.exists():
            raise FileNotFoundError(
                f"{THREADS_PARQUET} not found — run `python src/data_prep.py` first."
            )
        return cls(pd.read_parquet(THREADS_PARQUET))

    def without_ids(self, ids) -> "PrecedentIndex":
        """Return a copy whose index does not contain these tweet IDs (eval anti-leak)."""
        exclude = {_id_str(i) for i in ids if i is not None and str(i) != ""}
        if not exclude or "customer_tweet_id" not in self.df.columns:
            return self
        mask = ~self.df["customer_tweet_id"].isin(exclude)
        return PrecedentIndex(self.df.loc[mask].reset_index(drop=True))

    def search(self, query: str, top_k: int = 3, exclude_id: str | None = None) -> list[RetrievalHit]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        order = sims.argsort()[::-1]
        skip = _id_str(exclude_id) if exclude_id is not None else ""

        hits = []
        for idx in order:
            row = self.df.iloc[idx]
            row_id = _id_str(row.get("customer_tweet_id", ""))
            if skip and row_id == skip:
                continue
            hits.append(RetrievalHit(
                customer_message=row["customer_message"],
                brand_reply=row["brand_reply"],
                similarity=float(sims[idx]),
                customer_tweet_id=row_id,
            ))
            if len(hits) >= top_k:
                break
        return hits
