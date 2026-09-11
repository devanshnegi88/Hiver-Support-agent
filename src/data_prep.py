"""
Turns the raw Kaggle twcs.csv (flat tweet table) into reconstructed
(customer_message -> brand_reply) pairs for one brand.

The raw file has columns:
    tweet_id, author_id, inbound, created_at, text,
    response_tweet_id, in_response_to_tweet_id

`inbound=True`  -> tweet from a customer
`inbound=False` -> tweet from a brand account (author_id == brand handle)

We reconstruct threads by following in_response_to_tweet_id backwards, then
keep only (last customer message -> brand's reply) pairs where the brand is
BRAND_HANDLE. This is the "how this brand has historically resolved similar
issues" corpus used later for grounding.
"""
import re
import pandas as pd

from config import RAW_CSV, THREADS_PARQUET, BRAND_HANDLE


URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip URLs/handles/extra whitespace. Keep punctuation — it carries
    sentiment/urgency signal we want the classifier to see."""
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def load_raw(sample_n: int | None = None) -> pd.DataFrame:
    """Load the raw CSV. Use sample_n while iterating — the full file is
    ~3M rows and you don't need that to develop the pipeline."""
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found. Download twcs.csv from Kaggle "
            "(thoughtvector/customer-support-on-twitter) and place it in data/. "
            "See README.md."
        )
    df = pd.read_csv(RAW_CSV, dtype={"tweet_id": str, "in_response_to_tweet_id": str,
                                      "response_tweet_id": str, "author_id": str})
    if sample_n:
        # Sample by keeping whole threads' worth of rows near the front —
        # random row sampling would shred threads apart.
        df = df.head(sample_n)
    return df


def build_brand_pairs(df: pd.DataFrame, brand: str = BRAND_HANDLE) -> pd.DataFrame:
    """Return one row per (customer message -> brand's reply to it),
    for the given brand, with light cleaning applied.
    """
    df = df.set_index("tweet_id", drop=False)

    brand_replies = df[(df["inbound"] == False) & (df["author_id"] == brand)]
    if brand_replies.empty:
        raise ValueError(
            f"No rows found for author_id == '{brand}'. Check the handle "
            "matches exactly what's in the dataset (case-sensitive)."
        )

    pairs = []
    for _, reply_row in brand_replies.iterrows():
        parent_id = reply_row.get("in_response_to_tweet_id")
        if not isinstance(parent_id, str) or parent_id not in df.index:
            continue
        cust_row = df.loc[parent_id]
        if isinstance(cust_row, pd.DataFrame):  # duplicate index edge case
            cust_row = cust_row.iloc[0]
        if not bool(cust_row.get("inbound")):
            continue  # parent wasn't actually a customer message

        cust_text = clean_text(cust_row["text"])
        reply_text = clean_text(reply_row["text"])
        if len(cust_text) < 5 or len(reply_text) < 5:
            continue

        pairs.append({
            "customer_tweet_id": cust_row["tweet_id"],
            "brand_tweet_id": reply_row["tweet_id"],
            "customer_message": cust_text,
            "brand_reply": reply_text,
            "created_at": cust_row.get("created_at"),
        })

    out = pd.DataFrame(pairs).drop_duplicates(subset=["customer_tweet_id"])
    return out.reset_index(drop=True)


def main():
    print(f"Loading raw data and building pairs for brand={BRAND_HANDLE} ...")
    df = load_raw()
    pairs = build_brand_pairs(df, BRAND_HANDLE)
    THREADS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(THREADS_PARQUET, index=False)
    print(f"Wrote {len(pairs)} customer->brand pairs to {THREADS_PARQUET}")


if __name__ == "__main__":
    main()
