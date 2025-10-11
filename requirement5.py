import re
from math import inf
import argparse
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

'''Load functions from sentiment_analysis.py'''

def load_afinn_lexicon(afinn_path: str) -> dict:
    d = {}
    with open(afinn_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            word, score = line.rsplit("\t", 1)
            d[word] = int(score)
    return d

def calculate_sentiment_score(text: str, afinn_dict: dict) -> int:
    if not isinstance(text, str):
        return 0
    total = 0
    for w in text.lower().split():
        total += afinn_dict.get(w, 0)
    return total

'''End'''

'''Functions for requirement 5'''
def kadane_with_indices(arr):
    '''Kadane's algorithm to find max subarray sum with start/end indices.'''
    if not arr:
        return 0, -1, -1
    best_sum = -inf
    cur_sum = 0
    best_l = best_r = cur_l = 0
    for i, x in enumerate(arr):
        if cur_sum <= 0:
            cur_sum = x
            cur_l = i
        else:
            cur_sum += x
        if cur_sum > best_sum:
            best_sum, best_l, best_r = cur_sum, cur_l, i
    return best_sum, best_l, best_r

def best_paragraph_span(text: str, afinn_dict: dict):
    '''Find the best positive and negative paragraph spans in the text.
    Tokenize to sentences, score each sentence, then:
      - most positive span = Kadane(sent_scores)
      - most negative span = Kadane(-sent_scores)  (flip sign back)
    Returns two dicts: (pos_span, neg_span) each with:
      {text, score, start, end, sentences}
    '''
    sents = [s.strip() for s in sent_tokenize(text or "") if s.strip()]
    if not sents:
        return None, None

    sent_scores = [calculate_sentiment_score(s, afinn_dict) for s in sents]

    # positive
    pos_sum, pos_l, pos_r = kadane_with_indices(sent_scores)

    # negative 
    neg_sum_neg, neg_l, neg_r = kadane_with_indices([-x for x in sent_scores])
    neg_sum = -neg_sum_neg

    pos_span = {
        "text": " ".join(sents[pos_l:pos_r + 1]),
        "score": float(pos_sum),
        "start": int(pos_l),
        "end": int(pos_r),
        "sentences": sents[pos_l:pos_r + 1],
    }
    neg_span = {
        "text": " ".join(sents[neg_l:neg_r + 1]),
        "score": float(neg_sum),
        "start": int(neg_l),
        "end": int(neg_r),
        "sentences": sents[neg_l:neg_r + 1],
    }
    return pos_span, neg_span

MONTHS = ['January','February','March','April','May','June',
          'July','August','September','October','November','December']
MONTH_ORDER = {m: i for i, m in enumerate(MONTHS)}

def safe_sheet(name, maxlen=30):
    '''Excel-safe sheet name (<=30 chars, no / \\ : ? * [ ])'''
    name = re.sub(r'[:\\/?*\[\]]', ' ', str(name)).strip()
    name = re.sub(r'\s+', ' ', name)
    return name[:maxlen]

def best_overall_airline(df_group, text_col, afinn_dict):
    '''Iterate rows in a group; keep the single best + and - span across reviews.'''
    best_pos, best_neg = {"score": -inf}, {"score": inf}
    pos_meta = neg_meta = None

    for idx, r in df_group.iterrows():
        pos, neg = best_paragraph_span(r[text_col], afinn_dict)
        if pos and pos["score"] > best_pos["score"]:
            best_pos = pos
            pos_meta = {
                "row": idx,
                "name": r.get("Name"),
                "rating": r.get("Rating"),
                "date": r.get("Date Published"),
                "full_review": r.get(text_col),
            }
        if neg and neg["score"] < best_neg["score"]:
            best_neg = neg
            neg_meta = {
                "row": idx,
                "name": r.get("Name"),
                "rating": r.get("Rating"),
                "date": r.get("Date Published"),
                "full_review": r.get(text_col),
            }
    return best_pos, best_neg, pos_meta, neg_meta

def pack_row(common, best_pos, best_neg, pos_meta, neg_meta):
    '''Compose a single output row with Pos_/Neg_ blocks.'''
    return {
        **common,
        # Most Positive
        "Pos_Text Content": None if pos_meta is None else pos_meta["full_review"],
        "Pos_Most Positive Segment": None if pos_meta is None else best_pos["text"],
        "Pos_Score": None if pos_meta is None else best_pos["score"],
        "Pos_Start of Sentence": None if pos_meta is None else best_pos["start"],
        "Pos_End of Sentence": None if pos_meta is None else best_pos["end"],
        "Pos_Name": None if pos_meta is None else pos_meta["name"],
        "Pos_Rating": None if pos_meta is None else pos_meta["rating"],
        "Pos_Published Date": None if pos_meta is None else pos_meta["date"],
        # Most Negative
        "Neg_Text Content": None if neg_meta is None else neg_meta["full_review"],
        "Neg_Most Negative Segment": None if neg_meta is None else best_neg["text"],
        "Neg_Score": None if neg_meta is None else best_neg["score"],
        "Neg_Start of Sentence": None if neg_meta is None else best_neg["start"],
        "Neg_End of Sentence": None if neg_meta is None else best_neg["end"],
        "Neg_Name": None if neg_meta is None else neg_meta["name"],
        "Neg_Rating": None if neg_meta is None else neg_meta["rating"],
        "Neg_Published Date": None if neg_meta is None else neg_meta["date"],
    }

def rename_block(df, prefix):
    '''Rename Pos_/Neg_ columns to presentation names.'''
    mapping = {
        f"{prefix}_Text Content": "Text Content",
        f"{prefix}_Most {'Positive' if prefix=='Pos' else 'Negative'} Segment":
            f"Most {'Positive' if prefix=='Pos' else 'Negative'} Segment",
        f"{prefix}_Score": "Score",
        f"{prefix}_Start of Sentence": "Start of Sentence",
        f"{prefix}_End of Sentence": "End of Sentence",
        f"{prefix}_Name": "Name",
        f"{prefix}_Rating": "Rating",
        f"{prefix}_Published Date": "Date Published",
    }
    return df.rename(columns=mapping)

def build_overall(df, afinn_dict, text_col="Text Content", group_col="Airlines"):
    '''Build overall sentiment analysis results for each airline.
    Iterate groups (airlines), find most positive and negative spans across reviews.'''

    rows = []
    for airline, g in df.groupby(group_col):
        best_pos, best_neg, pos_meta, neg_meta = best_overall_airline(g, text_col, afinn_dict)
        rows.append(pack_row({"Airlines": airline}, best_pos, best_neg, pos_meta, neg_meta))
    out = pd.DataFrame(rows).sort_values("Airlines").reset_index(drop=True)
    pos = rename_block(out[["Airlines"] + [c for c in out.columns if c.startswith("Pos_")]], "Pos")
    neg = rename_block(out[["Airlines"] + [c for c in out.columns if c.startswith("Neg_")]], "Neg")
    return pos, neg

def build_year_month(df, afinn_dict, text_col="Text Content"):
    '''Build year-month sentiment analysis results for each airline.'''
    work = df.copy()
    work["Date Published"] = pd.to_datetime(work["Date Published"], errors="coerce")
    work["Year"] = work["Date Published"].dt.year
    work["Month Name"] = work["Date Published"].dt.month_name()
    work = work.dropna(subset=["Date Published"])

    rows = []
    for (airline, yr, mname), g in work.groupby(["Airlines", "Year", "Month Name"]):
        best_pos, best_neg, pos_meta, neg_meta = best_overall_airline(g, text_col, afinn_dict)
        rows.append(
            pack_row({"Airlines": airline, "Year": yr, "Month Name": mname},
                     best_pos, best_neg, pos_meta, neg_meta)
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["Airlines", "Year", "Month Name"],
            key=lambda col: col.map(MONTH_ORDER) if col.name == "Month Name" else col
        ).reset_index(drop=True)

    pos = rename_block(out[["Airlines","Year","Month Name"] + [c for c in out.columns if c.startswith("Pos_")]], "Pos")
    neg = rename_block(out[["Airlines","Year","Month Name"] + [c for c in out.columns if c.startswith("Neg_")]], "Neg")
    return pos, neg
'''End of Requirement 5 functions'''

def main():
    '''Main function to parse arguments, load data, perform analysis, and export results.'''
    parser = argparse.ArgumentParser(description="Arbitrary-length sentiment spans by airline.")
    parser.add_argument("--afinn", default="data/AFINN-en-165.txt", help="Path to AFINN-en-165 lexicon.")
    parser.add_argument(
        "--csv",
        default="https://raw.githubusercontent.com/DyanelLancea/Airline-Review-Sentiement-Analysis/refs/heads/master/data/airlines_review_cleaned.csv",
        help="Input CSV with columns: Airlines, Name, Rating, Date Published, Text Content.",
    )
    parser.add_argument("--text-col", default="Text Content", help="Column name for review text.")
    parser.add_argument("--out", default="airline_arbitrary_length.xlsx", help="Output Excel file path.")
    args = parser.parse_args()

    try:
        _ = sent_tokenize("test.")
    except LookupError:
        nltk.download("punkt")

    # load resources
    afinn_dict = load_afinn_lexicon(args.afinn)
    df = pd.read_csv(args.csv)

    df = df.dropna(subset=[args.text_col, "Airlines"]).copy()

    # overall tabs
    overall_pos, overall_neg = build_overall(df, afinn_dict, text_col=args.text_col, group_col="Airlines")

    # monthly tabs
    monthly_pos, monthly_neg = build_year_month(df, afinn_dict, text_col=args.text_col)

    # export
    with pd.ExcelWriter(args.out, engine="openpyxl", mode="w", datetime_format="yyyy-mm-dd") as w:
        overall_pos.to_excel(w, sheet_name=safe_sheet("Overall Most Positive"), index=False)
        overall_neg.to_excel(w, sheet_name=safe_sheet("Overall Most Negative"), index=False)
        monthly_pos.to_excel(w, sheet_name=safe_sheet("Monthly Positive"), index=False)
        monthly_neg.to_excel(w, sheet_name=safe_sheet("Monthly Negative"), index=False)

    print(f"Exported to {args.out}")

if __name__ == "__main__":
    main()