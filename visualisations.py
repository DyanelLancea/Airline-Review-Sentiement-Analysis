import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from sentiment_analysis import load_afinn_lexicon
from nltk.tokenize import sent_tokenize

#Read csv from our github
df = pd.read_csv("data/airlines_review_analysis.csv")

'''
Evaluation Functions
Model Evaulation: We shall be evaluating our sentiment analysis based on 4 metrics Accuracy, 
Precision, Recall and F1 Score.
'''
#Function to convert our analysis scores from numbers to postive or negative
def score_postive_negative(score):
    if score < -0:
        return "Negative"
    elif score >= 0:
        return "Postive"
    
#Function to chnage airline review ratings to postive or negative
def rating_postive_negative(rating):
    if rating <= 5:
        return "Negative"
    elif rating >= 6:
        return "Postive"

#Funtion to check if your analysis is True or False 
def determine_correct_prediction(actual, pred):
    if actual == pred:
        return True
    else:
        return False
    
#Function to calculate accuracy
def calculate_aaccuracy(whether_correct_pred):
    return sum(whether_correct_pred)/len(whether_correct_pred)

#Funtion to calculate confusion matrix
def calculate_confusion_matrix(actual, pred):
    if actual == 'Postive' and pred == 'Postive':
        return 'TP'
    elif actual == 'Negative' and pred == 'Postive':
        return 'FP'
    elif actual == 'Negative' and pred == 'Negative':
        return 'TN'
    elif actual == 'Postive' and pred == 'Negative':
        return 'FN'

#Function to calculate precision
def calculate_precision(confusion_matric):
    return confusion_matric.count('TP')/(confusion_matric.count('TP')+confusion_matric.count('FP'))

#Funtion to calculate Recall
def calculate_recall(confusion_matric):
    return confusion_matric.count('TP')/(confusion_matric.count('TP')+confusion_matric.count('FN'))

#Applying the all the functions to our datafraame
df['score_postive_negative'] = df.apply(lambda x: score_postive_negative(x['Normalized Sentiment Score']), axis=1)
df['rating_postive_negative'] = df.apply(lambda x: rating_postive_negative(x['Rating']), axis=1)
df['whether_correct_pred'] = df.apply(lambda x: determine_correct_prediction(x['rating_postive_negative'],x["score_postive_negative"]), axis=1)
df['confusion_matrix'] = df.apply(lambda x: calculate_confusion_matrix(x['rating_postive_negative'],x["score_postive_negative"]), axis=1)
recall = calculate_recall(df['confusion_matrix'].tolist())
precision = calculate_precision(df['confusion_matrix'].tolist())

#Funtion to Calculate F1 Score
def calculate_F1score(precision,recall):
    return 2 * (precision * recall) / (precision + recall)

# Compute your metrics (as in your existing code)
accuracy = calculate_aaccuracy(df['whether_correct_pred'])
f1_score_val = calculate_F1score(precision, recall)

# Create a summary DataFrame
results_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Value": [accuracy, precision, recall, f1_score_val]
})

# Save to CSV
results_df.to_csv("visualisations/Sentiment_Analysis_Metrics.csv", index=False)

'''
Visualisations
Using a mix of matplotlib and seaborn we created visualisation to show insights into our sentiment data.
'''
# Create a dataframe of the official ranking for us to use later
skytrax_top10 = [
    "Qatar Airways",
    "Singapore Airlines",
    "Cathay Pacific Airlines",
    "Emirates",
    "All Nippon Airways",
    "Turkish Airlines",
    "Korean Air",
    "Air France",
    "Japan Airlines",
    "Hainan Airlines",
]
official = pd.DataFrame({"Airlines": skytrax_top10, "Official Rank": range(1, 11)})



avg = (
    df
      .groupby("Airlines")["Normalized Sentiment Score"]
      .mean()
      .reset_index()
      .rename(columns={"Normalized Sentiment Score": "Avg Sentiment"})
)

m = official.merge(avg, on="Airlines", how="left")
m["Sentiment Rank"] = (
    m["Avg Sentiment"].rank(ascending=False, method="dense").astype(int)
)

# Order Best sentiment (rank 1) at the top
m = m.sort_values(["Sentiment Rank", "Official Rank"], ascending=[True, True]).reset_index(drop=True)

#----Create Graph Official vs Sentiment Ranking-------

fig, ax = plt.subplots(figsize=(11, 6))
y = np.arange(len(m))
bar_h = 0.35


# Bars (no x-axis inversion; left=0, right=10)
ax.barh(y - bar_h/2, m["Official Rank"], height=bar_h, label="Official Rank")
ax.barh(y + bar_h/2, m["Sentiment Rank"], height=bar_h, label="Sentiment Rank")

# Y ticks: airline names
ax.set_yticks(y)
ax.set_yticklabels(m["Airlines"])

# Axis + grid t
ax.set_xlabel("Rank (Lower is Better)")
ax.set_xlim(0, 10.5)
ax.grid(axis="x", linestyle="--", alpha=0.5)

# Title + legend
ax.set_title("Official Ranking vs Sentiment-Derived Ranking", 
             fontsize=13, fontweight="bold", pad=10)
ax.legend(loc="upper right", frameon=True)

# print value labels on bar
def label_bar(xval, yval):
    ax.text(max(0.1, xval - 0.2), yval, f"{int(xval)}",
            va="center", ha="right", color="black", fontsize=9)

for i, row in m.iterrows():
    label_bar(row["Official Rank"], i - bar_h/2)
    label_bar(row["Sentiment Rank"], i + bar_h/2)

# Save graph to folder
plt.tight_layout()
plt.savefig("visualisations/Official_vs_Sentiment.png", dpi=150)


# --- Diverging Bar — Rank Difference (Sentiment Rank – Official Rank) ---

# Caluclate Rank Difference
m["Rank Diff"] = m["Official Rank"] - m["Sentiment Rank"]

# Sort by Rank Diff for the diverging bar
m = m.sort_values("Rank Diff", ascending=True)

# Plot Graph
plt.figure(figsize=(10, 6))
y_pos = np.arange(len(m))
plt.barh(m["Airlines"], m["Rank Diff"])

# Reference line at 0 (alignment point)
plt.axvline(0, linestyle="--")

# Labels & title
plt.title("Rank Difference: Sentiment Rank – Official Rank (Skytrax 2025)")
plt.xlabel("Positive = Sentiment ranks better than Official")
plt.ylabel("Airline")

# Annotate values at bar ends
for idx, val in enumerate(m["Rank Diff"].values):
    x = val + (0.15 if val >= 0 else -0.15)
    ha = "left" if val >= 0 else "right"
    plt.text(x, idx, f"{val:+d}", va="center", ha=ha)

#Save graph into folder
plt.tight_layout()
plt.savefig("visualisations/diverging_rank_difference.png", dpi=150)

# --- Scatterplot — Official Rank vs Average Sentiment ---

# Compute correlation between Official Rank and Average Sentiment
corr, pval = pearsonr(m["Official Rank"], m["Avg Sentiment"])

# --- Create Scatterplot ---
plt.figure(figsize=(8, 5))
plt.scatter(
    m["Official Rank"],
    m["Avg Sentiment"],
    color="teal",
    s=100,
    alpha=0.8,
    edgecolor="black"
)

# Add regression trendline
z = np.polyfit(m["Official Rank"], m["Avg Sentiment"], 1)
p = np.poly1d(z)
plt.plot(m["Official Rank"], p(m["Official Rank"]), color="gray", linestyle="--", linewidth=1.5)

# Annotate airline names
for i, row in m.iterrows():
    plt.text(
        row["Official Rank"] + 0.1,
        row["Avg Sentiment"],
        row["Airlines"],
        fontsize=9,
        va="center"
    )

# Invert X-axis so Rank 1 (best) appears on left
plt.gca().invert_xaxis()

# Labels and title
plt.title("Official Skytrax Rank vs Average Passenger Sentiment (2025)", fontsize=14, fontweight="bold")
plt.xlabel("Official Skytrax Rank (1 = Best)", fontsize=12)
plt.ylabel("Average Normalized Sentiment", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)

# Display correlation in subtitle
plt.text(
    9.8, m["Avg Sentiment"].min(),
    f"Correlation (r) = {corr:.2f}",
    fontsize=10, color="dimgray"
)

plt.tight_layout()
plt.savefig("visualisations/scatter_official_rank_vs_sentiment.png", dpi=150)

# --- Boxplot — Sentiment Consistency by Airline (Skytrax Top 10) ---

df["Date Published"] = pd.to_datetime(df["Date Published"], errors="coerce")

# Order airlines by median sentiment (high to low)
order = (
    df.groupby("Airlines")["Normalized Sentiment Score"]
       .median()
       .sort_values(ascending=False)
       .index.tolist()
)

# Prepare data in order
data_by_airline = [df.loc[df["Airlines"] == a, "Normalized Sentiment Score"].values for a in order]

# --- Create Boxplot ---
plt.figure(figsize=(10, 6))
box = plt.boxplot(
    data_by_airline,
    vert=False,                   
    tick_labels=order,
    showmeans=True,
    patch_artist=True
)

# Set colours for easier visualisation
for patch in box['boxes']:
    patch.set_facecolor("#A7C7E7")
    patch.set_alpha(0.7)
for mean in box['means']:
    mean.set(marker="^", color="green", markersize=8)


plt.title("Sentiment Consistency by Airline — Boxplot (Skytrax Top 10)", fontsize=14, fontweight="bold")
plt.xlabel("Normalized Sentiment Score", fontsize=12)
plt.ylabel("Airline", fontsize=12)
plt.grid(axis="x", linestyle="--", alpha=0.6)

# Save and display
plt.tight_layout()
plt.savefig("visualisations/boxplot_sentiment_consistency.png", dpi=150)

# --- Monthly Sentiment: Spotlight One Sinagpore Airline vs Others ---

focus_airline = "Singapore Airlines"   
start = "2020-01-01"

mask = (df["Airlines"].isin(skytrax_top10)) & df["Date Published"].notna()
mask &= df["Date Published"] >= pd.to_datetime(start)
dft = df.loc[mask, ["Date Published","Airlines","Normalized Sentiment Score"]].dropna()

dft["Month"] = dft["Date Published"].dt.to_period("M").dt.to_timestamp()
m = (dft.groupby(["Month","Airlines"])["Normalized Sentiment Score"]
        .mean().reset_index())
wide = m.pivot(index="Month", columns="Airlines", values="Normalized Sentiment Score").sort_index()
wide_sm = wide.rolling(window=3, min_periods=1).mean()

plt.figure(figsize=(12, 5))

# Plot others airlines in light grey
for col in wide_sm.columns:
    if col == focus_airline: 
        continue
    plt.plot(wide_sm.index, wide_sm[col], linewidth=1.2, alpha=0.3)

# Spotlight the chosen airline
if focus_airline in wide_sm.columns:
    plt.plot(wide_sm.index, wide_sm[focus_airline], linewidth=2.8)

plt.title(f"Monthly Sentiment Trend — Spotlight: {focus_airline} (3-mo avg)", fontsize=14, fontweight="bold")
plt.xlabel("Month"); plt.ylabel("Average Normalized Sentiment")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualisations/trend_spotlight.png", dpi=150)

# --- Keyword Maps: TF-IDF Top Terms per Airline (Positive vs Negative) ---

# Simple tokenizer for TF-IDF
def tokenizer(s):
    return [w for w in s.split() if len(w) > 2]

# Vectorizer with domain stopwords
domain_stop = {
    'flight','flights','airline','airlines','plane','travel',
    'singapore','qatar','emirates','air','airways','japan','korean','turkish','france','hainan',
    'nippon','cathay','pacific','review','reviews'
}
vectorizer = TfidfVectorizer(tokenizer=tokenizer, token_pattern=None, stop_words='english', min_df=3)

rows = []
for airline in skytrax_top10:
    sub = df[df["Airlines"] == airline]
    for label, mask in [("Positive", sub["Normalized Sentiment Score"] > 0),
                        ("Negative", sub["Normalized Sentiment Score"] < 0)]:
        docs = sub.loc[mask, "Text Content Lower Case"].tolist()
        # Filter out domain words manually
        docs = [" ".join([w for w in d.split() if w not in domain_stop]) for d in docs]
        if len(docs) < 5:  # not enough text to compute TF-IDF robustly
            rows.append([airline, label, "(insufficient data)"])
            continue
        X = vectorizer.fit_transform(docs)
        terms = vectorizer.get_feature_names_out()
        # Average TF-IDF per term across docs
        avg_scores = X.mean(axis=0).A1
        top_idx = avg_scores.argsort()[-12:][::-1]  # top 12 terms
        top_terms = [terms[i] for i in top_idx]
        rows.append([airline, label, ", ".join(top_terms)])

# Present as a tidy table
tfidf_table = pd.DataFrame(rows, columns=["Airline", "Polarity", "Top Terms"]).sort_values(["Airline","Polarity"])

# Save to CSV for your appendix
tfidf_table.to_csv("visualisations/keyword_map_tfidf_top10.csv", index=False)

# --- Negative Postive Word Cloud ---

# Initialize lists to hold positive and negative words
positive_words = []
negative_words = []

# Load dictionary 
afinn_path = "data/AFINN-en-165.txt"
afinn_dict = load_afinn_lexicon(afinn_path)

# Iterate through each sentence in the DataFrame
for sentence in df['Text Content Lower Case']:
    words = sentence.split()  # Split sentence into words
    
    for word in words:
        score = afinn_dict.get(word, 0)  # Get the sentiment score of the word
        
        if score > 0:
            positive_words.append(word)  # Positive word
        elif score < 0:
            negative_words.append(word)  # Negative word

# Join the positive and negative words into single strings
positive_text = ' '.join(positive_words)
negative_text = ' '.join(negative_words)

# Create subplots for both positive and negative word clouds
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

# Create the positive word cloud
positive_wc = WordCloud(width=400, height=400, background_color="white").generate(positive_text)
axes[0].imshow(positive_wc, interpolation='bilinear')
axes[0].set_title("Positive Words")
axes[0].axis('off')

# Create the negative word cloud
negative_wc = WordCloud(width=400, height=400, background_color="white").generate(negative_text)
axes[1].imshow(negative_wc, interpolation='bilinear')
axes[1].set_title("Negative Words")
axes[1].axis('off')

plt.tight_layout()
plt.savefig("visualisations/negative_postive_wordcloud.png", dpi=150)