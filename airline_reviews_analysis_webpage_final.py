# %% [markdown]
# # 7. Web Application (Flask)

# %%
from flask import Flask, render_template, request
import pandas as pd
import requests
from io import StringIO
nltk.download('punkt_tab')

app = Flask(__name__, template_folder='templates')

# URL of the CSV file
#url = 'https://raw.githubusercontent.com/DyanelLancea/Airline-Review-Sentiement-Analysis/refs/heads/master/airlines_review.csv'

# Fetch and load the CSV data
#response = requests.get(url)
#csv_data = StringIO(response.text)
'''
To present segmentation, replace 'Text Content' with 'Text Content Segmented'
'''
 

df = pd.read_csv('airlines_review_analysis.csv')

from nltk.tokenize import sent_tokenize

# Apply AFINN sentiment analysis
df["sentiment_score"] = df["Text Content"].apply(
    lambda x: normalize_score(
        calculate_sentiment_score(str(x), afinn_dict),
        len(str(x).split())
    )
)

# Convert 'Date' column to datetime to facilitate month selection
df['Date Published'] = pd.to_datetime(df['Date Published'])

#list of months in order
order_of_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
# Extract available airlines and months for the dropdowns
airlines = df['Airlines'].unique().tolist()
months = sorted(df['Date Published'].dt.month_name().unique().tolist(), key=lambda x: order_of_months.index(x))



@app.route('/', methods=['GET', 'POST'])
def index():
    selected_airline = None
    selected_month = None
    t_reviews = []
    b_reviews = []
    positive_window = None
    negative_window = None
    arb_pos = None
    arb_neg = None
    top_airlines = []  # For top airlines across all months
    top_airlines_month = []  # For top airlines based on the selected month


    if request.method == 'POST':
        selected_airline = request.form.get('airline')
        selected_month = request.form.get('month')

        # ** If "All" is selected, show top airlines across all months **
        if selected_month == 'All':
            # Calculate top 3 airlines across all months (no filtering by month)
            avg_scores_all = df.groupby('Airlines')['Normalized Sentiment Score'].mean()
            top_airlines = avg_scores_all.sort_values(ascending=False).head(3).items()

            # Show reviews for selected airline (if any)
            if selected_airline:
                filtered_df = df[df['Airlines'] == selected_airline]
            else:
                filtered_df = df  # Show all reviews if no specific airline is selected

        else:
            # ** If a specific month is selected, show reviews for that month **
            filtered_df = df[(df['Airlines'] == selected_airline) & 
                             (df['Date Published'].dt.month_name() == selected_month)]

            # Calculate top 3 airlines for the selected month
            monthly_data = df[df['Date Published'].dt.month_name() == selected_month]
            avg_scores_month = monthly_data.groupby('Airlines')['Normalized Sentiment Score'].mean()
            top_airlines_month = avg_scores_month.sort_values(ascending=False).head(3).items()

            # Show reviews for the selected airline (if any)
            if selected_airline:
                filtered_df = df[df['Airlines'] == selected_airline]

        # Sort reviews by sentiment score for top and bottom reviews
        top_reviews_df = filtered_df.sort_values(by='Sentiment Score', ascending=False)
        bottom_reviews_df = filtered_df.sort_values(by='Sentiment Score', ascending=True)

        top_reviews = top_reviews_df.head(3)
        bottom_reviews = bottom_reviews_df.head(3)

        
        t_reviews = top_reviews[['Text Content', 'Sentiment Score']].values.tolist()
        b_reviews = bottom_reviews[['Text Content', 'Sentiment Score']].values.tolist()

        # Sliding window analysis for positive and negative text segments
        all_text = ' '.join(filtered_df['Text Content'].dropna().astype(str).tolist())
        pos, neg = sliding_window_analysis_words(all_text, afinn_dict, window_size=10)

        positive_window = pos  # {'paragraph': ..., 'score': ...}
        negative_window = neg
  
    
    return render_template('index.html', airlines=airlines, months=months, t_reviews=t_reviews, b_reviews=b_reviews, selected_airline=selected_airline, selected_month=selected_month, positive_window=positive_window,
        negative_window=negative_window,pos_segment=arb_pos, neg_segment=arb_neg, top_airlines=top_airlines,
                                  top_airlines_month=top_airlines_month)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5002, use_reloader=False)



