"""
main.py
--------
Flask application for the Airline Review Sentiment Dashboard.
Loads preprocessed sentiment data and serves the interactive web interface.
"""

print("🔧 Importing necessary libraries...")
from flask import Flask, render_template, request
import pandas as pd
from sentiment_analysis import full_pipeline  
import os


print("🚀 Starting the Flask application...")
app = Flask(__name__, template_folder='templates')

# --- Run the sentiment analysis pipeline once ---
if not os.path.exists("data/airlines_review_analysis.csv"):  # Checks if .csv already exists
    print("🔍 Running data analysis pipeline...") 
    full_pipeline() # If not, it calls full_pipeline(), (the entire sentiment analysis workflow), to generate it.
else:
    print("✅ Analysis file found. Loading existing data...") # Saves time as if it already exists, it skips the heavy computation from rerunning 

    

# --- Load processed dataset ---
print("📊 Loading data...")
df = pd.read_csv("data/airlines_review_analysis.csv")
df['Date Published'] = pd.to_datetime(df['Date Published'])

# --- Define dropdown options ---
order_of_months = ['January','February','March','April','May','June','July',
                   'August','September','October','November','December']
airlines = df['Airlines'].unique().tolist()
months = sorted(df['Date Published'].dt.month_name().unique().tolist(),
                key=lambda x: order_of_months.index(x))


# --- Flask route ---
print("🚀 Application is ready!")
@app.route('/', methods=['GET', 'POST'])
def index():
    selected_airline = None
    selected_month = None
    t_reviews = []
    b_reviews = []
    top_airlines = []
    top_airlines_month = []

    if request.method == 'POST':
        selected_airline = request.form.get('airline')
        selected_month = request.form.get('month')

        if selected_month == 'All':
            avg_scores_all = df.groupby('Airlines')['Normalized Sentiment Score'].mean()
            top_airlines = avg_scores_all.sort_values(ascending=False).head(3).items()

            if selected_airline == 'All' or selected_airline is None:
                filtered_df = df
            else:
                filtered_df = df[df['Airlines'] == selected_airline]
        else:
            if selected_airline == 'All' or selected_airline is None:
                filtered_df = df[df['Date Published'].dt.month_name() == selected_month]
            else:
                filtered_df = df[
                    (df['Airlines'] == selected_airline) &
                    (df['Date Published'].dt.month_name() == selected_month)
                ]

            monthly_data = df[df['Date Published'].dt.month_name() == selected_month]
            avg_scores_month = monthly_data.groupby('Airlines')['Normalized Sentiment Score'].mean()
            top_airlines_month = avg_scores_month.sort_values(ascending=False).head(3).items()

        top_reviews_df = filtered_df.sort_values(by='Sentiment Score', ascending=False)
        bottom_reviews_df = filtered_df.sort_values(by='Sentiment Score', ascending=True)

        t_reviews = top_reviews_df.head(3)[['Most Positive Senctence', 'Sentiment Score', 'Airlines', 'Text Content']].values.tolist()
        b_reviews = bottom_reviews_df.head(3)[['Most Negative Senctence', 'Sentiment Score', 'Airlines', 'Text Content']].values.tolist()

    return render_template(
        'index.html',
        airlines=airlines,
        months=months,
        t_reviews=t_reviews,
        b_reviews=b_reviews,
        selected_airline=selected_airline,
        selected_month=selected_month,
        top_airlines=top_airlines,
        top_airlines_month=top_airlines_month
    )

# --- Entry point ---
if __name__ == "__main__":
    if not os.path.exists("data/airlines_review_analysis.csv"):
        full_pipeline()
    app.run(debug=False, host="127.0.0.1", port=5000)
