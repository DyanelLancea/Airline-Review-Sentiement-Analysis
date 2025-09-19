from flask import Flask, render_template_string, request
import pandas as pd
import requests
from io import StringIO

app = Flask(__name__)

# URL of the CSV file
url = 'https://raw.githubusercontent.com/DyanelLancea/Airline-Review-Sentiement-Analysis/refs/heads/master/airlines_review.csv'

# Fetch and load the CSV data
response = requests.get(url)
csv_data = StringIO(response.text)
df = pd.read_csv(csv_data)

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

# Extract available airlines and months for the dropdowns
airlines = df['Airlines'].unique().tolist()
months = df['Date Published'].dt.month_name().unique().tolist()

df['Text Content Tokenized'] = df['Text Content'].apply(tokenize_sentences)

#Apply tonkenize function into Dataframe
df['Text Content Tokenized'] = df['Text Content'].apply(tokenize_sentences)
#Apply Sentiment scoring function into Dataframe
df['Sentiment Score'] = df.apply(lambda x: calculate_sentiment_score(x['Text Content'],afinn_dict), axis=1)
#Apply normaalize function to Dataframe
df['Normalized Sentiment Score'] = df.apply(lambda x: normalize_score(x['Sentiment Score'],len(x["Text Content"].split())), axis=1)#Apply tonkenize function into Dataframe
df['Text Content Tokenized'] = df['Text Content'].apply(tokenize_sentences)
#Apply Sentiment scoring function into Dataframe
df['Sentiment Score'] = df.apply(lambda x: calculate_sentiment_score(x['Text Content'],afinn_dict), axis=1)
#Apply normaalize function to Dataframe
df['Normalized Sentiment Score'] = df.apply(lambda x: normalize_score(x['Sentiment Score'],len(x["Text Content"].split())), axis=1)

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_airline = None
    selected_month = None
    t_reviews = []
    b_reviews = []

    if request.method == 'POST':
        selected_airline = request.form.get('airline')
        selected_month = request.form.get('month')

        if selected_airline and selected_month:
            # Filter reviews based on selected airline and month
            filtered_df = df[(df['Airlines'] == selected_airline) &
                             (df['Date Published'].dt.month_name() == selected_month)]
        
            # Sort the filtered DataFrame by 'Normalized Sentiment Score' in descending order
            top_reviews_df = filtered_df.sort_values(by='Sentiment Score', ascending=False)
            bottom_reviews_df = filtered_df.sort_values(by='Sentiment Score', ascending = False)

            # Select the top 3 reviews based on the highest sentiment score
            top_reviews = top_reviews_df.head(3)
            bottom_reviews = bottom_reviews_df.tail(3)

            # Store the reviews with the highest sentiment scores
            t_reviews = top_reviews[['Text Content', 'Sentiment Score']].values.tolist()
            b_reviews = bottom_reviews[['Text Content', 'Sentiment Score']].values.tolist()

    # HTML template directly in the Python code using render_template_string
    html_content = '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Airline Reviews</title>
            <style>
                .review-list {
                    font-size: 10px;
                    line-height: 1.5;
                    font-family: Arial, sans-serif;
                }

                .review-list li {
                    margin-bottom: 10px;
                }

                .review-header {
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 10px;
                }

                /* Style the sentiment score section */
                .score {
                    color: #007BFF;
                    font-weight: bold;
                }

                /* Add some padding and style for form */
                form {
                    margin-bottom: 20px;
                }

                label {
                    font-weight: bold;
                    margin-right: 10px;
                }

                select, button {
                    margin: 5px 10px;
                }

            </style>
        </head>
        <body>
            <h1>Airline Review Sentiment Analysis</h1>

            <form method="POST">
                <label for="airline">Select Airline:</label>
                <select name="airline" id="airline" required>
                    <option value="">--Select Airline--</option>
                    {% for airline in airlines %}
                        <option value="{{ airline }}" {% if airline == selected_airline %}selected{% endif %}>{{ airline }}</option>
                    {% endfor %}
                </select>

                <label for="month">Select Month:</label>
                <select name="month" id="month" required>
                    <option value="">--Select Month--</option>
                    {% for month in months %}
                        <option value="{{ month }}" {% if month == selected_month %}selected{% endif %}>{{ month }}</option>
                    {% endfor %}
                </select>

                <button type="submit">Filter Reviews</button>
            </form>

            <hr>

            {% if t_reviews %}
                <h2 class="review-header">Top Reviews for {{ selected_airline }} in {{ selected_month }}:</h2>
                <div class="review-list">
                    <ul>
                        {% for t_reviews, score in t_reviews %}
                            <li>
                                <p><strong>Review:</strong> {{ t_reviews }}</p>
                                <p><strong>Score:</strong> <span class="score">{{ score }}</span></p>
                            </li>
                        {% endfor %}
                    </ul>
                </div>
             {% endif %}   
            
             {% if b_reviews %}
                <h2 class="review-header">Lowest Reviews for {{ selected_airline }} in {{ selected_month }}:</h2>
                <div class="review-list">
                    <ul>
                        {% for b_reviews, score in b_reviews %}
                            <li>
                                <p><strong>Review:</strong> {{ b_reviews }}</p>
                                <p><strong>Score:</strong> <span class="score">{{ score }}</span></p>
                            </li>
                        {% endfor %}
                    </ul>
                </div>
            {% endif %}
            {% if not t_reviews and not b_reviews %}
                <p>No reviews found for the selected filters.</p>
            {% endif %}
        </body>
        </html>
        '''
    
    return render_template_string(html_content, airlines=airlines, months=months, t_reviews=t_reviews, b_reviews=b_reviews, selected_airline=selected_airline, selected_month=selected_month)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5002, use_reloader=False)
