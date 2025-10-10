"""
sentiment_analysis.py
----------------------
Contains all data preprocessing and sentiment analysis functions
used for the airline review Flask dashboard.

Functions:
- load_afinn_lexicon(file_path)
- clean_text(df)
- full_pipeline()
"""

# --- Core Data Manipulation and NLP ---
import pandas as pd
import os
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer


''' ---------------------- Data Processing ---------------------- '''
# Load data
PATH = 'data/airlines_review.csv'
df = pd.read_csv(PATH)


def load_afinn_lexicon(afinn_path):
    """
    2. AFINN/Lexicon Loading from a text file
    Each line of the file contains a word and its sentiment score,
    separated by a tab. Returns a dictionary mapping words to scores. (afinn_path)
    """
    
    afinn_dict = {} # initialise an empty dictionary to hold the AFINN lexicon
    try:
        with open(afinn_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                word, score = line.rsplit('\t', 1)
                afinn_dict[word] = int(score)
    except FileNotFoundError:
        print(f"File not found: {afinn_path}")
    return afinn_dict


def replace_missing_value(df):
    """
    Function used in -> load_and_clean_data().
    Imputes missing values (NaN) in the DataFrame based on column data type.
    - String columns (object type) are filled with 'Unknown'.
    - All other column types (numeric, datetime, etc.) are filled with "NA".

    Args:
        df (pd.DataFrame): The DataFrame to modify in place.
    Returns:
        None: The function modifies the input DataFrame directly.
    """
    for col in df.columns:
        if df[col].dtype == 'O':  # Object type (string)
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna("NA")
    return 


def remove_special_characters(df, removechar, char):
    """
    Function used in -> load_and_clean_data().
    Removes a predefined list of noise characters from specific text columns.

    - Characters are replaced with a single space in the 'Airlines' column 
      to prevent word concatenation.
    - Characters are completely removed (replaced with an empty string) in 
      the 'Text Content' column.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        removechar (list): A list of strings/characters to search for and remove.
        char (str): An unused parameter (retained from original signature).
        
    Returns:
        pd.DataFrame: The DataFrame with characters removed from specified columns.
    """
    for char in removechar:
        df['Airlines'] = df['Airlines'].str.replace(char, ' ', regex=False)
        df['Text Content'] = df['Text Content'].str.replace(char, '', regex=False)
    return df


def load_and_clean_data(PATH):
    """
    Data Loading & Cleaning
    Cleans the input csv by dropping duplicates and NaNs, removing noise characters, 
    and standardizing text casing. 
    Args:
        PATH (str): The path or PATH of the CSV file containing the data.
    
    Returns: a tuple containing..
        df (pd.DataFrame): The cleaned and preprocessed DataFrame 
        sid (SentimentIntensityAnalyzer): Initialized VADER SentimentIntensityAnalyzer for analysis.
    """
    sid = SentimentIntensityAnalyzer()
    df = pd.read_csv(PATH)
    
    # Remove duplicates
    df = df.drop_duplicates()
    # Remove rows with missing values
    df = df.dropna()
    
    # characters to remove
    removechar = [ '@', '#', '$', '%', '^', '&', '*', '(', ')',
                   '-', '_', '=', '+', '{', '}', '[', ']', '|',
                   '\\', ':', ';', '"', "'", '<', '>', '?',
                   '/', '~', '`', '✅ Trip Verified', 'Not Verified',
                   'Â Â', '✅ Verified Review']
    
    # Apply your cleaning steps
    replace_missing_value(df)
    remove_special_characters(df, removechar, char='')

    # Standardize text case
    df['Airlines'] = df['Airlines'].str.title()
    df['Name'] = df['Name'].str.title()
    df['Text Content Lower Case'] = df['Text Content'].str.lower()

    # Remove leading spaces
    df['Airlines'] = df['Airlines'].str.lstrip()
    df['Name'] = df['Name'].str.lstrip()
    df['Date Published'] = df['Date Published'].str.lstrip()
    df['Text Content'] = df['Text Content'].str.lstrip()
    
    df.to_csv('data/airlines_review_cleaned.csv', index=False)
    
    return df, sid



''' ---------------------- Sentiment Scoring Algorithm ---------------------- '''

'''Start of Req 6: Word Segmentation using Dynamic Programming'''
def load_dictionary(dict_path):
    """
    Function used in -> call_dynamic_prog().
    Load words.txt into a set for O(1) membership tests.
    Normalize to lowercase. 
    """
    words = set()
    with open(dict_path, 'r', encoding='utf-8', errors='ignore') as f:  
        # use ignore to skip bad chars, 'open(..) as f' is to auto close when done
        for line in f:
            w = line.strip() # strip whitespace/newline
            if not w:
                continue
            words.add(w.lower()) # store lowercase for matching
    return words


def segment_text(text, dictionary, max_word_length=30, unknown_penalty=1):
    """
    Function used in -> call_dynamic_prog().
    and is theCore DP segmentation function
    Segment `text` (string without spaces) into words using dictionary.
    Returns a segmented string (have spaces). Keeps original casing of input,
    but algorithm works on lowercase.
    
    - max_word_length: limit to consider for last word length (speeds up).
    - unknown_penalty: penalty (cost) for each character that is not matched to a dictionary word.
    
    Strategy:
      DP over positions. best[i] = (score, last_split_index)
      Score is total matched characters (so higher is better). Unknown chars are penalized.
    """
    if text is None:
        return ""
    s = text.strip()
    if not s:
        return ""
    
    s_lower = s.lower() #converts whole input string lowercase for matching
    n = len(s_lower)
    best_score = [-10**9] * (n + 1) # best score so far, ending position i
    prev_idx = [-1] * (n + 1)   # records where last word started, for backtracking
    best_score[0] = 0  # empty prefix score 0
    max_len = min(max_word_length, n)  # for speed, precompute maximum possible word length (min of provided max and n)
    length_bias = 0.3  # bias towards longer words
    
    
    
    for i in range(1, n + 1):       
        start_j = max(0, i - max_len)
        # loops through each position i
        
        for j in range(start_j, i):
            #checks for best segmentation up to j within max_len
            chunk = s_lower[j:i]
            L = len(chunk)

            if chunk in dictionary:
                # reward/score: matched characters count, with extra bonus for longer words
                # base = L ; multiplier increases with length_bias
                multiplier = 1.0 + length_bias * (L - 1)
                score = best_score[j] + (L * multiplier)
            else:
                # penalise unknown chunk by its length * penalty
                score = best_score[j] - (L * unknown_penalty)

            if score > best_score[i]:
                best_score[i] = score
                prev_idx[i] = j

    # reconstruct segmentation
    if prev_idx[n] == -1:
        # fallback: no segmentation found, return original
        return s

    tokens = []
    i = n #start from the end of the string
    while i > 0:
        j = prev_idx[i]
        if j == -1:
            # if something odd, push the remainder and break
            tokens.append(s[j:i])
            break
        tokens.append(s[j:i])
        i = j
    tokens.reverse()
    # Optionally, try to recover original casing by mapping tokens back to original text slice
    # We'll return tokens joined by spaces
    return ' '.join(tokens) #joins them all in a single string


def call_dynamic_prog():
    """
    This function is called and used to present Requirement 6 of the project.
    """
     # Example: load dictionary and apply
    dict_path = os.path.join('data', 'words.txt')   # adjust path if your words.txt is elsewhere
    dictionary = load_dictionary(dict_path)         # uses words.txt you uploaded. :contentReference[oaicite:2]{index=2}

    # Optional: determine a reasonable max_word_length from the dictionary (speeds DP)
    if dictionary:
        max_word_length = max(len(w) for w in dictionary)
        # clamp to a reasonable upper bound (e.g., 30) for performance
        max_word_length = min(max_word_length, 30)
    else:
        max_word_length = 30

    # Add segmented column to your dataframe
    # Assuming df['Text Content'] contains the no-space strings

    df['Text Content Segmented'] = df['Text Content'].astype(str).apply(
        lambda t: segment_text(t, dictionary, max_word_length=max_word_length, unknown_penalty=1)
    )

    # Save a quick sample to inspect
    df[['Text Content', 'Text Content Segmented']].head(20).to_csv('data/segmentation_sample.csv', index=False)

    # (Then you can use 'Text Content Segmented' for later sentence tokenization or sentiment)
'''End of Req 6: Word Segmentation using Dynamic Programming'''

# save airlines_review_cleaned.csv??

def tokenize_sentences(text):
    '''
    Function for Sentence tokanization. This function takes some text and returns a list of sentences. 
    If you don’t give it a string, it safely returns an empty list instead of crashing.
    '''
    if not isinstance(text, str):
        return []
    return sent_tokenize(text)


def calculate_sentiment_score(sentences, afinn_dict):
    '''
    Calculates the sentiment score
    '''
    score = 0
    words = sentences.lower().split()    # breaks down the sentence into individual words
    for word in words:  # goes through each word in the list 
        score += afinn_dict.get(word, 0)    # looks up each word and adds the score
    return score


def normalize_score(score, text_length):
    '''
    Calculates the normalization score
    '''
    if text_length == 0:    # sentences with no words will return a score of 0 (preventing an error of dividing by zero)
        return 0
    # normalization to get a score per word, then clamping to [-1, 1]
    normalized = score / text_length    # calculates the average score per word
    return max(-1.0, min(1.0, normalized))  
    # makes sure the score doesn't go over 1.0 and below -1.0. 
    # (if the score is -2.5, this will return -1.0)


def find_extreme_sentences(sentences, afinn_dict):
    '''
    Finds extreme sentences with highest and lowest normalized sentiment scores
    1. Calculates sentiment scores for each sentence using the AFINN lexicon.
    2. Normalizes the sentiment scores based on sentence length.
    3. Identifies and returns the sentences with the highest and lowest normalized scores.
    '''
    
    if not sentences:   # to prevent errors if the list is empty
        return None, None

    # initialises an empty list to hold sentences and their scores
    scored_sentences = []
    for sent in sentences:
        score = calculate_sentiment_score(sent, afinn_dict)     # uses the sentiment score function
        normalized_score = normalize_score(score, len(sent.split()))    # uses the normalization function
        scored_sentences.append({'sentence': sent, 'score': normalized_score})

    # find the sentences with max and min normalized scores
    most_positive = max(scored_sentences, key=lambda x: x['score'])
    most_negative = min(scored_sentences, key=lambda x: x['score'])
    
    return most_positive, most_negative


def sliding_window_analysis_words(text, afinn_dict, window_size=10):
    """
    Applies a sliding window to find the most positive and negative text segments,
    based on a word-level window size.
    
    Args:
        text (str): The full text to analyze.
        afinn_dict (dict): The AFINN sentiment lexicon.
        window_size (int): The number of words to include in each window.

    Returns:
        tuple: A tuple containing the most positive and most negative paragraphs.
    """
    if not text:
        return None, None
        
    words = text.lower().split()
    if len(words) < window_size:
        return None, None
    
    scored_windows = []
    # Slide the window across the words
    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        paragraph_text = ' '.join(window)
        
        # Calculate score for the paragraph window
        score = calculate_sentiment_score(paragraph_text, afinn_dict)
        normalized_score = normalize_score(score, len(paragraph_text.split()))

        scored_windows.append({'paragraph': paragraph_text, 'score': normalized_score})

    if not scored_windows:
        return None, None

    # Find the paragraphs with max and min scores
    most_positive_paragraph = max(scored_windows, key=lambda x: x['score'])
    most_negative_paragraph = min(scored_windows, key=lambda x: x['score'])
    
    return most_positive_paragraph, most_negative_paragraph


'''---------------------- Applying analysis to the dataframe ---------------------- '''
def run_sentiment_analysis(df, afinn_dict):
    '''
    Apply Analysis to DataFrame
    '''
    print(" Running Sentiment Analysis...")
    
    #Apply tonkenize function into Dataframe
    df['Text Content Tokenized'] = df['Text Content'].apply(tokenize_sentences)

    #Apply Sentiment scoring function into Dataframe
    df['Sentiment Score'] = df.apply(lambda x: calculate_sentiment_score(x['Text Content'],afinn_dict), axis=1)

    #Apply normalize function to Dataframe
    df['Normalized Sentiment Score'] = df.apply(lambda x: normalize_score(x['Sentiment Score'],len(x["Text Content"].split())), axis=1)

    #Apply finding extreme sentences function to Dataframe
    df['Extreme Senctences'] = df.apply(lambda x: find_extreme_sentences(x['Text Content Tokenized'],afinn_dict), axis=1)

    #Creating Columns for Most and Least Extreme Sentences
    # df['Most Positive Senctence'] = df['Extreme Senctences'].apply(lambda x: x[0]['sentence'])
    # df['Most Positive Senctence Score'] = df['Extreme Senctences'].apply(lambda x: x[0]['score'])
    # df['Most Negative Senctence'] = df['Extreme Senctences'].apply(lambda x: x[1]['sentence'])
    # df['Most Negative Senctence Score'] = df['Extreme Senctences'].apply(lambda x: x[1]['score'])

    #Creating Columns for Most and Least Extreme Sentences with checks
    df['Most Positive Senctence'] = df['Extreme Senctences'].apply(
        lambda x: x[0]['sentence'] if x and isinstance(x, tuple) and x[0] is not None else None)
    df['Most Positive Senctence Score'] = df['Extreme Senctences'].apply(
        lambda x: x[0]['score'] if x and isinstance(x, tuple) and x[0] is not None else None)
    df['Most Negative Senctence'] = df['Extreme Senctences'].apply(
        lambda x: x[1]['sentence'] if x and isinstance(x, tuple) and x[1] is not None else None)
    df['Most Negative Senctence Score'] = df['Extreme Senctences'].apply(
        lambda x: x[1]['score'] if x and isinstance(x, tuple) and x[1] is not None else None)


    #Applying Sliding Window to Dataframe
    df['Sliding Window Results'] = df.apply(lambda x: sliding_window_analysis_words(x['Text Content'],afinn_dict, window_size=10), axis=1)


    '''
    Use this if you are testing with 'airlines_review_no_space.csv
    '''
    #Creating Columns for Most Positive and Negative lines of words from sliding window analysis
    # df['Most Positive Line'] = df['Sliding Window Results'].apply(
    #     lambda x: x[0]['sentence'] if x and isinstance(x, tuple) and x[0] is not None else None)
    # df['Most Positive Line Score'] = df['Sliding Window Results'].apply(
    #     lambda x: x[0]['sentence'] if x and isinstance(x, tuple) and x[0] is not None else None)
    # df['Most Negative Line'] = df['Sliding Window Results'].apply(
    #     lambda x: x[0]['sentence'] if x and isinstance(x, tuple) and x[0] is not None else None)
    # df['Most Negative Line Score'] = df['Sliding Window Results'].apply(
    #     lambda x: x[0]['sentence'] if x and isinstance(x, tuple) and x[0] is not None else None)

    #Creating Columns for Most Positive and Negative lines of words from sliding window analysis
    df['Most Positive Line'] = df['Sliding Window Results'].apply(lambda x: x[0]['paragraph'])
    df['Most Positive Line Score'] = df['Sliding Window Results'].apply(lambda x: x[0]['score'])
    df['Most Negative Line'] = df['Sliding Window Results'].apply(lambda x: x[1]['paragraph'])
    df['Most Negative Line Score'] = df['Sliding Window Results'].apply(lambda x: x[1]['score'])
    #Output results into csv
    df.to_csv('airlines_review_analysis.csv', index=False)
    
    print(" Sentiment Analysis Completed. Saved to airlines_review_analysis.csv.")



def full_pipeline():
    afinn_dict = load_afinn_lexicon("data/AFINN-en-165.txt")
    df, sid = load_and_clean_data("data/airlines_review.csv")
    df = run_sentiment_analysis(df, afinn_dict)
    #create_visualizations(df)
    #requirement6Function
    df.to_csv("data/airlines_review_analysis.csv", index=False)

