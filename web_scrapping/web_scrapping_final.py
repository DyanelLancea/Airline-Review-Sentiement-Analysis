import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path  

# Function for web scraping (Qatar Airways)
def scrape_page_QA(page_number):
    url = f'https://www.airlinequality.com/airline-reviews/qatar-airways/page/{page_number}/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    reviews = []
    for review_block in soup.find_all('article', itemprop='review'):  # use the full review container
        name_tag = review_block.find('span', itemprop='name')
        date_tag = review_block.find('time', itemprop='datePublished')
        text_content_div = review_block.find('div', class_='text_content')
        rating_tag = review_block.find('span', itemprop='ratingValue')

        # Extract review details with error handling
        name = name_tag.text.strip() if name_tag else 'N/A'
        date_published = date_tag['datetime'] if date_tag else 'N/A'
        text_content = text_content_div.text.strip() if text_content_div else 'N/A'
        rating = rating_tag.get_text(strip=True) if rating_tag else 'N/A'

        # Append extracted data to the list
        reviews.append({
            'Name': name,
            'Date Published': date_published,
            'Text Content': text_content,
            'Rating': rating
        })

    return reviews

# Main loop for qatar airways to scrape multiple pages and track progress
qatar_airways_reviews = []
total_pages = 264
print("Scrapping for Qatar Airways")
for page_number in range(1, total_pages + 1):
    page_reviews = scrape_page_QA(page_number)
    qatar_airways_reviews.extend(page_reviews)

    # Print progress message
    if page_number % 10 == 0:
        print(f"{page_number} pages extraction completed")

qatar_airways_reviews_df = pd.DataFrame(qatar_airways_reviews)

# Function for web scraping (Singapore Airlines)
def scrape_page_SQ(page_number):
    url = f'https://www.airlinequality.com/airline-reviews/singapore-airlines/page/{page_number}/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    reviews = []
    for review_block in soup.find_all('article', itemprop='review'):  # use the full review container
        name_tag = review_block.find('span', itemprop='name')
        date_tag = review_block.find('time', itemprop='datePublished')
        text_content_div = review_block.find('div', class_='text_content')
        rating_tag = review_block.find('span', itemprop='ratingValue')

        # Extract review details with error handling
        name = name_tag.text.strip() if name_tag else 'N/A'
        date_published = date_tag['datetime'] if date_tag else 'N/A'
        text_content = text_content_div.text.strip() if text_content_div else 'N/A'
        rating = rating_tag.get_text(strip=True) if rating_tag else 'N/A'

        # Append extracted data to the list
        reviews.append({
            'Name': name,
            'Date Published': date_published,
            'Text Content': text_content,
            'Rating': rating
        })

    return reviews

# Main loop for singapore airlines to scrape multiple pages and track progress
singapore_airlines_reviews = []
total_pages = 167
print("Scrapping for Singapore Airlines")
for page_number in range(1, total_pages + 1):
    page_reviews = scrape_page_SQ(page_number)
    singapore_airlines_reviews.extend(page_reviews)

    # Print progress message
    if page_number % 10 == 0:
        print(f"{page_number} pages extraction completed")


singapore_airlines_reviews_df = pd.DataFrame(singapore_airlines_reviews)

#Function for web scraping (Cathay pacific Airways)
def scrape_page_CP(page_number):
    url = f'https://www.airlinequality.com/airline-reviews/cathay-pacific-airways/page/{page_number}/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    reviews = []
    for review_block in soup.find_all('article', itemprop='review'):  # use the full review container
        name_tag = review_block.find('span', itemprop='name')
        date_tag = review_block.find('time', itemprop='datePublished')
        text_content_div = review_block.find('div', class_='text_content')
        rating_tag = review_block.find('span', itemprop='ratingValue')

        # Extract review details with error handling
        name = name_tag.text.strip() if name_tag else 'N/A'
        date_published = date_tag['datetime'] if date_tag else 'N/A'
        text_content = text_content_div.text.strip() if text_content_div else 'N/A'
        rating = rating_tag.get_text(strip=True) if rating_tag else 'N/A'

        # Append extracted data to the list
        reviews.append({
            'Name': name,
            'Date Published': date_published,
            'Text Content': text_content,
            'Rating': rating
        })

    return reviews

# Main loop for cathay pacific airways to scrape multiple pages and track progress
cathay_pacific_airways_reviews = []
total_pages = 151
print("Scrapping for Cathay Pacific Airways")
for page_number in range(1, total_pages + 1):
    page_reviews = scrape_page_CP(page_number)
    cathay_pacific_airways_reviews.extend(page_reviews)

    # Print progress message
    if page_number % 10 == 0:
        print(f"{page_number} pages extraction completed")

cathay_pacific_airways_reviews_df = pd.DataFrame(cathay_pacific_airways_reviews)

#Function for web scraping (Emirates)
def scrape_page_EM(page_number):
    url = f'https://www.airlinequality.com/airline-reviews/emirates/page/{page_number}/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    reviews = []
    for review_block in soup.find_all('article', itemprop='review'):  # use the full review container
        name_tag = review_block.find('span', itemprop='name')
        date_tag = review_block.find('time', itemprop='datePublished')
        text_content_div = review_block.find('div', class_='text_content')
        rating_tag = review_block.find('span', itemprop='ratingValue')

        # Extract review details with error handling
        name = name_tag.text.strip() if name_tag else 'N/A'
        date_published = date_tag['datetime'] if date_tag else 'N/A'
        text_content = text_content_div.text.strip() if text_content_div else 'N/A'
        rating = rating_tag.get_text(strip=True) if rating_tag else 'N/A'

        # Append extracted data to the list
        reviews.append({
            'Name': name,
            'Date Published': date_published,
            'Text Content': text_content,
            'Rating': rating
        })

    return reviews

# Main loop for emirates to scrape multiple pages and track progress
emirates_reviews = []
total_pages = 247
print("Scrapping for Emirates")
for page_number in range(1, total_pages + 1):
    page_reviews = scrape_page_EM(page_number)
    emirates_reviews.extend(page_reviews)

    # Print progress message
    if page_number % 10 == 0:
        print(f"{page_number} pages extraction completed")

emirates_reviews_df = pd.DataFrame(emirates_reviews)

#Function for web scraping (ANA All Nippon Airways)
def scrape_page_ANA(page_number):
    url = f'https://www.airlinequality.com/airline-reviews/ana-all-nippon-airways/page/{page_number}/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    reviews = []
    for review_block in soup.find_all('article', itemprop='review'):  # use the full review container
        name_tag = review_block.find('span', itemprop='name')
        date_tag = review_block.find('time', itemprop='datePublished')
        text_content_div = review_block.find('div', class_='text_content')
        rating_tag = review_block.find('span', itemprop='ratingValue')

        # Extract review details with error handling
        name = name_tag.text.strip() if name_tag else 'N/A'
        date_published = date_tag['datetime'] if date_tag else 'N/A'
        text_content = text_content_div.text.strip() if text_content_div else 'N/A'
        rating = rating_tag.get_text(strip=True) if rating_tag else 'N/A'

        # Append extracted data to the list
        reviews.append({
            'Name': name,
            'Date Published': date_published,
            'Text Content': text_content,
            'Rating': rating
        })

    return reviews

# Main loop for ANA All Nippon Airways to scrape multiple pages and track progress
ANA_reviews = []
total_pages = 62
print("Scrapping for ANA")
for page_number in range(1, total_pages + 1):
    page_reviews = scrape_page_ANA(page_number)
    ANA_reviews.extend(page_reviews)

    # Print progress message
    if page_number % 10 == 0:
        print(f"{page_number} pages extraction completed")

ANA_reviews_df = pd.DataFrame(ANA_reviews)

#Function for web scraping (Turkish Airlines)
def scrape_page_Turk(page_number):
    url = f'https://www.airlinequality.com/airline-reviews/turkish-airlines/page/{page_number}/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    reviews = []
    for review_block in soup.find_all('article', itemprop='review'):  # use the full review container
        name_tag = review_block.find('span', itemprop='name')
        date_tag = review_block.find('time', itemprop='datePublished')
        text_content_div = review_block.find('div', class_='text_content')
        rating_tag = review_block.find('span', itemprop='ratingValue')

        # Extract review details with error handling
        name = name_tag.text.strip() if name_tag else 'N/A'
        date_published = date_tag['datetime'] if date_tag else 'N/A'
        text_content = text_content_div.text.strip() if text_content_div else 'N/A'
        rating = rating_tag.get_text(strip=True) if rating_tag else 'N/A'

        # Append extracted data to the list
        reviews.append({
            'Name': name,
            'Date Published': date_published,
            'Text Content': text_content,
            'Rating': rating
        })

    return reviews

# Main loop for turkish airlines to scrape multiple pages and track progress
turkish_airlines_reviews = []
total_pages = 280
print("Scrapping for Turkish Airlines")
for page_number in range(1, total_pages + 1):
    page_reviews = scrape_page_Turk(page_number)
    turkish_airlines_reviews.extend(page_reviews)

    # Print progress message
    if page_number % 10 == 0:
        print(f"{page_number} pages extraction completed")

turkish_airlines_reviews_df = pd.DataFrame(turkish_airlines_reviews)

#Function for web scraping (Korean Air)
def scrape_page_KA(page_number):
    url = f'https://www.airlinequality.com/airline-reviews/korean-air/page/{page_number}/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    reviews = []
    for review_block in soup.find_all('article', itemprop='review'):  # use the full review container
        name_tag = review_block.find('span', itemprop='name')
        date_tag = review_block.find('time', itemprop='datePublished')
        text_content_div = review_block.find('div', class_='text_content')
        rating_tag = review_block.find('span', itemprop='ratingValue')

        # Extract review details with error handling
        name = name_tag.text.strip() if name_tag else 'N/A'
        date_published = date_tag['datetime'] if date_tag else 'N/A'
        text_content = text_content_div.text.strip() if text_content_div else 'N/A'
        rating = rating_tag.get_text(strip=True) if rating_tag else 'N/A'

        # Append extracted data to the list
        reviews.append({
            'Name': name,
            'Date Published': date_published,
            'Text Content': text_content,
            'Rating': rating
        })

    return reviews

# Main loop for korean air to scrape multiple pages and track progress
korean_air_reviews = []
total_pages = 60
print("Scrapping for Korean Air")
for page_number in range(1, total_pages + 1):
    page_reviews = scrape_page_KA(page_number)
    korean_air_reviews.extend(page_reviews)

    # Print progress message
    if page_number % 10 == 0:
        print(f"{page_number} pages extraction completed")

korean_air_reviews_df = pd.DataFrame(korean_air_reviews)

#Function for web scraping (Air France)
def scrape_page_AF(page_number):
    url = f'https://www.airlinequality.com/airline-reviews/air-france/page/{page_number}/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    reviews = []
    for review_block in soup.find_all('article', itemprop='review'):  # use the full review container
        name_tag = review_block.find('span', itemprop='name')
        date_tag = review_block.find('time', itemprop='datePublished')
        text_content_div = review_block.find('div', class_='text_content')
        rating_tag = review_block.find('span', itemprop='ratingValue')

        # Extract review details with error handling
        name = name_tag.text.strip() if name_tag else 'N/A'
        date_published = date_tag['datetime'] if date_tag else 'N/A'
        text_content = text_content_div.text.strip() if text_content_div else 'N/A'
        rating = rating_tag.get_text(strip=True) if rating_tag else 'N/A'

        # Append extracted data to the list
        reviews.append({
            'Name': name,
            'Date Published': date_published,
            'Text Content': text_content,
            'Rating': rating
        })

    return reviews

# Main loop for air france to scrape multiple pages and track progress
air_france_reviews = []
total_pages = 146
print("Scrapping for Air France")
for page_number in range(1, total_pages + 1):
    page_reviews = scrape_page_AF(page_number)
    air_france_reviews.extend(page_reviews)

    # Print progress message
    if page_number % 10 == 0:
        print(f"{page_number} pages extraction completed")

air_france_reviews_df = pd.DataFrame(air_france_reviews)

#Function for web scraping (japan Airlines)
def scrape_page_JA(page_number):
    url = f'https://www.airlinequality.com/airline-reviews/japan-airlines/page/{page_number}/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    reviews = []
    for review_block in soup.find_all('article', itemprop='review'):  # use the full review container
        name_tag = review_block.find('span', itemprop='name')
        date_tag = review_block.find('time', itemprop='datePublished')
        text_content_div = review_block.find('div', class_='text_content')
        rating_tag = review_block.find('span', itemprop='ratingValue')

        # Extract review details with error handling
        name = name_tag.text.strip() if name_tag else 'N/A'
        date_published = date_tag['datetime'] if date_tag else 'N/A'
        text_content = text_content_div.text.strip() if text_content_div else 'N/A'
        rating = rating_tag.get_text(strip=True) if rating_tag else 'N/A'

        # Append extracted data to the list
        reviews.append({
            'Name': name,
            'Date Published': date_published,
            'Text Content': text_content,
            'Rating': rating
        })

    return reviews

# Main loop for japan airlines to scrape multiple pages and track progress
japan_airlines_reviews = []
total_pages = 44
print("Scrapping for Japan Airlines")
for page_number in range(1, total_pages + 1):
    page_reviews = scrape_page_JA(page_number)
    japan_airlines_reviews.extend(page_reviews)

    # Print progress message
    if page_number % 10 == 0:
        print(f"{page_number} pages extraction completed")

japan_airlines_reviews_df = pd.DataFrame(japan_airlines_reviews)

#Function for web scraping (Hainan Airlines)
def scrape_page_HA(page_number):
    url = f'https://www.airlinequality.com/airline-reviews/hainan-airlines/page/{page_number}/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    reviews = []
    for review_block in soup.find_all('article', itemprop='review'):  # use the full review container
        name_tag = review_block.find('span', itemprop='name')
        date_tag = review_block.find('time', itemprop='datePublished')
        text_content_div = review_block.find('div', class_='text_content')
        rating_tag = review_block.find('span', itemprop='ratingValue')

        # Extract review details with error handling
        name = name_tag.text.strip() if name_tag else 'N/A'
        date_published = date_tag['datetime'] if date_tag else 'N/A'
        text_content = text_content_div.text.strip() if text_content_div else 'N/A'
        rating = rating_tag.get_text(strip=True) if rating_tag else 'N/A'

        # Append extracted data to the list
        reviews.append({
            'Name': name,
            'Date Published': date_published,
            'Text Content': text_content,
            'Rating': rating
        })

    return reviews

# Main loop for hainan airlines to scrape multiple pages and track progress
hainan_airlines_reviews = []
total_pages = 43
print("Scrapping for Hainan Airlines")
for page_number in range(1, total_pages + 1):
    page_reviews = scrape_page_HA(page_number)
    hainan_airlines_reviews.extend(page_reviews)

    # Print progress message
    if page_number % 10 == 0:
        print(f"{page_number} pages extraction completed")


hainan_airlines_reviews_df = pd.DataFrame(hainan_airlines_reviews)

# Combine all individual airlines dataframes
qatar_airways_reviews_df.insert(0, 'Airlines', 'qatar_airways')
singapore_airlines_reviews_df.insert(0, 'Airlines', 'singapore_airlines')
cathay_pacific_airways_reviews_df.insert(0, 'Airlines', 'cathay_pacific_airlines')
emirates_reviews_df.insert(0, 'Airlines', 'emirates')
ANA_reviews_df.insert(0, 'Airlines', 'all_nippon_airways')
turkish_airlines_reviews_df.insert(0, 'Airlines', 'turkish_airlines')
korean_air_reviews_df.insert(0, 'Airlines', 'korean_air')
air_france_reviews_df.insert(0, 'Airlines', 'air_france')
japan_airlines_reviews_df.insert(0, 'Airlines', 'japan_airlines')
hainan_airlines_reviews_df.insert(0, 'Airlines', 'hainan_airlines')

airlines_review = pd.concat([
    qatar_airways_reviews_df,
    singapore_airlines_reviews_df,
    cathay_pacific_airways_reviews_df,
    emirates_reviews_df,
    ANA_reviews_df,
    turkish_airlines_reviews_df,
    korean_air_reviews_df,
    air_france_reviews_df,
    japan_airlines_reviews_df,
    hainan_airlines_reviews_df
], axis=0, ignore_index=True)

# Export out data to csv
airlines_review.to_csv('data/airlines_review.csv', index=False)