# %%
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path  

# %% [markdown]
# #1. Qatar Airways

# %%
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

# %%
# Main loop to scrape multiple pages and track progress
qatar_airways_reviews = []
total_pages = 264
for page_number in range(1, total_pages + 1):
    page_reviews = scrape_page_QA(page_number)
    qatar_airways_reviews.extend(page_reviews)

    # Print progress message
    if page_number % 10 == 0:
        print(f"{page_number} pages extraction completed")


# %%
qatar_airways_reviews_df = pd.DataFrame(qatar_airways_reviews)

# %%
qatar_airways_reviews_df.sample(5)

# %%
qatar_airways_reviews_df.shape

# %% [markdown]
# #2. Singapore Airlines

# %%
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

# %%
singapore_airlines_reviews = []
total_pages = 167
for page_number in range(1, total_pages + 1):
    page_reviews = scrape_page_SQ(page_number)
    singapore_airlines_reviews.extend(page_reviews)

    # Print progress message
    if page_number % 10 == 0:
        print(f"{page_number} pages extraction completed")


# %%
singapore_airlines_reviews_df = pd.DataFrame(singapore_airlines_reviews)

# %%
singapore_airlines_reviews_df.sample(5)

# %%
singapore_airlines_reviews_df.shape

# %% [markdown]
# #3. Cathay Pacific Airways

# %%
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

# %%
cathay_pacific_airways_reviews = []
total_pages = 151
for page_number in range(1, total_pages + 1):
    page_reviews = scrape_page_CP(page_number)
    cathay_pacific_airways_reviews.extend(page_reviews)

    # Print progress message
    if page_number % 10 == 0:
        print(f"{page_number} pages extraction completed")


# %%
cathay_pacific_airways_reviews_df = pd.DataFrame(cathay_pacific_airways_reviews)

# %%
cathay_pacific_airways_reviews_df.sample(5)

# %%
cathay_pacific_airways_reviews_df.shape

# %% [markdown]
# #4. Emirates

# %%
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

# %%
emirates_reviews = []
total_pages = 247
for page_number in range(1, total_pages + 1):
    page_reviews = scrape_page_EM(page_number)
    emirates_reviews.extend(page_reviews)

    # Print progress message
    if page_number % 10 == 0:
        print(f"{page_number} pages extraction completed")


# %%
emirates_reviews_df = pd.DataFrame(emirates_reviews)

# %%
emirates_reviews_df.sample(5)

# %%
emirates_reviews_df.shape

# %% [markdown]
# #5.ANA All Nippon Airways

# %%
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

# %%
ANA_reviews = []
total_pages = 62
for page_number in range(1, total_pages + 1):
    page_reviews = scrape_page_ANA(page_number)
    ANA_reviews.extend(page_reviews)

    # Print progress message
    if page_number % 10 == 0:
        print(f"{page_number} pages extraction completed")


# %%
ANA_reviews_df = pd.DataFrame(ANA_reviews)

# %%
ANA_reviews_df.sample(5)

# %%
ANA_reviews_df.shape

# %% [markdown]
# #6. Turkish Airlines

# %%
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

# %%
turkish_airlines_reviews = []
total_pages = 280
for page_number in range(1, total_pages + 1):
    page_reviews = scrape_page_Turk(page_number)
    turkish_airlines_reviews.extend(page_reviews)

    # Print progress message
    if page_number % 10 == 0:
        print(f"{page_number} pages extraction completed")


# %%
turkish_airlines_reviews_df = pd.DataFrame(turkish_airlines_reviews)

# %%
turkish_airlines_reviews_df.sample(5)

# %%
turkish_airlines_reviews_df.shape

# %% [markdown]
# #7. Korean Air

# %%
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

# %%
korean_air_reviews = []
total_pages = 60
for page_number in range(1, total_pages + 1):
    page_reviews = scrape_page_KA(page_number)
    korean_air_reviews.extend(page_reviews)

    # Print progress message
    if page_number % 10 == 0:
        print(f"{page_number} pages extraction completed")


# %%
korean_air_reviews_df = pd.DataFrame(korean_air_reviews)

# %% [markdown]
# #8. Air France

# %%
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

# %%
air_france_reviews = []
total_pages = 146
for page_number in range(1, total_pages + 1):
    page_reviews = scrape_page_AF(page_number)
    air_france_reviews.extend(page_reviews)

    # Print progress message
    if page_number % 10 == 0:
        print(f"{page_number} pages extraction completed")


# %%
air_france_reviews_df = pd.DataFrame(air_france_reviews)

# %% [markdown]
# #9. Japan Airlines

# %%
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

# %%
japan_airlines_reviews = []
total_pages = 44
for page_number in range(1, total_pages + 1):
    page_reviews = scrape_page_JA(page_number)
    japan_airlines_reviews.extend(page_reviews)

    # Print progress message
    if page_number % 10 == 0:
        print(f"{page_number} pages extraction completed")


# %%
japan_airlines_reviews_df = pd.DataFrame(japan_airlines_reviews)

# %% [markdown]
# #10. Hainan Airlines

# %%
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

# %%
hainan_airlines_reviews = []
total_pages = 43
for page_number in range(1, total_pages + 1):
    page_reviews = scrape_page_HA(page_number)
    hainan_airlines_reviews.extend(page_reviews)

    # Print progress message
    if page_number % 10 == 0:
        print(f"{page_number} pages extraction completed")


# %%
hainan_airlines_reviews_df = pd.DataFrame(hainan_airlines_reviews)

# %% [markdown]
# Combining Datasets together

# %%
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

# %%
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

# %%
airlines_review.sample(10)

# %%
airlines_review.shape

# %%
airlines_review.to_csv('airlines_review.csv', index=False)



