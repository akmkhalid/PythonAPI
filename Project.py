import requests  
from bs4 import BeautifulSoup  
import time  
import polars as pl  
import re  

mainpage_url = "https://www.thedailystar.net/life-living/relationships-family" # storing the url in a string
print(mainpage_url)
print("{mainpage_url}/robots.txt") # print URL and robots.txt location
r = requests.get(mainpage_url) # send HTTP request 

print(r.status_code)
if r.status_code == 200:  # connection successful
    print(" Success!")
else:
    print(r.status_code)  # connection failed
    exit()

mainpage = BeautifulSoup(r.text, "html.parser") # parse HTML
print(mainpage.prettify()[:500]) # Formatted, readable HTML
print(mainpage.title) # extract and print a title of a page
print(mainpage.find('title')) # extract title using find function
print(mainpage.select('title')) #extract title using select function

all_links = mainpage.find_all("a", href=True) # Extract All Links 
print(len(all_links))

for i, link in enumerate(all_links[:10], 1): #Show first 10 links as preview  
    print(f"  {i}. {link.get('href')}")

article_links = []
for link in all_links: 
    href = link.get("href")
    if "/life-living/relationships-family/" in href and "page=" not in href:  #"/life-living/relationships-family/" in href: Build absolute URL
        if href.startswith("/"): 
            full_url = "https://www.thedailystar.net" + href 
        else:
            full_url = href
        if full_url not in article_links: #Add to article_links list if not duplicate
            article_links.append(full_url)

print(len(article_links))            
for i, url in enumerate(article_links[:5], 1): #print the first 5 published 
    print(i, url)

def scrape_article(url): # takes URL as input
    """Scrape a single article and extract:
    - Title
    - Publication date
    - Article content    """
    print("Scraping: "+ url[:80])
    
    response = requests.get(url)
    if response.status_code != 200:
        print(response.status_code)
        return None
    
    article_soup = BeautifulSoup(response.text, "html.parser")
    title_tag = article_soup.find("h1")
    if title_tag:
        title = title_tag.get_text(strip=True)
        print(" Title: {title[:60]}...")
    else:
        title = "No title found"
        print(" No title found")
    date = None
    date_selectors = [
        article_soup.find("span", class_="date"),
        article_soup.find("time"),
        article_soup.find("div", class_="submitted-date"),
        article_soup.find("meta", property="article:published_time")
    ]
    for selector in date_selectors:
        if selector:
            if selector.name == "meta":
                date = selector.get("content")
            else:
                date = selector.get_text(strip=True)
            if date:
                print(date)
                break
    if not date:
        print("No date found")
    content = None
    content_areas = [
        article_soup.find("div", class_="field--name-body"),
        article_soup.find("div", class_="node__content"),
        article_soup.find("article"),
        article_soup.find("main")
    ]
    for area in content_areas:
        if area:
            paragraphs = area.find_all("p")
            if paragraphs:
                content = "\n\n".join([p.get_text(strip=True) for p in paragraphs])
                if len(content) > 200:  # Valid article has substantial text
                    print(f"Content: {len(content)} characters")
                    break
    if not content:
        print(f"No content found")
        return None

    return {
        "title": title,
        "date": date if date else "Unknown",
        "url": url,
        "content": content,
        "content_length": len(content)
    }

if article_links:
    print("\nTesting scrape_article() on first article:")
    test_article = scrape_article(article_links[0])
    if test_article:
        print("\n Function works correctly!")
    time.sleep(1)  

articles_data = []
for i, url in enumerate(article_links[:10], 1):  # Limiting to 10 for demo
    print(f"\n Article {i}/{min(10, len(article_links))}")
    article_info = scrape_article(url)

    if article_info:
        articles_data.append(article_info)
        print(f"  Added to dataset")
    else:
        print(f" Failed to scrape")

    time.sleep(1)

print(f"\n Successfully scraped {len(articles_data)} out of {min(10, len(article_links))} articles")

if articles_data:
    df = pl.DataFrame(articles_data)
    print("DataFrame created successfully!")
    print("\nDataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("Columns: {df.columns}")
    print("\nFirst 3 rows of our data:")
    print(df.head(3))

    print("\n Basic statistics:")
    print("  • Total articles: {len(df)}")
    print("  • Average content length: {df['content_length'].mean():.0f} characters")
    print("  • Articles with dates: {df['date'].null_count()}")

if articles_data:
    csv_file = "dailystar_relationships_articles.csv"
    df.write_csv(csv_file)
    print(f" Saved as CSV: {csv_file}")

    parquet_file = "dailystar_relationships_articles.parquet"
    df.write_parquet(parquet_file)
    print(f"Saved as Parquet: {parquet_file}")

    json_file = "dailystar_relationships_articles.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        import json

        json.dump(articles_data, f, indent=2, ensure_ascii=False)
        print(f" Saved as JSON: {json_file}")

if articles_data:
    print("Here are some things you can do with your scraped data:")

    print("\n1. Find the longest article:")
    longest_idx = df['content_length'].arg_max()
    longest_article = df[longest_idx]
    print(f" Title: {longest_article['title'][:60]}...")
    print(f" Length: {longest_article['content_length']} characters")

    print("\n2. List all article titles:")
    for i, title in enumerate(df['title'].head(5), 1):
        print(f"   {i}. {title[:70]}...")

    print("\n3. Check date range (if dates available):")
    if df['date'].null_count() < len(df):
        print("Date information is available for most articles")
    else:
        print("Date information limited - website may use dynamic loading")

if articles_data:
    print(f" {csv_file} - Open in Excel/Google Sheets")
    print(f" {parquet_file} - Efficient format for large datasets")
    print(f" {json_file} - Easy to use with JavaScript/APIs")

print("""
To scrape multiple pages of articles, you can use:

for page in range(2, 6):  # Pages 2 through 5
    page_url = f"https://www.thedailystar.net/life-living/relationships-family?page={page}"
    response = requests.get(page_url)
    soup = BeautifulSoup(response.text, "html.parser")
    # Extract links from this page...
    time.sleep(1) """)
