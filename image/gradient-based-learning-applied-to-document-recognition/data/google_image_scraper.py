import argparse
import requests
import base64
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
    
import time
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm




#TODO: pipeline the for loops and use multi processing to speed up the searching and downloading. probably need to just do async requests idk. 
def scrape_images(query, num_images, output_path, num_processes):
    print("setting up web scraper...")

    # Set up Chrome options for headless mode
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in headless mode (no GUI)
    chrome_options.add_argument('--disable-gpu')  # Disable GPU acceleration (useful for headless mode)\\
    # using the firefox web browser and making sure it is the headleass version
    print("starting browser...")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
    print("browser started")
    
    driver.get("https://www.google.com/imghp") # going to google images
    # print("finding the search box")
    from selenium.webdriver.common.by import By

    search_box = driver.find_element(by=By.NAME, value="q") # finding the search box in the html
    search_box.send_keys(query) # entering the search query
    search_box.send_keys(Keys.RETURN)
    print("web scraper set up")
    
    print("searching for images...")
    for _ in tqdm(range(num_images // 100)): # scrolls the page multiple time to load more results
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        
    print("downloading images...")
    soup = BeautifulSoup(driver.page_source, "html.parser")
    img_tags = soup.find_all("img", class_="rg_i") # rg_i typically contains image urls
    
    
    def download_img(tag, num):
        """this first check if 'src' is base 64 string or url and downloads as so"""
        img_url = tag.get("src")
        if img_url:
            if "base64" in img_url:
                base64_str = img_url.split(",", 1)[1]
                img_data = base64.b64decode(base64_str)
            elif "https://" in img_url:
                img_data = requests.get(img_url).content # downloads the image

            with open(f"{output_path}/cat_{num}.jpg", "wb") as file:
                file.write(img_data)
                
    
    for i in tqdm(range(len(img_tags))):
        tag = img_tags[i]
        download_img(tag, i)
        
    driver.quit()
    
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--query", required=True, type=str, help="something to search like 'pictures of real life cats'")
    parser.add_argument("--num-images", type=int, default=100, help="number of images to scrape")
    parser.add_argument("--output-path", type=str, required=True, help="path to folder to store images")
    parser.add_argument("--num-processes", type=int, default=8, help="number of processes to run at once")
    
    args = parser.parse_args()
    
    scrape_images(args.query, args.num_images, args.output_path, args.num_processes)
                
        
    