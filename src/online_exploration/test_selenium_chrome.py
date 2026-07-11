import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

print("--- Starting Minimal WebDriver Test ---")

# 1. 使用最简化的 Options
options = Options()
# Headless Linux environments commonly need these options.
options.add_argument("--headless=new") 
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# 2. 不使用 Service 对象，让 Selenium 自动管理
# 这可以排除 Service 配置带来的问题
driver = None
try:
    print("Attempting to initialize WebDriver with minimal options...")
    # Selenium Manager resolves a compatible driver from the local environment.
    driver = webdriver.Chrome(options=options)
    
    print("WebDriver initialized successfully!")
    
    driver.get("https://www.google.com")
    print(f"Navigated to: {driver.title}")
    
    driver.save_screenshot("minimal_test_screenshot.png")
    print("Screenshot saved.")

except Exception as e:
    print(f"\n--- TEST FAILED ---")
    print(f"An error occurred: {e}")

finally:
    if driver:
        driver.quit()
        print("WebDriver quit.")

print("--- Minimal WebDriver Test Finished ---")
