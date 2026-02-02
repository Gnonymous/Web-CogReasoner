import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

print("--- Starting Minimal WebDriver Test ---")

# 强制清理
import os
os.system("pkill -9 -f 'chromedriver' || true")
os.system("pkill -9 -f 'chrome' || true")

# 1. 使用最简化的 Options
options = Options()
# 在 Docker 或无 GUI 环境中，这三项是必需的
options.add_argument("--headless=new") 
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# 2. 不使用 Service 对象，让 Selenium 自动管理
# 这可以排除 Service 配置带来的问题
driver = None
try:
    print("Attempting to initialize WebDriver with minimal options...")
    # Selenium 4.10+ 会自动下载和管理驱动，但我们可以先用系统路径里的
    # 为了确保使用我们指定的 chromedriver，可以创建一个 Service 对象
    from selenium.webdriver.chrome.service import Service
    service = Service(executable_path='/usr/local/bin/chromedriver', log_output='chromedriver_minimal.log')
    
    driver = webdriver.Chrome(service=service, options=options)
    
    print("WebDriver initialized successfully!")
    
    driver.get("https://www.google.com")
    print(f"Navigated to: {driver.title}")
    
    driver.save_screenshot("minimal_test_screenshot.png")
    print("Screenshot saved.")

except Exception as e:
    print(f"\n--- TEST FAILED ---")
    print(f"An error occurred: {e}")
    print("\nCheck the 'chromedriver_minimal.log' file for detailed driver logs.")

finally:
    if driver:
        driver.quit()
        print("WebDriver quit.")

print("--- Minimal WebDriver Test Finished ---")