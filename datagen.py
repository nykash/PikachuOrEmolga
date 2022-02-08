#Imports Packages
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
#Opens up web driver and goes to Google Images

train = True

pokemon = "pikachu"
q = "image" if train else "drawings"

driver = webdriver.Chrome(executable_path=r"C:/Users/dassu/Downloads/chromedriver_win32/chromedriver.exe")
driver.get('https://images.google.com/?gws_rd=ssl')

box = driver.find_element_by_xpath('//*[@id="sbtc"]/div[2]/div[2]/input')

box.send_keys('{pokemon} {q}')
box.send_keys(Keys.ENTER)

last_height = driver.execute_script('return document.body.scrollHeight')
while True:
    driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
    time.sleep(2)
    new_height = driver.execute_script('return document.body.scrollHeight')
    try:
        driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[5]/input').click()
        time.sleep(2)
    except:
        pass
    if new_height == last_height:
        break
    last_height = new_height

for i in range(100):
    try:
        driver.find_element_by_xpath('//*[@id="islrg"]/div[1]/div['+str(i)+']/a[1]/div[1]/img').screenshot(f"test/{pokemon}/{i}.png")
    except:
        pass

while(True):
    pass
