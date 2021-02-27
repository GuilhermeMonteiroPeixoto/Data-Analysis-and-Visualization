import selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
from time import sleep
from selenium.webdriver.common.keys import Keys
#from selenium.webdriver.chrome.service import Service as ChromeService
#from subprocess import CREATE_NO_WINDOW

#service = ChromeService('path/to/chromedriver')
#service.creationflags = CREATE_NO_WINDOW

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--incognito")
print("\n\n")
informacao = input("Sobre o que você deseja saber? ")

driver = webdriver.Chrome(chrome_options=chrome_options)
driver.get("https://pt.wikipedia.org/wiki/")

sleep(1)
search = driver.find_element_by_xpath('//*[@id="searchInput"]')
search.send_keys(informacao)
click = driver.find_element_by_xpath('//*[@id="searchButton"]')
click.click()

sleep(1)
head1 = driver.find_element_by_xpath('//*[@id="mw-content-text"]/div[1]/p[1]')
head2 = driver.find_element_by_xpath('//*[@id="mw-content-text"]/div[1]/p[2]')
head3 = driver.find_element_by_xpath('//*[@id="mw-content-text"]/div[1]/p[3]')

print(head1.text)
print(head2.text)
print(head3.text)

driver.quit()
