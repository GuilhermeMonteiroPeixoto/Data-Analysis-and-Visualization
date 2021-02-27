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

def print_arquivo_texto(string1, string2, string3, string4):
    file = open('arquivo_saida_scrap.txt','w+')
    file.write(string4.upper()+'\n\n')
    file.write('R: '+string1+'\n')
    file.write(string2+'\n')
    file.write(string3+'\n')
    file.close()

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--incognito")

def leitura():
    print("Tell me a topic and I'll search on Wikipedia!\n")
    informacao = input("What do you want to know about? ")
    return informacao

def main_func():

    print("\n\n")
    informacao = leitura()

    driver = webdriver.Chrome(chrome_options=chrome_options)
    driver.get("https://en.wikipedia.org/wiki/")

    sleep(1)
    search = driver.find_element_by_xpath('//*[@id="searchInput"]')
    search.send_keys(informacao)
    click = driver.find_element_by_xpath('//*[@id="searchButton"]')
    click.click()

    sleep(1)
    head1 = driver.find_element_by_xpath('//*[@id="mw-content-text"]/div[1]/p[1]')
    head2 = driver.find_element_by_xpath('//*[@id="mw-content-text"]/div[1]/p[2]')
    head3 = driver.find_element_by_xpath('//*[@id="mw-content-text"]/div[1]/p[3]')

    print_arquivo_texto(head1.text, head2.text, head3.text, informacao)

    driver.quit()

main_func()
