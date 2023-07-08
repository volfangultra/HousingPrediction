#Ovaj fajl uzme te linkove i prolazi kroz njih iz iz svakog izvuce sta mu treba i stavi to u google sheets.

import pandas as pd
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
import gspread
from google.oauth2.service_account import Credentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Namjestiti googleSheets
scopes = ['https://www.googleapis.com/auth/spreadsheets',
          'https://www.googleapis.com/auth/drive']
credentials = Credentials.from_service_account_file("driven-stage-388723-c5f88347ee1f.json", scopes=scopes)
gc = gspread.authorize(credentials)
gauth = GoogleAuth()
drive = GoogleDrive(gauth)
gs = gc.open_by_key('12vvKPVZZOyVXwoa5MephdExISR6s2p_gavjEIWQBo80')
worksheet1 = gs.worksheet('Sheet1')

# Uzeti sve linkove zgrada iz fajla
file1 = open("linkovi_zgrada.txt", "r")
linije = file1.readlines()
options = Options()
options.add_argument("--headless")
service = Service()
driver = webdriver.Chrome(service=service, options=options)


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


for i in range(len(linije)):
    linije[i] = linije[i].rstrip()


brojac = 0
# Iz svake zgrade izvuci sve bitne podatke te vecinu pretvoriti u odgovarajuci format
for url in linije:
    driver.get(url)
    driver.implicitly_wait(5)

    broj_soba = 1
    namjesten = 0
    sprat = 0
    vrsta_grijanja = "nista"
    kvadrata = 0
    godina_izgradnje = 0
    lift = 0
    parking = 0
    cijena = 0

    try:
        required_atributes = driver.find_element(By.CLASS_NAME, "required-attributes").find_elements(By.CLASS_NAME,
                                                                                                     "required-wrap")
    except Exception:
        continue
    for atribute in required_atributes:
        temp = atribute.text.lower()
        if 'kvadrata' in temp:
            if is_float(temp.split(" ")[1]):
                kvadrata = float(temp.split(" ")[1])

        if 'broj soba' in temp:
            broj_soba = temp.split(" ")[2].lower()
            if broj_soba == 'garsonjera':
                broj_soba = 0
            elif broj_soba == 'jednosoban':
                broj_soba = 1
            elif broj_soba == "jednoiposoban":
                broj_soba = 1.5
            elif broj_soba == "dvosoban":
                broj_soba = 2
            elif broj_soba == 'dvoiposoban':
                broj_soba = 2.5
            elif broj_soba == "trosoban":
                broj_soba = 3
            elif broj_soba == "troiposoban":
                broj_soba = 3.5
            elif broj_soba == 'cetverosoban':
                broj_soba = 4
            elif broj_soba == "cetveroiposoban":
                broj_soba = 4.5
            elif broj_soba == "petosoban":
                broj_soba = 5

            elif broj_soba.isnumeric():
                broj_soba = int(broj_soba)
        if 'namješten' in temp:
            if temp.split(" ")[1] == "namješten":
                namjesten = 2
            elif temp.split(" ")[1] == "polunamješten":
                namjesten = 1
        if 'sprat' in temp:
            sprat = temp.split(" ")[1]
            if sprat.isnumeric():
                sprat = int(sprat)

        if 'vrsta grijanja' in temp:
            vrsta_grijanja = temp.split(" ")[2].lower()

    table = driver.find_element(By.TAG_NAME, "table").find_elements(By.TAG_NAME, "tr")
    for element in table:
        temp = element.text.lower()
        if 'godina izgradnje' in temp:
            godina_izgradnje = temp.split(" ")[2].lower()
            if (godina_izgradnje == 'novogradnja') or (godina_izgradnje == "2015+") or (godina_izgradnje == "2015"):
                godina_izgradnje = 1
            else:
                godina_izgradnje = 0

        if 'lift' in temp:
            if temp.split(" ")[1] == '✓':
                lift = 1

        if 'parking' in temp:
            if temp.split(" ")[1] == '✓':
                parking = 1

    grad = driver.find_element(By.XPATH,
                               '//*[@id="__layout"]/div/div[1]/div[2]/div/div/div[2]/div[2]/div[1]/div[2]/div/div['
                               '1]/div[3]/label[1]').text

    cijena = driver.find_element(By.CLASS_NAME, 'price-heading').text.split(" ")[0]
    if is_float(cijena):
        cijena = cijena.replace(".", "")
        cijena = int(cijena)

    brojac += 1
    print(grad, kvadrata, sprat, broj_soba, vrsta_grijanja, lift, parking, godina_izgradnje, namjesten, cijena, brojac)

    df = pd.DataFrame({'grad': [grad], 'kvadrata': [kvadrata], 'sprat': [sprat], 'broj_soba': [broj_soba],
                       'vrsta_grijanja': [vrsta_grijanja], 'lift': [lift], 'parking': [parking],
                       'novogradnja': [godina_izgradnje], 'namjesten': [namjesten], 'cijena': [cijena]})
    df_values = df.values.tolist()
    gs.values_append('Sheet1', {'valueInputOption': 'RAW'}, {'values': df_values})

file1.close()
driver.close()
