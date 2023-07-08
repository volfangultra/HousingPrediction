#OVaj fajl povlači linkove za sve stanove na piku


from selenium.webdriver.common.by import By
from selenium import webdriver


# pokreniti Selenium driver
driver = webdriver.Chrome()

# Otvoriti stranicu
driver.get("https://olx.ba")

# Postaviti HTML page(uzet rucno) jer nije moguce prebacivati stranice
html_content = """"""
driver.execute_script("document.documentElement.innerHTML = arguments[0];", html_content)

file1 = open("linkovi_zgrada.txt", "a")

#pronaci kartice artikala
listings_container = driver.find_element(By.CLASS_NAME, "articles")
listings = listings_container.find_elements(By.CLASS_NAME, "cardd")

brojac = 0
#pronaci link gdje svaka kartica vodi i te linkove staviti u fajl
for listing in listings:
    title_element = listing.find_element(By.TAG_NAME, "a")
    link = title_element.get_attribute("href")
    file1.write(link + '\n')


driver.quit()
file1.close()