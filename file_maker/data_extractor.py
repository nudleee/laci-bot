from urllib.request import urlopen
from bs4 import BeautifulSoup, NavigableString, Tag
import pandas as pd

def create_soup_from_url(url:str) -> BeautifulSoup:
    html = urlopen(url)
    return BeautifulSoup(html, 'lxml')

def save_csv(data: list, dest:str, separator: str, index: bool): 
    df = pd.DataFrame(data)
    df.to_csv(dest, separator,index=index)

def save_file(data: str, dest:str):
    with open(dest, 'w', encoding='UTF-8') as file:
        file.write(data)

soup = create_soup_from_url('https://www.aut.bme.hu/SzakmaiGyakorlat/GyIK')
questions = soup.find_all('p', {'class': 'question'})

questions_with_answers = []
for q in questions:
    questions_with_answers.append({'question': q.text.strip(), 'answer': q.find_next_sibling().text.strip().replace('\n', '')})

save_csv(questions_with_answers, 'files/FQA.csv', ',', False)


soup = create_soup_from_url('https://www.aut.bme.hu/SzakmaiGyakorlat/Teendok')
table = soup.table
table_rows = table.find_all('tr')

todo_table = []
header = ['Tanszéki felelős', 'Hallgató', 'Cég', 'Mikor']
for tr in table_rows:
    td = tr.find_all('td')
    if len(td) != 4 : continue
    row = {}
    for idx, item in enumerate(td):
        stripped_lines = [line.strip() for line in td[idx].text.strip().splitlines()]
        result_string = ' '.join(stripped_lines)
        row.update({header[idx]: result_string})
    
    todo_table.append(row)
   
save_csv(todo_table, 'files/todos.csv', ',', False)


soup = create_soup_from_url('https://www.vik.bme.hu/kepzes/gyakorlat/442.html')
info = soup.find('div', {'id': 'main'})
save_file(info.text.strip(), 'files/practice.txt')

soup = create_soup_from_url('https://www.aut.bme.hu/SzakmaiGyakorlat/')
info = soup.find('div', {'role': 'main'})
result_string = ''
for item in info:
    text = item.text
    stripped_lines = [line.strip() for line in text.splitlines() if line.strip()]
    lines = '\n'.join(stripped_lines)
    result_string = result_string + lines

save_file(result_string, 'files/aut_practice.txt')