import os
import json
import time
from selenium import common
from selenium import webdriver


class Poem:

    #VALIDLABELS = ['LOVE', 'NATURE', 'SOCIAL COMMENTARIES', 'RELIGON', 'LIVING', 'RELATIONSHIPS',
    # 'ACTIVITIES', 'ARTS & SCIENCE', 'MYTHOLOGY & FLOKLORE']

    def __init__(self, title, author, content, label):
        self.title = title
        self.author = author
        self.content = content
        self.label = label

    def clean(self):
        newLabels = [entity for entity in self.label if not (entity == "")]
        self.label = newLabels
        return self


class PoemScrapy:

    '''
    Initilize 2 driverChrome where
    [1] self.driver: scrapy the outer url, with title, author, and its labels
    [2] self.driverPoem : scrapy the inner url with the poem's content
    '''
    def __init__(self, driverPath):
        self.driver = webdriver.Chrome(executable_path=driverPath)
        self.driverPoem = webdriver.Chrome(executable_path=driverPath)

    def scrapy(self, outerUrl):
        self.driver.get(outerUrl)
        # sleep for at least 10 seconds in order to avoid potential conflict
        time.sleep(10)
        resPoemList = []
        try:
            innerPoemList = self.driver.find_elements_by_css_selector('div.c-feature.c-mix-feature_shrinkwrap')
        except Exception:
            innerPoemList = []

        count = 0
        for poemEntity in innerPoemList:

            count += 1
            if count == 10:
                time.sleep(2)

            try:
                # outer information extraction
                author = poemEntity.find_element_by_css_selector('span.c-txt.c-txt_attribution').text
                title = poemEntity.find_element_by_css_selector('div.c-feature-hd').text
                innerUrl = poemEntity.find_element_by_css_selector('div.c-feature-hd').find_element_by_tag_name('a').get_attribute('href')
                webLabels = poemEntity.find_element_by_css_selector('span.u-obscuredAfter8').find_elements_by_tag_name('span')
                labels = [webText.text for webText in webLabels]

                if not labels:
                    continue

                # inner information extraction
                self.driverPoem.get(innerUrl)
                content = self.driverPoem.find_element_by_css_selector('div.o-poem.isActive').text
                resPoemList.append(Poem(title, author, content, labels).clean())

            except common.exceptions.NoSuchElementException:
                continue
            except common.exceptions.TimeoutException:
                self.driver = webdriver.Chrome(executable_path=driverPath)
                self.driverPoem = webdriver.Chrome(executable_path=driverPath)
                time.sleep(10)
                break
            except common.exceptions.NoSuchWindowException:
                self.driver = webdriver.Chrome(executable_path=driverPath)
                self.driverPoem = webdriver.Chrome(executable_path=driverPath)
                time.sleep(10)
                break

        return resPoemList

if __name__ == '__main__':

    driverPath = os.path.abspath(r'E:\Poetry\chromedriver.exe')
    startPage, endPage = 1, 2000
    outputJsonPrefix = 'result_'
    outputJsonSuffix = '_json.json'

    urlPrefix = r'https://www.poetryfoundation.org/poems/browse#page='
    urlSuffix = r'&sort_by=recently_added'

    poemDriver = PoemScrapy(driverPath=driverPath)
    for pageNumber in range(startPage, endPage):
        url = urlPrefix + str(pageNumber) + urlSuffix
        outputJson = outputJsonPrefix + str(pageNumber) + outputJsonSuffix
        poemList = poemDriver.scrapy(outerUrl=url)

        with open(outputJson, 'w') as jsonfile:
            dataPoems = [{
                'title': poem.title,
                'author': poem.author,
                'content': poem.content,
                'label': poem.label,
            } for poem in poemList]
            json.dump(dataPoems, jsonfile, indent=4)
    time.sleep(1)

