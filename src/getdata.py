import os
import requests
import zipfile, io

"""
Creates a  folder named Dataset for all the datasets 
"""
if not os.path.exists('dataset'):
    os.makedirs('dataset')
os.chdir('dataset')


"""
Func:Downloads a zip file and extracts it 
param: url
"""
def download_zip(url):
    req = requests.get(url)
    zip = zipfile.ZipFile(io.BytesIO(req.content))
    zip.extractall()


"""
Function: Fetches the csv file from the url and overwrites any pre existing file with same name 
param: url
"""
def download_csv(url):
    file = requests.get(url)
    if url.find('/'):
        file_name = url.rsplit('/', 1)[1]
    open('OxCGRT_latest.csv', 'wb').write(file.content)


"""
Fetching the most up to data and updating the dataset folder 
"""
download_zip("https://data.icos-cp.eu/licence_accept?ids=%5B%22-fQOdyZAYNWX77K_YVw6r-2d%22%5D&isColl=true")
download_csv('https://github.com/OxCGRT/covid-policy-tracker/raw/master/data/OxCGRT_latest.csv')
