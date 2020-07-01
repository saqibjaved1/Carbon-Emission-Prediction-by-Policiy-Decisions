import requests
import zipfile, io


class Dataloader:
    def __init__(self,url):
        self.url = url

    # Download and extract zip file
    def download_zip(self):
        req = requests.get(self.url)
        zip = zipfile.ZipFile(io.BytesIO(req.content))
        zip.extractall()

    #Download any other file type like csv,tsv,xlsx,etc
    def download_csv(self):
        file = requests.get(self.url)
        if self.url.find('/'):
            file_name = self.url.rsplit('/', 1)[1]
        open(file_name, 'wb').write(file.content)







