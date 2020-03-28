import requests
import tqdm
import tarfile
import src
import os

archive_link = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
archive_path = 'cifar100.tar.gz'

response = requests.get(archive_link, stream=True)

block_size = 1024

if response.status_code == 200:
    total_size = int(response.headers['content-length'])
    progress_bar = tqdm.tqdm(total=total_size, unit='block')

    with open(archive_path, 'wb') as archive:
        for block in response.iter_content(block_size):
            archive.write(block)
            progress_bar.update(len(block))

with tarfile.open(archive_path) as archive:
    archive.extractall()

os.remove(archive_path)

