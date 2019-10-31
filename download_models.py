import os
import requests

from config import CHECKPOINT_DIR

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


if not os.path.isdir(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

print('Dowloading Trained Model (63Mb)...')
download_file_from_google_drive('1yXeEh2zbP4NQ9ogOO-r7GO9pdrV7yXzr', CHECKPOINT_DIR + '/checkpoint')
download_file_from_google_drive('1yl3mMkvXBZf19XoM38lmDyUuJkYt-Rgb', CHECKPOINT_DIR + '/model.ckpt.index')
download_file_from_google_drive('1YQP0zzbkGH-EaqU3eMIX6MpWB7dD3l6l', CHECKPOINT_DIR + '/model.ckpt.meta')
download_file_from_google_drive('1YbiBNm2iIRuSm4Jb3xSVJ5UrJspIE9cw', CHECKPOINT_DIR + '/model.ckpt.data-00000-of-00001')
print('Done.')