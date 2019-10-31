import requests
import os

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


if not os.path.isdir('0_data'):
    os.mkdir('0_data')

print('Dowloading Camera 1 Example data... ')
download_file_from_google_drive('1qES6teQUprs-cgL-nACzFighjm_Dho0L', '0_data/Cam1.zip')

print('Dowloading Camera 2 Example data... ')
download_file_from_google_drive('1wiz5XPricNj-sFwqfj7tVI1IomlyO6xM', '0_data/Cam2.zip')

print('Dowloading Outdoor Example data... ')
download_file_from_google_drive('1UFah1QWltNDzUlsQ4HCMnuiyAAUh-9EE', '0_data/Outdoor.zip')

os.system('unzip 0_data/Cam1.zip -d 0_data')
os.system('unzip 0_data/Cam2.zip -d 0_data')
os.system('unzip 0_data/Outdoor.zip -d 0_data')
