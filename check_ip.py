import requests
import time

def check():
    url = "https://checkip.amazonaws.com/"
    payload = {}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
    print(response.text)
    return response.text

while True :
    check()
    time.sleep(2)
    




