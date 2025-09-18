import requests

url = "http://127.0.0.1:8000/predict"
with open(r"D:\\cow-breed-final\\Brown_Swiss_8_jpg.rf.a5610e202f315f18a65708199ceb3e39.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.json())


# import requests

# url = "http://127.0.0.1:8000/predict"
# with open(r"D:\\cow-breed-final\Brown_Swiss_8_jpg.rf.a5610e202f315f18a65708199ceb3e39.jpg", "rb") as f:
#     files = {"file": f}
#     response = requests.post(url, files=files)

# print(response.json())
