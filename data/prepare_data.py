import numpy as np
import json
import os

def split_train_and_eval(origin_file = "./train_origin.json"):
    with open(origin_file, "r") as f:
        content = f.read()
    data = json.loads(content)
    ids = list(data.keys())

    ids_eval = sorted(list(np.random.choice(ids, int(len(ids)/10), replace=False)), key=lambda x:int(x))
    ids_train = [id for id in ids if id not in ids_eval]

    data_train = {id:data[id] for id in ids_train}
    data_eval = {id:data[id] for id in ids_eval}
    with open("./train.json", "w") as f:
        f.write(json.dumps(data_train, ensure_ascii=False))
    with open("./eval.json", "w") as f:
        f.write(json.dumps(data_eval, ensure_ascii=False))

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination)

if __name__ == "__main__":
    if not os.path.exists("eval.json"):
        if not os.path.exists("train_origin.json"):
            if not os.path.exists("MLHomowork_FoodPredictDataset.rar"):
                print("Downloading MLHomowork_FoodPredictDataset.rar ...")
                download_file_from_google_drive(
                    id = "1zpRj2VgNltSDzG9b5-IiMlCgpYzunbXC",
                    destination = "./MLHomowork_FoodPredictDataset.rar"
                )
            os.system("unrar e MLHomowork_FoodPredictDataset.rar")
            os.system("mv train.json train_origin.json")
        split_train_and_eval()
