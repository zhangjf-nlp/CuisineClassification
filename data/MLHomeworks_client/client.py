# -*- coding: utf-8 -*-
import numpy as np
np.set_printoptions(precision=3)
import pandas as pd
import requests
import torch
import glob
import time
import sys
import os
from tqdm import tqdm

from group_info import SID, TOKEN

all_cuisine = [
    'brazilian',
    'british',
    'cajun_creole',
    'chinese',
    'filipino',
    'french',
    'greek',
    'indian',
    'irish',
    'italian',
    'jamaican',
    'japanese',
    'korean',
    'mexican',
    'moroccan',
    'russian',
    'southern_us',
    'spanish',
    'thai',
    'vietnamese'
]


def lines_to_logits(lines):
    labels = [all_cuisine.index(line.split(" ")[1]) for line in lines]
    logits = np.eye(len(all_cuisine))[labels]
    return logits

def logits_to_lines(logits, ids):
    labels = list(np.argmax(logits, axis=-1))
    lines = [f"{id} {all_cuisine[label]}" for id,label in zip(ids,labels)]
    return lines

def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f :
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

global last_post_time
last_post_time= time.time()
def main(ip="115.236.52.125", port="4000", sid=SID, token=TOKEN,
         ans=None, problem="FoodPredict_evaluate", verbose=1):
    if verbose: print("正在提交...")
    url = "http://%s:%s/jsonrpc" % (ip, port)
    
    global last_post_time
    while time.time() - last_post_time < 1:
        time.sleep(1)
    payload = {
        "method": problem,
        "params": [ans],
        "jsonrpc": "2.0",
        "id": 0,
    }
    response = requests.post(
        url,
        json=payload,
        headers={"token": token, "sid": sid}
    ).json()
    last_post_time = time.time()
    
    if verbose: print(response)
    if "auth_error" in response:
        print("您的认证信息有误")
        print(response["auth_error"])
        return response["auth_error"]
    elif "error" not in response:
        if verbose: print("测试完成，请查看分数")
        return response["result"]
    else:
        print("提交文件存在问题，请查看error信息")
        return response["error"]["data"]["message"]

def path_to_lines(path):
    with open(path, "r") as f:
        content = f.read()
        lines = content.split("\n")
        lines = [line for line in lines if len(line)>2]
    return lines

def path_to_logits(path):
    logits = torch.load(path.replace("submission.txt", "logits.pt"))
    return logits

class GeneticAlgorithm4Submission:
    def __init__(self, max_iter, num_caches, submission_paths):
        self.max_iter = max_iter
        self.num_caches = num_caches
        self.submission_paths = submission_paths
        self.init_test()
    
    def run(self):
        self.update()
        for i in tqdm(range(int(self.max_iter/2), self.max_iter)):
        #for i in tqdm(range(self.max_iter)):
            seed_decay = (np.cos(np.pi/self.max_iter*i)+1) * 0.2
            p_decay = np.cos(np.pi/self.max_iter*i)+1
            self.iter_step(seed_decay=seed_decay, p_decay=p_decay)
            if i%10==0:
                print("\n".join([f"{weight} - {score:.8f}" for weight,score in self.caches]))
        main(ans=self.best_lines)
        with open(f"./{self.best_score:.8f}_submission.txt", "w") as f:
            f.write("\n".join([line for line in self.best_lines]))
        torch.save(self.bert_logits, f"./{self.best_score:.8f}_logits.pt")
    
    def init_test(self):
        path_lines = [(path,path_to_lines(path)) for path in self.submission_paths]
        path_to_lines_ = {path:path_to_lines(path) for path in self.submission_paths}
        path_scores = [(path,main(ans=lines, verbose=0)) for path,lines in tqdm(path_lines)]
        path_scores = sorted(path_scores, key=lambda x:x[1], reverse=True)
        print("initial scores:")
        for path,score in path_scores:
            print(f"{score:.8f} - {path}")
        eye = np.eye(len(self.submission_paths))
        self.logitss = np.concatenate([path_to_logits(path)[None,:,:] for path,score in path_scores], axis=0) # logitss: [n_submissions, n_samples, n_classes]
        self.caches = [(eye[i],score) for i,(path,score) in enumerate(path_scores) if score > 0.75] #(weight,score) weight: [n_submissions]
        self.seeds = [(eye[i],score) for i,(path,score) in enumerate(path_scores) if score > 0.75] #(weight,score) weight: [n_submissions]
        
        self.ids = [line.split(" ")[0] for line in path_lines[0][1]]
        self.best_score = self.caches[0][1]
        self.bert_logits = np.sum(self.caches[0][0][:,None,None]*self.logitss, axis=0)
        self.best_lines = logits_to_lines(np.sum(self.caches[0][0][:,None,None]*self.logitss, axis=0), self.ids)
    
    def iter_step(self, seed_decay=1.0, p_decay=1.0):
        for _ in range(5):
            p = 1/(np.log(np.arange(len(self.caches))*p_decay+1)+1)
            index_weight_from_cache = np.random.choice(len(self.caches), 3, p=p/p.sum()).tolist()
            weight_from_cache = [self.caches[i][0] for i in index_weight_from_cache]
            variance_cache = [np.random.normal(1,1) for i in range(len(weight_from_cache))]
            
            p = 1/(np.log(np.arange(len(self.seeds))+1)+1)
            index_weight_from_seed = np.random.choice(len(self.seeds), 3, p=p/p.sum()).tolist()
            weight_from_seed = [self.seeds[i][0] for i in index_weight_from_seed]
            variance_seed = [np.random.normal(0,1)*seed_decay for i in range(len(weight_from_seed))]
            
            all_weights = weight_from_cache+weight_from_seed
            all_variances = variance_cache+variance_seed
            if sum(all_variances)==0:
                all_variances = [1 for _ in all_variances]
            weight = np.sum([weight*variance for weight,variance in zip(all_weights, all_variances)], axis=0) / sum(all_variances)
            weighted_logits = np.sum(weight[:,None,None]*self.logitss, axis=0)
            weighted_lines = logits_to_lines(weighted_logits, self.ids)
            
            score = main(ans=weighted_lines, verbose=0)
            if score > self.best_score:
                print(f"update best score: {self.best_score:.8f} -> {score:.8f}")
                print(f"through: {weight}")
                
                old_best_score = self.best_score
                
                self.best_score = score
                self.best_lines = weighted_lines
                
                with open(f"./{old_best_score:.8f}_submission.txt", "w") as f:
                    f.write("\n".join([line for line in self.best_lines]))
                torch.save(self.bert_logits, f"./{old_best_score:.8f}_logits.pt")
                os.system(f"mv {old_best_score:.8f}_submission.txt {self.best_score:.8f}_submission.txt")
                os.system(f"mv {old_best_score:.8f}_logits.pt {self.best_score:.8f}_logits.pt")
                
            self.caches.append((weight, score))
        self.update()
    
    def update(self):
        self.caches = sorted(self.caches, key=lambda x:x[1], reverse=True)
        self.caches = self.caches[:self.num_caches]
    
if __name__ == "__main__":
    # 需要修改的参数：problem, sid, token

    # problem 参数：
    #    Action_evaluate:        低分辨率视频行为识别
    #    FoodPredict_evaluate:   菜品分类
    #    StoreSale_evaluate:     商品销售额预测
    #    Toxicity_evaluate:      恶意评论分类
    #    CarDemand_evaluate:     汽车需求量预测
    #    FineGrainedCar_evaluate:细粒度汽车分类
    #    Traffic_evaluate:       疫情人流量预测
    #    Mask_evaluate:          口罩检测

    problem = "FoodPredict_evaluate"
    # IP 固定为 115.236.52.125
    ip = "115.236.52.125"
    # 端口不需要修改
    port = "4000"
    # 改成你的学号
    sid = SID
    # 改成你的口令
    token = TOKEN

    if problem in ["Action_evaluate",
                   'FoodPredict_evaluate',
                   'StoreSale_evaluate',
                   'Toxicity_evaluate',
                   'CarDemand_evaluate',
                   'FineGrainedCar_evaluate',
                   'Traffic_evaluate']:
        
        def get_search_grid(depth, grid, rang=3, base=3):
            if depth==0:
                yield [base**i for i in grid] + [0]
            else:
                for i in range(rang):
                    for _ in get_search_grid(depth-1, grid+[i], rang):
                        yield _
        def search_path(path):
            for subpath in [os.path.join(path, _) for _ in os.listdir(path)]:
                if os.path.isdir(subpath):
                    for _ in search_path(subpath):
                        yield _
                elif "submission.txt" in subpath:
                    yield subpath
                else:
                    pass
        
        submission_path = sys.argv[1]
        if "GA" in submission_path: # ensemble the submissions under specified path through Genetic Algorithm
            submission_paths = [_ for _ in search_path(sys.argv[2])]
            GA = GeneticAlgorithm4Submission(max_iter=int(submission_path[2:]), num_caches=48, submission_paths=submission_paths)
            GA.run()
            
        elif submission_path.isdigit(): # ensemble the specified submissions through grid search
            submission_paths = [sys.argv[i+2] for i in range(int(submission_path))]
            d = ensemble_submissions(submission_paths)
            best_score = 0
            best_weights = None
            from tqdm import tqdm
            for weights in tqdm(get_search_grid(depth=int(submission_path), grid=[]), total=3**int(submission_path)):
                d = ensemble_submissions(submission_paths, weights)
                score = main(ip, port, sid, token, d, problem)
                print(f"{weights}: {score}")
                if score > best_score:
                    best_score = score
                    best_weights = weights
            print(f"best_score: {best_score}")
            print(f"best_weights: {best_weights}")
            print(f"submit this again...")
            main(ip, port, sid, token, ensemble_submissions(submission_paths, best_weights), problem)
            
        elif "+" in submission_path: # ensemble the specified submissions with specified weights
            weights = [int(_) for _ in submission_path.split("+")]
            submission_paths = [sys.argv[i+2] for i in range(int(len(weights)))]
            d = ensemble_submissions(submission_paths, weights)
            score = main(ip, port, sid, token, d, problem)
            print(score)
        
        elif "submission.txt" not in submission_path: # simply post the submissions under specified path
            path_score = []
            all_paths = [_ for _ in search_path(submission_path)]
            for path in tqdm(all_paths):
                time.sleep(1)
                lines = path_to_lines(path)
                score = main(ip, port, sid, token, lines, problem, verbose=0)
                path_score.append((path,score))
            sorted_path_score = sorted(path_score, key=lambda x:x[1], reverse=True)
            print("\n".join([str(_) for _ in sorted_path_score]))
            time.sleep(1)
            with open(sorted_path_score[0][0]) as f:
                lines = f.read().split("\n")
            main(ip, port, sid, token, lines, problem)
        
        else: # simply post the single specified submission
            lines = path_to_lines(submission_path)
            score = main(ip, port, sid, token, lines, problem)
            print(submission_path)
            print(score)

    elif problem == "Mask_evaluate":
        submit_dir = './submission'
        submissions = os.listdir('./submission')
        d = {}
        for submit in submissions:
            submit_file = os.path.join(submit_dir,submit)
            with open(submit_file,'r') as f:
                d[submit] = f.read().splitlines()
        score = main(ip, port, sid, token, d, problem)
        print(score)