#import main
import pandas as pd
import os
import cvxpy as cp
import numpy as np
import math
from tqdm import tqdm,trange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from cvxpylayers.torch import CvxpyLayer
from collections import Counter
import torch.nn as nn
import argparse
"""
Solve the dual problem to optimality using cvxpy and mosek, and print the optimal function value.
We provide GPU-version and CPU-version
"""


def sigmoid(x):
    return 1/(1+np.exp(-x))



class GPU_layer(nn.Module):
    def __init__(self,  p_size, lambd, rho):
        super(GPU_layer, self).__init__()
        self.rho = rho
        self.A = torch.triu(torch.ones((p_size,p_size)))
        self.d = torch.ones(p_size)
        self.lambd = lambd
        self.p_size = p_size


    def forward(self,x):
        sorted_args = torch.argsort(x*self.rho.to(x.device),dim=-1)
        sorted_x = x.gather(dim=-1,index=sorted_args)

        rho = self.rho.gather(dim=-1,index=sorted_args).cpu()
        answer = cp.Variable(self.p_size)
        para_ordered_tilde_dual = cp.Parameter(self.p_size)
        constraints = []
        constraints += [cp.matmul(cp.multiply(rho,answer),self.A) + self.lambd * self.d >= 0]
        objective = cp.Minimize(cp.sum_squares(cp.multiply(rho,answer) - cp.multiply(rho, para_ordered_tilde_dual)))
        problem = cp.Problem(objective, constraints)
        #assert problem.is_dpp()
        self.cvxpylayer = CvxpyLayer(problem, parameters=[para_ordered_tilde_dual], variables=[answer])

        solution, = self.cvxpylayer(sorted_x)
        re_sort = torch.argsort(sorted_args,dim=-1)
        return solution.to(x.device).gather(dim=-1,index=re_sort)



def CPU_layer(ordered_tilde_dual, rho, lambd):
    m = len(rho)
    answer = cp.Variable(m)
    
    
    objective = cp.Minimize(cp.sum_squares(cp.multiply(rho, answer) - cp.multiply(rho, ordered_tilde_dual)))
    
    
    constraints = []
    for i in range(1, m + 1):
        constraints.append(cp.sum(cp.multiply(rho[:i], answer[:i])) >= -lambd)
    
    
    constraints.append(answer >= -1)  # Lower bound constraint
    constraints.append(answer <= 1)   # Upper bound constraint
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    # Debugging information
    # print("Problem status:", prob.status)
    # if prob.status not in ["optimal", "optimal_inaccurate"]:
    #     print("Problem not solved to optimality. Status:", prob.status)
    #     return None
    
    # print("Optimal value of the objective function:", prob.value)
    # print("Optimal solution:", answer.value)
    
    return answer.value



def compute_next_dual(eta, rho, dual, gradient, lambd):
    tilde_dual = dual - eta*gradient/rho/rho
    order = np.argsort(tilde_dual*rho)
    ordered_tilde_dual = tilde_dual[order]
    # print("tilde_dual:", tilde_dual)
    # print("order:", order)
    # print("ordered_tilde_dual:", ordered_tilde_dual)
    ordered_next_dual = CPU_layer(ordered_tilde_dual, rho[order], lambd)
    return ordered_next_dual[order.argsort()]

def P_MMF_CPU(lambd,args):
    T = args.Time

    trained_preference_scores = np.load("/content/drive/MyDrive/Capstone/Dataset/Office_Products_Filtered_5/result/CCFCRec/best_model_ratings.npy", allow_pickle=True)
    data = np.load("/content/drive/MyDrive/Capstone/Dataset/Office_Products_Filtered_5/test_cold_interactions_provider_formatted.npy", allow_pickle=True)   
    data = pd.DataFrame(data)
    data[3] += 1
    uid_field, iid_field, time_field, provider_field = data.columns     
    n_provider = 1
    provider_map = {-1: 0}
    key = tuple(data[3])
    for interaction in data:
        if key not in provider_map:
            provider_map[key] = n_provider
            n_provider += 1

    num_providers = len(data[provider_field].unique())

    user_num, item_num = np.shape(trained_preference_scores)
    providerLen = np.array(data.groupby(provider_field).size().values)
    rho = (1+1/num_providers)*providerLen/np.sum(providerLen)
    #print("rho", rho)    

    data.sort_values(by=[time_field], ascending=True,inplace=True)
    batch_size = int(len(data)* 0.1//T)

    data_val = np.array(data[uid_field].values[-batch_size*T:]).astype(int)
    UI_matrix = trained_preference_scores[data_val]

    #normalize user-item perference score to [0,1]
    UI_matrix = sigmoid(UI_matrix)
    tmp = data[[iid_field,provider_field]].drop_duplicates()
    item2provider = {x:y for x,y in zip(tmp[iid_field],tmp[provider_field])}

    #A is item-provider matrix
    A = np.zeros((item_num,num_providers))
    for i in range(len(item2provider)):
      if item2provider[i] >= num_providers:
          item2provider[i] = num_providers - 1
            
    iid2pid = []
    for i in range(item_num):
      # if item2provider[i] < num_providers:
        iid2pid.append(item2provider[i])
        A[i, item2provider[i]] = 1 
 
    print("iid2pid", iid2pid)
    #print("len iid2pid", len(iid2pid))    
    W_batch = []
    RRQ_batch, MMF_batch = [], []

    K = args.topk

    for b in trange(batch_size):
        min_index = b * T
        max_index = (b+1) * T
        batch_UI = UI_matrix[min_index:max_index,:]
        nor_dcg = []
        UI_matrix_sort = np.sort(batch_UI,axis=-1)
        for i in range(T):
            nor_dcg.append(0)
            for k in range(K):
                nor_dcg[i] = nor_dcg[i] + UI_matrix_sort[i,item_num-k-1]/np.log2(k+2)

        mu_t = np.zeros(num_providers)
        B_t = T*K*rho
        #print(np.float(B_t>0))
        sum_dual = 0
        result_x = []
        eta = args.eta / np.sqrt(T)
        gradient_cusum = np.zeros(num_providers)
        gradient_list = []
        for t in range(T):
            alpha = args.alpha
            x_title = batch_UI[t,:] - np.matmul(A,mu_t)
            mask = np.matmul(A,(B_t>0).astype(float))

            mask = (1.0-mask) * -10000.0
            x = np.argsort(x_title+mask,axis=-1)[::-1]
            x_allocation = x[:K]
            re_allocation = np.argsort(batch_UI[t,x_allocation])[::-1]
            x_allocation = x_allocation[re_allocation]
            result_x.append(x_allocation)
            B_t = B_t - np.sum(A[x_allocation],axis=0,keepdims=False)
            gradient = -np.mean(A[x_allocation],axis=0,keepdims=False) + B_t/(T*K)

            gradient = alpha * gradient + (1-alpha) * gradient_cusum
            gradient_cusum = gradient
  
            for g in range(1):
                mu_t = compute_next_dual(eta, rho, mu_t, gradient, lambd)
          
            #exit(0)
            sum_dual += mu_t
        ndcg = 0

        base_model_provider_exposure = np.zeros(num_providers)
        #print("base_model_provider_exposure", base_model_provider_exposure)
        result = 0
        for t in range(T):
            dcg = 0
            x_recommended = result_x[t]
            # print("x_recommend", x_recommended)
            # print("iid2pid", iid2pid)
            #x_recommended = np.random.choice(list(range(0,item_num)),size=K,replace=False,p=x_value[t,:]/K)
            for k in range(K):
                base_model_provider_exposure[iid2pid[x_recommended[k]]] += 1
                dcg = dcg + batch_UI[t,x_recommended[k]]/np.log2(k+2)
                result = result + batch_UI[t,x_recommended[k]]

            ndcg = ndcg + dcg/nor_dcg[t]
        ndcg = ndcg/T
        rho_reverse = 1/(rho*T*K)
        MMF = np.min(base_model_provider_exposure*rho_reverse)
        W = result/T + lambd * MMF

        W_batch.append(W)
        RRQ_batch.append(ndcg)
        MMF_batch.append(MMF)

    W, RRQ, MMF = np.mean(W_batch), np.mean(RRQ_batch), np.mean(MMF_batch)
    print("W:%.4f RRQ: %.4f MMF: %.4f "%(W,RRQ ,MMF))
    return W, RRQ, MMF


def P_MMF_GPU(lambd,args):
    T = args.Time
    trained_preference_scores = np.load("/content/drive/MyDrive/Capstone/Dataset/Office_Products_Filtered_5/result/CCFCRec/best_model_ratings.npy", allow_pickle=True)
    data = np.load("/content/drive/MyDrive/Capstone/Dataset/Office_Products_Filtered_5/test_cold_interactions_provider_formatted.npy", allow_pickle=True)   
    data = pd.DataFrame(data)
    data[3] += 1
    uid_field, iid_field, time_field, provider_field = data.columns     
    num_providers = len(data[provider_field].unique())
    user_num, item_num = np.shape(trained_preference_scores)
    providerLen = np.array(data.groupby(provider_field).size().values)
    rho = (1+1/num_providers)*providerLen/np.sum(providerLen)
        

    data.sort_values(by=[time_field], ascending=True,inplace=True)
    batch_size = int(len(data)* 0.1//T)

    data_val = np.array(data[uid_field].values[-batch_size*T:]).astype(int)
    UI_matrix = trained_preference_scores[data_val]

    #normalize user-item perference score to [0,1]
    UI_matrix = sigmoid(UI_matrix)
    tmp = data[[iid_field,provider_field]].drop_duplicates()
    item2provider = {x:y for x,y in zip(tmp[iid_field],tmp[provider_field])}

    #A is item-provider matrix
    A = np.zeros((item_num,num_providers))
    iid2pid = []
    for i in range(item_num):
        iid2pid.append(item2provider[i])
        A[i,item2provider[i]] = 1
    W_batch = []
    RRQ_batch, MMF_batch = [], []

    K = args.topk
    device = 'cuda'
    rho = torch.FloatTensor(rho).to(device)

    update_mu_function = GPU_layer(p_size=num_providers,lambd=lambd,rho=rho)


    # mu_t = torch.zeros((batch_size,num_provider)).to(device)
    # b_t = torch.FloatTensor(np.array([T * rho * K for j in range(batch_size)])).to(device)
    A_sparse = torch.FloatTensor(A).to(device).to_sparse()
    A = torch.FloatTensor(A).to(device)

    for b in trange(batch_size):
        min_index = b * T
        max_index = (b+1) * T
        batch_UI = UI_matrix[min_index:max_index,:]
        nor_dcg = []
        UI_matrix_sort = np.sort(batch_UI,axis=-1)
        for i in range(T):
            nor_dcg.append(0)
            for k in range(K):
                nor_dcg[i] = nor_dcg[i] + UI_matrix_sort[i,item_num-k-1]/np.log2(k+2)

        batch_UI = torch.FloatTensor(batch_UI).to(device)
        mu_t = torch.zeros(num_providers).to(device)
        B_t = T*K*rho
        #print(np.float(B_t>0))
        sum_dual = 0
        result_x = []
        eta = args.eta / np.sqrt(T)
        gradient_cusum = torch.zeros(num_providers).to(device)

        for t in range(T):
            alpha = args.alpha
            x_title = batch_UI[t,:] - A_sparse.matmul(mu_t.t()).t()
            mask = A_sparse.matmul((B_t>0).float().t()).t()

            mask = (1.0-mask) * -10000.0
            values,items = torch.topk(x_title+mask,k=K,dim=-1)
            #x = np.argsort(x_title+mask,axis=-1)[::-1]
            #
            re_allocation = torch.argsort(batch_UI[t,items],descending=True)
            x_allocation = items[re_allocation]
            result_x.append(x_allocation)
            B_t = B_t - torch.sum(A[x_allocation],dim=0,keepdims=False)
            gradient_tidle = -torch.mean(A[x_allocation],dim=0,keepdims=False) + B_t/(T*K)

            gradient = alpha * gradient_tidle + (1-alpha) * gradient_cusum
            gradient_cusum = gradient

            for g in range(1):
                mu_t = update_mu_function(mu_t-eta*gradient/rho/rho)
            #print(mu_t)
            #exit(0)
            sum_dual += mu_t
        ndcg = 0

        base_model_provider_exposure = np.zeros(num_providers)
        result = 0
        for t in range(T):
            dcg = 0
            x_recommended = result_x[t].cpu().detach().numpy().astype(int)
            #x_recommended = np.random.choice(list(range(0,item_num)),size=K,replace=False,p=x_value[t,:]/K)
            for k in range(K):
                base_model_provider_exposure[iid2pid[x_recommended[k]]] += 1
                dcg = dcg + batch_UI[t,x_recommended[k]]/np.log2(k+2)
                result = result + batch_UI[t,x_recommended[k]]

            ndcg = ndcg + dcg/nor_dcg[t]
        ndcg = ndcg/T
        ndcg = ndcg.cpu().numpy()
        rho_reverse = 1/(rho*T*K)
        MMF = np.min(base_model_provider_exposure*rho_reverse.cpu().numpy())
        W = result/T + lambd * MMF
        W = W.cpu().numpy()

        W_batch.append(W)
        RRQ_batch.append(ndcg)
        MMF_batch.append(MMF)

    W, RRQ, MMF = np.mean(W_batch), np.mean(RRQ_batch), np.mean(MMF_batch)
    print("W:%.4f RRQ: %.4f MMF: %.4f "%(W,RRQ ,MMF))
    return W, RRQ, MMF


