import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import *
def createC1(dataset):
    """
    Create C1 itemset from the dataset.
    """
    C1 = []
    for transaction in dataset:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))

def scanD(D, Ck, min_support):
    """
    Scan the dataset D to find all frequent itemsets of size k.
    """
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= min_support:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData

def aprioriGen(Lk, k):
    """
    Generate all possible itemsets of size k from frequent itemset Lk.
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            L1 = tuple(L1)
            L2 = tuple(L2)
            if L1 == L2:
                retList.append(Lk[i]|Lk[j])
    return retList

def apriori(dataset, min_support=0.5,n=3):
    """
    Apriori algorithm.
    """
    C1 = createC1(dataset)
    D = list(map(set, dataset))
    L1, supportData = scanD(D, C1, min_support)
    L = [L1]
    for k in range(2, n+2):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, min_support)
        supportData.update(supK)
        L.append(Lk)
    return L, supportData

def generateRules(L, supportData, minConf=0.7):
    """
    Generate association rules from frequent itemsets.
    """
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    """
    Calculate the confidence of a rule.
    """
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    """
    Generate rules from frequent itemsets of size k-1.
    """
    m = len(H[0])
    if (len(freqSet) > (m+1)):
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

def FeatureSelection(dataset, min_support=0.005, n=3, minConf=0.1):
    L, supportData = apriori(dataset, min_support=min_support, n=n)
    rules = generateRules(L, supportData, minConf=minConf)
    Investment = {};OutCome = {}
    data = pd.DataFrame(columns=['NOC', 'Sport', 'Investment'])
    for rule in rules:
        member1, member2, conf = map(lambda x: list(x)[0] if not isinstance(x, float) else x, rule)
        if member1 in noc_to_full_name.values():
            Investment.update({member1: (member2, conf)})
        else:
            data = data._append({'NOC': member2, 'Sport': member1, 'Investment': conf}, ignore_index=True)
            OutCome.update({member2: (member1, conf)})
    data.to_csv('./data/influence.csv', index=False)
    return rules,Investment,OutCome


if __name__ == '__main__':
    dataset = pd.read_csv('./data/summerOly_athletes.csv', encoding='utf-8')
    dataset = dataset[dataset['Medal'] != 'No medal']
    dataset = dataset[dataset['Year'] == 2024]
    dataset['NOC'] = dataset['NOC'].apply(noc_to_country)
    dataset = dataset[['NOC','Sport']].to_numpy().tolist()
    rules,Investment,OutCome = FeatureSelection(dataset, min_support=0.01, n=3, minConf=0.1)
