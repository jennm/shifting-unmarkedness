"""
Running this file obtains the words that distinguish a target group from the corresponding
unmarked ones.
Example usage: (To obtain the words that differentiate the 'Asian F' category)
python3 marked_words.py ../generated_personas.csv --target_val 'an Asian' F --target_col race gender --unmarked_val 'a White' M
"""

import pandas as pd
import numpy as np
from collections import Counter
import argparse
from collections import defaultdict
import heapq
import math
import sys

def get_log_odds(df1, df2, df0,verbose=False,lower=True, prior=True, frac_words=1):
    """Monroe et al. Fightin' Words method to identify top words in df1 and df2
    against df0 as the background corpus"""

    overall_common_words = {'the', 'of', 'and', 'to', 'a', 'in', 'for', 'is', 'on', 'that', 'by', 'this', 'with', 'i', 'you', 'it', 'not', 'or', 'be', 'are', 'from', 'at', 'as', 'your', 'am', 'an', 'my', 'are'}

    if lower:
        counts1 = defaultdict(int,[[i,j] for i,j in df1.str.lower().str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
        counts2 = defaultdict(int,[[i,j] for i,j in df2.str.lower().str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
        prior = defaultdict(int,[[i,j] for i,j in df0.str.lower().str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
    else:
        counts1 = defaultdict(int,[[i,j] for i,j in df1.str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
        counts2 = defaultdict(int,[[i,j] for i,j in df2.str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
        prior = defaultdict(int,[[i,j] for i,j in df0.str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
    
    sigmasquared = defaultdict(float)
    sigma = defaultdict(float)
    delta = defaultdict(float)

    reg1 = sum(prior.values())
    reg2 = sum(prior.values())


    prior_min_heap = []
    counts1_min_heap = []
    counts2_min_heap = []
    num_common_words = 20

    for word in prior.keys():
        prior[word] = float(prior[word] + 0.5) #/ reg 
        if not prior:
            prior[word] = 0
        else:
            heapq.heappush(prior_min_heap, (prior[word], word))
            if len(prior_min_heap) > num_common_words:
                heapq.heappop(prior_min_heap)

    for word in counts1.keys():
        counts1[word] = int(counts1[word] + 0.5)
        if prior and prior[word] == 0:
            prior[word] = float(1)
        else:
            heapq.heappush(counts1_min_heap, (counts1[word], word))
            if len(counts1_min_heap) > num_common_words:
                heapq.heappop(counts1_min_heap)


    for word in counts2.keys():
        counts2[word] = int(counts2[word] + 0.5)
        if prior and prior[word] == 0:
            prior[word] = float(1) #/ reg
        else:
            heapq.heappush(counts2_min_heap, (counts2[word], word))
            if len(counts2_min_heap) > num_common_words:
                heapq.heappop(counts2_min_heap)

    n1 = sum(counts1.values())
    n2 = sum(counts2.values())
    nprior = sum(prior.values()) 

    if n1 == 0 or n2 == 0:
        return delta
    prior_top_words_set = set(map(lambda x: x[1], prior_min_heap))
    counts1_top_words_set = set(map(lambda x: x[1], counts1_min_heap))
    counts2_top_words_set = set(map(lambda x: x[1], counts2_min_heap))
    common_words = (prior_top_words_set & counts1_top_words_set & counts2_top_words_set) | overall_common_words

    c = 0.005078125

    p_word = 0
    g1_word = 0
    g2_word = 0

    for word in common_words:
        p_word += float(prior[word])
        g1_word += float(counts1[word])
        g2_word += float(counts2[word])


    reg1 = c * p_word / g1_word
    reg2 = c * p_word / g2_word

    print(f'c: {c}, p: {p_word}, g1: {g1_word}, g2: {g2_word}, r1: {reg1}, r2: {reg2}')
    
    for word in prior.keys():
        if n1 - counts1[word] > 0 and n2 - counts2[word] > 0 and counts1[word] > 0 and counts2[word] > 0:
            l1 = float(counts1[word] + float(prior[word]) / reg1) / (( n1 + float(nprior) / reg1 ) - (counts1[word] + float(prior[word])/reg1))
            l2 = float(counts2[word] + float(prior[word])/reg2) / (( n2 + float(nprior)/reg2 ) - (counts2[word] + float(prior[word])/reg2))

            sigmasquared[word] =  1/(float(counts1[word]) + float(prior[word])/reg1) + 1/(float(counts2[word]) + float(prior[word])/reg2)

            ll1 = math.log(l1)
            ll2 = math.log(l2)
            diff_logs = ll1 - ll2
            sigma[word] =  math.sqrt(sigmasquared[word])
            delta[word] = diff_logs / sigma[word]
            
    if verbose:
        for word in sorted(delta, key=delta.get)[:10]:
            print("%s, %.3f" % (word, delta[word]))

        for word in sorted(delta, key=delta.get,reverse=True)[:10]:
            print("%s, %.3f" % (word, delta[word]))
    return delta




def marked_words(df, target_val, target_col, unmarked_val,verbose=False, prior=True, frac_words=1, prompt=None):

    """Get words that distinguish the target group (which is defined as having 
    target_group_vals in the target_group_cols column of the dataframe) 
    from all unmarked_attrs (list of values that correspond to the categories 
    in unmarked_attrs)"""

    grams = dict()
    thr = 1.96 #z-score threshold

    subdf = df.copy()
    for i in range(len(target_val)):
        subdf = subdf.loc[subdf[target_col[i]]==target_val[i]]
    if prompt:
        r = target_val[0]
        gen = target_val[1]
        if gen == 'F':
            if r == 'a Latino':
                r = 'a Latina'
            g = 'woman'
            p = 'her'
            p2 = 'she'
        elif gen =='N':
            g = 'nonbinary person'
            p = 'them'
            p2 = 'they'
        else:
            g = 'man'
            p = 'him'
            p2 = 'he'
        prompt = prompt % (r, g)
        subdf = subdf.loc[subdf['prompt'] == prompt]
        
    for i in range(len(unmarked_val)):
        thr  = 1.96
        delt = get_log_odds(subdf['text'], df.loc[df[target_col[i]]==unmarked_val[i]]['text'],df['text'],verbose, prior=prior, frac_words=frac_words) #first one is the positive-valued one
        c1 = []
        c2 = []
        for k,v in delt.items():
            if v > thr:
                c1.append([k,v])
            elif v < -thr:
                c2.append([k,v])

        if 'target' in grams:
            grams['target'].extend(c1)
        else:
            grams['target'] = c1
        if unmarked_val[i] in grams:
            grams[unmarked_val[i]].extend(c2)
        else:
            grams[unmarked_val[i]] = c2
    grams_refine = dict()
    

    for r in grams.keys():
        temp = []
        thr = len(unmarked_val) # must satisfy all intersections
        for k,v in Counter([word for word, z in grams[r]]).most_common():
            if v >= thr:
                z_score_sum = np.sum([z for word, z in grams[r] if word == k])
                temp.append([k, z_score_sum])

        grams_refine[r] = temp
    return grams_refine['target']


def main():
    parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("filename", help="Generated personas file")
    parser.add_argument("--target_val",nargs="*", 
    type=str,
    default=[''], help="List of demographic attribute(s) for target group of interest")
    parser.add_argument("--target_col", nargs="*",
    type=str,
    default=[''],help="List of demographic categories that distinguish target group")
    parser.add_argument("--unmarked_val", nargs="*",
    type=str,
    default=[''],help="List of unmarked default values for relevant demographic categories")
    parser.add_argument("--verbose", action='store_true',help="If set to true, prints out top words calculated by Fightin' Words")

    args = parser.parse_args()

    filename = args.filename
    target_val = args.target_val
    target_col = args.target_col
    unmarked_val = args.unmarked_val

    assert len(target_val) == len(target_col) == len(unmarked_val)
    assert len(target_val) > 0
    df = pd.read_csv(filename)

    # Optional: filter out unwanted prompts
    # df = df.loc[~df['prompt'].str.contains('you like')]
    top_words = marked_words(df, target_val, target_col, unmarked_val,verbose=args.verbose)
    print("Top words:")
    print(top_words)

if __name__ == '__main__':
    
    main()
