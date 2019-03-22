import pandas as pd
import os
import sys
import numpy as np

#CSV_PATH_1 = "./results/exp_spiral_1/"
#CSV_PATH_2 = "./results/exp_spiral_2/"
#CSV_PATH_3 = "./results/exp_spiral_3/"

FILES_TO_JOIN = ["./results/exp_gaussian_recalculated_2/"+ "results_pid="+ str(i) + ".csv" for i in range(8)]
#FILES_TO_JOIN.extend([CSV_PATH_2 + "results_pid="+ str(i) + ".csv" for i in range(8)])
#FILES_TO_JOIN.extend([CSV_PATH_3 + "results_pid="+ str(i) + ".csv" for i in range(8)])


OUTPUT_FILE = "./results/joined_gaussian_dynamic_v2.csv"

def calculate_mean_sd(df):
    df['experiments'] = 0
    df['mean_acc'] = 0
    df['sd_acc'] = 0
    df['each_acc'] = 0
    df['min_acc'] = 0
    df['max_acc'] = 0
    df['median_acc'] = 0
    
    
    
    acc_dict = {}
    index_dict = {}
    
    rel_cols = [x not in ['index','id','acc','elapsed_time'] for x in df.columns]
    
    for i in range(df.shape[0]):
        key = str(df.loc[df.index[i],rel_cols].values)
        val = df.loc[df.index[i],"acc"]
        
        L = acc_dict.get(key,[])
        L.append(val)
        acc_dict[key] =  L
        
        index_dict[key] =  i
    
    
    key_list = list(acc_dict.keys())    
    new_df = pd.DataFrame(index=range(len(key_list)),columns=df.columns)

    for i in range(new_df.shape[0]):
        key = key_list[i]
        accs = acc_dict[key]
        new_df.iloc[i,:] = df.iloc[index_dict[key],:]
        new_df.loc[df.index[i],"mean_acc"] = np.mean(accs)
        new_df.loc[df.index[i],"sd_acc"] = np.std(accs)
        new_df.loc[df.index[i],"experiments"] = len(accs)
        new_df.loc[df.index[i],"each_acc"] = str(sorted(accs))
        new_df.loc[df.index[i],"min_acc"] = min(accs)
        new_df.loc[df.index[i],"max_acc"] = max(accs)
        new_df.loc[df.index[i],"median_acc"] = np.median(accs)
        
    new_df = new_df.loc[:,[x not in ['acc','id','index'] for x in new_df.columns]]
    return(new_df)
    
def main():
    print(FILES_TO_JOIN)

    joined_df = None

    for i in range(len(FILES_TO_JOIN)):
        some_f = FILES_TO_JOIN[i]
        if not os.path.isfile(some_f):
            raise FileNotFoundError("Did not find " + str(some_f))

        new_df = pd.read_csv(some_f,delimiter=",",header=0)
            

        if i == 0:
            joined_df = new_df
        else:
            joined_df = joined_df.append(new_df)
    joined_df = joined_df.reset_index()
    joined_df = joined_df.loc[:,[x not in ['index','Unnamed: 0'] for x in joined_df.columns]]
    
    
    summarized_df = calculate_mean_sd(joined_df)
    print(summarized_df.iloc[0:10,])
    print(summarized_df.shape)
    
    summarized_df.to_csv(OUTPUT_FILE)      
    print()
main()