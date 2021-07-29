"""
An example of how to use:: 
==============================

from cluster_evaluate import *

cluster_labels = get_clusters_label(... , ...)

cluster_top_labels = evaluate_cluster_labels(cluster_labels, threshold=30)

... and so on

----------------------------

Example to import in Google Colab::
==================================
## No need to mount your drive to the notebook , just upload it directly from your local

from google.colab import files
src = list(files.upload().values())[0]  ## Here you see a pop-up command to upload the file from your local directory

open('cluster_evaluate.py','wb').write(src)
from cluster_evaluate import *

"""

import pandas as pd
import numpy as np

import nltk
import seaborn as sns
from nltk.corpus import stopwords
from tqdm import tqdm as tq
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics
from scipy.stats import spearmanr

#%matplotlib inline

## Load the resources
nltk.download('stopwords')
nltk.download('punkt')


stop_words = stopwords.words('english')


def get_clusters_label(cluster_predictions , actual_labels):
  """
      Get the cluster and labels distribution
      
     ::Input Params::
     * cluster_predictions: "array-like" of all cluster predictions from your clustering algorithm
     * actual_labels: "array-like" of all actual labels used CORRESPONDING the cluster predictions  (book_name , title , author ...etc)

     ::Output::
     * Get the cluster and labels distribution (MultiIndex DataFrame)
  """

  clusters_labels = pd.DataFrame( data= {'clusters': cluster_predictions,
                                'actual_labels': actual_labels})
  
  clusters_labels_count =  pd.DataFrame(clusters_labels.groupby('clusters').actual_labels.value_counts(normalize=True)*100)
  return clusters_labels_count
  
  

def evaluate_cluster_labels(cluster_labels,
                            threshold=50):
  """
     Plot the clusters labels distribution for each label 
     
     ::Input Params::
     * cluster_labels: "DataFrame-like" cluster and labels distribution (MultiIndex DataFrame)
     * threshold: "Integer" of all actual labels used CORRESPONDING the cluster predictions  (book_name , title , author ...etc)

     ::Output::
     * Get the top label based on the threshold 
  """

  cluster_top_labels=[]

  for cls in cluster_labels.index.get_level_values('clusters').unique():
    if threshold:
      if cluster_labels['actual_labels'][cls][0]>threshold:
        cluster_top_labels.append({'cluster': cls,
                                   'label': cluster_labels['actual_labels'][cls].keys()[0],
                                   'percentage': cluster_labels['actual_labels'][cls][0]})
    else:
      cluster_top_labels.append({'cluster': cls,
                                 'label': cluster_labels['actual_labels'][cls].keys()[0],
                                 'percentage': cluster_labels['actual_labels'][cls][0]})
      
  return cluster_top_labels
  
  

def cluster_labels_dist_plot(cluster_predictions , actual_labels):
  """
    Plot the clusters labels distribution for each label 
    
    ::Input Params::
    * cluster_predictions: "array-like" of all cluster predictions from your clustering algorithm
    * actual_labels: "array-like" of all actual labels used CORRESPONDING the cluster predictions  (book_name , title , author ...etc)

    ::Output::
    * analysis of plot for each cluster with the distribution of the labels found in it 
  """
  
  clusters_labels = pd.DataFrame(data= {'clusters': cluster_predictions,
                                        'actual_labels': actual_labels})
  
  num_clusters = len(np.unique(cluster_predictions))
  ## Aggregating the data of clusters
  all_label_dist_dict =[]
  for label in clusters_labels.actual_labels.unique():
    label_dist = pd.DataFrame(clusters_labels.groupby('actual_labels').clusters.value_counts()).loc[label]['clusters']
    num_cls = len(np.unique(cluster_predictions))
    label_dist_dict = label_dist.to_dict()
    for cls in np.unique(cluster_predictions):
      if cls not in label_dist_dict:
        label_dist_dict.update({cls: 0})

    label_dist_dict.update({'label': label})

    all_label_dist_dict.append(label_dist_dict)

  all_label_dist_df = pd.DataFrame(all_label_dist_dict)
  all_label_dist_df = all_label_dist_df[[i for i in range(0,num_clusters)]+['label']]

  sns.heatmap(all_label_dist_df[[i for i in range(0,num_clusters)]] , yticklabels=all_label_dist_df['label'])

  ## Plotting steps
  plt.figure(figsize=(15,10))
  # all_cls = [all_label_dist_df[i] for i in range(num_clusters)]
  width = .08
  Pos = np.array(range(num_clusters))
  height=0.1
  plt.bar(Pos - height, all_label_dist_df.loc[0][:-1], width = width, label = all_label_dist_df['label'][0])

  plt.bar(Pos, all_label_dist_df.loc[1][:-1], width = width, label = all_label_dist_df['label'][1])
  plt.bar(Pos + height, all_label_dist_df.loc[2][:-1], width = width, label = all_label_dist_df['label'][2])

  plt.bar(Pos - 2*height, all_label_dist_df.loc[3][:-1], width = width, label = all_label_dist_df['label'][3])
  plt.bar(Pos + 2*height, all_label_dist_df.loc[4][:-1], width = width, label = all_label_dist_df['label'][4])

  plt.xticks(Pos, [f'cluster_{i}' for i in range(num_clusters)])
  plt.legend()

  plt.show()
  
  return all_label_dist_df
  
  
  

def top_frequent_analysis(cluster_predictions , actual_labels, partitions,
                          exclude_stopwords=False, top_n= 10):
  
  """
     Get the top N frequent words in each cluster for the right-clustered vs wrong-clustered
     assuming that the right-clustered label is the highest percentage label in the cluster and other labels considered as wrong clustered
     
     ::Input Params::
     * cluster_predictions: "array-like" of all cluster predictions from your clustering algorithm
     * actual_labels: "array-like" of all actual labels used CORRESPONDING the cluster predictions  (book_name , title , author ...etc)
     * paritions: "array-like" of all the partitions text CORRESPONDING the cluster predictions
     * exclude_stopwords: "Boolean" wether you want to exclude the stop words from the top words or not
     * top_n: "Integer" number of top frequent words you want to search for (default= 10)
     
     ::Output Params::
     * analysis of plot for each cluster with the distribution of the top_n words 
  """
  
  clusters_labels = pd.DataFrame(data= {'clusters': cluster_predictions,
                                      'actual_labels': actual_labels,
                                      'partitions': partitions})
  cluster_groups = clusters_labels.groupby('clusters')
  for name,group in cluster_groups:

    label_distribution = group.actual_labels.value_counts()
    right_label = label_distribution.keys()[0]
    wrong_labels = label_distribution.keys()[1:]

    right_partitions = group[group.actual_labels==right_label]
    wrong_partitions = group[group.actual_labels.isin(wrong_labels)]
    
    right_text = " ".join(right_partitions.partitions.values)
    wrong_text = " ".join(wrong_partitions.partitions.values)

    if exclude_stopwords:
      right_tokens = [word for word in nltk.word_tokenize(right_text) if word not in stop_words]
      wrong_tokens = [word for word in nltk.word_tokenize(wrong_text) if word not in stop_words]
    else:
      right_tokens = nltk.word_tokenize(right_text)
      wrong_tokens = nltk.word_tokenize(wrong_text)

    top_n_right = Counter(right_tokens).most_common(top_n)
    top_n_wrong = Counter(wrong_tokens).most_common(top_n)

    
    top_rights = {word[0]:word[1] for word in top_n_right}
    top_wrongs = {word[0]:word[1] for word in top_n_wrong}
    all_labels = set(list(top_rights.keys()) + list(top_wrongs.keys()))
    
    ## For plotting
    right_labels_count = []
    wrong_labels_count = []
    for label in all_labels:
      if label in top_rights:
        right_labels_count.append(top_rights[label])
      else:
        right_labels_count.append(0)

      if label in top_wrongs:
        wrong_labels_count.append(top_wrongs[label])
      else:
        wrong_labels_count.append(0)
    
    common_tokens = [word for word in top_n_wrong if word[0] in top_rights]

    ## Plotting steps
    plt.figure(figsize=(15,5))
    plt.title(f'Cluster ({name}) top {top_n} words distribution.')
    all_labels_count = [right_labels_count , wrong_labels_count]
    width = .1
    Pos = np.array(range(len(all_labels)))
    height=0.2
    plt.bar(Pos - height, all_labels_count[0], width = width, label = 'Right Clustered')
    plt.bar(Pos, all_labels_count[1], width = width, label = 'Wrong Clustered')
    plt.xticks(Pos, all_labels)
    plt.legend()

    plt.show()

    print(f'Top {top_n} words in the right clustered\n {top_n_right}\n')
    print(f'Top {top_n} words in the wrong clustered\n {top_n_wrong}\n')
    print(f'Found {len(common_tokens)} words in both right and wrong clustered top words {common_tokens}\n\n')
    
  
def map_actual_labels(cluster_labels,actual_labels):
  df=pd.DataFrame(evaluate_cluster_labels(get_clusters_label(cluster_labels,actual_labels),threshold=0))
  cluster_labels=cluster_labels.astype(str)
  
  for cluster_num in df.cluster:
     cluster_labels = np.where(cluster_labels == str(cluster_num) , df.label[cluster_num], cluster_labels)

  return cluster_labels  
  
  
def calculate_metrics(actual,predicted,X):
  ## Homogeneity scores: a cluster should contain only samples belonging to a single class
  print("Homogeneity score: ",np.round(metrics.homogeneity_score(actual, predicted),2))

  ## Completeness score: if all the data points that are members of a given class are elements of the same cluster
  print("Completeness score: ",np.round( metrics.completeness_score(actual, predicted),2))

  ##The V-Measure is defined as the harmonic mean of homogeneity and completeness
  print("V-measure score: ",np.round( metrics.v_measure_score(actual ,predicted),2))

  ## The Rand Index computes a similarity measure between two clusterings by considering all pairs 
  print("Adjusted rand score: ",np.round( metrics.adjusted_rand_score(actual,predicted),2))

  ## Cohen’s kappa: a statistic that measures inter-annotator agreement
  print("Kappa score: ",np.round( metrics.cohen_kappa_score(actual, predicted,weights='linear'),2))

  ##The Silhouette Coefficient is calculated using the mean intra-cluster distance and the mean nearest-cluster distance for each sample
  print("Silhouette score: ",np.round( metrics.silhouette_score(X, predicted),2))

  ## Calculate a Spearman correlation coefficient with associated p-value.
  print("Correlation: ",str(spearmanr(actual,predicted)))  
  
