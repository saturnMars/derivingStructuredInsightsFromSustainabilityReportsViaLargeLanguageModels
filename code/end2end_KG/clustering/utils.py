from langchain.embeddings import HuggingFaceInstructEmbeddings, FakeEmbeddings
from sentence_transformers import util
from torch import cuda, Tensor
from tqdm.auto import tqdm
from collections import defaultdict   
from os import path
from json import load
import pandas as pd

def read_triples(folder_path, file_names):
    triple_data = dict()
    
    # Load the files
    for file_name in file_names:
        with open(path.join(folder_path, file_name), mode = 'r') as fp:
            triple_data.update(load(fp))
    return triple_data

def load_triples(folder_path, file_names) -> pd.DataFrame:
    triple_data = read_triples(folder_path, file_names)
    
    # Convert to dataframe
    raw_data = []
    for companyName, triples in triple_data.items():
        for triple in triples:
            raw_data.append({
                'company': companyName,
                'category': triple['esg_category'],
                'predicate': triple['predicate'],
                'object': triple['object']
            })
    df = pd.DataFrame(raw_data)
    return df

def load_embedding_model():
    #production = True
    #if not production:
    if not cuda.is_available():
        model = FakeEmbeddings(size = 768)
    else:
        model = HuggingFaceInstructEmbeddings(
            model_name = "hkunlp/instructor-xl", 
            model_kwargs = {'device': 'cuda'},
            embed_instruction = "Represent the title: ")
    print("DEVICE: GPU", cuda.get_device_name(0),  "\n")
    return model

def mine_clusters(model, companyElements, min_cluster_size = 2, cluster_threshold = 0.8):

    # Compute the embeddings
    print(f"\n[INFO] Computing the embeddings for {len(companyElements)} elements...")
    embedded_elements = Tensor(model.embed_documents(companyElements))
    print("SHAPE:", embedded_elements.shape)
    
    # Mine the clusters: DENSE REGIONS --> embeddings that are closer than threshold are assigned to the same cluster --> CLUSTER: [cluster_threshold: 1]
    print(f"\n[INFO] Computing community (acc: {cluster_threshold} || cluster_size: {min_cluster_size})...")
    clusters = util.community_detection(embedded_elements, min_community_size = min_cluster_size, threshold = cluster_threshold)
    
    clustered_elements = dict()
    for cluster_item_ids in clusters:
        
        # Retrieve the actual 
        items = [companyElements[item_id] for item_id in cluster_item_ids]
        
        # the first element in each list is the central point in the community.
        clustered_elements[items[0]] = items
        
    return clustered_elements

def substitute_element_with_clusterName(folder_path, file_names, clustered_elements):

    # Load the triples
    triple_data = read_triples(folder_path, file_names)
    
    # Sort the data according to the company name
    triple_data = dict(sorted(triple_data.items(), key = lambda item : item[0]))
    
    # Define the function
    find_clusterName = lambda element, clustered_elements: [cluster_name for cluster_name, cluster_items in clustered_elements.items() 
                                                            if element in cluster_items]
    
    # Post-process the triples
    processed_triples = defaultdict(list)
    triple_att = None
    for companyName, triples in tqdm(triple_data.items(), desc = "Companies"):
        for triple in triples:
            
            if not triple_att:
                triple_att = list(triple.keys())
            
            # Save the original names
            _original_esg_category = triple['esg_category']
            _original_predicate = triple['predicate']
            
            # Substitute the elements with the cluster name 
            #[cluster_name for cluster_name, cluster_items in clustered_elements['category'].items() if triple['_original_esg_category'] in cluster_items]
            category_centroid = find_clusterName(_original_esg_category, clustered_elements['category'])
            if len(category_centroid) > 0:
                triple['esg_category'] = category_centroid[0]
                
                if triple['esg_category'] != _original_esg_category:
                     triple['original_esg_category'] = _original_esg_category
            
            #[cluster_name for cluster_name, cluster_items in clustered_elements['predicate'].items() if triple['_original_predicate'] in cluster_items]
            predicate_centroid = find_clusterName(_original_predicate, clustered_elements['predicate'])
            if len(predicate_centroid) > 0:
                triple['predicate'] = predicate_centroid[0]
                
                if triple['predicate'] != _original_predicate:
                     triple['original_predicate'] = _original_predicate

            # Save the processed triple
            #triple = dict(sorted(triple.items(), key = lambda dict_item: triple_att.index(dict_item[0]) if dict_item[0] in triple_att else len(triple_att) + 1))
            processed_triples[companyName].append(triple)
    return processed_triples
            