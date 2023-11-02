from os import path, makedirs
from json import dump

import utils

if __name__ == "__main__":
    print("STARTED")
    
    params = {
        "min_cluster_size": 2, 
        "cluster_threshold": 0.8
    }
    
    # Import paths
    folder_path = path.join('outputs', 'genSRL') #  '_old', 'top20'
    file_names = [
        'triples_wizardLM_filtering_setA.json', 
        'triples_wizardLM_filtering_setB.json', 
        'triples_wizardLM_filtering_setC.json',
        'triples_wizardLM_filtering_setD.json', 
        'triples_wizardLM_filtering_setE.json'
        ]
    
    # Load triples
    df = utils.load_triples(folder_path, file_names)
    print(df)
    
    # Load the embedding model 
    embedding_model = utils.load_embedding_model()
    
    # Mine the clusters
    clustered_elements = {}
    num_values = {}
    for col in ['category', 'predicate', 'object']: # , 'object'
        
        # Mine the clusters
        clustered_elements[col] = utils.mine_clusters(
            model = embedding_model, 
            companyElements = df[col].unique(),
            min_cluster_size = params['min_cluster_size'], 
            cluster_threshold= params['cluster_threshold'])
        
        print(f'\n{col}: {len(clustered_elements[col])} clusters (total items: {len(df[col].unique())}) \n')
        
        num_values[col] = len(df[col].unique())
        
        # Post-processing the triples
        #df[col].map(lambda element: [cluster_name for cluster_name, cluster_items in clustered_elements[col].items() if element in cluster_items])
    
    # Post-processing the triples
    postprocessed_triples = utils.substitute_element_with_clusterName(folder_path, file_names, clustered_elements)
    
    # Create the saving folder
    saving_folder = path.join('outputs', 'genSRL', f"clustered_t{int(params['cluster_threshold'] * 100)}")
    if not path.exists(saving_folder):
        makedirs(saving_folder)
        
    # Save the clusters
    with open(path.join(saving_folder, 'clustered_elements.json'), mode = 'w') as fp:
        dump(clustered_elements, fp, indent = 4)
        
    # Save the post-processed the triples
    with open(path.join(saving_folder, 'triples_wizardLM_filtering.json'), mode = 'w') as fp:
        dump(postprocessed_triples, fp, indent = 4)
    #df.to_excel(path.join(saving_folder, 'triples_wizardLM_filtering.xlsx'), index = False)
        
    print('\n[Ended]')
    