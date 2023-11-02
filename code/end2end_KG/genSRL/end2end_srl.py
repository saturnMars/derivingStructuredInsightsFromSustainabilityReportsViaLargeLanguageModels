
import itertools
import pandas as pd
import torch.cuda as cuda

from os import  path
from tqdm import tqdm
from more_itertools import batched
from psutil import Process
from time import perf_counter, strftime
from kor import create_extraction_chain
from json import load

# LOCAL IMPORTS
from utils.document_loader import NonFinancialDisclosuresDataset, SemanticSearcher
from models.wizardLM_langchain import WizardLM
from models.prompt.srl_schema import schema, prompt_template

if __name__ == '__main__':
    print("PROCESS PID:", Process().pid)
    print('DEVICE:', '(GPU) ' + cuda.get_device_name() if cuda.is_available() else 'CPU')
    
    # Read settings
    with open(file = path.join('end2end_KG', 'genSRL', 'configs.json')) as json_file:
        configs = load(json_file)
        params = configs['params']
        selected_companies = configs['companies']
        
        if params['debug']:
            selected_companies = selected_companies[:2]
      
    # Start the counter
    program_start = perf_counter()
    timing_stats = dict()
    
    data_path = path.join('..', '..','data')
    nonFinancialFiles_path = path.join(data_path, 'processed', 'nonFinancialReports_REGEX')
    
    dataset = NonFinancialDisclosuresDataset(
        data_path = nonFinancialFiles_path, 
        languages = params['document_languages'],
        selected_companies = selected_companies, 
        select_most_recent_companyDoc = params['select_most_recent_companyDoc'],
        sentence_minWords = params['sentence_minWords'], 
        sentences_per_input = params['sentences_per_input'],
        max_sentences = 30 if params['debug'] else -1
    )
    
    if params['sentence_filtering']['topk'] != -1:
        semantic_searcher = SemanticSearcher()
        selected_indices = semantic_searcher.filter_sentences(
            sentences = dataset, 
            sentence_per_keyword = params['sentence_filtering']['topk'], 
            sim_threshold = params['sentence_filtering']['sim_threshold'], 
            verbose = True)
        dataset = dataset.select_subset(selected_indices)
        
        del semantic_searcher
        cuda.empty_cache()
        
    print("CUDA:", str(round(cuda.max_memory_allocated() * (10 ** -9), 2)) + ' GB')
        
    # Load the language model
    print("\nLoading the language model...")
    llm_model = WizardLM()
        
    # 2) Initialize the LLM chain
    chain = create_extraction_chain(llm_model, schema, encoder_or_encoder_class='json') #, instruction_template=prompt_template) 
    print("*" * 100 + "\n" + "*" * 100 + "\n" + "*" * 100 + "\n", 
          chain.prompt.format_prompt(text="[user input]").to_string(), 
          "\n" + "*" * 100 + "\n" + "*" * 100 + "\n" + "*" * 100, "\n")    
    
    # Process the documents    
    print(f"Computing {len(dataset)} documents...")  
    start_processing = perf_counter() 
    
    # Get the model outputs
    #data_loader = DataLoader(dataset, batch_size = params['batch_dim'], num_workers = 0,
                            #collate_fn = lambda sentences: [sentence for sentence in sentences]) 
                            
    outputs = itertools.chain.from_iterable(
        chain.apply(batch) for batch in tqdm(batched(dataset, n = params['batch_dim']),  # type: ignore
                                             desc = "batches", total = len(dataset) // params['batch_dim']))
    
    # Save the triples into the dataframe
    dataset.store_triples(list_triples = outputs)
    
    # Stop the timer
    timing_stats['sentence_processing'] = perf_counter() - start_processing
    timing_stats['total'] = perf_counter() - program_start
    time_elapsed = divmod(timing_stats['total'], 60)
    
    # Save the dataset on the disk
    saving_folder = path.join('outputs', 'genSRL')
    file_name = 'triples_wizardLM_' + params['test_name']
    dataset.save(saving_folder, file_name, sort_triples = True if not params['debug'] else False)
    
    # Save some technical stats
    stats_df = pd.Series({
        'timestamp': strftime("%Y-%m-%d, %H:%M"),
        'script_params': params,
        'inputs':{
            'num_docs': dataset.stats['documents_loaded'],
            'companies': ', '.join(dataset.stats['companies']),
            'total_sentences': len(dataset),
            'sentence_coverage': dataset.stats['sentence_coverage']
        },
        'run_stats':{
            'computational_time': {
                'total': f'{round(time_elapsed[0])} minutes and {round(time_elapsed[1])} seconds',
                'sentence_processing': timing_stats['sentence_processing'],
                'avg_sentence': round(timing_stats['sentence_processing'] / len(dataset), 1)
            },
            'cuda':{
                'max_memory_allocated': str(round(cuda.max_memory_allocated() * (10 ** -9), 2)) + ' GB',
                'max_memory_reserved': str(round(cuda.max_memory_reserved() * (10 ** -9), 2)) + ' GB',
            }
        },
        'model_params':llm_model._identifying_params
    })
    
    with open(path.join(saving_folder, file_name + '_stats.json'), mode = 'w', encoding = 'utf-8') as json_file:
        stats_df.to_json(json_file, orient = 'index', indent = 4, force_ascii = False)
        
    print("||| END |||")