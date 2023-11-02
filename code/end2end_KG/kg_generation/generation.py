from utils import AuraDBApp
from os import path
from json import load
from tqdm import tqdm

if __name__ == '__main__':
    
    # Load JSON data
    folder_path = path.join('outputs', 'genSRL')
    file_names = ['triples_wizardLM_filtering_setA.json', 'triples_wizardLM_filtering_setB.json']
        
    # Load the files
    triple_data = dict()
    for file_name in file_names:
        with open(path.join(folder_path, file_name), mode = 'r') as fp:
            triple_data.update(load(fp))
    triple_data = dict(sorted(triple_data.items(), key = lambda x: len(x[0])))
    print("\nCOMPANIES:", len(triple_data.keys()), "(e.g.,",', '.join(list(triple_data.keys())[:3]), "...)\n")
    
    # Connect to the database
    uri = "neo4j+s://607920af.databases.neo4j.io"
    app = AuraDBApp(uri, user = "neo4j", password = "iOkuhageeDfvgL_4ukPK7Z1jW8ijBlyu6x4UEDiPmi8") 
    
    # Clear the database
    app.clear_db()
    
    # Populate the database
    verbose = False
    for companyName, actions in triple_data.items():
        print("COMPANY:", companyName)
        
        for action in tqdm(actions):
            app.create_company_relationship(companyName.capitalize(), action['esg_category'], verbose = verbose)
            app.create_esg_relationship(action['esg_category'], action['predicate'], action['object'], action['properties'], verbose = verbose)
            
    # Close the database
    app.close()
    print("\nFINISHED!")