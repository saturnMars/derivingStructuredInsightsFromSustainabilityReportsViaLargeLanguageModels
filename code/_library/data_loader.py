from os import path, scandir, listdir
from pandas import DataFrame
import re
from json import load

def loader_nonFinancialReports(folderPath, companies = [], read_text = False):
    companies = [company.lower() for company in companies]
    
    reports = []
    for yearlyFolder in scandir(folderPath):
        if not yearlyFolder.is_dir():
            print('A file was found in the folder', folderPath, 'and it was ignored')
            continue

        # Extract information
        year = yearlyFolder.name
        try: 
            year = int(year)
        except:
            pass
        files = listdir(yearlyFolder.path)
        for fileName in files:
   
            documentName, fileExtension  = path.splitext(fileName)

            if fileExtension.lstrip('.') not in ['pdf', 'txt', 'html', 'xhtml']:
                print('\nWARNING! Wrong extension for file "', fileName, '" in folder', yearlyFolder.path, 'and it was ignored')
                continue

            # Extract company name and document type
            partialComponents = documentName.split('-')
            
            if len(partialComponents) < 2 or len(partialComponents) > 3:
                raise Exception("\n" + documentName + "-->" + str(partialComponents) + '\nThe file name <<'+ fileName + '>> is not in the correct format!<companyName>-<documentType>[-<documentLanguage>]')

            # Parse the company name and document type
            clean_str = lambda name: re.sub('_+', ' ', name).strip()
            companyName = clean_str(partialComponents[0])
            documentType = clean_str(partialComponents[1])
            
            # Parse the document language
            if len(partialComponents) == 3:
                documentLanguage = partialComponents[2]
            else:
                documentLanguage = None
                
            file_path = path.join(yearlyFolder.path, fileName)
                    
            companyInfo = {
                'companyName': companyName,
                'year': year,
                'documentName': documentType,
                'documentLanguage': documentLanguage,
                'fileExtension': fileExtension,
                'path': file_path
            }
            
            if len(companies) > 0 and companyName.lower() not in companies:
                continue
            
            # TODO Delate: Select only one document per company
            #if companyName in [report['companyName'] for report in reports]:
            #    continue
            
            if read_text:
                with open(file_path) as txt_file:
                    companyInfo['text'] = txt_file.read()
                    
                # Save information
                if companyInfo['text'] != "":
                    reports.append(companyInfo)
            else:
                reports.append(companyInfo)
            
    # Crate the dataframe
    documents = DataFrame(reports)
    documents = documents.sort_values(by = ['companyName', 'year']).reset_index(drop = True)

    return documents


def get_reportTexts(reports, language = None, verbose = True):

    # Sort the reports by year
    report_metadata = sorted(reports, key =  lambda report: report['year'], reverse = True)
    
    # Read the reports
    documents = dict()
    for report in report_metadata:
        
        # Check if the language requested is the same as the one in the report
        if language != None:
            if isinstance(language, list):
                differentLanguage = report['documentLanguage'] not in list(map(str.upper, language))
            else:
                differentLanguage = report['documentLanguage'] not in language.upper()
            
            if differentLanguage:
                continue
        
        if verbose:
            print(f"[{report['year']}] {report['documentType']} ({report['documentLanguage']})")
        
        with open(report['path'], 'r') as f:
            documents[report['documentType']] = f.read()
    return documents

def load_esg_categories(file_path = path.join('models', 'prompt', 'esg_topics.json')):
    with open(file_path) as esg_categories:
        return load(esg_categories)
