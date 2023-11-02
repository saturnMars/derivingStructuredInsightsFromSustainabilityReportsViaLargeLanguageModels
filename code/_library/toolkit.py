import Levenshtein

def print_title(str, prefix = '', capitalize = True):
    if capitalize:
        str = str.capitalize()
    
    print('\n' + '-' * 50)
    print('-' * (19 - (len(str) // 2)), prefix, str, '-' * (19 - (len(str) // 2)))
    print('-' * 50)
    
def lookup_companyName(company_names, name, fuzzy_match = False):
    fuzzyMatches = [companyName for companyName in company_names if name.lower() in companyName.lower()]
    fuzzyMatches = sorted(fuzzyMatches, key = lambda companyName: Levenshtein.distance(companyName, name))
    
    if fuzzy_match:
        return fuzzyMatches
    else:
        best_match = fuzzyMatches[0] if len(fuzzyMatches) > 0 else None
        return best_match