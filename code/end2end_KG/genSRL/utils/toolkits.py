from ujson import loads, JSONDecodeError
import dirtyjson

def parse_output(output):
    output = output['text'].lstrip('<json>').rstrip('</json>')
    
    try:
        parsed_output = loads(output)
    except JSONDecodeError as parsing_error:
        try:
            parsed_output = dirtyjson.loads(output)
        except dirtyjson.error.Error:
            print("FAILED PARSING:\n", output)
            return []
        
    if isinstance(parsed_output, dict) and 'esg_actions' in parsed_output.keys():
        return parsed_output['esg_actions']
    
    return []

def attach_tripleProperties(df_row):
    to_add = {'source': str(df_row['year']) + '-' + df_row['documentName'], 'original_sentence': df_row['text']}
                                                                         
    for triple in df_row['triples']:
        if len(triple.keys()) > 0:
            if 'properties' in triple.keys():
                triple['properties'].update(to_add)
            else:
                triple['properties'] = to_add
    return df_row['triples']