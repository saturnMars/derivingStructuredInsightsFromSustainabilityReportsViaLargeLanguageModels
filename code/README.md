# End-to-end knowledge graph generation
1) **DATA WRANGLING & TRIPLE GENERATION**
    - ./end2end_KG/genSRL/end2end_srl.py
        - INPUT: textual reports (.txt)
        - OUTPUT: company triples (.json) 
            - outputs/genSRL/triples_wizardLM_filtering_setX.json
        - params: configs.json

2) **POST-PROCESSING** (./end2end_KG/.)
    - ***Semantic clustering***: ./clustering/clustering.py
        - INPUT: company triples (.json)
        - OUTPUT: company triples (.json) 
            -  outputs/genSRL/clustered_tXX/triples_wizardLM_filtering.json
    - ***Knowledge Graph generation***: ./kg_generation/generation.py
        - INPUT: company triples (.json)
        - OUTPUT: Neo4j graph (Neo4j AuraDB: cloud instance)

3) **ANALYSES** (./data_exploration)
    1) ***Descriptive statistics***: ./bipartite/triple_stats_pycondor.ipynb
        - INPUT: company triples (.json) 
        - OUTPUTS: Graph statistics regarding three different bipartite graphs
            - outputs/bipartite_analyses/pycondor/[bipartiteGraph_version]
            - FORMAT: excel files, graphs
        - NOTE: it contains a lot of code related to another task (community detection)

    2) ***Similarity***: ./triple_jaccard_similarity.ipynb
        - INPUT: company triples (.json), company information (i.e., sector)
        - OUTPUT: company similarities (outputs/graph_analyses/jaccard_clustered/.)
            - TABULAR DATA: ./rawJaccardSimilarities_subtractNullModel.xlsx
            - GRAPH (heatmap): ./jaccardSimilarities_subtractNullModel.png
            - NETWORK (most similar companies): ./actionscompanySimilarities.gexf

    3) ***Correlation***: ./companySimilarities_correlations.ipynb
        - INPUT: company triples (.json), company information and ESG scores (xlsx, from data/refinitiv.ipynb)
        - OUTPUT: feature correlations (outputs/ESG_correlations/.)
            - ACTION-FOCUSED CORRELATION: ./kendallCorrelations.xlsx
            - TABULAR DATA: ./raw/. 
            - GRAPHS: ./graphs/kendall/.

    4) ***ESG score interpretability***: ./regression.ipynb
        - INPUT: company triples (.json), company information and ESG scores (xlsx, from data/refinitiv.ipynb)
        - OUTPUT: (outputs/ESG_regression/poly1)
            - TABULAR DATA: ./regressionCoeff.xlsx
            - Global interpetability GRAPHS: ./explainability/featureImportanceWithEffects/.
            - Local interpetability GRAPHS: ./explainability/observationPlots/.
