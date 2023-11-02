# Glitter or Gold? Deriving Structured Insights from Sustainability Reports via Large Language Models

by
*Marco Bronzini*, 
*Carlo Nicolini*, 
*Bruno Lepri*, 
*Andrea Passerini* and 
*Jacopo Staiano*

This paper has been submitted for publication in *EPJ Data Science*.
A pre-print version can be found on [ArXiv](https://arxiv.org/abs/2310.05628)

## Abstract
Over the last decade, several regulatory bodies have started requiring the disclosure of non-financial information from publicly listed companies, in light of the investors' increasing attention to Environmental, Social, and Governance (ESG) issues.
Such information is publicly released in a variety of non-structured and multi-modal documentation.
Hence, it is not straightforward to aggregate and consolidate such data in a cohesive framework to further derive insights about sustainability practices across companies and markets.

Given these premises, it is natural to resort to Information Extraction (IE) techniques to provide concise, informative, and actionable data to the stakeholders.
Moving beyond traditional text processing techniques, in this work we leverage Large Language Models (LLMs), along with the prominent in-context learning technique and the Retrieved Augmented Generation (RAG) paradigm, to extract semantically structured ESG-related information from companies' sustainability reports.

We then adopt graph-based representations to conduct meaningful statistical, similarity and correlation analyses concerning the ESG-related actions disclosed by companies in their sustainability reports.  
These analyses unveiled that companies address ESG-related issues through several actions encompassing recognition, compliance, and partnerships; highlighting the complexity and joint efforts needed to address them. 
Moreover, disclosure similarities emerged among companies from the same region or sector. 

Lastly, we investigate which factual aspects impact the most on companies' ESG scores using our findings and other company information.
This analysis unveiled that companies' disclosures affect ESG scores more than other financial or company characteristics.

KEYWORDS: *ESG Dimensions*, *Non-financial Disclosures*, *Large Language Models*, *In-context Learning*, *Knowledge Graphs*, *Bipartite Graph Analyses*, *Interpretability*

## Repository structure
### Data
This folder includes the sustainability reports, our inputs, in their original format (PDFs) and after text extraction (textual files).

### Code
This folder includes the minimal code to extract stuctured data from textual documents 
and conduct the analyses reported in the paper.

### Output
Here can be found the triples extracted using our methdology representing the ESG-related actions disclosed by the selected companies. 
