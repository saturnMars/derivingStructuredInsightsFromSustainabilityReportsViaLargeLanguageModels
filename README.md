# Glitter or Gold? Deriving Structured Insights from Sustainability Reports via Large Language Models

by
*Marco Bronzini*, 
*Carlo Nicolini*, 
*Bruno Lepri*, 
*Andrea Passerini* and 
*Jacopo Staiano*

The minimal implementation of the paper "[Glitter or Gold? Deriving Structured Insights from Sustainability Reports via Large Language Models](https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-024-00481-2)".
Published in the *EPJ Data Science* journal.

## Abstract
Over the last decade, several regulatory bodies have started requiring the disclosure of non-financial information from publicly listed companies, in light of the investors’ increasing attention to Environmental, Social, and Governance (ESG) issues. Publicly released information on sustainability practices is often disclosed in diverse, unstructured, and multi-modal documentation. This poses a challenge in efficiently gathering and aligning the data into a unified framework to derive insights related to Corporate Social Responsibility (CSR). Thus, using Information Extraction (IE) methods becomes an intuitive choice for delivering insightful and actionable data to stakeholders. In this study, we employ Large Language Models (LLMs), In-Context Learning, and the Retrieval-Augmented Generation (RAG) paradigm to extract structured insights related to ESG aspects from companies’ sustainability reports. We then leverage graph-based representations to conduct statistical analyses concerning the extracted insights. These analyses revealed that ESG criteria cover a wide range of topics, exceeding 500, often beyond those considered in existing categorizations, and are addressed by companies through a variety of initiatives. Moreover, disclosure similarities emerged among companies from the same region or sector, validating ongoing hypotheses in the ESG literature. Lastly, by incorporating additional company attributes into our analyses, we investigated which factors impact the most on companies’ ESG ratings, showing that ESG disclosure affects the obtained ratings more than other financial or company data.

KEYWORDS: *ESG Dimensions*, *Non-financial Disclosures*, *Large Language Models*, *In-context Learning*, *Knowledge Graphs*, *Bipartite Graph Analyses*, *Interpretability*

[Supplementary Material document](https://static-content.springer.com/esm/art%3A10.1140%2Fepjds%2Fs13688-024-00481-2/MediaObjects/13688_2024_481_MOESM1_ESM.pdf) 

## Repository structure
### Data
This folder includes the sustainability reports, and our inputs, in their original format (PDFs) and after text extraction (textual files).

### Code
This folder includes the minimal code to extract structured data from textual documents 
and conduct the analyses reported in the paper.

### Output
Here the triples extracted using our methodology represent the ESG-related actions disclosed by the selected companies. 
