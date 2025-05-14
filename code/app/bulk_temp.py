"""
Author: Fernando Casabán Blasco and Pablo Hernández Carrascosa
Script to run the tool with several text files at the same time. Used to evaluate the tool.
"""

import argparse
import glob
import os
import time
import spacy
import coreferee
from rdflib.term import URIRef

import utils.build_RDF_triples as brt
import utils.preprocess_sentences as pps
import utils.process_triples as pt
import utils.triples_extraction as te
import utils.lookup_tables_services as lts
from utils.log_generator import tracking_log
import pandas as pd
import tqdm

timestr = time.strftime("%Y%m%d-%H%M%S")

EVALUATION = False
PROP_LEXICALIZATION_TABLE = "datasets/verb_prep_property_lookup.json"
CLA_LEXICALIZATION_TABLE = "datasets/classes_lookup.json"
OUTPUT_FOLDER = "code/output/"
SPOTLIGHT_ONLINE_API = "https://api.dbpedia-spotlight.org/en/annotate"
SPOTLIGHT_LOCAL_URL = "http://localhost:2222/rest/annotate/"


def pipeline(nlp, raw_text, dbo_graph, prop_lex_table, cla_lex_table):
    raw_text = pps.clean_text(raw_text)
    doc = nlp(raw_text)

    # Coreference resolution
    if doc._.coref_chains:
        rules_analyzer = nlp.get_pipe('coreferee').annotator.rules_analyzer
        interchange_tokens_pos = []

        for token in doc:
            if bool(doc._.coref_chains.resolve(token)):
                mention_head = doc._.coref_chains.resolve(token)
                if full_mention := rules_analyzer.get_propn_subtree(doc[mention_head[0].i]):
                    mention_text = ''.join([token.text_with_ws for token in full_mention])
                    interchange_tokens_pos.append((token.i, mention_text))
                else:
                    interchange_tokens_pos.append((token.i, doc[mention_head[0].i].text))

        if interchange_tokens_pos:
            resultado = ''
            pointer = 0
            for tupla in interchange_tokens_pos:
                resultado += doc[pointer:tupla[0]].text_with_ws + tupla[1] + ' '
                pointer = tupla[0] + 1
            resultado += doc[pointer:].text_with_ws
            doc = nlp(resultado)

        tracking_log(doc, level=1)

    sentences = pps.get_sentences(doc)
    n_sent_spacy = len(sentences)
    tracking_log(sentences, level=2)

    triples, n_sent_simples = te.get_all_triples(nlp, sentences)
    triples = pt.split_amod_conjunctions_subj(nlp, triples)
    triples = pt.split_amod_conjunctions_obj(nlp, triples)

    rdf_triples = []
    try:
        term_URI_dict, term_types_dict = brt.get_annotated_text_dict(raw_text, service_url=SPOTLIGHT_ONLINE_API)
        rdf_triples = brt.replace_text_URI(triples, term_URI_dict, term_types_dict, prop_lex_table, cla_lex_table, dbo_graph)
    except Exception as e:
        print(f"[ERROR] Spotlight or RDF processing failed: {e}")

    return triples, rdf_triples, n_sent_spacy, n_sent_simples


def print_debug(triples):
    debug_info = ""
    if triples:
        sent = triples[0].sent
        debug_info += "-"*50 + "  \n"
        debug_info += f"**{sent}**\n"
        for t in triples:
            if t.sent != sent:
                sent = t.sent
                debug_info += "-"*50 + "  \n"
                debug_info += f"**{sent}**\n"
            debug_info += t.__repr__() + "  \n"
            debug_info += t.get_rdf_triple() + "  \n"
    return debug_info


def get_only_triples_URIs(rdf_triples):
    return [t for t in rdf_triples if isinstance(t.pred_rdf, URIRef) and isinstance(t.objct_rdf, URIRef)]


def init():
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe('coreferee')
    prop_lex_table = lts.load_lexicalization_table(PROP_LEXICALIZATION_TABLE)
    cla_lex_table = lts.load_lexicalization_table(CLA_LEXICALIZATION_TABLE)
    dbo_graph = brt.load_dbo_graph(DBPEDIA_ONTOLOGY)
    return nlp, prop_lex_table, cla_lex_table, dbo_graph


if __name__ == "__main__":
    local_ontology_path = 'datasets/'
    local_ontology_files = glob.glob(f'{local_ontology_path}*.owl')
    names = [os.path.basename(x) for x in local_ontology_files]
    namesSorted = sorted(names, reverse=True)
    DBPEDIA_ONTOLOGY = local_ontology_path + namesSorted[0]

    nlp, prop_lex_table, cla_lex_table, dbo_graph = init()

    print('DBpedia abstracts to RDF')
    print('This app translates any kind of text into RDF!')

    from_index = 9
    to_index = 10
    df = pd.read_csv('datasets/long-abstracts-sample.csv')
    df = df.to_dict(orient='records')
    df = df[from_index:to_index]

    for elem in tqdm.tqdm(df):
        raw_text = elem['abstract'].replace('\n', '')
        try:
            text_triples, rdf_triples, nsent_spacy, nsent_simples = pipeline(nlp, raw_text, dbo_graph, prop_lex_table, cla_lex_table)

            elem['nsent_spacy'] = nsent_spacy
            elem['nsent_simples'] = nsent_simples

            if text_triples:
                elem['relations'] = [t.__repr__() for t in text_triples]
            else:
                elem['relations'] = ['None']

            if rdf_triples:
                elem['rdf_triples'] = list(set(t.get_rdf_triple() for t in rdf_triples))
            else:
                elem['rdf_triples'] = ['None']

        except Exception as e:
            print(f"error: {e}")
            elem['nsent_spacy'] = -1
            elem['nsent_simples'] = -1
            elem['relations'] = ['None']
            elem['rdf_triples'] = ['None']

    df = pd.DataFrame.from_records(df)
    df.to_csv(f'code/output/text2rdf_triples{from_index}_{to_index}.csv', index=False)
