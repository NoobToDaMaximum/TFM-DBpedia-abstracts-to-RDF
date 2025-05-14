"""
Functions to extract triples from a set of sentences
Author: Fernando Casabán Blasco and Pablo Hernández Carrascosa
"""
from spacy.matcher import DependencyMatcher, Matcher
from utils.log_generator import tracking_log, log_failed_extraction
from spacy.tokens import Span

class Triple:
    def __init__(self, subj, pred, objct, sent):
        """list of tokens"""
        self.subj = subj
        self.pred = pred
        self.objct = objct
        self.sent = sent

    def get_copy(self):
        return Triple(self.subj.copy(), self.pred.copy(), self.objct.copy(), self.sent)

    def set_rdf_triples(self, subj, pred, objct):
        self.subj_rdf = subj
        self.pred_rdf = pred
        self.objct_rdf = objct

    def get_rdf_triple(self):
        return f"{self.subj_rdf} | {self.pred_rdf} | {self.objct_rdf}"

    def __repr__(self):
        return f"{' '.join([x.text for x in self.subj])} | {' '.join([x.text for x in self.pred])} | {' '.join([x.text for x in self.objct])}"

    def __str__(self):
        return f"{' '.join([x.text for x in self.subj])} {' '.join([x.text for x in self.pred])} {' '.join([x.text for x in self.objct])}"

################################
# Triples extraction functions #
################################


def get_simple_triples(nlp, sentence):
    """
    Get the triples from each sentence in <subject, predicate, object> format.
    First extract all complements of the verb that form the predicate of the triple.
    Then look for subject of the triplet with a matcher rule.
    Finally we get the object as the span from the last token of the predicate until the end of the sentence.
    With subject, predicate and object build an object Triple
    This function only works with simple sentences.
    """
    doc = nlp(sentence)

    ## PREDICATES
    patt_ROOT = [{"DEP": "ROOT"}]
    patt_ROOT_xcomp = [{"DEP": "ROOT", "POS": "VERB"}, {"DEP": "aux"}, {"DEP": "xcomp"}]

    matcher = Matcher(nlp.vocab)
    matcher.add("patt_ROOT", [patt_ROOT])
    matcher.add("patt_ROOT_xcomp", [patt_ROOT_xcomp])

    matches = matcher(doc)
    predicate_spans = []
    predicate_indices = []

    if not matches:
        log_failed_extraction(sentence, "No predicate (ROOT/xcomp) found")
        return []

    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]
        if string_id == "patt_ROOT":
            span = doc[start:end]
            predicate_spans.append(span)
            predicate_indices.append(span[0].i)
        elif string_id == "patt_ROOT_xcomp":
            span = doc[start + 2:end]
            predicate_spans.append(span)
            predicate_indices.append(span[0].i)

    ## SUBJECTS
    subj_matcher = Matcher(nlp.vocab)
    subj_matcher.add("patt_SUBJS", [[{"DEP": {"IN": ["nsubj", "nsubjpass"]}}]])
    subj_matches = subj_matcher(doc)
    all_subjs = []

    for match_id, start, end in subj_matches:
        subj_token = doc[start:end][0]
        subj = get_sentence_subtree_from_token(subj_token, ["cc", "conj"], inner=False)
        all_subjs.append(subj)
        conjs = [t for t in doc[end:doc[-1].i] if t.dep_ == "conj"]
        for c in conjs:
            all_subjs.append(get_sentence_subtree_from_token(c, ["cc", "conj"], inner=False))

    ## OBJECTS via DependencyMatcher
    dep_matcher = DependencyMatcher(nlp.vocab)
    dep_matcher.add("patt_attr", [[{"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "be"}},
                                    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "attr", "RIGHT_ATTRS": {"DEP": "attr"}}]])
    dep_matcher.add("patt_advmod_obj", [[{"RIGHT_ID": "verb", "RIGHT_ATTRS": {"DEP": {"IN": ["ROOT", "xcomp"]}}},
                                          {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "advmod", "RIGHT_ATTRS": {"DEP": {"IN": ["advmod", "dobj"]}}}]])
    dep_matcher.add("patt_prep_obj", [[{"RIGHT_ID": "verb", "RIGHT_ATTRS": {"DEP": {"IN": ["ROOT", "xcomp"]}}},
                                       {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": {"DEP": "prep"}},
                                       {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "obj", "RIGHT_ATTRS": {"DEP": "pobj"}}]])
    dep_matcher.add("patt_be_acomp", [[{"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "be"}},
                                       {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "acomp", "RIGHT_ATTRS": {"DEP": "acomp"}}]])
    dep_matcher.add("patt_agent_obj", [[{"RIGHT_ID": "verb", "RIGHT_ATTRS": {"DEP": "ROOT"}},
                                        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "agent", "RIGHT_ATTRS": {"DEP": "agent"}},
                                        {"LEFT_ID": "agent", "REL_OP": ">", "RIGHT_ID": "obj", "RIGHT_ATTRS": {"DEP": "pobj"}}]])

    dep_matches = dep_matcher(doc)

    triples = []
    for pred_span, pred_index in zip(predicate_spans, predicate_indices):
        pred_token = pred_span[0]
        valid_subjs = all_subjs

        objs = []
        for match_id, token_ids in dep_matches:
            if not check_ascending_order_token_id(token_ids):
                continue
            verb_token = doc[token_ids[0]]
            if verb_token.lemma_ != pred_token.lemma_ and abs(verb_token.i - pred_token.i) > 2:
                continue

            string_id = nlp.vocab.strings[match_id]
            if string_id in ["patt_attr", "patt_advmod_obj", "patt_be_acomp"]:
                obj_token = doc[token_ids[1]]
                conjs = obj_token.conjuncts
                objs.append(get_sentence_subtree_from_token(obj_token, ["cc", "conj"], inner=False))
                for c in conjs:
                    objs.append(get_sentence_subtree_from_token(c, ["cc", "conj"], inner=False))

            elif string_id == "patt_prep_obj":
                prep = doc[token_ids[1]]
                pobj = doc[token_ids[2]]
                prep_object = get_sentence_subtree_from_token(prep, ["cc", "conj"], inner=False)
                simpler_objs = split_conjunctions_obj(prep_object, prep, pobj)
                objs.extend(simpler_objs)

            elif string_id == "patt_agent_obj":
                agent = doc[token_ids[1]]
                pobj = doc[token_ids[2]]
                agent_object = get_sentence_subtree_from_token(agent, ["cc", "conj"], inner=False)
                simpler_objs = split_conjunctions_obj(agent_object, agent, pobj)
                objs.extend(simpler_objs)

        for s in valid_subjs:
            for o in objs:
                if not o:
                    continue
                if o[0].pos_ == "ADP" and o[0].text == "by":
                    o = o[1:]
                    subject_token_list = list(o)
                    object_token_list = list(s)
                    triple = Triple(subject_token_list, [pred_token], object_token_list, sentence)
                else:
                    pred_prep = [pred_token, o[0]] if o[0].pos_ == "ADP" else [pred_token]
                    o = o[1:] if o[0].pos_ == "ADP" else o
                    subject_token_list = list(s)
                    object_token_list = [tk for span in o for tk in (span if isinstance(span, Span) else [span])]
                    triple = Triple(subject_token_list, pred_prep, object_token_list, sentence)

                triples.append(triple)

    # Fallback extraction if no triples matched
    if not triples:
        subj = next((t for t in doc if t.dep_ in ["nsubj", "nsubjpass"]), None)
        pred = next((t for t in doc if t.dep_ == "ROOT"), None)
        obj = next((t for t in doc if t.dep_ in ["attr", "dobj", "pobj", "oprd"]), None)

        if subj and pred and obj:
            subj_span = get_sentence_subtree_from_token(subj, ["cc", "conj"], inner=False)
            obj_span = get_sentence_subtree_from_token(obj, ["cc", "conj"], inner=False)
            triples.append(Triple(list(subj_span), [pred], list(obj_span), sentence))

    return triples



def simplify_sentence(nlp, sentence):
    subsentences = []

    if type(sentence) == str:
        doc = nlp(sentence)
        sentence = doc[:]
    else:
        doc = nlp(sentence.text)

    patt_relcl_attr = [
        {
            "RIGHT_ID": "relcl",
            "RIGHT_ATTRS": {"DEP": "relcl"}
        },
        {
            "LEFT_ID": "relcl",
            "REL_OP": ">",
            "RIGHT_ID": "pron",
            "RIGHT_ATTRS": {"LOWER": {"IN": ["who", "which"]}}
        },
        {
            "LEFT_ID": "relcl",
            "REL_OP": "<<",
            "RIGHT_ID": "attr",
            "RIGHT_ATTRS": {"DEP": "attr"}
        },
        {
            "LEFT_ID": "attr",
            "REL_OP": ">>",
            "RIGHT_ID": "suj",
            "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}
        }
    ]
    patt_relcl_obj = [
        {
            "RIGHT_ID": "relcl",
            "RIGHT_ATTRS": {"DEP": "relcl"}
        },
        {
            "LEFT_ID": "relcl",
            "REL_OP": ">",
            "RIGHT_ID": "pron",
            "RIGHT_ATTRS": {"LOWER": {"IN": ["who", "which"]}}
        },
        {
            "LEFT_ID": "relcl",
            "REL_OP": "<<",
            "RIGHT_ID": "obj",
            "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "pobj", "appos"]}}
        }
    ]
    patt_generic_relcl = [
        {
            "RIGHT_ID": "relcl",
            "RIGHT_ATTRS": {"DEP": "relcl"}
        },
        {
            "LEFT_ID": "relcl",
            "REL_OP": ">",
            "RIGHT_ID": "pron",
            "RIGHT_ATTRS": {"LOWER": {"IN": ["who", "which", "where"]}}
        }
    ]

    matcher = DependencyMatcher(nlp.vocab)
    matcher.add("patt_generic_relcl", [patt_generic_relcl])
    matcher.add("patt_relcl_attr", [patt_relcl_attr])
    matcher.add("patt_relcl_obj", [patt_relcl_obj])

    matches = matcher(doc)

    if not matches:
        subsentences.append(sentence.text)
        return subsentences

    match_id, token_ids = matches[-1]
    string_id = nlp.vocab.strings[match_id]
    span = doc[token_ids[0]]

    if sentence[token_ids[1] - 1].dep_ == 'punct':
        new_sentence = ' '.join([sentence[0:token_ids[1] - 1].text, sentence[token_ids[1]:-1].text])
        sentence = nlp(new_sentence)[0:-1]
    main_sentence = get_sentence_subtree_from_token(sentence.root, ["relcl"], nlp)
    second_sentence = get_sentence_subtree_from_token(span, nlp=nlp)

    new_sentence = []
    if string_id == "patt_relcl_attr":
        subj = [tkn for tkn in main_sentence if "subj" in tkn.dep_]
        if subj:
            subj = get_sentence_subtree_from_token(subj[0], nlp=nlp)
        else:  # for sentences including "there is" or "there are", which haven't subjects
            subj = get_sentence_subtree_from_token(sentence[token_ids[2]], ["relcl"], nlp)
        [new_sentence.append(subj) if tkn.i == doc[token_ids[1]].i else new_sentence.append(tkn) for tkn in second_sentence]
    elif string_id == "patt_relcl_obj":
        obj = doc[token_ids[2]]
        obj = get_sentence_subtree_from_token(obj, ["relcl"], nlp)
        [new_sentence.append(obj) if tkn.i == doc[token_ids[1]].i else new_sentence.append(tkn) for tkn in
         second_sentence]
    elif string_id == "patt_generic_relcl":
        pron = doc[token_ids[1]].text
        if pron in ["who", "which"]:
            subj = [tkn for tkn in main_sentence if "subj" in tkn.dep_]
            if subj:
                subj = get_sentence_subtree_from_token(subj[0], ["relcl"], nlp=nlp)
                [new_sentence.append(subj) if tkn.i == doc[token_ids[1]].i else new_sentence.append(tkn) for tkn in second_sentence]
        elif pron in ["where"]:
            [new_sentence.append(tkn) for tkn in second_sentence if tkn.i != doc[token_ids[1]].i]  # remove 'where'
    if new_sentence:
        second_sentence = ''.join([t.text_with_ws for t in new_sentence])
        subsentences.append(main_sentence.text)
        subsentences.append(second_sentence)
        return subsentences
    else:
        return []

def get_sentence_subtree_from_token(token, stop_condition=None, nlp=None, inner=True):
    if stop_condition is None:
        stop_condition = []
    sent = []
    for child in token.subtree:
        if inner:
            if child.dep_ in stop_condition and child != token:
                continue
        else:
            if (child.dep_ in stop_condition) and (child.i > token.i):
                break
        ancestors = [t for t in child.ancestors if t in token.subtree]
        if any([t for t in ancestors if t.dep_ in stop_condition and t != token]):
            if inner:
                continue
            else:
                break
        sent.append(child)
    sent.sort(key=lambda tkn: tkn.i)
    if inner and len(sent) == (sent[-1].i + 1 - sent[0].i):
        try:
            return sent[0].doc[sent[0].i: sent[-1].i + 1]
        except:
            return []
    elif inner and nlp:
        result = ''.join([tkn.text_with_ws for tkn in sent])
        return nlp(result)
    elif not inner:
        try:
            result = sent[0].doc[sent[0].i: sent[-1].i + 1]
            return result[:-1] if result[-1].is_punct else result
        except:
            return []
    return []


def check_ascending_order_token_id(token_id):
    """
    Check whether positions of tokens matched are in ascending order. The first token must be a verb,
    so matches must be at right side of the verb, avoiding some token occurring before the verb
    """
    up = True
    for i in range(1, len(token_id)):
        if token_id[i] < token_id[i - 1]:
            up = False
            break
    return up


def split_conjunctions_obj(span, prep, object_token):
    """
    Search for conjunctions in the object and splits in simpler objects, here is an example:
    Original: in China, India, the Muslim world, and Europe
    Result: in China | in India | in the Muslim world | in Europe
    Returns a list of objects
    """
    new_objects = []

    conjunctions = object_token.conjuncts
    if conjunctions:
        # Build the first object
        obj = get_sentence_subtree_from_token(object_token, ["cc", "conj"])
        if obj:
            new_objects.append([prep, obj])
        for conjunction in conjunctions:
            obj = get_sentence_subtree_from_token(conjunction, ["cc", "conj"])
            if obj:
                new_objects.append([prep, obj])
    else:
        # No conjunction tokens at the object part of the triple
        new_objects.append(span)
    return new_objects


def split_conjunctions_verbs(nlp, span):
    """
    Search for chained verbs by conjunctions and splits in simpler sentences, here is an example:
    Original: An engineer designs and directs projects
    Result: An engineer designs projects, An engineer directs projects
    """
    doc = nlp(span)

    first_verb = [token for token in doc if token.dep_ == "ROOT"]

    if first_verb[0].pos_ in ['VERB', 'AUX']:
        conjunctions = first_verb[0].conjuncts
        if conjunctions:
            simpler_sentences = []
            shared_text_left = doc[:first_verb[0].i]  # span previous to the first verb
            shared_text_right = doc[(conjunctions[-1].i + 1):]  # span after last verb

            new = [token for token in shared_text_left]
            new.append(first_verb[0])
            new.extend([token for token in shared_text_right])
            simpler_sentences.append(''.join([t.text_with_ws for t in new]))

            for c in conjunctions:
                new = [token for token in shared_text_left]
                new.append(c)
                new.extend([token for token in shared_text_right])
                simpler_sentences.append(''.join([t.text_with_ws for t in new]))

            return simpler_sentences
        else:
            return [span]
    else:
        return [span]


def get_all_triples(nlp, sentences):
    """ 
    Extract all the triples from the input list of sentences. Triples can be extracted from simple and complex senteces.
    Returns a list of objects of class Triple.
    """
    triples = []
    simple_sentences_tracking = []  # tracking
    for sentence in sentences:
        sententes_list = simplify_sentence(nlp, sentence)
        if len(sententes_list) > 1:
            i = 0  # number of sentences processed
            while len(sententes_list) != i:  # len(lista) is the number of sentences waiting to be simplified
                result = simplify_sentence(nlp, sententes_list[i])
                if len(result) == 1:  # check if the sentences was decomposed in simpler sentences
                    i = i + 1
                else:
                    del sententes_list[i]  # substitute complex sentence by simpler sentences
                    sententes_list[i:i] = result
        for sent in sententes_list:
            sent_simple_list = split_conjunctions_verbs(nlp, sent)

            [simple_sentences_tracking.append(simple) for simple in sent_simple_list]  # tracking
            for s in sent_simple_list:
                tps = get_simple_triples(nlp, s)
                if not tps:
                    log_failed_extraction(s, "No triples extracted")
                triples.extend(tps)

    additional_triples = []
    for t in triples:
        if len(t.objct) > 1:
            adj_tokens = [tok for tok in t.objct if tok.pos_ == "ADJ"]
            noun_tokens = [tok for tok in t.objct if tok.pos_ == "NOUN"]
            if adj_tokens and noun_tokens:
                is_from_tokens = list(nlp("is from"))
                nationality_triple = Triple(t.subj, is_from_tokens, adj_tokens, t.sent)
                additional_triples.append(nationality_triple)

    triples.extend(additional_triples)

    unique_triples = []
    seen = set()
    for t in triples:
        key = repr(t).lower()
        if key not in seen:
            seen.add(key)
            unique_triples.append(t)

    tracking_log(simple_sentences_tracking, level=3)  # tracking
    return unique_triples, len(simple_sentences_tracking)

