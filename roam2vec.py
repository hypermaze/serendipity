# -*- coding: utf-8 -*-
"""roam2vec.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1e5CmsV_ZY7WHh1y5Kw2MP49R4-KlVqtq
"""

import os
import shutil
from sentence_transformers import SentenceTransformer, models, util
import torch
import faiss
import numpy as np
import pandas as pd
import itertools
import re
import json
from toolz import thread_first, thread_last
from collections.abc import Iterable
from datetime import date

"""## Utilities"""

from typing import List, Dict, Union
def pipe(*funcs:List[callable], thread="first"):
    thread = thread_first if thread == "first" else thread_last
    return lambda data: thread(data, *funcs)

URL_REGEX = '\(?((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*\)?'

def remove_html_tags(form, content="HTML_TAGS"):
    html_tags = re.compile('<.*?>')
    return re.sub(html_tags, '', form)

def remove_buttons(form):
    button_elems = re.compile('\{\{\[\[(TODO|DONE|slider)\]\]\}\}')
    return re.sub(button_elems, '', form)

def remove_url(form):
    url = re.compile(URL_REGEX)
    return re.sub(url, '', form)

def remove_attr(form):
    attr = re.compile('^[^:\r\n]+:*')
    return re.sub(attr, '', form)

def replace_block_ref(form, lookup):
    """good job here... my implementation in js was much worse"""
    block_ref = re.compile('\(\((.*)\)\)')
    block = re.search(block_ref, form)
    if block:
        return lookup.get(re.sub('[()]', '', block.group(0)))
    return form

def remove_duplicates(form: Iterable):
    if isinstance(form, list):
        return list(set(form))
    return form

def roam_graph_to_blocks(roam_graph):
    roam_blocks_map = {}

    def extract_strings(roam_block):
        if type(roam_block) == list:
            roam_block = roam_block[0]
        if roam_block.get("string") and roam_block.get("uid"):
            roam_blocks_map[roam_block.get("uid")] = roam_block.get("string")
        if roam_block.get("children"):
            for child in roam_block.get("children"):
                extract_strings(child)

    for block in roam_graph:
        extract_strings(block)

    return roam_blocks_map

STRING_FUNCS = ["capitalize", "count", "isalnum", "isalpha", "isascii", "isdecimal", "isdigit", "isidentifier", "islower",
                "isnumeric", "isspace", "istitle", "isupper", "lower", "lstrip", "replace", "rstrip", "split", "strip", "upper"]

for s_func in STRING_FUNCS: #PYTHON MAGIC
    exec("%s=getattr(str, s_func)" %s_func)

clean_sentence = pipe(
#                       replace_block_ref,
                      remove_buttons,
                      remove_html_tags,
                      remove_url,
#                       remove_attr, #attention !!! buggy!
                      (replace, "  ", " "),
                      (replace, "[", ""),
                      (replace, "]", ""),
                      (replace, "#", ""),
                      (replace, "`", ""),
                      (replace, "__", ""),
                      (replace, "~~", ""),
                      (replace, "**", ""),
                      (replace, "^^", ""),
                      strip,
                      lower
                      )

def is_too_short(s, length=10):
    return len(s.split(" ")) < length

stop_symbols = ["TODO", "DONE", "::", "```", "!["]
def has_stop_symbols(s):
    return any(symbol in s for symbol in stop_symbols)

"""## Roam2Vec

TODO
* compress embeddings
"""


with open("roam.json", "r") as f:
    roam_data = json.loads(f.read())

roam_blocks = roam_graph_to_blocks(roam_data)

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

blocks = [(uid, clean_sentence(sentence)) for i, (uid, sentence) in enumerate(roam_blocks.items())  if not is_too_short(sentence) and not has_stop_symbols(sentence)]
#blocks = [(uid, sents), (uid, sents) ]

uids = [uid for uid, sentence in blocks]
sentences = [sentence for uid, sentence in blocks]
embeddings = model.encode(sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
embeddings = embeddings.cpu().detach().numpy()

len(sentences), len(roam_blocks.items())

"""Save the text + embeddings as JSON for API use"""

with open("../roam_index.json", "w") as outfile:
    roam_index = [
        {
            "uid": uids[idx],
            "sentence": sentences[idx],
            "embedding": embeddings[idx],
        }
        for idx in range(len(sentences))
    ]

    json.dump(index_list, outfile)

"""## FAISS QUERIES

This is basically what the server is running. Quite simple and fast
"""

def create_index(embeddings):
    faiss_index = faiss.IndexFlatL2(len(embeddings[0]))
    faiss_index.add(embeddings)

#     print(faiss_index.ntotal)

    return faiss_index
def query_index(text, model, target_list, index, with_distance=False, k=10):
    embedding = model.encode([text])
    distances, indices = index.search(embedding, k)
    if with_distance:
        return [(target_list[index], distances[0][i]) for i, index in enumerate(indices[0])]
    return [target_list[i] for i in indices[0]]

#when loading: embeddings = np.array([obj.get("embedding") for obj in roam_index], dtype=np.float32)
index = create_index(embeddings)
print(query_index("Debating program languages is for nerds", model, sentences, index, with_distance=True, k =8))
