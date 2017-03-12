#!/usr/bin/env python
# -*- coding: utf-8 -*-import re

import os
import re
import string

import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

csv_dir = 'csv/'


def read_csv(dir):
    dataframes = {
        "cooking": pd.read_csv(csv_dir + "cooking.csv"),
        "crypto": pd.read_csv(csv_dir + "crypto.csv"),
        "robotics": pd.read_csv(csv_dir + "robotics.csv"),
        "biology": pd.read_csv(csv_dir + "biology.csv"),
        "travel": pd.read_csv(csv_dir + "travel.csv"),
        "diy": pd.read_csv(csv_dir + "diy.csv"),
    }
    return dataframes


def strip_tags_and_uris(x):
    uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
    if x:
        # BeautifulSoup on content
        soup = BeautifulSoup(x, "html.parser")
        # Stripping all <code> tags with their content if any
        if soup.code:
            soup.code.decompose()
        # Get all the text out of the html
        text = soup.get_text()
        # Returning text stripping out all uris
        return re.sub(uri_re, "", text)
    else:
        return ""


def strip_dataframes_content(dataframes):
    for df in dataframes.values():
        df["content"] = df["content"].map(strip_tags_and_uris)


def remove_special(x, punctuation=True):
    # Lowercasing all words
    x = x.lower()
    # Removing non ASCII chars
    x = re.sub(r'[^\x00-\x7f]', r' ', x)
    if punctuation:
        return x
    else:
        # Removing (replacing with empty spaces actually) all the punctuations
        return re.sub("[" + string.punctuation + "]", " ", x)


def remove_dataframes_special(dataframes, punctuation=True):
    for df in dataframes.values():
        df["title"] = [remove_special(t, punctuation) for t in df["title"]]
        df["content"] = [remove_special(c, punctuation) for c in df["content"]]
        #     df["title"] = df["title"].map(remove_special)
        # df["content"] = df["content"].map(savePunctuation)


def remove_stopwords(x):
    stops = set(stopwords.words("english"))
    # Removing all the stopwords
    filtered_words = [word for word in x.split() if word not in stops]
    return " ".join(filtered_words)


def remove_dataframes_stopwords(dataframes):
    for df in dataframes.values():
        df["title"] = df["title"].map(remove_stopwords)
        df["content"] = df["content"].map(remove_stopwords)


def tags_to_list(dataframes):
    for df in dataframes.values():
        # From a string sequence of tags to a list of tags
        df["tags"] = df["tags"].map(lambda x: x.split())


def save(dataframes, dir, appended_name):
    dir = fix_dir(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    for name, df in dataframes.items():
        # Saving to file
        df.to_csv(dir + name + "_" + appended_name + ".csv", index=False)


def fix_dir(dir):
    if not dir.endswith('/'):
        dir += '/'
    return dir


def preprocess(dir, html=False, special=False, punctuation=True, stopwords=True, taglist=True):
    dir = fix_dir(dir)
    dataframe = read_csv(dir)

    sep = '_'
    s_html = 'html' + sep
    s_special = 'special' + sep
    s_punctuation = 'punctuation' + sep
    s_stopwords = 'stopwords' + sep
    s_taglist = 'taglist' + sep

    if html:
        s_html = s_html.upper()
    else:
        strip_dataframes_content(dataframe)

    if punctuation:
        s_punctuation = s_punctuation.upper()

    if special:
        s_special = s_special.upper()
    else:
        remove_dataframes_special(dataframe, punctuation)

    if stopwords:
        s_stopwords = s_stopwords.upper()
    else:
        remove_dataframes_stopwords(dataframe)

    if taglist:
        tags_to_list(dataframe)
        s_taglist = s_taglist.upper()

    return dataframe, s_html+s_special+s_punctuation+s_stopwords+s_taglist
