from youtube_transcript_api import YouTubeTranscriptApi
#import PySimpleGUI as sg
#import nltk
import spacy 
from collections import Counter
import numpy as np
import pandas as pd
import streamlit as st
from transformers import pipeline


st.title("YouTube Transcript Summarizer")

url = "https://www.youtube.com/watch?v=UF8uR6Z6KLc"
url = st.text_input("Enter the url")


video_id = url.split("=")[1]

#print(video_id)

transcript = YouTubeTranscriptApi.get_transcript(video_id)

final_text = ""

for line in transcript:
    for key in line:
        if key=="text":
            final_text += line[key]

#print(final_text)


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = (summarizer(final_text, max_length=130, min_length=30, do_sample=False,truncation = True))
print(type(summary))
print(summary)

summary_list = []

for i in summary:
    summary_list.append(i["summary_text"])
final_summary = " ".join(summary_list)

st.write(final_summary)

