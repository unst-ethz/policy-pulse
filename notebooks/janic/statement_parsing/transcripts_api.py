""" This is a helper file to work with https://transcripts.un.org/. We define custom classes and 
fetching logic

TODO: quality checks
"""
## Check out the documentation here: -> https://transcripts.un.org/openapi
## Example request: https://transcripts.un.org/en/asset/k1m/k1mw02yjrj.json


#########################################################################
# Imports ###############################################################
#########################################################################

from pydantic import BaseModel, Field, ValidationError
from pathlib import Path

#########################################################################
# Classes ###############################################################
#########################################################################

class Speaker(BaseModel):
    affiliation: str | None # ISO alpha-3 country code or short affiliation label
    affiliation_full: str | None # Official UN country name resolved from the ISO code
    group: str | None
    function: str | None

class Topic(BaseModel):
    key: str
    label: str
    description: str

class Sentence(BaseModel):
    text: str
    start: float
    end: float
    topics: list[Topic]

class Paragraph(BaseModel):
    sentences: list[Sentence]

class Statement(BaseModel):
    statement_number: int
    start: float
    #page_url
    paragraphs: list[Paragraph]
    speaker: Speaker

class Transcript(BaseModel):
    transcript_id: str
    language: str
    data: list[Statement]
    topics: list[Topic]

class Video(BaseModel):
    slug: str
    title: str
    date: str
    body: str
    session_number: str | None
    pv_symbol: str | None

class MeetingResponse(BaseModel):
    disclaimer: str
    url: str
    video: Video
    transcript: Transcript | None


#########################################################################
# Constants #############################################################
#########################################################################

#BASE = "https://transcripts.un.org"
#DATA = "./transcripts_data"
# GA_BODIES = {"General Assembly"} | {
#     f"{n} Committee" for n in ("First", "Second", "Third", "Fourth", "Fifth", "Sixth")
# }

#########################################################################
# Helper Functions ######################################################
#########################################################################




#########################################################################
# Main Functions ########################################################
#########################################################################

