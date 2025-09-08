# UN Security Council Voting Data: Description and Terms of Use
## Title
United Nations Security Council Voting Data: resolution 1 (24 January 1946) to resolution 2787 (17 July 2025)
## Contributor
UN. Dag Hammarskjöld Library
## Publisher
United Nations
## Date of Publication
2025-07-21
## Summary
This dataset is a compilation of the voting data of the Security Council until 21 July 2025. It is derived from the UN Digital Library voting records and provides information on Member States' votes related to adopted resolutions up to resolution 2787 (2025) adopted on 17 July 2025. Votes on paragraphs of draft resolutions or drafts that failed to be adopted are not included. The dataset comprises 40929 entries, representing the vote of one Member State for a particular draft resolution. The dataset is updated several times a year; the next update is scheduled for the end of September 2025.
Copyright, United Nations; non-commercial use with attribution.
## Citation
United Nations Dag Hammarskjöld Library, Security Council Voting Data, United Nations, 2025, Version 3 (21 July 2025), downloaded from https://digitallibrary.un.org/record/4055387, [download date]
# Versions
Version 3 (21 July 2025)
Older versions: Version 1 (30 June 2024), version 2 (31 March 2025)
## Data Dictionary
* undl_id: Unique identifier for the voting record.
* ms_code: Member State three-letter code.
* ms_name: Member State official name at the time of the vote.
* permanent_member: True/False indicates if the Member State is a permanent member of the Council.
* ms_vote: Member State vote (Y for yes, N for no, A for abstention, X for non-voting); when no vote was taken and the resolution was adopted unanimously/by acclamation, votes are recorded as "Y".
* date: Date of the vote.
* resolution: Symbol of the adopted resolution.
* draft: Symbol of the draft resolution; 14078 rows without values; this data point is completed from 1998 onwards.
* meeting: Symbol of the meeting record during which the resolution was adopted.
* description: Resolution number and short description of the topic added by the Library.
* agenda: Agenda item of the Security Council; 6149 rows without values.
* subjects: Concepts that describe the subject of the adopted resolution and/or draft; 7509 rows without values; this data point is complete for votes dated 1985 onwards.
* vote_note: A note is only added to indicate peculiarities: a Member State that was not present and/or did not participate in the vote, adoption of the resolution without a vote or by acclamation, etc; 38415 rows without values.
* modality: Indicates if the resolution was adopted with or without a vote.
* total_yes: Number of "yes" votes.
* total_no: Number of "no" votes.
* total_abstentions: Number of abstentions.
* total_non_voting: Number of non-voting members.
* total_ms: Total number of Security Council members.
* modality: Indicates if the resolution was adopted with or without a vote.
* undl_link: Link to the voting record in the UN Digital Library.
## Applicable Terms of Use
* UN Digital Library terms of use: https://digitallibrary.un.org/pages/?ln=en&page=tos
* Terms and conditions of use of United Nations websites: https://www.un.org/en/about-us/terms-of-use
## Disclaimer
Content provided by the Dag Hammarskjöld Library is offered AS IS. Refer to the UN Library disclaimers: http://www.un.org/en/sections/about-website/terms-use/index.html#Disclaimers. The Dag Hammarskjöld Library disclaims any and all warranties, whether expressed or implied, and is providing this content, data, and metadata AS IS. By using it, you acknowledge that the Library does not make any representations or warranties about the content, metadata, data, material, and information, and you agree that you are solely responsible for your reuse of content made available by the Library.