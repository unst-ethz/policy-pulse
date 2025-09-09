# UN General Assembly Voting Data: Description and Terms of Use

## Title

United Nations General Assembly Voting Data: resolutions 1 (11 December 1946) to 79/317 (18 July 2025)

## Contributor

UN Dag Hammarskjöld Library

## Publisher

United Nations

## Date of Publication

2025-07-23

## Summary

This dataset is a compilation of the voting data related to the resolutions adopted by the General Assembly at its regular, special, emergency special sessions from its inception until July 2025. It is derived from the UN Digital Library voting records and provides information on Member States' votes related to resolutions adopted through a recorded vote. Votes on individual paragraphs of draft resolutions or drafts that failed to be adopted are not included. The dataset comprises 914,624 entries, with each entry representing the vote of one Member State on a particular resolution.
Copyright, United Nations; Non commercial use, with attribution.
## Additional Dataset: Resolution-Level Voting Data
In addition to the Member State-level voting dataset, a transformed dataset has been created with one entry per resolution. This dataset aggregates voting information for each resolution and includes columns for resolution metadata and voting results. For each country, there is a dedicated column indicating its vote (Y/N/A/X) for that resolution.

## Citation

United Nations Dag Hammarskjöld Library, General Assembly Voting Data, United Nations, version 3 (July 2025), 2025, downloaded from https://digitallibrary.un.org/record/4060887, [download date]

## Version

Version 3 (July 2025)
Older versions: Version 2 (March 2025)

## Data Dictionary

* undl_id: UN Digital Library control number.
* ms_code: Member State three-letter code.
* ms_name: Member State official name at the time of the vote.
* ms_vote: Member State vote (Y for yes, N for no, A for abstention, X for non-voting).
* date: date of the vote.
* session: session number.
* resolution: symbol of the adopted resolution.
* draft: symbol of the draft resolution (433,659 empty rows).
* committee_report: symbol of the report of the main General Assembly Committee, transmitting texts of draft resolutions (184,394 empty rows).
* meeting: symbol of the meeting record during which the resolution was adopted.
* title: title of the resolution.
* agenda_title: title of the agenda item (12,249 empty rows).
* subjects: concept or combination of concepts describing the subject of the resolution (234,897 empty rows).
* vote_note: additional voting information (914,265 empty rows).
* total_yes: number of "yes" votes.
* total_no: number of "no" votes.
* total_abstentions: number of abstentions.
* total_non_voting: number of non-voting members.
* total_ms: total number of Member States.
* undl_link: link to the voting record in the UN Digital Library.

## Data Dictionary: Transformed Resolution-Level Dataset
* undl_id: UN Digital Library control number for the resolution.
* resolution: Symbol of the adopted resolution.
* date: Date of the vote.
* session: Session number.
* draft: symbol of the draft resolution (433,659 empty rows).
* committee_report: Symbol of the report of the main General Assembly Committee, transmitting texts of draft resolutions.
* meeting: Symbol of the meeting record during which the resolution was adopted.
* title: Title of the resolution.
* agenda_title: Title of the agenda item.
* subjects: Concept or combination of concepts describing the subject of the resolution.
* total_yes: Number of "yes" votes.
* total_no: Number of "no" votes.
* total_abstentions: Number of abstentions.
* total_non_voting: Number of non-voting members.
* total_ms: Total number of Member States.
* undl_link: Link to the voting record in the UN Digital Library.
* For each country: A column with the country name or code, indicating the specific vote (Y/N/A/X) for that resolution.

## Applicable Terms of Use

* UN Digital Library terms of use: https://digitallibrary.un.org/pages/?ln=en&page=tos
* Terms and conditions of use of United Nations websites: https://www.un.org/en/about-us/terms-of-use

## Disclaimer

Content provided by the Dag Hammarskjöld Library is offered AS IS. Refer to the UN Library disclaimers: http://www.un.org/en/sections/about-website/terms-use/index.html#Disclaimers. The Dag Hammarskjöld Library disclaims any and all warranties, whether expressed or implied, and is providing this content, data, and metadata AS IS. By using it, you acknowledge that the Library does not make any representations or warranties about the content, metadata, data, material, and information, and you agree that you are solely responsible for your reuse of content made available by the Library.