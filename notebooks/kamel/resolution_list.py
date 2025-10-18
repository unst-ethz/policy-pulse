
import streamlit as st
import pandas as pd
import datetime 

from resolution_analyzer import UNResolutionAnalyzer
analyzer = UNResolutionAnalyzer(config_path='config/data_sources.yaml')



PAGE_SIZE = 100

if 'results' not in st.session_state:
    st.session_state.results = pd.DataFrame()
if 'num_shown' not in st.session_state:
    st.session_state.num_shown = PAGE_SIZE


st.title('United Nations Resolution Finder')
st.markdown("Select your criteria below to find relevant resolutions.")


subject_options = analyzer.subject_df["label_en"]

title_to_id_map = pd.Series(
            analyzer.subject_df.subject_id.values,
            index=analyzer.subject_df.label_en
        ).to_dict()


selected_titles = st.multiselect(
    label='Choose subjects:',
    options=subject_options
)


country_options = []

for c in analyzer.resolution_table.columns:
    if c not in ["undl_id", "date", "session", "resolution", "draft", "committee_report", "meeting", "title", "agenda_title", "subjects", "total_yes", "total_no", "total_abstentions", "total_non_voting", "total_ms", "undl_link"]:
        country_options.append(c)
selected_countries = st.multiselect(
    "Choose up to 2 countries to see their votes:",
    options=country_options,
    max_selections=2
)

vote_agreement_filter = "All resolutions"
if len(selected_countries) == 2:
    vote_agreement_filter = st.radio(
        "Filter by vote agreement:",
        ("All resolutions", "Agreements", "Disagreements", "Strong disagreements"),
        horizontal=True,
    )

country_vote_filter = "All resolutions"
if len(selected_countries) == 1:
    vote_agreement_filter = st.radio(
        "Filter by vote:",
        ("All resolutions", "Yes", "Abstained", "No", "Didn't Vote"),
        horizontal=True,
    )

min_un_date = datetime.date(1945, 1, 1)


col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "Start date (optional)",
        value=None,
        min_value=min_un_date,
        format="YYYY/MM/DD"
    )

with col2:
    end_date = st.date_input(
        "End date (optional)",
        value=None,
        min_value=min_un_date,
        format="YYYY/MM/DD"
    )

find_button = st.button('Find Resolutions', type="primary")



if find_button:

    st.session_state.results = analyzer.query(
        subject_ids=[title_to_id_map[title] for title in selected_titles], start_date=start_date, end_date=end_date
    )

    st.session_state.num_shown = PAGE_SIZE


if not st.session_state.results.empty:
    results_df = st.session_state.results.copy()

    if len(selected_countries) == 1:
        country = selected_countries[0]
        if vote_agreement_filter == "Yes":
            results_df = results_df[results_df[country] == "Y"]
        elif vote_agreement_filter == "Abstained":
            results_df = results_df[results_df[country] == "A"]
        elif vote_agreement_filter == "No":
            results_df = results_df[results_df[country] == "N"]
        elif vote_agreement_filter == "Didn't Vote":
            results_df = results_df[results_df[country] == "X"]
        results_df.dropna(subset=[country], inplace=True)





    if len(selected_countries) == 2:
        country1, country2 = selected_countries
        if vote_agreement_filter == "Agreements":
            results_df = results_df[results_df[country1] == results_df[country2]]
        elif vote_agreement_filter == "Disagreements":
            results_df = results_df[results_df[country1] != results_df[country2]]
        elif vote_agreement_filter == "Strong disagreements":
            results_df = results_df.query(f'({country1} == "Y" and {country2} == "N") or ({country1} == "N" and {country2} == "Y")')
        results_df.dropna(subset=[country1, country2], inplace=True)   


    total_results = len(results_df)

    st.subheader("Results")
    st.success(f"Found {total_results} resolution(s):")

    df_to_display = results_df.head(st.session_state.num_shown)
    for index, row in df_to_display.iterrows():
        display_text = f"**{row['resolution']} -- {row['date']}**: {row['title']}: {row['agenda_title']}"

        if selected_countries:
            vote_parts = []
            for country in selected_countries:
                vote = row[country]
                if vote == 'Y':
                    color = 'green'
                elif vote == 'A':
                    color = 'orange'
                elif vote == 'X':
                    color = 'blue'
                else: # 'N'
                    color = 'red'
                vote_parts.append(f":{color}[● {country}]")

            
            display_text += f"&nbsp;&nbsp;{'&nbsp;&nbsp;'.join(vote_parts)}"

        st.markdown(display_text, unsafe_allow_html=True)
    if total_results > st.session_state.num_shown:
        if st.button("Load More"):
            st.session_state.num_shown += PAGE_SIZE
            st.rerun()
else:
    if find_button:
        st.info("No resolutions found with the selected criteria.")






