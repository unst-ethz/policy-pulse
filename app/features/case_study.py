from dash import get_relative_path, html, dcc


_share_step = lambda n: html.Div([
    html.H3(f"Step {n}: Share, export, or start over"),
    html.P(
        [
            "Found something interesting? The URL in your browser always reflects your current filters — "
            "copy and paste it to share the exact view with someone else. "
            "You can also hit ",
            html.Strong("Download CSV"),
            " (top right of the filter panel) to export the matching resolutions for further analysis in Excel, Python, or any other tool. "
            "When you want to start fresh, ",
            html.Strong("Reset Filters"),
            " (next to the download button) clears everything back to the default view.",
        ]
    ),
])

_tip = html.P(
    [
        html.Em("Tip: "),
        "We recommend opening the ",
        dcc.Link("Trends page", href=get_relative_path("/trends")),
        " in a separate tab so you can follow along step by step without losing your place.",
    ],
    style={"fontStyle": "italic", "color": "#666"},
)

layout = html.Div(
    [
        dcc.Tabs(
            id="case-study-tabs",
            value="case-study-1",
            children=[
                dcc.Tab(
                    label="Simple Case Study",
                    value="case-study-1",
                    children=[
                        html.Div(
                            [
                                _tip,
                                html.H2("How has Switzerland voted at the UN?"),
                                html.P(
                                    [
                                        "Want to see what the Policy Pulse platform can do? Let us walk through a concrete example. ",
                                        "Switzerland is an interesting case to start with: besides being our location of study, it hosts the UN's European headquarters in Geneva and the Human Rights Council, ",
                                        "but only became a full UN member in ",
                                        html.Strong("2002"),
                                        ", making it one of the last countries to join. ",
                                        "Let's look at how has it voted since then, and if its voting behavior aligns with its neighbours.",
                                    ]
                                ),
                                html.H3("Step 1: Open the Trends page"),
                                html.P(
                                    [
                                        "Head to the ",
                                        dcc.Link("Trends page", href=get_relative_path("/trends")),
                                        ". This is where all the analysis tools live. "
                                        "You'll see a filter panel at the top and a set of tabs below it, let's walk through them.",
                                    ]
                                ),
                                html.H3("Step 2: Select Switzerland as your primary country"),
                                html.P(
                                    [
                                        "In the filter panel, use the ",
                                        html.Strong("Main country"),
                                        " dropdown to select ",
                                        html.Strong("Switzerland"),
                                        ". This tells the platform whose voting record you want to explore. "
                                        "The ",
                                        html.Strong("Resolutions"),
                                        " tab (which is active by default) will now show how Switzerland voted on each resolution — "
                                        "you can sort by date and filter by vote type (Yes, No, Abstain).",
                                    ]
                                ),
                                html.H3("Step 3: Focus on a subject area"),
                                html.P(
                                    [
                                        "By default, you're looking at all resolutions. Use the ",
                                        html.Strong("Subjects"),
                                        " dropdown to narrow the view. For example, select ",
                                        html.Strong("Human rights"),
                                        " to see only resolutions tagged with that topic. "
                                        "You can also select multiple subjects at once. "
                                        "Where applicable, most features — e.g., the resolution list, the map, the timeline — will update to reflect your subject selection.",
                                    ]
                                ),
                                html.P(
                                    [
                                        "This works the other way too. If you want to exclude a dominant topic to get a clearer picture of "
                                        "the rest, you can select every subject ",
                                        html.Em("except"),
                                        " that one. For instance, resolutions on the Question of Palestine make up a large share of "
                                        "the dataset. Filtering them out can reveal patterns that would otherwise be hidden. "
                                        "You can also use the ",
                                        html.Strong("Keyword Search"),
                                        " to find resolutions by title (e.g., searching \"disarmament\" or \"climate\").",
                                    ]
                                ),
                                html.H3("Step 4: See who agrees with Switzerland"),
                                html.P(
                                    [
                                        "Switch to the ",
                                        html.Strong("Agreement Map"),
                                        " tab. A world map appears, colour-coded from red (disagreement) to blue (agreement), "
                                        "showing how closely each country's voting aligns with Switzerland's ("
                                        "now filtered to your selected subject area). "
                                        "You might notice that, given you are comparing to Switzerland, surrounding Western European countries tend to cluster in blue, "
                                        "while other regions show more variation.",
                                    ]
                                ),
                                html.H3("Step 5: Compare with Germany over time"),
                                html.P(
                                    [
                                        "Go back to the filter panel and add ",
                                        html.Strong("Germany"),
                                        " as a comparison country using the ",
                                        html.Strong("Compare with"),
                                        " dropdown. Now switch to the ",
                                        html.Strong("Agreement Timeline"),
                                        " tab. You'll see a smoothed line chart showing how closely Switzerland and Germany "
                                        "have voted over the years. If you observe any dips, these may correspond to periods "
                                        "where the two countries diverged on specific issues.",
                                    ]
                                ),
                                html.H3("Step 6: Dive deeper into subject areas"),
                                html.P(
                                    [
                                        "Open the ",
                                        html.Strong("Alignment by Subject"),
                                        " tab. This breaks down the Switzerland–Germany comparison by UN subject area "
                                        "(e.g., human rights, disarmament, economic development). "
                                        "You can see at a glance which topics they agree on most and where they diverge. "
                                        "Subjects need at least 30 common votes to appear, so very niche topics are filtered out.",
                                    ]
                                ),
                                html.H3("Step 7: Try a group preset"),
                                html.P(
                                    [
                                        "Want a broader comparison? Use the ",
                                        html.Strong("Quick Select"),
                                        " dropdown next to the comparison country to load a preset group. "
                                        "For example, ",
                                        html.Strong("P5"),
                                        " (the five permanent Security Council members) or ",
                                        html.Strong("GRULAC"),
                                        " (Latin America & Caribbean). The Agreement Timeline will now show multiple lines, "
                                        "letting you compare Switzerland's alignment with several countries at once.",
                                        "Note that this is also possible by simply selecting multiple countries in the ",
                                        html.Strong("Compare with"),
                                        " dropdown.",
                                    ]
                                ),
                                html.H3("Step 8: Narrow by time period"),
                                html.P(
                                    [
                                        "Use the ",
                                        html.Strong("Year Range"),
                                        " filter or pick an era preset like ",
                                        html.Strong("Recent (2015–present)"),
                                        " to focus on a specific period. "
                                        "This is useful if you want to see how relationships have shifted in recent years "
                                        "without older Cold War-era votes affecting the averages.",
                                    ]
                                ),
                                html.H3("What to take away"),
                                html.P(
                                    "Even this quick exploration shows that voting records at the UN are rich with insights of both historical and contemporary relevance, "
                                    " a single country's voting behavior, as well as global patterns and a pulse of the state of international relations. "
                                    "They might reflect certain diplomatic strategies or regional alliances, as well as the specific "
                                    "issues on the table at any given time. The Policy Pulse platform makes these patterns visible, "
                                    "allowing for data-driven exploration of international relations through the lens of UN voting behavior."
                                ),
                            ]
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Advanced Case Study",
                    value="case-study-2",
                    children=[
                        html.Div(
                            [
                                _tip,
                                html.H2("Diving Deeper: Tracing a Political Transition Through Votes"),
                                html.P(
                                    [
                                        "Now that you have mastered the basics, let's attempt to explore some more detailed analysis: this case study compares the voting records of ",
                                        html.Strong("Bulgaria"),
                                        " (an EU member state in southeastern Europe) and ",
                                        html.Strong("Angola"),
                                        " (a southern African nation). "
                                        "Let's use the Policy Pulse platform to investigate where their votes align and where they diverge.",
                                    ]
                                ),
                                html.H3("Step 1: Set up the comparison"),
                                html.P(
                                    [
                                        "Head to the ",
                                        dcc.Link("Trends page", href=get_relative_path("/trends")),
                                        ". Set ",
                                        html.Strong("Bulgaria"),
                                        " as the main country and ",
                                        html.Strong("Angola"),
                                        " as the comparison country.",
                                    ]
                                ),
                                html.H3("Step 2: Look at how agreement has changed over time"),
                                html.P(
                                    [
                                        "Switch to the ",
                                        html.Strong("Agreement Timeline"),
                                        " tab. Each point on the chart represents the average agreement score for one UN General Assembly session. The chart spans from ",
                                        html.Strong("1976"),
                                        " (when Angola joined the UN) to the present. "
                                        "Through the late 1970s and 1980s, the two countries voted together on roughly ",
                                        html.Strong("75–100%"),
                                        " of resolutions per session. This changed sharply around ",
                                        html.Strong("1990"),
                                        ": agreement dropped from ~98% to ~63% within just a few sessions, coinciding with Bulgaria's transition away from its Soviet-aligned government following the end of the Cold War. "
                                        "Since then, it has mostly remained in a band of around ",
                                        html.Strong("65–75%"),
                                        ", with a notable dip to around ",
                                        html.Strong("57%"),
                                        " in a most recent special session.",
                                    ]
                                ),
                                html.P(
                                    [
                                        "Use the ",
                                        html.Strong("Year Range"),
                                        " filter to zoom in on specific periods — for example, compare the pre-1990 era to the post-1995 one.",
                                    ]
                                ),
                                html.H3("Step 3: Compare agreement across subject areas"),
                                html.P(
                                    [
                                        "Open the ",
                                        html.Strong("Alignment by Subject"),
                                        " tab. The overall agreement rate varies considerably depending on the topic.",
                                    ]
                                ),
                                html.P(
                                    [
                                        html.Strong("Highest agreement: "),
                                        html.Strong("Population"),
                                        " (~94%) and ",
                                        html.Strong("Industry"),
                                        " (~94%) show the strongest alignment, followed by ",
                                        html.Strong("Geographical Descriptors"),
                                        " (~90%), ",
                                        html.Strong("Economic Development & Development Finance"),
                                        " (~89%) and ",
                                        html.Strong("Science & Technology"),
                                        " (~88%). These subject areas consistently see broad cross-regional consensus in the General Assembly.",
                                    ]
                                ),
                                html.P(
                                    [
                                        html.Strong("Lowest agreement: "),
                                        html.Strong("Social Conditions & Equity"),
                                        " has the lowest score at ~57% compared to the other subjects. ",
                                        html.Strong("International Trade"),
                                        " (~70%) and ",
                                        html.Strong("Political & Legal Questions"),
                                        " (~74%), which is by far the largest category by number of votes, also fall below the overall average.",
                                    ]
                                ),
                                # html.H3("Step 4: Coming soon"),
                                html.H3("Step 4: Trace the transition through Political & Legal Questions"),
                                html.P(
                                    [
                                        "The subject area where the transition is most visible is ",
                                        html.Strong("Political & Legal Questions"),
                                        " — also the largest category in the dataset by vote count. "
                                        "Select it in the ",
                                        html.Strong("Subjects"),
                                        " dropdown, then use the ",
                                        html.Strong("Year Range"),
                                        " filter to step through consecutive five-year windows around Bulgaria's political transition:",
                                    ]
                                ),
                                html.Ul([
                                    html.Li([html.Strong("1980–1985: "), "95% agreement (217 votes)"]),
                                    html.Li([html.Strong("1985–1990: "), "94% agreement (387 votes) — the transition begins"]),
                                    html.Li([html.Strong("1990–1995: "), "67% agreement (151 votes) — post transition dip"]),
                                    html.Li([html.Strong("1995–2000: "), "71% agreement (159 votes)"]),
                                ]),
                                html.P(
                                    [
                                        "The drop from ",
                                        html.Strong("95% to 67%"),
                                        " in the space of a few years illustrates how directly a change in a country's political system "
                                        "can register in its UN voting record. However, also note that this score is also heavily influenced "
                                        "by which exact votes are being held during a time period and therefore only represents a snapshot "
                                        "of a countries voting pattern.",
                                    ]
                                ),
                                html.H3("Step 5: Per-vote exploration"),
                                html.P(
                                    [
                                        "But numbers only tell one story, to get an insight on an actual per-vote basis, head back to the ",
                                        html.Strong("Resolutions"),
                                        " tab. With ",
                                        html.Strong("Political & Legal Questions"),
                                        " and the time window ",
                                        html.Strong("1990–1995"),
                                        " still selected, you can see the list of resolutions pertaining to this subject that were voted on during this period. "
                                        "Additionally, you can filter by vote type to see, for example, only the resolutions where Bulgaria and Angola cast opposing votes. "
                                        "Click any resolution to open its full record on the UN Digital Library — including the per-country vote breakdown.",
                                    ]
                                ),
                                _share_step(6),
                                html.H3("What to take away"),
                                html.P(
                                    "The Bulgaria–Angola comparison shows how two countries may converge strongly on some topics while diverging significantly on others, "
                                    "and how those patterns can shift over time. "
                                    "The subject breakdown and timeline features of the Policy Pulse platform are designed to surface exactly these kinds of differences."
                                ),
                            ]
                        ),
                    ],
                ),
            ],
        ),
    ]
)
