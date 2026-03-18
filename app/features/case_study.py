from dash import html, dcc


layout = html.Div(
    [
        html.Div(
            [
                html.P(
                    [
                        "Want to see what Policy Pulse can do? Let us walk through a concrete example. ",
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
                        dcc.Link("Trends page", href="/trends"),
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
                # Step 4
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
                # Step 5
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
                        html.Strong("Nordic"),
                        " countries or ",
                        html.Strong("BRICS"),
                        ". The Agreement Timeline will now show multiple lines, "
                        "letting you compare Switzerland's alignment with several countries at once.",
                        "Note that this is also possible by simply selecting multiple countries in the ",html.Strong("Compare with")," dropdown.",
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
                    "issues on the table at any given time. Policy Pulse makes these patterns visible, "
                    "allowing for data-driven exploration of international relations through the lens of UN voting behavior."
                ),
            ]
        ),
    ]
)