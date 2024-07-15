import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import base64
import warnings
import dash
from dash import dcc, html, dash_table, Dash
from dash import dcc, html, Input, Output, State  # Add State here
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
warnings.simplefilter(action='ignore', category=FutureWarning)


# Read the datasets
ball_by_ball_data = pd.read_csv("datasets/IPL_Ball_by_Ball_2008_2022.csv")
match_data = pd.read_csv("datasets/IPL_Matches_2008_2022.csv")

# Merge ball by ball data with match data
ball_by_ball_data = pd.merge(ball_by_ball_data, match_data[['ID', 'Team1', 'Team2', 'Season', 'Venue']], how='left',
                             on='ID')

# Rename venues for consistency
ball_by_ball_data["Venue"].replace({
    'Wankhede Stadium': 'Wankhede Stadium, Mumbai',
    'M Chinnaswamy Stadium': 'M.Chinnaswamy Stadium',
    'Eden Gardens, Kolkata': 'Eden Gardens',
    'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium',
    'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium',
    'MA Chidambaram Stadium': 'MA Chidambaram Stadium, Chepauk, Chennai',
    'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium, Chepauk, Chennai',
    'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium',
    'Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium'
}, inplace=True)

top_batsmen = ball_by_ball_data.groupby('batter')['batsman_run'].sum().nlargest(10).index.tolist()

# List of players
players = ['AB de Villiers', 'CH Gayle', 'DA Warner', 'KD Karthik', 'MS Dhoni', 'RG Sharma', 'RV Uthappa', 'S Dhawan',
           'SK Raina', 'V Kohli']

# Initialize a list to store statistics for all players
all_players_stats = {}

# Iterate over each player
for player in players:
    # Initialize a list to store statistics for all seasons of the player
    all_seasons_stats = []
    # Iterate over each season
    for season in ball_by_ball_data['Season'].unique():
        # Filter data for the current season
        season_data = ball_by_ball_data[
            (ball_by_ball_data['batter'] == player) & (ball_by_ball_data['Season'] == season)]

        # Calculate the number of fours for the current season
        fours_count = season_data[season_data['batsman_run'] == 4].groupby(['Season', 'ID']).size().reset_index(
            name='fours')
        total_fours = fours_count['fours'].sum()

        sixes_count = season_data[season_data['batsman_run'] == 6].groupby(['Season', 'ID']).size().reset_index(
            name='sixes')
        total_sixes = sixes_count['sixes'].sum()

        # Calculate innings score for the current season
        innings_score = season_data.groupby(['Season', 'ID']).agg(score=('batsman_run', 'sum')).reset_index()
        total_runs = innings_score['score'].sum()
        matches_played = len(innings_score)
        highest_score = innings_score['score'].max()
        if matches_played == 0:
            avg_score = 0
        else:
            avg_score = total_runs / matches_played

        # Calculate the number of fifties and hundreds
        no_of_fifties = len(innings_score[(innings_score['score'] >= 50) & (innings_score['score'] < 100)])
        no_of_hundreds = len(innings_score[innings_score['score'] >= 100])

        # Append statistics for the current season to the list
        all_seasons_stats.append({
            'Season': season,
            'Matches_Played': matches_played,
            'Total_Runs': total_runs,
            'Highest_Score': highest_score,
            '50s': no_of_fifties,
            '100s': no_of_hundreds,
            '4s': total_fours,
            '6s': total_sixes,
            'Avg_Score': round(avg_score, 2)
        })

    # Convert the list of dictionaries to a DataFrame
    all_seasons_stats_df = pd.DataFrame(all_seasons_stats)

    # Calculate career statistics
    career_stats = {
        'Season': 'CAREER',
        'Matches_Played': all_seasons_stats_df['Matches_Played'].sum(),
        'Total_Runs': all_seasons_stats_df['Total_Runs'].sum(),
        'Highest_Score': all_seasons_stats_df['Highest_Score'].max(),
        '50s': all_seasons_stats_df['50s'].sum(),
        '100s': all_seasons_stats_df['100s'].sum(),
        '4s': all_seasons_stats_df['4s'].sum(),
        '6s': all_seasons_stats_df['6s'].sum(),
        'Avg_Score': round(all_seasons_stats_df['Avg_Score'].mean(), 2)
    }

    career_stats_df = pd.DataFrame([career_stats])
    # Concatenate the career statistics DataFrame with the existing DataFrame
    all_seasons_stats_df = pd.concat([career_stats_df, all_seasons_stats_df], ignore_index=True)
    all_players_stats[player] = all_seasons_stats_df

all_players_stats = {player : all_players_stats[player] for player in players}

# Dictionary to map player names to image file paths
player_images = {
    'AB de Villiers': 'assets/AB de Villiers.webp',
    'CH Gayle': 'assets/CH Gayle.webp',
    'DA Warner': 'assets/DA Warner.webp',
    'KD Karthik': 'assets/KD Karthik.webp',
    'MS Dhoni': 'assets/MS Dhoni.webp',
    'RG Sharma': 'assets/RG Sharma.webp',
    'RV Uthappa': 'assets/RV Uthappa.webp',
    'S Dhawan': 'assets/S Dhawan.webp',
    'SK Raina': 'assets/SK Raina.webp',
    'V Kohli': 'assets/V Kohli.webp',
    'R Ashwin':'assets/R Ashwin.webp',
    'SP Narine': 'assets/SP Narine.webp',
    'Harbhajan Singh':'assets/Harbhajan Singh.webp',
    'B Kumar':'assets/B Kumar.webp',
    'SL Malinga':'assets/SL Malinga.webp',
    'JJ Bumrah':'assets/JJ Bumrah.webp',
    'JC Archer':'assets/JC Archer.webp',
    'Rashid Khan':'assets/Rashid Khan.webp',
    'YS Chahal':'assets/YS Chahal.webp',
    'KA Pollard':'assets/KA Pollard.webp'
}

def get_player_description(player_name):
    file_path = f"assets/{player_name}.txt"
    with open(file_path, 'r') as file:
        description = file.read()
    return description

# Initialize a dictionary to store statistics for each player
bowler_stats = {}

bowlers = ['R Ashwin', 'SP Narine', 'Harbhajan Singh', 'B Kumar', 'SL Malinga',
           'JJ Bumrah', 'JC Archer', 'Rashid Khan', 'YS Chahal', 'KA Pollard']

# Iterate over each player
for player in bowlers:
    # Initialize a list to store statistics for each season of the player
    player_season_stats = []
    # Filter data for the current player
    player_data = ball_by_ball_data[ball_by_ball_data['bowler'] == player]
    # Iterate over each season
    for season in player_data['Season'].unique():
        # Filter data for the current season
        season_data = player_data[player_data['Season'] == season]
        # Number of matches played by the player in the current season
        matches_played = season_data['ID'].nunique()
        # Number of balls bowled by the player in the current season
        balls_bowled = season_data.shape[0]
        # Total runs conceded by the player in the current season
        total_runs_conceded = season_data['total_run'].sum()
        # Number of wickets taken by the player in the current season
        wickets_count = season_data[season_data['isWicketDelivery'] == 1].shape[0]
        # Calculate the economy rate for the current season
        total_overs = balls_bowled / 6
        economy_rate = round((total_runs_conceded / total_overs), 2)
        # Store the statistics for the current season in the list
        player_season_stats.append({
            'Season': season,
            'Matches_Played': matches_played,
            'Balls_Bowled': balls_bowled,
            'Total_Runs_Conceded': total_runs_conceded,
            'Wickets_Taken': wickets_count,
            'Economy_Rate': economy_rate
        })
    # Convert the list of dictionaries to a DataFrame
    # all_seasons_stats_df = pd.DataFrame(player_season_stats)
    # bowler_stats[player] = all_seasons_stats_df
    all_seasons_stats_df = pd.DataFrame(player_season_stats)
    career_stats = {
        'Season': 'CAREER',
        'Matches_Played': all_seasons_stats_df['Matches_Played'].sum(),
        'Balls_Bowled': all_seasons_stats_df['Balls_Bowled'].sum(),
        'Total_Runs_Conceded': all_seasons_stats_df['Total_Runs_Conceded'].sum(),
        'Wickets_Taken': all_seasons_stats_df['Wickets_Taken'].sum(),
        'Economy_Rate': round(all_seasons_stats_df['Economy_Rate'].mean(), 2),
    }
    career_stats_df = pd.DataFrame([career_stats])
    all_seasons_stats_df = pd.concat([career_stats_df, all_seasons_stats_df], ignore_index=True)
    bowler_stats[player] = all_seasons_stats_df


app = Dash(__name__)
server = app.server


app.layout = html.Div([
    html.H1("IPL Data Visualization", style={'text-align': 'center'}),
    dcc.Tabs(id='tabs', value='batsman', children=[
        dcc.Tab(label='Batsman Stats', value='batsman', children=[
            html.Div([
                html.H1("Batsman Stats", style={'text-align': 'center'}),
                html.Div([
                    html.Div([
                        html.H3("Select a Player", style={'text-align': 'center'}),
                        dcc.Dropdown(
                            id='player-dropdown',
                            options=[{'label': player, 'value': player} for player in all_players_stats.keys()],
                            value='RG Sharma',  # Default value
                            clearable=False
                        )
                    ], style={'width': '30%', 'margin': 'auto'})
                ], style={'text-align': 'center'}),
                html.Div([
                    html.H2(id='player-name', style={'text-align': 'center'}),
                    html.Img(id='player-image', src='', style={'width': '400px', 'height': '400px', 'display': 'block', 'margin': 'auto'}),
                    html.H3(id='player-description', style={'text-align': 'center'}),
                    dash_table.DataTable(
                        id='player-stats',
                        columns=[{'name': col, 'id': col} for col in all_players_stats['RG Sharma'].columns],
                        style_table={'overflowX': 'scroll'},
                        style_cell={'minWidth': '100px', 'width': '100px', 'maxWidth': '100px'},
                        page_size=10
                    )
                ], style={'width': '70%', 'display': 'inline-block'}),
                dcc.Graph(id='batsman-runs-line-chart'),
                dcc.Graph(id='batsman-runs-pie-chart'),
                dcc.Graph(id='batsman-strike-rate'),
                dcc.Graph(id='batsman-4s-6s-count')
            ])
        ]),
        dcc.Tab(label='Bowlers Stats', value='bowlers', children=[
            html.Div([
                html.H1("Bowler Stats", style={'text-align': 'center'}),
                html.Div([
                    html.Div([
                        html.H3("Select a Bowler", style={'text-align': 'center'}),
                        dcc.Dropdown(
                            id='bowler-dropdown',
                            options=[{'label': bowler, 'value': bowler} for bowler in bowler_stats.keys()],
                            value='JJ Bumrah',  # Default value
                            clearable=False
                        )
                    ], style={'width': '30%', 'margin': 'auto'})
                ], style={'text-align': 'center'}),
                html.Div([
                    html.H2(id='bowler-name', style={'text-align': 'center'}),
                    html.Img(id='bowler-image', src='',
                             style={'width': '400px', 'height': '400px', 'display': 'block', 'margin': 'auto'}),
                    html.H3(id='bowler-description', style={'text-align': 'center'}),
                    dash_table.DataTable(
                        id='bowler-stats',
                        columns=[{'name': col, 'id': col} for col in bowler_stats['JJ Bumrah'].columns],
                        style_table={'overflowX': 'scroll'},
                        style_cell={'minWidth': '100px', 'width': '100px', 'maxWidth': '100px'},
                        page_size=10
                    )
                ], style={'width': '70%', 'display': 'inline-block'}),
                dcc.Graph(id = 'bowling-area-plot'),
                html.Br(),
                dcc.Graph(id = 'bowling-scatter-plot'),
                html.Br(),
                dcc.Graph(id='bowling-heat-plot'),
                html.Br(),
                dcc.Graph(id = 'bowling-bar-plot')
            ])
        ]),
        dcc.Tab(label='Compare Player Stats', children=[
            html.H1("Batsman Strike Rate Comparison"),
            html.Label("Select Batsmen:"),
            dcc.Checklist(
                id='player-checkboxes',
                options=[{'label': player, 'value': player} for player in top_batsmen],
                value=top_batsmen[:1],  # Select the first player by default
                inline=True
            ),
            html.Label("Select Overs Range:"),
            dcc.RangeSlider(
                id='over-slider',
                min=0,
                max=19,
                step=1,
                marks={i: str(i) for i in range(0, 20)},
                value=[0, 19]  # Default range from over 0 to over 19
            ),
            dcc.Graph(id='strike-rate-plot'),

            # Download button for strike rate graph
            html.A(
                html.Button("Download Strike Rate Graph"),
                id="download-link",
                href="",  # No href defined
                target="_blank"
            ),

            # Text area and submit button for feedback
            html.Div([
                html.Label("Feedback:"),
                dcc.Textarea(
                    id="feedback-text",
                    placeholder="Type your feedback here...",
                    style={"width": "100%", "height": "100px"}
                ),
                html.Button("Submit", id="submit-feedback-button"),
                html.Div(id="feedback-output")
            ])
        ])
    ])
])

# Callback to handle feedback submission
@app.callback(
    Output("feedback-output", "children"),
    [Input("submit-feedback-button", "n_clicks")],
    [State("feedback-text", "value")]
)
def submit_feedback(n_clicks, feedback_text):
    if n_clicks is not None:
        return f"You submitted the following feedback: {feedback_text}"
    else:
        return ""


# Define callback to update player stats, image, and line chart
@app.callback(
    [Output('player-stats', 'data'),
     Output('player-image', 'src'),
     Output('batsman-runs-line-chart', 'figure'),
     Output('batsman-runs-pie-chart', 'figure'),
     Output('batsman-strike-rate', 'figure'),
     Output('batsman-4s-6s-count', 'figure'),
     Output('player-description', 'children')],
    [Input('player-dropdown', 'value')]
)
def update_player_stats_image_and_line_chart(player_name):
    # 1. Get player stats dataframe
    player_stats_df = all_players_stats[player_name]

    # 2. Get player image file path
    image_path = player_images[player_name]

    # Encode image to base64
    encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')

    #-------------------------------------------------------------------------------------
    # 1. Get line chart data
    #-------------------------------------------------------------------------------------

    batsman_runs_df = ball_by_ball_data[ball_by_ball_data['batter'] == player_name]
    batsman_runs_df = batsman_runs_df.groupby(['Season', 'batter']).agg(batsman_total=('batsman_run', 'sum')).reset_index()

    fig_1 = px.line(
        batsman_runs_df,
        x='Season',
        y='batsman_total',
        markers=True,
        title="<b>Batsman Runs Comparison Across Seasons</b>"
    )

    fig_1.update_layout(hovermode="x unified")
    fig_1.update_xaxes(categoryorder='category ascending')
    fig_1.update_yaxes(showgrid=True)
    fig_1.update_traces(line=dict(width=3.0))

    fig_1.update_layout(
        title='<b>Batsman Performance across seasons</b>',
        font_family="Courier New",
        title_font_family="Courier New",
        title_font_color="red",
        title_font_size=20,
        xaxis_title="<b>Season</b>",
        yaxis_title="<b>Runs</b>",
        plot_bgcolor='#FFFFFF'
    )

    #-------------------------------------------------------------------------------------
    # 2. PIE CHART
    #-------------------------------------------------------------------------------------

    player_data = all_players_stats[player_name]
    total_runs_career = player_data[player_data['Season'] == 'CAREER']['Total_Runs'].values[0]
    total_fours_career = player_data[player_data['Season'] == 'CAREER']['4s'].values[0]
    total_sixes_career = player_data[player_data['Season'] == 'CAREER']['6s'].values[0]
    total_other_runs_career = total_runs_career - (total_fours_career * 4) - (total_sixes_career * 6)
    pie_data = pd.DataFrame({
        'Type': ['4s', '6s', 'Others'],
        'Percentage': [round(((total_fours_career * 4 / total_runs_career) * 100), 2),
                       round(((total_sixes_career * 6 / total_runs_career) * 100), 2),
                       round(((total_other_runs_career / total_runs_career) * 100), 2)]
    })
    fig_2 = px.pie(pie_data, names='Type', values='Percentage', title='Percentage of Runs Distribution (CAREER)',
                 hole=0.5)

    fig_2.update_layout(
        title='<b>Percentage of Runs Distribution (CAREER)</b>',
        title_font_family="Courier New",
        title_font_color="red",
        title_font_size=20,
    )

    #-------------------------------------------------------------------------------------
    # 3. SCATTER PLOT WITH REGRESSION LINE
    #-------------------------------------------------------------------------------------

    batsman_data = ball_by_ball_data[ball_by_ball_data["batter"] == player_name]
    overwise_data = batsman_data.groupby('overs').agg({'ballnumber': 'count', 'total_run': 'sum'}).reset_index()
    overwise_data['strike_rate'] = round((overwise_data['total_run'] / overwise_data['ballnumber']) * 100, 2)
    regressor = LinearRegression()
    regressor.fit(overwise_data[['overs']], overwise_data['strike_rate'])

    # Create scatter plot for strike rate
    scatter_trace = go.Scatter(
        x=overwise_data['overs'],
        y=overwise_data['strike_rate'],
        mode='markers',
        name='Strike Rate',
        marker=dict(color='blue')
    )

    # Generate predictions for regression line
    regression_line = regressor.predict(overwise_data[['overs']])

    # Create line plot for regression line
    line_trace = go.Scatter(
        x=overwise_data['overs'],
        y=regression_line,
        mode='lines',
        name='Regression Line',
        line=dict(color='black')
    )

    # Combine both plots into a single figure
    fig_3 = go.Figure([scatter_trace, line_trace])

    fig_3.update_layout(
        title=f'<b>Strike Rate vs Overs for {player_name}<b>',
        xaxis_title='Overs',
        title_font_family="Courier New",
        title_font_color="red",
        title_font_size=20,
        yaxis_title='Strike Rate',
        showlegend=True
    )

    # -------------------------------------------------------------------------------------
    # 4. Count Plot
    # -------------------------------------------------------------------------------------
    player_df = all_players_stats[player_name]
    player_df = player_df[player_df['Season'] != 'CAREER']
    data_long = pd.melt(player_df, id_vars=['Season'], value_vars=['4s', '6s'], var_name='Type', value_name='Count')
    fig_4 = px.bar(data_long, x='Season', y='Count', color='Type', barmode='group',
                 title=f"Count of 4's and 6's for {player_name} Across Seasons")
    fig_4.update_xaxes(categoryorder='category ascending')
    fig_4.update_yaxes(title="Count")
    fig_4.update_layout(
        title=f"<b>Count of 4's and 6's for {player_name} Across Seasons<b>",
        xaxis_title='Season',
        title_font_family="Courier New",
        title_font_color="red",
        title_font_size=20,
        yaxis_title='Count',
        showlegend=True
    )
    player_description = get_player_description(player_name)
    return player_stats_df.to_dict('records'), f'data:image/png;base64,{encoded_image}', fig_1, fig_2, fig_3, fig_4, player_description

# Define callback to update bowler stats
@app.callback(
    [Output('bowler-stats', 'data'),
    Output('bowler-image', 'src'),
    Output('bowler-description', 'children'),
    Output('bowling-area-plot', 'figure'),
    Output('bowling-scatter-plot', 'figure'),
    Output('bowling-heat-plot', 'figure'),
    Output('bowling-bar-plot', 'figure')],
    [Input('bowler-dropdown', 'value')]
)
def update_bowler_stats(bowler_name):
    # 2. Get player image file path
    image_path = player_images[bowler_name]
    # Encode image to base64
    encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
    bowler_description = get_player_description(bowler_name)

    # -------------------------------------------------------------------------------------
    # 1. Area Plot
    # -------------------------------------------------------------------------------------
    player_df = bowler_stats[bowler_name]
    player_df = player_df[player_df['Season'] != 'CAREER']
    player_df = player_df[['Balls_Bowled', 'Season', 'Total_Runs_Conceded']]
    player_df.set_index('Season', inplace=True)
    traces = []
    for column in player_df.columns:
        traces.append(go.Scatter(
            x=player_df.index,
            y=player_df[column],
            mode='lines',
            name=column,
            fill='tozeroy'
        ))

    # Define layout
    layout = go.Layout(
        title="Trends Across Seasons",
        xaxis=dict(title='Season'),
        yaxis=dict(title='Values'),
        hovermode='closest'
    )

    # Create the figure
    figure_area_plot = go.Figure(data=traces, layout=layout)
    figure_area_plot.update_layout(
        title=f"<b>Trend Across Seasons for {bowler_name}</b>",
        title_font_family="Courier New",
        title_font_color="red",
        title_font_size=20,
        showlegend=True
    )

    # -------------------------------------------------------------------------------------
    # 2. Scatter Plot
    # -------------------------------------------------------------------------------------
    player_df = bowler_stats[bowler_name]
    bins = [0, 5, 10, 15, 20, 25, float('inf')]
    labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25+']
    player_df['category'] = pd.cut(player_df['Wickets_Taken'], bins=bins, labels=labels)
    player_df = player_df.dropna(subset=['category'])
    fig_scatter_colored = px.scatter(player_df,
                     x='Economy_Rate',
                     y='Total_Runs_Conceded',
                     color='category',
                     facet_col='category',
                     template='plotly_dark')
    fig_scatter_colored.update_layout(
        title=f"<b>Economy rate Vs Runs Conceded Vs Wickets Taken for {bowler_name}</b>",
        title_font_family="Courier New",
        title_font_color="red",
        title_font_size=20,
        showlegend=True
    )

    # -------------------------------------------------------------------------------------
    # 3. Heat Plot
    # -------------------------------------------------------------------------------------
    # Calculate correlation matrix
    matches_data = ball_by_ball_data[ball_by_ball_data['bowler'] == bowler_name]
    matches_data = matches_data.groupby('ID').agg({'isWicketDelivery': 'sum', 'ballnumber': 'count', 'batsman_run': 'sum'})
    matches_data['economy'] = round(matches_data['batsman_run'] / (matches_data['ballnumber'] / 6), 2)
    correlation_matrix = matches_data[['batsman_run', 'ballnumber', 'isWicketDelivery', 'economy']].corr()

    # Create heatmap using Plotly Express
    fig_heat = px.imshow(correlation_matrix.values,
                    labels=dict(x="Metrics", y="Metrics", color="Correlation"),
                    x=['Runs Conceded', 'Balls Bowled', 'No of Wickets Taken', 'Economy'],
                    y=['Runs Conceded', 'Balls Bowled', 'No of Wickets Taken', 'Economy'],
                    color_continuous_scale='RdBu',
                    # Using RdBu colorscale for better visualization of positive and negative correlations
                    zmin=-1, zmax=1,  # Set color scale limits for correlation values
                    )

    # Annotate cells with correlation values
    for i in range(len(correlation_matrix.index)):
        for j in range(len(correlation_matrix.columns)):
            fig_heat.add_annotation(
                x=['Runs Conceded', 'Balls Bowled', 'No of Wickets Taken', 'Economy'][i],
                y=['Runs Conceded', 'Balls Bowled', 'No of Wickets Taken', 'Economy'][j],
                text=str(np.round(correlation_matrix.values[i, j], 2)),
                showarrow=False
            )

    # Update layout
    # fig_heat.update_layout(
    #     title='Correlation Heatmap of Wickets Taken, Runs Conceded, and Economy Rate',
    #     xaxis_title='Metrics',
    #     yaxis_title='Metrics',
    #     template='plotly_dark'
    # )
    fig_heat.update_layout(
        title=f"<b>Correlation Heatmap of Wickets Taken, Runs Conceded, and Economy Rate for  {bowler_name}</b>",
        title_font_family="Courier New",
        title_font_color="red",
        title_font_size=20,
        showlegend=True
    )

    # -------------------------------------------------------------------------------------
    # 4. bar plot
    # -------------------------------------------------------------------------------------
    player_data = ball_by_ball_data[ball_by_ball_data['bowler'] == bowler_name]
    player_wickets = player_data[player_data['isWicketDelivery'] == 1].groupby('BattingTeam').size().reset_index(
        name='Wickets')
    player_wickets_sorted = player_wickets.sort_values(by='Wickets', ascending=False)
    fig_wickets_bar = px.bar(player_wickets_sorted, x='BattingTeam', y='Wickets',
                 title=f'Number of Wickets taken by {bowler_name} Against Each Team',
                 labels={'BattingTeam': 'Opponent Team', 'Wickets': 'Number of Wickets'})

    #fig_wickets_bar.update_layout(xaxis_title='Opponent Team', yaxis_title='Number of Wickets')
    fig_wickets_bar.update_layout(
        title=f"<b>Wickets taken against each team by {bowler_name}</b>",
        title_font_family="Courier New",
        title_font_color="red",
        title_font_size=20,
        showlegend=True,
        xaxis_title='Opponent Team',
        yaxis_title='Number of Wickets'
    )
    return bowler_stats[bowler_name].to_dict('records'), f'data:image/png;base64,{encoded_image}', bowler_description, figure_area_plot, fig_scatter_colored, fig_heat, fig_wickets_bar


# Define callback to update the strike rate plot
@app.callback(
    Output('strike-rate-plot', 'figure'),
    [Input('player-checkboxes', 'value'),
     Input('over-slider', 'value')]
)
def update_strike_rate_plot(selected_players, selected_overs):
    # Initialize an empty list to store individual DataFrames for each player
    player_data = []

    # Iterate over each selected player
    for player in selected_players:
        # Filter data for the current player and selected overs range
        filtered_data = ball_by_ball_data[(ball_by_ball_data['batter'] == player) &
                                          (ball_by_ball_data['overs'].between(selected_overs[0], selected_overs[1]))]

        # Calculate strike rate for the current player
        overwise_data = filtered_data.groupby('overs').agg({'ballnumber': 'count', 'total_run': 'sum'}).reset_index()
        overwise_data['strike_rate'] = (overwise_data['total_run'] / overwise_data['ballnumber']) * 100

        # Add player name to the data
        overwise_data['batter'] = player

        # Append data for the current player to the list
        player_data.append(overwise_data)

    # Concatenate all player DataFrames into a single DataFrame
    combined_data = pd.concat(player_data, ignore_index=True)

    # Plot combined strike rate data for all selected players
    fig = px.line(
        combined_data,
        x='overs',
        y='strike_rate',
        color='batter',  # Use different colors for each player
        markers=True,
        title="Batsman Strike Rate Comparison"
    )

    fig.update_layout(xaxis_title="Over", yaxis_title="Strike Rate")

    return fig

# Callback to generate downloadable strike rate graph
@app.callback(
    Output("download-link", "href"),
    [Input("strike-rate-plot", "figure")]
)
def update_download_link(figure):
    # Do nothing for now
    return ""
# Run the app

# if __name__ == '__main__':
#     app.run_server(debug=True)

if __name__ == "__main__":
    app.run_server(debug = True, host = '0.0.0.0', port = 8030)
