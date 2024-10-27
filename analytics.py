import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import plotly.express as px
import plotly.io as pio
import dash
from dash import dcc, html, Input, Output
from dash.exceptions import PreventUpdate

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Team Statistics Dashboard"),
    
    # Input field for team name
    html.Label("Enter Team Name:"),
    dcc.Input(id='team-input', type='text', placeholder='Enter team name'),
    html.Button('Submit', id='submit-button', n_clicks=0),
    
    # Area to display the visualizations
    html.Div(id='output-container')
])

@app.callback(
    Output('output-container', 'children'),
    [Input('submit-button', 'n_clicks')],
    [Input('team-input', 'value')]
)


def update_output(n_clicks,teamName):
    def determine_result(row):
        if ((row['HomeTeam'] == teamName) & (row['FTR'] == 'H')) | ((row['AwayTeam'] == teamName) & (row['FTR'] == 'A')):
            return 'W'
        elif ((row['HomeTeam'] == teamName) & (row['FTR'] == 'A')) | ((row['AwayTeam'] == teamName) & (row['FTR'] == 'H')):
            return 'L'
        else :
            return 'D'
    def determine_season(row):
        if (pd.to_datetime('2020-09-10') < row['Date'] < pd.to_datetime('2021-05-25')):
                return '20_21'
        elif (pd.to_datetime('2021-08-10') < row['Date'] < pd.to_datetime('2022-05-25')):
                return '21_22'
        elif (pd.to_datetime('2022-08-03') < row['Date'] < pd.to_datetime('2023-05-30')):
                return '22_23'
    def determine_result_HT(row):
        if ((row['HomeTeam'] == teamName) & (row['HTR'] == 'H')) | ((row['AwayTeam'] == teamName) & (row['HTR'] == 'A')):
            return 'W'
        elif ((row['HomeTeam'] == teamName) & (row['HTR'] == 'A')) | ((row['AwayTeam'] == teamName) & (row['HTR'] == 'H')):
            return 'L'
        else :
            return 'D'
    def determine_goals_scored(row):
        if row['HomeTeam'] == teamName:
            return row['FTHG']
        else:
            return row['FTAG']

    def determine_goals_conceded(row):
        if row['HomeTeam'] == teamName:
            return row['FTAG']
        else:
            return row['FTHG']
    def determine_goals_scored_HT(row):
        if row['HomeTeam'] == teamName:
            return row['HTHG']
        else:
            return row['HTAG']

    def determine_goals_conceded_HT(row):
        if row['HomeTeam'] == teamName:
            return row['HTAG']
        else:
            return row['HTHG']
    def determine_redcards(row):
        if row['HomeTeam'] == teamName:
            return row['HR']
        else:
            return row['AR']
        
    if n_clicks == 0 or not teamName:
        raise PreventUpdate
    
    
    f1 = r'2020-2021.csv'
    f2 = r'2021-2022.csv'
    f3 = r'epl_results_2022-23.csv'

    csv_files = [f1,f2,f3]  # List your specific CSV file paths

    df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)

    df=df.iloc[:,:24]

    df_team = df[ (df['HomeTeam'] == teamName) | (df['AwayTeam'] == teamName) ]

    if df_team.empty:
        return html.Div(f"No data found for team '{teamName}'")
    
    df_team['Date'] = pd.to_datetime(df['Date'],format='%d/%m/%Y')
    df_team['FT_Result'] = df_team.apply(determine_result,axis=1)
    df_team['HT_Result'] = df_team.apply(determine_result_HT,axis=1)
    df_team['Season'] = df_team.apply(determine_season,axis=1)
    df_team['FT_goals_scored'] = df_team.apply(determine_goals_scored,axis=1)
    df_team['FT_goals_conceded'] = df_team.apply(determine_goals_conceded,axis=1)
    df_team['HT_goals_scored'] = df_team.apply(determine_goals_scored_HT,axis=1)
    df_team['HT_goals_conceded'] = df_team.apply(determine_goals_conceded_HT,axis=1)
    df_team['RedCards'] = df_team.apply(determine_redcards,axis=1)

    WinPercentage = df_team['FT_Result'].value_counts()['W']/df_team.shape[0]*100
    pct = ((df_team['FT_Result'].value_counts()['W']*3)+(df_team['FT_Result'].value_counts()['D']*1))/ (df_team.shape[0]*3)*100

    fig1 = px.bar(df_team.groupby('Season')['FT_Result'].value_counts().unstack(),barmode='group',title = 'Victories, Defeats, and Ties Across Seasons')

    home_wins = df_team[(df_team['HomeTeam'] == teamName) & (df_team['FT_Result'] == 'W')].groupby('Season').size()
    away_wins = df_team[(df_team['AwayTeam'] == teamName) & (df_team['FT_Result'] == 'W')].groupby('Season').size()
    wins_df = pd.DataFrame({'Home Wins': home_wins, 'Away Wins': away_wins})
    fig2 = px.bar(wins_df,barmode='group',color_discrete_map={'Home Wins': 'green','Away Wins': 'blue'},title='Home and Away Wins')

    d2w = df_team[(df_team['HT_Result'] == 'D') & (df_team['FT_Result'] == 'W')].groupby('Season').size()
    l2w = df_team[(df_team['HT_Result'] == 'L') & (df_team['FT_Result'] == 'W')].groupby('Season').size()
    l2d = df_team[(df_team['HT_Result'] == 'L') & (df_team['FT_Result'] == 'D')].groupby('Season').size()
    d2l = df_team[(df_team['HT_Result'] == 'D') & (df_team['FT_Result'] == 'L')].groupby('Season').size()
    w2l = df_team[(df_team['HT_Result'] == 'W') & (df_team['FT_Result'] == 'L')].groupby('Season').size()
    w2d = df_team[(df_team['HT_Result'] == 'W') & (df_team['FT_Result'] == 'D')].groupby('Season').size()

    cb_df = pd.DataFrame({'Draw->Win': d2w, 'Lose->Win': l2w})

    # ax = cb_df.plot(kind='bar',color=['Blue','Green'])
    fig3 = px.bar(cb_df,barmode='group',color_discrete_map={'Draw->Win': 'blue','Lose->Win': 'green'},title='Comebacks')


    gs_df = pd.DataFrame({'Lose->Draw': l2d})

    # ax = gs_df.plot(kind='bar',color=['Orange'])
    # plt.title('Gaining atleast Something')
    fig4 = px.bar(gs_df,color_discrete_map={'Lose->Draw': 'orange'},title='Gaining atleast Something')

    ga_df = pd.DataFrame({'Win->Loss': w2l, 'Win->Draw': w2d,'Draw->Loss':d2l})

    # ax = ga_df.plot(kind='bar',color=['Red','Grey','pink'])
    # plt.title('GiveAway')
    fig5 = px.bar(ga_df,barmode='group',color_discrete_map={'Win->Loss': 'red','Win->Draw': 'grey','Draw->Loss':'pink'},title='GiveAway')

    ft_goals = df_team['FT_goals_scored'].sum()
    ft_goals_conceded = df_team['FT_goals_conceded'].sum()

    ftgs_df = df_team.groupby('Season')['FT_goals_scored'].sum()
    # ax = ftgs_df.plot(kind='pie', 
    #                   autopct=lambda p: f'{int(p * ftgs_df.sum() / 100)}', 
    #                   colors=['violet', 'Blue', 'Orange'])
    # plt.title('Goals Scored over the Years')
    fig6 = px.pie(
        ftgs_df.reset_index(),  # Resetting the index to convert the Series into a DataFrame
        values='FT_goals_scored',  # The column with the goal counts
        names='Season',  # The column with season names
        title='FT Goals Scored by Season',
        color_discrete_sequence=['violet', 'blue', 'orange']  # Custom colors
    )

    fig6.update_traces(textposition='inside',  texttemplate='%{label}<br>%{value} goals')

    ftgsHome_df = df_team[df_team['HomeTeam'] == teamName]['FT_goals_scored'].sum()
    ftgsAway_df = df_team[df_team['AwayTeam'] == teamName]['FT_goals_scored'].sum()

    labels = ['Home Goals', 'Away Goals']
    sizes = [ftgsHome_df, ftgsAway_df]
    colors = ['Blue', 'Violet']

    fig7 = px.pie(
        values=sizes,  # The column with the goal counts
        names=labels,  # The column with season names
        title=f'Goals Scored by {teamName} - Home vs Away',
        color_discrete_sequence=colors  # Custom colors
    )
    fig7.update_traces(
        textposition='inside',
        texttemplate='%{label}<br>%{value} goals<br>%{percent}'
    )

    with open(f'final_report_{teamName}.html','w') as f:
        f.write(pio.to_html(fig1))
        f.write(pio.to_html(fig2))
        f.write(pio.to_html(fig3))
        f.write(pio.to_html(fig4))
        f.write(pio.to_html(fig5))
        f.write(pio.to_html(fig6))
        f.write(pio.to_html(fig7))

    return [
        html.Div(f"Winning Percentage of {teamName}: {WinPercentage:.3f}%", style={
            'font-size': '24px', 
            'font-weight': 'bold', 
            'padding': '10px', 
            'border': '2px solid black', 
            'border-radius': '5px', 
            'background-color': '#f0f0f0',
            'margin-top': '20px'
        }),
        html.Div(f"Points Percentage of {teamName}: {pct:.3f}%", style={
            'font-size': '24px', 
            'font-weight': 'bold', 
            'padding': '10px', 
            'border': '2px solid black', 
            'border-radius': '5px', 
            'background-color': '#f0f0f0',
            'margin-top': '20px'
        }),
        html.Img(src=f'/assets/{teamName}_logo.webp', style={
            'width': '10%',  # Adjust width as needed
            'height': 'auto',
            'display':'block',
            'margin-bottom': '20px',# Space below the image
            'margin-left':'auto',
            'margin-right':'auto'
        }),

        dcc.Graph(figure=fig1),
        html.Div("The above bar chart shows the count of Wins, Losses, and Draws by season", style={'margin-bottom': '20px'}),
        dcc.Graph(figure=fig2),
        html.Div("The above bar chart shows the count of Wins in Home and Away", style={'margin-bottom': '20px'}),
        dcc.Graph(figure=fig3),
        html.Div(f"The above bar chart illustrates how many times {teamName} managed to turn the game in their favor during the second half when they were either trailing or tied at halftime.", style={'margin-bottom': '20px'}),
        dcc.Graph(figure=fig4),
        html.Div(f"The above bar chart demonstrates how often {teamName} was able to secure at least one point by full-time after being behind at halftime.", style={'margin-bottom': '20px'}),
        dcc.Graph(figure=fig5),
        html.Div(f"The above bar chart above depicts the number of instances in which {teamName} lost points even though they were leading or tied with the opposing team at halftime.", style={'margin-bottom': '20px'}),
        dcc.Graph(figure=fig6),
        html.Div(f"The above pie chart depicts the proportion of goals scored each season by {teamName}", style={'margin-bottom': '20px'}),
        dcc.Graph(figure=fig7),
        html.Div(f"The above pie chart depicts the distribution of goals scored by {teamName} in home and away matches.", style={'margin-bottom': '20px'})
        
    ]
    
if __name__ == '__main__':
    app.run_server(debug=True)
