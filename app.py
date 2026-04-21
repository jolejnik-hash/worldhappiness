
import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from plotly.subplots import make_subplots
import os

# --- Advanced Data Engine ---
def load_data():
    # Data will be placed in a 'data' subfolder within the Hugging Face Space
    data_folder = "data"
    csv_files = [
        os.path.join(data_folder, '2015.csv'),
        os.path.join(data_folder, '2016.csv'),
        os.path.join(data_folder, '2017.csv'),
        os.path.join(data_folder, '2018.csv'),
        os.path.join(data_folder, '2019.csv')
    ]
    mapping = {
        'Country': 'Country', 'Country or region': 'Country', 'Happiness Score': 'Score',
        'Happiness.Score': 'Score', 'Economy (GDP per Capita)': 'GDP',
        'Economy..GDP.per.Capita.': 'GDP', 'GDP per capita': 'GDP',
        'Family': 'Social support', 'Health (Life Expectancy)': 'Health',
        'Health..Life.Expectancy.': 'Health', 'Healthy life expectancy': 'Health',
        'Freedom': 'Freedom', 'Freedom to make life choices': 'Freedom',
        'Trust (Government Corruption)': 'Corruption', 'Trust..Government.Corruption.': 'Corruption',
        'Perceptions of corruption': 'Corruption', 'Region': 'Region'
    }

    all_dfs = []
    for file in csv_files:
        try:
            yr = int(os.path.basename(file).split('.')[0])
            temp = pd.read_csv(file)
            temp['Year'] = yr
            temp = temp.rename(columns=mapping)
            all_dfs.append(temp)
        except FileNotFoundError:
            print(f"Warning: Data file not found: {file}. Skipping this year.")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not all_dfs:
        print("Error: No data loaded. Returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.dropna(subset=['Country'])

    # Only fillna if 'Corruption' column exists
    if 'Corruption' in df.columns:
        df['Corruption'] = df['Corruption'].fillna(df['Corruption'].mean())
    else:
        df['Corruption'] = pd.NA # Or handle as per requirement if it's a critical factor

    if 'Region' not in df.columns:
        df['Region'] = 'Unknown'

    # Attempt to map regions based on available data
    # Ensure there are non-NA regions to create a mapping
    if df['Region'].notna().any():
        region_map = df[df['Region'].notna()].drop_duplicates('Country').set_index('Country')['Region'].to_dict()
        df['Region'] = df['Region'].fillna(df['Country'].map(region_map)).fillna('Other')
    else:
        df['Region'] = 'Other' # Default if no region information at all

    cont_map = {
        'Western Europe': 'Europe', 'North America': 'Americas', 'Australia and New Zealand': 'Oceania',
        'Middle East and Northern Africa': 'Africa/ME', 'Latin America and Caribbean': 'Americas',
        'Southeastern Asia': 'Asia', 'Central and Eastern Europe': 'Europe', 'Eastern Asia': 'Asia',
        'Sub-Saharan Africa': 'Africa', 'Southern Asia': 'Asia', 'Other': 'Other'
    }
    df['Continent'] = df['Region'].map(cont_map).fillna('Other')

    for factor in ['GDP', 'Social support', 'Health', 'Freedom', 'Generosity', 'Corruption']:
        if factor not in df.columns:
            df[factor] = pd.NA # Handle missing factor columns

    return df

df = load_data()
if df.empty:
    print("Dashboard cannot run without data. Please ensure data files are present in the 'data' folder.")
    # Exit or handle gracefully if no data is loaded
    # In Gradio Spaces, an empty DataFrame might lead to errors or a non-functional app,
    # so it's better to explicitly handle it.
    # For this example, we'll allow the app to launch but warn if no data was found.
    # A more robust solution might involve sys.exit(1) or raising an exception.
    pass

factors = ['GDP', 'Social support', 'Health', 'Freedom', 'Generosity', 'Corruption']
pale_colors = ['#FADBD8', '#EBDEF0', '#D4E6F1', '#D1F2EB', '#FCF3CF', '#F5CBA7']

# --- UI Callbacks ---
def update_globe(year):
    dff = df[df['Year'] == int(year)]
    fig = px.choropleth(dff, locations="Country", locationmode='country names', color="Score",
                        hover_data=factors, color_continuous_scale="RdYlGn", projection="orthographic")
    fig.update_layout(margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor='rgba(0,0,0,0)', height=600)
    return fig

def update_correlation(year, continent, region, factor):
    dff = df[df['Year'] == int(year)]
    if continent != "All": dff = dff[dff['Continent'] == continent]
    if region != "All": dff = dff[dff['Region'] == region]

    if dff.empty or len(dff) < 2:
        empty_plot = go.Figure().update_layout(title="Insufficient data for correlation analysis.")
        empty_plot.add_annotation(text="No data or insufficient data for selected filters.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return empty_plot, pd.DataFrame(columns=["Factor", "Correlation"])

    # Filter out NA values for the selected factor and score before correlation
    dff_filtered = dff.dropna(subset=[factor, 'Score'])
    if dff_filtered.empty or len(dff_filtered) < 2:
        empty_plot = go.Figure().update_layout(title="Insufficient valid data for correlation analysis after dropping NaNs.")
        empty_plot.add_annotation(text="No valid data points for selected factor and score.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return empty_plot, pd.DataFrame(columns=["Factor", "Correlation"])

    X, y = dff_filtered[[factor]].values, dff_filtered['Score'].values
    model = LinearRegression().fit(X, y)

    fig = px.scatter(dff_filtered, x=factor, y="Score", color="Region", hover_name="Country", height=600)
    fig.add_traces(go.Scatter(x=dff_filtered[factor], y=model.predict(X), mode='lines', name='Trend', line=dict(color='black', dash='dot')))
    plot_corr_coef = np.corrcoef(dff_filtered[factor].astype(float), dff_filtered['Score'].astype(float))[0,1]
    fig.update_layout(title=f"Plot Correlation (Score vs. {factor}): {plot_corr_coef:.2f}", template="plotly_white", yaxis=dict(range=[2, 8]))

    all_corrs = []
    for f in factors:
        temp_dff = dff.dropna(subset=[f, 'Score'])
        if f in temp_dff.columns and 'Score' in temp_dff.columns and len(temp_dff) > 1:
            c = np.corrcoef(temp_dff['Score'].astype(float), temp_dff[f].astype(float))[0,1]
            all_corrs.append([f, round(c, 3)])
        else:
            all_corrs.append([f, np.nan])
    all_correlations_df = pd.DataFrame(all_corrs, columns=["Factor", "Correlation"]).sort_values("Correlation", ascending=False)

    return fig, all_correlations_df

def update_trajectory(continent, region, country):
    dff = df.copy()
    name = "Global"
    if continent != "All": dff, name = dff[dff['Continent'] == continent], continent
    if region != "All": dff, name = dff[dff['Region'] == region], region
    if country != "None": dff, name = dff[dff['Country'] == country], country

    timeline = dff.groupby('Year')[['Score'] + factors].mean().reset_index()

    if timeline.empty:
        empty_plot = go.Figure().update_layout(title="Insufficient data for trajectory analysis.")
        empty_plot.add_annotation(text="No data for selected filters.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return empty_plot, pd.DataFrame(columns=["Factor", "Correlation"]), f"""### Total Change in Happiness
## No Data"""


    score_change = timeline['Score'].iloc[-1] - timeline['Score'].iloc[0] if len(timeline) > 1 else 0

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for i, f in enumerate(factors):
        if f in timeline.columns:
            fig.add_trace(go.Scatter(x=timeline['Year'], y=timeline[f], name=f, line=dict(color=pale_colors[i], width=2)), secondary_y=True)
    if 'Score' in timeline.columns:
        fig.add_trace(go.Scatter(x=timeline['Year'], y=timeline['Score'], name="Happiness Score", line=dict(color='#00b894', width=5)), secondary_y=False)

    fig.update_layout(title=f"{name} Progress: {'+' if score_change>=0 else ''}{score_change:.3f} Points", template="plotly_white", xaxis=dict(type='category'), height=600)
    fig.update_yaxes(range=[2, 8], secondary_y=False)

    corrs = []
    for f in factors:
        temp_timeline = timeline.dropna(subset=[f, 'Score'])
        if f in temp_timeline.columns and 'Score' in temp_timeline.columns and len(temp_timeline) > 1:
            c = np.corrcoef(temp_timeline['Score'].astype(float), temp_timeline[f].astype(float))[0,1]
            corrs.append([f, round(c, 3)])
        else:
            corrs.append([f, np.nan])

    corr_df = pd.DataFrame(corrs, columns=["Factor", "Correlation"]).sort_values("Correlation", ascending=False)
    return fig, corr_df, f"""### Total Change in Happiness
## {'📈' if score_change >=0 else '📉'} {score_change:.3f}"""

# Initial calls to populate dropdowns (assuming df is not empty)
initial_continents = ["All"] + sorted(df['Continent'].unique().tolist()) if not df.empty and 'Continent' in df.columns else ["All"]
initial_regions = ["All"] + sorted(df['Region'].unique().tolist()) if not df.empty and 'Region' in df.columns else ["All"]
initial_countries = ["None"] + sorted(df['Country'].unique().tolist()) if not df.empty and 'Country' in df.columns else ["None"]

with gr.Blocks() as demo:
    gr.Markdown("<center><h1>World Happiness Analytics Engine</h1></center>")

    with gr.Row(): # Outer Row for centering
        gr.Column(scale=1) # Left spacer column

        with gr.Column(scale=3): # Main content column
            with gr.Column(): # Replaced gr.Box() with gr.Column()
                with gr.Tabs():
                    with gr.Tab("Globe"):
                        with gr.Row():
                            gr.Column(scale=1, min_width=100) # Spacer for centering
                            with gr.Column(scale=2, min_width=300):
                                globe_year = gr.Radio([2015, 2016, 2017, 2018, 2019], value=2019, label="View Year")
                            gr.Column(scale=1, min_width=100) # Spacer for centering
                        globe_plot = gr.Plot(elem_id="full-width-globe")

                    with gr.Tab("Correlations"):
                        with gr.Row():
                            gr.Column(scale=1, min_width=50) # Spacer for centering
                            with gr.Column(scale=3, min_width=600):
                                with gr.Row(): # Inner row for specific components to align
                                    scat_year = gr.Radio([2015, 2016, 2017, 2018, 2019], value=2019, label="Select Year")
                                    scat_cont = gr.Dropdown(initial_continents, value="All", label="Continent")
                                    scat_reg = gr.Dropdown(initial_regions, value="All", label="Region")
                                    scat_fac = gr.Dropdown(factors, value="GDP", label="Factor to Plot")
                            gr.Column(scale=1, min_width=50) # Spacer for centering
                        with gr.Row():
                            with gr.Column(scale=3):
                                scat_plot = gr.Plot()
                            with gr.Column(scale=1, variant="panel"):
                                gr.Markdown("**All Factor Correlations**")
                                all_factor_corr_table = gr.DataFrame(headers=["Factor", "Correlation"], label="Correlation with Score")

                    with gr.Tab("Trajectories"):
                        with gr.Row():
                            gr.Column(scale=1, min_width=50) # Spacer for centering
                            with gr.Column(scale=3, min_width=600):
                                with gr.Row(): # Inner row for specific components to align
                                    t_cont = gr.Dropdown(initial_continents, value="All", label="Continent")
                                    t_reg = gr.Dropdown(initial_regions, value="All", label="Region")
                                    t_cty = gr.Dropdown(initial_countries, value="None", label="Country")
                            gr.Column(scale=1, min_width=50) # Spacer for centering
                        with gr.Row():
                            with gr.Column(scale=3):
                                trend_plot = gr.Plot()
                            with gr.Column(scale=1, variant="panel"):
                                gr.Markdown("**Summary & Factor Impact**")
                                change_md = gr.Markdown()
                                corr_table = gr.DataFrame(headers=["Factor", "Correlation"], label="Factor Impact")
        gr.Column(scale=1) # Right spacer column

    # Events
    globe_year.change(update_globe, globe_year, globe_plot)

    # Correlations Tab Events
    scat_cont.change(lambda c: gr.update(choices=(["All"] + sorted(df[df['Continent']==c]['Region'].unique().tolist())) if c != "All" and not df.empty else initial_regions, value="All"), scat_cont, scat_reg)
    for inp in [scat_year, scat_cont, scat_reg, scat_fac]:
        inp.change(update_correlation, [scat_year, scat_cont, scat_reg, scat_fac], [scat_plot, all_factor_corr_table])

    # Trajectories Tab Events
    t_cont.change(lambda c: gr.update(choices=(["All"] + sorted(df[df['Continent']==c]['Region'].unique().tolist())) if c != "All" and not df.empty else initial_regions, value="All"), t_cont, t_reg)
    t_reg.change(lambda r: gr.update(choices=(["None"] + sorted(df[df['Region']==r]['Country'].unique().tolist())) if r != "All" and not df.empty else initial_countries, value="None"), t_reg, t_cty)
    for inp in [t_cont, t_reg, t_cty]:
        inp.change(update_trajectory, [t_cont, t_reg, t_cty], [trend_plot, corr_table, change_md])

    # Initial Loads (needed for Gradio to display content on first load)
    demo.load(update_globe, globe_year, globe_plot)
    demo.load(update_correlation, [scat_year, scat_cont, scat_reg, scat_fac], [scat_plot, all_factor_corr_table])
    demo.load(update_trajectory, [t_cont, t_reg, t_cty], [trend_plot, corr_table, change_md])

demo.launch(share=False, theme=gr.themes.Soft(primary_hue="teal")) # share=False for Hugging Face Spaces
