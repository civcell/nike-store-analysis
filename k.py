import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import folium
import plotly.graph_objs as go
from statsmodels.tsa.api import ExponentialSmoothing
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------
# Data Generation Using Nike Store Locations
# ------------------------------------------------

# List of Nike stores with coordinates
nike_stores = [
    {
        'Store_Name': 'Nike Factory Store - San Jose',
        'lat': 37.29284440730021,
        'lon': -121.98794716456636
    },
    {
        'Store_Name': 'Nike Factory Store - Milpitas',
        'lat': 37.41498541613848,
        'lon': -121.8967474132278
    },
    {
        'Store_Name': 'Nike San Francisco',
        'lat': 37.7886408431095,
        'lon': -122.40666443985953
    },
    {
        'Store_Name': 'Nike Clearance Store - San Leandro',
        'lat': 37.70970732463595,
        'lon': -122.16334928563877
    },
    {
        'Store_Name': 'Nike Well Collective - Santana Row',
        'lat': 37.32210435541055,
        'lon': -121.94833339519059
    },
]

# dataFrame creation
stores_df = pd.DataFrame(nike_stores)

# geometry column from coordinates
stores_df['geometry'] = stores_df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)

# geoDataFrame creation
gdf = gpd.GeoDataFrame(stores_df, geometry='geometry', crs='EPSG:4326')

# Add synthetic sales data
np.random.seed(42)  # For reproducibility
gdf['sales_volume'] = np.random.uniform(500000, 2000000, len(gdf))  # Higher sales volumes
gdf['growth_rate'] = np.random.uniform(-0.05, 0.15, len(gdf))  # -5% to +15%
gdf['customer_count'] = np.random.randint(1000, 10000, len(gdf))  # Higher customer counts

# store_id assignment
gdf = gdf.reset_index(drop=True)
gdf['store_id'] = gdf.index + 1

# ------------------------------------------------
# Generate Time Series Data for Forecasting
# ------------------------------------------------

def generate_time_series_data(store_names, num_months=12):
    time_series_data = pd.DataFrame()
    for store_name in store_names:
        base_sales = np.random.uniform(500000, 2000000)
        monthly_growth_rates = np.random.uniform(-0.02, 0.05, num_months)
        sales_volumes = [base_sales]
        for growth in monthly_growth_rates:
            sales_volumes.append(sales_volumes[-1] * (1 + growth))
        sales_volumes = sales_volumes[1:]  
        dates = pd.date_range(start='2024-01-01', periods=num_months, freq='MS')
        df = pd.DataFrame({
            'store_name': store_name,
            'month': dates,
            'sales_volume': sales_volumes
        })
        time_series_data = pd.concat([time_series_data, df], ignore_index=True)
    return time_series_data

# generation of time series data
store_names = gdf['Store_Name'].tolist()
ts_data = generate_time_series_data(store_names)

# ------------------------------------------------
# Dash Application Setup
# ------------------------------------------------

# dash app creation
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  

# app layout
app.layout = dbc.Container([
    html.H1("Geospatial Analysis of Synthetic Sales Data for Nike Stores with Forecasting"),
    dbc.Row([
        dbc.Col([
            html.H4("Filters"),
            html.Label("Sales Volume Range ($):"),
            dcc.RangeSlider(
                id='sales_range',
                min=500000, max=2000000, step=50000,
                value=[500000, 2000000],
                marks={i: f'${i//1000}k' for i in range(500000, 2000001, 250000)}
            ),
            html.Label("Growth Rate (%):"),
            dcc.RangeSlider(
                id='growth_rate',
                min=-5, max=15, step=1,
                value=[-5, 15],
                marks={i: f'{i}%' for i in range(-5, 16, 5)}
            ),
            html.Label("Customer Count:"),
            dcc.RangeSlider(
                id='customer_count',
                min=1000, max=10000, step=500,
                value=[1000, 10000],
                marks={i: str(i) for i in range(1000, 10001, 2000)}
            ),
            html.Br(),
            dbc.Button("Apply Filters", id='apply_filters', color='primary'),
            html.Hr(),
            html.H4("Forecasting"),
            html.Label("Select Store for Forecasting:"),
            dcc.Dropdown(
                id='selected_store',
                options=[{'label': name, 'value': name} for name in gdf['Store_Name']],
                value=gdf['Store_Name'].iloc[0]
            ),
            dbc.Button("Run Forecast", id='run_forecast', color='primary'),
        ], width=3),
        dbc.Col([
            dcc.Tabs(id='tabs', value='tab-map', children=[
                dcc.Tab(label='Sales Map', value='tab-map'),
                dcc.Tab(label='Forecast', value='tab-forecast'),
            ]),
            html.Div(id='tabs-content')
        ], width=9)
    ])
], fluid=True)

# ------------------------------------------------
# Callbacks for Interactivity
# ------------------------------------------------

# callback to update the map or forecast tab content
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value'),
     Input('apply_filters', 'n_clicks'),
     Input('run_forecast', 'n_clicks')],
    [State('sales_range', 'value'),
     State('growth_rate', 'value'),
     State('customer_count', 'value'),
     State('selected_store', 'value')]
)
def render_content(tab, apply_clicks, forecast_clicks, sales_range, growth_rate_range, customer_count_range, selected_store):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if tab == 'tab-map':
        # filtering the data
        filtered_gdf = gdf[
            (gdf['sales_volume'] >= sales_range[0]) & (gdf['sales_volume'] <= sales_range[1]) &
            (gdf['growth_rate'] * 100 >= growth_rate_range[0]) & (gdf['growth_rate'] * 100 <= growth_rate_range[1]) &
            (gdf['customer_count'] >= customer_count_range[0]) & (gdf['customer_count'] <= customer_count_range[1])
        ]

        # map creation
        m = folium.Map(location=[37.7749, -122.4194], zoom_start=9)

        # color scale for growth rate
        min_growth = gdf['growth_rate'].min() * 100
        max_growth = gdf['growth_rate'].max() * 100
        growth_colormap = folium.LinearColormap(['red', 'yellow', 'green'], vmin=min_growth, vmax=max_growth)
        growth_colormap.caption = 'Growth Rate (%)'

        for idx, row in filtered_gdf.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],  # latitude, longitude
                radius=(row['sales_volume'] - gdf['sales_volume'].min()) /
                       (gdf['sales_volume'].max() - gdf['sales_volume'].min()) * 10 + 5,
                color=growth_colormap(row['growth_rate'] * 100),
                fill=True,
                fill_color=growth_colormap(row['growth_rate'] * 100),
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"Store: {row['Store_Name']}<br>"
                    f"Sales Volume: ${row['sales_volume']:.2f}<br>"
                    f"Growth Rate: {row['growth_rate'] * 100:.2f}%<br>"
                    f"Customer Count: {row['customer_count']}"
                ),
            ).add_to(m)

        # adding the color legend
        growth_colormap.add_to(m)

        # saving the map to html and embedding in dash
        map_html = m._repr_html_()

        return html.Div([
            html.Iframe(srcDoc=map_html, width='100%', height='800')
        ])

    elif tab == 'tab-forecast':
        if triggered_id == 'run_forecast' and forecast_clicks:
            selected_store = str(selected_store)
            store_ts = ts_data[ts_data['store_name'] == selected_store]
            forecast_df = forecast_sales(store_ts, forecast_months=3)
            # combining data
            combined_df = pd.concat([
                store_ts[['month', 'sales_volume']],
                forecast_df.rename(columns={'forecast_sales_volume': 'sales_volume'})
            ], ignore_index=True)
            # plot creation
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=store_ts['month'],
                y=store_ts['sales_volume'],
                mode='lines+markers',
                name='Historical Sales',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['month'],
                y=forecast_df['forecast_sales_volume'],
                mode='lines+markers',
                name='Forecasted Sales',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title=f"Sales Forecast for {selected_store}",
                xaxis_title='Month',
                yaxis_title='Sales Volume ($)',
                template='plotly_white'
            )
            return dcc.Graph(figure=fig)
        else:
            return html.Div("Select a store and click 'Run Forecast' to see the forecast.")

# updating the options for the store dropdown based on filtered data
@app.callback(
    Output('selected_store', 'options'),
    [Input('apply_filters', 'n_clicks')],
    [State('sales_range', 'value'),
     State('growth_rate', 'value'),
     State('customer_count', 'value')]
)
def update_store_options(n_clicks, sales_range, growth_rate_range, customer_count_range):
    filtered_gdf = gdf[
        (gdf['sales_volume'] >= sales_range[0]) & (gdf['sales_volume'] <= sales_range[1]) &
        (gdf['growth_rate'] * 100 >= growth_rate_range[0]) & (gdf['growth_rate'] * 100 <= growth_rate_range[1]) &
        (gdf['customer_count'] >= customer_count_range[0]) & (gdf['customer_count'] <= customer_count_range[1])
    ]
    options = [{'label': name, 'value': name} for name in filtered_gdf['Store_Name']]
    return options

# ------------------------------------------------
# Forecasting Function
# ------------------------------------------------

def forecast_sales(store_ts, forecast_months=3):
    # ensuring the sales_volume is sorted by month
    store_ts = store_ts.sort_values('month')
    # fitting the model
    model = ExponentialSmoothing(store_ts['sales_volume'], trend='add', seasonal=None)
    model_fit = model.fit()
    # forecasting future sales
    forecast = model_fit.forecast(forecast_months)
    # preparing the forecast DataFrame
    forecast_dates = pd.date_range(start=store_ts['month'].iloc[-1] + pd.DateOffset(months=1), periods=forecast_months, freq='MS')
    forecast_df = pd.DataFrame({
        'month': forecast_dates,
        'forecast_sales_volume': forecast.values
    })
    return forecast_df



if __name__ == '__main__':
    app.run_server(debug=True)