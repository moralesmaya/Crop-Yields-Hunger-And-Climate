
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler

import geopandas as gpd
import json
import joblib

from bokeh.io import  show, output_file, curdoc
from bokeh.plotting import figure
#from bokeh.embed import json_item
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, Slider, HoverTool, Column, ColumnDataSource, RangeTool
from bokeh.palettes import brewer
from bokeh.layouts import row, column
from bokeh.plotting import figure
import plotly.express as px
import plotly.graph_objects as go

import pickle

from PIL import Image
#import cv2


#Read in Data
forecasts = pd.read_csv('data/final_projections_df.csv')
undernourishment = pd.read_csv('data/Data2/prevalence-of-undernourishment.csv')

shapefile = 'data/countries_110m/ne_110m_admin_0_countries.shp'
#Read shapefile using Geopandas
gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]




@st.cache
def load_data(filepath):
    data = pd.read_csv(filepath)
    df = data[['Country', 'Year', 'Population', 'total_yield', 'Average monthly precipitation', 'AverageTemperature']]
    df['population_scaled'] = np.log2(df['Population'])
    df.rename(columns = {'Average monthly precipitation': 'AvgPrecipitation'}, inplace = True)
    df["Country"].replace({"United States": "United States of America", "Congo":'Democratic Republic of the Congo',
    'Tanzania': 'United Republic of Tanzania'}, inplace = True)
    df_country = df.groupby(["Country", "Year"]).mean()
    df_country['temp_change'] = df_country['AverageTemperature'].diff()
    df_country.reset_index(inplace = True)
    df = df_country
    return df

df = load_data('data/tomap.csv')
data = pd.read_csv('data/tomap.csv')

def get_data_for_map(year):
    year_df = df[df['Year'] == year]
    #sales = df.loc[df["InvoiceDate"].dt.year == year].groupby("Country")[["total_yield"]].sum()
    #sales["scaled_revenue"] = np.log(sales.Revenue)
    merged = sf.merge(year_df, how='left', left_on='Country', right_on='Country')
    merged.fillna(0, inplace = True)
    merged_json = json.loads(merged.to_json())
    json_data = json.dumps(merged_json)
    return json_data

def get_data_for_hunger_map(year):
    year_df = hunger[hunger['Year'] == year]
    #sales = df.loc[df["InvoiceDate"].dt.year == year].groupby("Country")[["total_yield"]].sum()
    #sales["scaled_revenue"] = np.log(sales.Revenue)
    merged = sf.merge(year_df, how='left', left_on='Country', right_on='Country')
    merged.fillna(0, inplace = True)
    merged_json = json.loads(merged.to_json())
    json_data = json.dumps(merged_json)
    return json_data



@st.cache
def read_shape_file():
    shapefile = gpd.read_file('https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip')
    shapefile = shapefile.filter(['ADMIN', 'geometry'])
    shapefile.rename(columns = {"ADMIN": "Country"}, inplace = True)
    shapefile.drop(shapefile.loc[shapefile["Country"] == "Antarctica"].index, axis = 0, inplace = True)
    return shapefile

st.sidebar.title("Explore Climate Change and Food scarcity ")
select_display = st.sidebar.radio("Choose your analysis", ("Background", "Important Climate Indicators by Country","Hunger & Food Scarcity","Future Projections", "Make Your Own Predictions","My Models", "Take Aways"))

if select_display == 'Background':

    st.image('foodie.jpg', use_column_width = True)


    st.write("""
    #### Climate change is having a severe impact on crop yields. Living in the developed world we often have the privilege of importing food when we need it so we do not feel yield shortaged as strongly as other countries.
    ####
    """)
    st.image('world.jpg', use_column_width = True)

    st.write("""
     The goal of this project is to analyze how different climate variables impact crop yields, and see how that translated into hunger. This project makes predictions based on machine learning algorithms. The code can all be found on my github.


    """)

    st.image('planting.jpg', use_column_width = True)

    st.write("""
     ### Find me on Github
     https://github.com/moralesmaya

    """)

#use_column_width=True


if select_display == "Important Climate Indicators by Country":
    st.write("""
    ## Choose an Indicator to see how it's changed over time around the world



    """)
    analysis_type = st.selectbox("Indicators", ("Yields", "Temperature", "Precipitation", "Population"))

    ## YIELDS WORLD MAP ##
    if analysis_type == "Yields":
        st.write("""
        #### Hover over a Country to view it's yields for year chosen on slider



        """)

        year = st.slider("Year", 1990, 2000, 2010)
        sf = read_shape_file()

        geosource = GeoJSONDataSource(geojson = get_data_for_map(year))
        color_mapper = LinearColorMapper(palette =  brewer['YlGn'][9][::-1], low = 5000, high = 500000, nan_color = '#d9d9d9')
        tick_labels = {
        "0": "0", "1.77": "5.87", "3.54": "34.5", '5.31': "202.5",
        "7.08" : "1190", "8.85" : "7,000", "10.62": "50,000",
        "12.39": "240,385", "14.16":"1,00,00", "16":"8,886,110"}
        color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,
                             border_line_color=None,location = (0,0), orientation = 'horizontal', major_label_overrides = tick_labels)
        hover = HoverTool(tooltips = [ ('Country/region','@Country'),('Yield', "@total_yield" )])

        p = figure(title = 'Yield by Year', plot_height = 580 , plot_width = 950, toolbar_location = None, tools = [hover])
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.patches('xs','ys', source = geosource, fill_color = {'field' : "total_yield", 'transform' : color_mapper},
              line_color = 'black', line_width = 0.25, fill_alpha = 1)
        p.add_layout(color_bar, 'below')

        layout = column(p)
        st.bokeh_chart(layout)


        goal_actual = pd.read_csv('data/goal_actual.csv')

        goal_country = goal_actual.groupby('Country').mean()

        goal_country.reset_index(inplace = True)

        fig = px.line(goal_country, x='Country', y=['yield_actual', 'goal_yield'], title = "Actual yield VS. Goal Yield by Country")
        fig.update_layout(width=900,height=600)
        st.plotly_chart(fig)
        ## TEMPERATURE WORLD MAP ##
    if analysis_type == "Temperature":
        st.write("""
        #### Hover over a Country to view it's temperature change



        """)
        year = st.slider("Year", 1990, 2000, 2010)

        sf = read_shape_file()

        geosource = GeoJSONDataSource(geojson = get_data_for_map(year))
        color_mapper = LinearColorMapper(palette =  brewer['OrRd'][9][::-1], low = 0, high = 20, nan_color = '#d9d9d9')
        tick_labels = {
        "0": "0", "1.77": "5.87", "3.54": "34.5", '5.31': "202.5",
        "7.08" : "1190", "8.85" : "7,000", "10.62": "50,000",
        "12.39": "240,385", "14.16":"1,00,00", "16":"8,886,110"}
        color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,
                             border_line_color=None,location = (0,0), orientation = 'horizontal', major_label_overrides = tick_labels)
        hover = HoverTool(tooltips = [ ('Country/region','@Country'),('Temperature', "@AverageTemperature" )])

        p = figure(title = 'Temperature by Year', plot_height = 580 , plot_width = 950, toolbar_location = None, tools = [hover])
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.patches('xs','ys', source = geosource, fill_color = {'field' : "AverageTemperature", 'transform' : color_mapper},
              line_color = 'black', line_width = 0.25, fill_alpha = 1)
        p.add_layout(color_bar, 'below')

        layout = column(p)
        st.bokeh_chart(layout)

        ##RAINFALL world map##
    if analysis_type == "Precipitation":
        st.write("""
        #### Hover over a Country to view it's precipitation change
        """)
        year = st.slider("Year", 1990, 2000, 2010)
        sf = read_shape_file()

        geosource = GeoJSONDataSource(geojson = get_data_for_map(year))
        color_mapper = LinearColorMapper(palette =  brewer['Blues'][9][::-1], low = 1, high = 15, nan_color = '#d9d9d9')
        tick_labels = {
        "0": "0", "1.77": "5.87", "3.54": "34.5", '5.31': "202.5",
        "7.08" : "1190", "8.85" : "7,000", "10.62": "50,000",
        "12.39": "240,385", "14.16":"1,00,00", "16":"8,886,110"}
        color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,
                             border_line_color=None,location = (0,0), orientation = 'horizontal', major_label_overrides = tick_labels)
        hover = HoverTool(tooltips = [ ('Country/region','@Country'),('Rainfall', "@AvgPrecipitation" )])

        p = figure(title = 'Rainfall  by Year', plot_height = 580 , plot_width = 950, toolbar_location = None, tools = [hover])
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.patches('xs','ys', source = geosource, fill_color = {'field' : "AvgPrecipitation", 'transform' : color_mapper},
              line_color = 'black', line_width = 0.25, fill_alpha = 1)
        p.add_layout(color_bar, 'below')

        layout = column(p)
        st.bokeh_chart(layout)


        ##Populaton world map##
    if analysis_type == "Population":
        st.write("""
        #### Hover over a Country to view it's precipitation change
        """)
        year = st.slider("Year", 1990, 2000, 2010)
        sf = read_shape_file()

        geosource = GeoJSONDataSource(geojson = get_data_for_map(year))
        color_mapper = LinearColorMapper(palette =  brewer['PuRd'][9][::-1], low = 13, high = 35, nan_color = '#d9d9d9')
        tick_labels = {
        "0": "0", "1.77": "5.87", "3.54": "34.5", '5.31': "202.5",
        "7.08" : "1190", "8.85" : "7,000", "10.62": "50,000",
        "12.39": "240,385", "14.16":"1,00,00", "16":"8,886,110"}
        color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,
                             border_line_color=None,location = (0,0), orientation = 'horizontal', major_label_overrides = tick_labels)
        hover = HoverTool(tooltips = [ ('Country/region','@Country'),('Population', "@Population" )])

        p = figure(title = 'Population by Year', plot_height = 580 , plot_width = 950, toolbar_location = None, tools = [hover])
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.patches('xs','ys', source = geosource, fill_color = {'field' : "population_scaled", 'transform' : color_mapper},
              line_color = 'black', line_width = 0.25, fill_alpha = 1)
        p.add_layout(color_bar, 'below')

        layout = column(p)
        st.bokeh_chart(layout)


if select_display == "Future Projections":

    #analysis_type2 = st.selectbox("Indicators", ("Yields", "Temperature", "Precipitation", "Population"))

    ##  ##
    #if analysis_type2 == "Yields":
    st.write("""

        ## How will Yields change through time?

        #### Even though population increases exponetially yields cannot keep up

    """)
    fig = px.line(forecasts, x='Year_1', y=['Population Scaled', 'Yields', 'temperature with mitigation'])
    fig = fig.update_layout(width=1000,height=500)
    st.plotly_chart(fig)
        #fig.show()
    st.write("""

    ## What does this mean for food scarcity?


    Crop yields are failing to meet the 2.4% annual increase scientists predict is needed to increase food security by 2050
    """)
    group_yield = data.groupby('Year').mean()
    group_yield.reset_index(inplace = True)

    group_yield['Yield Percent Change'] = group_yield['total_yield'].pct_change()

    fig = px.line(group_yield[1:], x="Year", y= ["Yield Percent Change", 'goal_percent_change'])
    st.plotly_chart(fig)

    st.write("""

    ## Humans need calories, not yields. How can we translate this?


    Let's take a look at hunger over the past few years and try to understand if there's a correlation with yield.
    """)


    st.write("""

    ## Looking at yields by country alone we can predict hunger with about 90% accuracy


    """)

    st.image('download.jpg', use_column_width = True)



    st.write("""

    This model was created simply by using Country and total yields as predictors for % of the population undernourished

    """)

    st.write("""

    ## When we combine this with climate, population, and projected yield data we have a fairly accurate model to predict future hunger rates by country.
    #### Check out the "Make Your Own Predictions section on the left" to see what hunger will be like in each country.

    """)

if select_display == "Hunger & Food Scarcity":

    st.write("""

    ### The ultamite goal for this project is to use climate data to predict yields and eventually to predict hunger
    #### To build a model which could accurately predict hunger I had to gather historical hunger data.

    Explore this historical data it below using the interactive world map.
    """)
    hunger = pd.read_csv('data/Data2/hunger_app.csv')
    year = st.slider("Year", 2000, 2005, 2010)
    sf = read_shape_file()

    geosource = GeoJSONDataSource(geojson = get_data_for_hunger_map(year))
    color_mapper = LinearColorMapper(palette =  brewer['PuRd'][9][::-1], low = 3, high = 25, nan_color = '#d9d9d9')
    tick_labels = {
    "0": "0", "1.77": "5.87", "3.54": "34.5", '5.31': "202.5",
    "7.08" : "1190", "8.85" : "7,000", "10.62": "50,000",
    "12.39": "240,385", "14.16":"1,00,00", "16":"8,886,110"}
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,
                         border_line_color=None,location = (0,0), orientation = 'horizontal', major_label_overrides = tick_labels)
    hover = HoverTool(tooltips = [ ('Country/region','@Country'),('Population', "@Undernourishment" )])

    p = figure(title = '% of Population Undernourished', plot_height = 580 , plot_width = 950, toolbar_location = None, tools = [hover])
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.patches('xs','ys', source = geosource, fill_color = {'field' : "Undernourishment", 'transform' : color_mapper},
          line_color = 'black', line_width = 0.25, fill_alpha = 1)
    p.add_layout(color_bar, 'below')

    layout = column(p)
    st.bokeh_chart(layout)

    st.write("""

    Climate change and Hunger disproportionately impact developing countries while the developed world countributes to the majority of green house gas emissions.

    ### Below shows Countries with the greatest % Undernourished in hunger_2013
    #### Hover over to see data.
    """)

    hunger_2013 = hunger[hunger['Year'] == 2013]

    fig = px.sunburst(hunger_2013, path=['Country'], values='Undernourishment',
                  color='Undernourishment', hover_data=['total_yield'],
                  color_continuous_scale='RdBu',
                  color_continuous_midpoint=np.average(hunger_2013['Undernourishment'], weights=hunger_2013['total_yield']))
    fig.update_layout(width=600,height=600)
    st.plotly_chart(fig)

    st.write("""

    Climate change and Hunger disproportionately impact developing countries while the developed world countributes to the majority of green house gas emissions.

    ### Hunger is correlated with a wide range of both temperature and yield variables.

    """)
    st.image('Correlations.jpg', use_column_width = True)

    st.write("""

    This may look confusing, many of these variables are engineered, aka multipled etc, with one another.
    Overall, you may notice that as temperature is positive correlated with hunger. Aka as temperatures increase, so does hunger.


    """)




if select_display == "Make Your Own Predictions":
    # loading the trained model##,
    st.write("""

    ### Explore future yield predictions by country and year


    """)

    pickle_in = open('preds.pkl', 'rb')

    #

    classifier = pickle.load(pickle_in)

    features = pd.read_csv('data/prefeatures.csv')
    #@st.cache()
    country = st.text_input("Please input the country name: ").lower()
    year = st.number_input("Please input the year to predict: ")


    country2 = country
    year2 = year
    #def prediction(country, year):
    from sklearn.preprocessing import OneHotEncoder

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(features)

    transformed = enc.transform([[year,country]]).toarray()
    # Making predictions

    prediction = classifier.predict(transformed)
    ##




    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result = prediction
        st.success('The predicted yield for {} in {} is {}'.format(country, year,result))
        #print(prediction)


    st.write("""

    ### How accurate is my prediction?

    """)
    random_x = np.linspace(0, 1, 2313)
    yield_acc = pd.read_csv('data/yield_predictions_acc.csv')
    fig = go.Figure(data = go.Scatter(x = yield_acc['yield_actual'], y=yield_acc['yield_predicted'],mode = 'markers',
                                      marker_color=yield_acc['yield_actual']))

    fig.update_layout(title='Yield Predicted Vs. Yield Actual',
                     xaxis_title="Yield Actual",
                    yaxis_title="Yield Predicted",)
    st.plotly_chart(fig)

    st.write("""

    Predictions are made on a model which predicts yield with about 80% accuracy, however the further into the future you get, the more likely the accuracy of the prediction is to go down.


    """)

    st.write("""

    ## Would you like to also predict hunger based on the Country and Year you chose?


    """)

    pickle_in_2 = open('hunger_mod.pkl', 'rb')

    classifier2 = pickle.load(pickle_in_2)

    features2 = pd.read_csv('data/hungerprefeatures.csv')


    enc2 = OneHotEncoder(handle_unknown='ignore')
    enc2.fit(features2)

    country2 = str(country)
    year2 = int(year2)
    prediction2 = int(prediction)

    transformed2 = enc2.transform([[year2, country2, prediction2]]).toarray()

    # Making predictions
    prediction2 = classifier2.predict(transformed2)


    if st.button("Predict Hunger"):
        result2 = prediction2
        st.success('The predicted hunger for {} in {} is {}'.format(country, year,result2))
        #print(prediction2)
    #pickle_in = open('hunger.pkl', 'rb')

        st.write("""

        #### This number represents the precentage of the population that is considered undernourished.


        """)
if select_display == 'My Models':
    st.write("""
    # My Models
    #### Decision Tree Regressors were used to predict both crop yields and then again to predict hunger.
    Choose a model below to find more details.
    """)
    analysis_type = st.selectbox("Models", ("Yields", "Hunger"))

    if analysis_type == 'Yields':
        st.write("""

        ### Models that predicted yields

        ## Starting with the Arima Time Series Model
        the yellow line is actual yield, the green line is the predicted yield.
        """)

        st.image('arima.jpg', use_column_width = True)
        st.write("""

        Arima is well known for predicting future data. I fit a SARIMAX (0, 2, 1), (1, 1, 1, 4) model. Though my predictions were fairly accurate, I soon realized I would have to sacrafice any hope of making prediction by country.

        """)

        st.write("""

        ### Forecasting with SARIMAX
        A perk of this model is that it was able to differentiate between climate condition data with Co2 mitigation and without.
        This model shows yields decreasing at a higher rate under future conditions where emission are not reduced and temperatures and precip grow increasingly uncertain.


        The blue line represents no mitigation predictions. The green line represents mitigation predictions.
        """)

        st.image('yieldpreds.jpg', use_column_width = True)


        st.write("""

        This model was based almost entirely on projected data. It used predicted temperatures and precipitation data from NASA for the years 2013 - 2100.
        Because these predictions were based entirely on unforeseen data, it was difficult to measure it's accuracy.
        """)


        st.write("""
        ## Decision Tree Regressor Models

        Because I wanted to fit models which would make predictions using countries without have to build hundreds of seperate models for each country. Decision tree regresssor was a perfect fit.

        My decision tree regressor could predict yield based on different climate conditions with up to 90% accuracy across train and test data.
        """)

        st.image('modelacc.jpg', use_column_width = True)
        ###
        st.write("""
        After one-hot-encoding Year and Country, I was able to transpose and reformat my data so that I could recieve my predictions along with their associated year and country.
        """)

        st.image('predscountry.jpg', use_column_width = False)

    if analysis_type == 'Hunger':
        st.write("""
        ## Decision Tree Regressor Models

        All hunger models were fit using a decision tree regressor.

        Accuracy ranged from 85 - 91%

        Accuracy cannot be calculated for predictions based entrirely on unforeseen data.
        """)

        st.image('anotha1.jpg', use_column_width = False )

        st.write("""
        Predictions based on Country and Year
        """)


        st.image('preds1.jpg', use_column_width = False )

if select_display == 'Take Aways':
    st.write("""
    ## To ensure global food sceurity climate change repercussions must be mitigatied.

    """)

    st.write("""
    ### At minimum we need to bridge the gap between the rate of increase between the population and between yields to prevent hunger.

    """)
    st.image('yield_pop.jpg')

    st.write("""
    ## Future Direction
    In the future I want to better predict yields across different mitigation data. This information can be used to better inform hunger intiatives, where they need to focus, as well as policy intiatives, how hot can our planet get without risking our global food supply?

    """)

    #st.image('direction.jpg', use_column_width = False)
    st.write("""

    # Thank you for watching!

    """)
    st.image('food_kid.jpg')
