"""
Class: CS230--Section 3
Name: Massimo Lorenzetti
Description: Final Project: Uber Data
I pledge that I have completed the programming assignment independently.
I have not copied the code from a student or any source.
I have not given my code to any student.
"""

import pandas as pd
import numpy as np
# import matplotlib as mpl
# import datetime as dt
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.basemap import Basemap
# from geopy.geocoders import Nominatim
from geopy import distance

def all_coords_valid(lat_1, lon_1, lat_2, lon_2) :
    """
    Checks validity of the given coordinates
    Parameters: coords (tuple of coordinates)
    Returns:    True if all coords exist and within -90 and 90, False otherwise
    """
    return all((coord != None and -90 <= coord <= 90) for coord in (lat_1, lon_1, lat_2, lon_2))

def calc_dist(lat_1, lon_1, lat_2, lon_2) :
    """
    Calculates distance between 2 given sets of coordinates
    Parameters: lat_1 (latitude of point 1)
                lon_1 (longitude of point 1)
                lat_2 (latitude of point 2)
                lon_2 (longitude of point 2)
    Returns:    The distance between the given points
    """
    return distance.distance((lat_1, lon_1), (lat_2, lon_2)).miles

def refine_names(data) :
    """
    Refines the names of the Uber dataframe's columns and drops the 'key' and
    index columns
    Parameters: data (dataframe of data)
    Returns:    A dataframe with refined column names and no 'key' column
    """
    data = data.reset_index()
    data = data.drop(['index', 'key'], axis=1)
    refined_names = {}
    for col in data.columns:
        refined_names[col] = col.replace('_', ' ').title()
    return data.rename(refined_names, axis=1)

def remove_invalid_rows(data) :
    """
    Removes rows that have invalid Uber data
    Parameters: data (dataframe of data)
    Returns:    A dataframe with no invalid data
    """
    for index, row in data.iterrows():
        if not (
            row['Fare Amount'] != None and row['Fare Amount'] > 0 and
            row['Pickup Datetime'] != None and
            all_coords_valid(row['Pickup Latitude'], row['Pickup Longitude'], row['Dropoff Latitude'], row['Dropoff Longitude']) and
            row['Passenger Count'] != None and row['Passenger Count'] > 0
        ):
            data = data.drop(labels=index)
    return data

def get_pickup_hours(data) :
    """
    Converts pickup timestamps to just hour of pickup
    Parameters: data (dataframe of data)
    Returns:    A dataframe with only hour of pickup, not exact pickup timestamp
    """
    data['Pickup Datetime'] = [time.hour for time in pd.to_datetime(data['Pickup Datetime'])]
    return data.rename({'Pickup Datetime' : 'Pickup Hour'}, axis=1)

def replace_coords_with_distances(data) :
    """
    Removes the coordinate columns and replaces them with a distance column
    Parameters: data (dataframe of data)
    Returns:    A dataframe with a distance column and no coordinate columns
    """
    distances = []
    for index, row in data.iterrows():
        distances.append(calc_dist(row['Pickup Latitude'], row['Pickup Longitude'], row['Dropoff Latitude'], row['Dropoff Longitude']))
    data['Distance (mi)'] = distances
    return data.drop(['Pickup Latitude', 'Pickup Longitude', 'Dropoff Latitude', 'Dropoff Longitude'], axis=1)

def load_data(nrows = -1) :
    """
    Loads and refines Uber data
    Parameters: data (dataframe of data)
    Returns:    An Uber dataframe with no invalid data
    """
    data = pd.read_csv('Uber Data CSV.csv', index_col=0, nrows=nrows) if nrows > 0 else pd.read_csv('Uber Data CSV.csv', index_col=0)
    data = refine_names(data)
    data = remove_invalid_rows(data)
    data = get_pickup_hours(data)
    data = replace_coords_with_distances(data)
    return data

def display_raw_data_analysis(data) :
    """
    Creates and displays an analysis of the raw data in an expander
    Parameters: data (dataframe of data)
    Returns:    An Uber dataframe with no invalid data
    """
    TRIPS_TO_SHOW = 5
    FAR_DISTANCE = 5
    EXPENSIVE_RIDE = 25
    with st.expander('Raw Data Analysis', expanded=True):
        fare_sorted = data.sort_values('Fare Amount')
        st.subheader('Cheapest Rides')
        st.dataframe(fare_sorted.head(TRIPS_TO_SHOW))
        st.subheader('Most Expensive Rides')
        st.dataframe(fare_sorted.tail(TRIPS_TO_SHOW))
        st.subheader('Rides Over $25')
        st.dataframe(data[data['Fare Amount'] > EXPENSIVE_RIDE].head(TRIPS_TO_SHOW))
        st.subheader('Rides Over 5 Miles')
        st.dataframe(data[data['Distance (mi)'] > FAR_DISTANCE].head(TRIPS_TO_SHOW))
        st.subheader('Expensive Solo Rides')
        st.dataframe(data[(data['Fare Amount'] > EXPENSIVE_RIDE) &
            (data['Passenger Count'] == 1)].head(TRIPS_TO_SHOW))
        st.subheader('Expensive Short Rides')
        st.dataframe(data[(data['Fare Amount'] > EXPENSIVE_RIDE) &
            (data['Distance (mi)'] < FAR_DISTANCE)].head(TRIPS_TO_SHOW))

def time_vs_fare(data) :
    """
    Isolates pickup time and fare data and creates a line plot to visualize the
    relationship between them
    Parameters: data (dataframe of data)
    Returns:    A line plot of fare vs pickup hour and relevant data
    """
    time_fare = pd.DataFrame.from_dict({
        'time' : data['Pickup Hour'],
        'fare' : data['Fare Amount']
    })
    time_fare_gb = time_fare.groupby('time').mean()
    fig, ax = plt.subplots()
    ax.plot(range(0, 24), time_fare_gb)
    ax.set_title('Fare Amount vs Number of Cars')
    plt.xlabel('Number of Cars')
    plt.ylabel('Fare Amount')
    return fig, time_fare_gb

def sort_size(num_pass) :
    """
    Categorizes numbers of passengers to size of care required
    Parameters: num_pass (number of passengers in the Uber)
    Returns:    The size of car required
    """
    if num_pass < 4:
        return 'Single Uber'
    if num_pass < 7:
        return 'Single Uber XL'
    else:
        return 'Multiple Ubers'

def num_passengers_vs_fare(data) :
    """
    Isolates passenger count and fare data and creates a bar graph to visualize
    the relationship between them
    Parameters: data (dataframe of data)
    Returns:    A bar graph of fare vs passenger count and relevant data
    """
    num_pass_fare = pd.DataFrame.from_dict({
        'num_pass' : [sort_size(num) for num in data['Passenger Count']],
        'fare' : data['Fare Amount']
    })
    num_pass_fare_gb = num_pass_fare.groupby('num_pass').mean()
    fig, ax = plt.subplots()
    ax.bar(num_pass_fare_gb.index, num_pass_fare_gb['fare'].values)
    ax.set_title('Fare Amount vs Time of Day')
    plt.xlabel('Number of Cars')
    plt.ylabel('Fare Amount')
    return fig, num_pass_fare_gb

def length_vs_fare(data) :
    """
    Isolates ride distance and fare data and creates a scatter plot to visualize
    the relationship between them. Also generates data for rounded ride distance
    Parameters: data (dataframe of data)
    Returns:    A scatter plot of fare vs ride distance and relevant data
    """
    coord_fare = pd.DataFrame.from_dict({
        'length' : data['Distance (mi)'],
        'fare' : data['Fare Amount']
    })
    coord_fare_rounded = pd.DataFrame.from_dict({
        'length' : np.round(data['Distance (mi)']),
        'fare' : data['Fare Amount']
    })
    coord_fare_rounded_gb = coord_fare_rounded.groupby('length').mean()

    fig, ax = plt.subplots()
    ax.scatter('length', 'fare', edgecolors='black', data=coord_fare, linewidths=0.75)
    ax.set_title('Fare Amount vs Distance Travelled in Miles')
    plt.xlabel('Distance (mi)')
    plt.ylabel('Fare Amount')
    return fig, coord_fare_rounded_gb

def get_trip_data() :
    """
    Creates a form for a user to input trip data
    Returns: The trip data that the user entered
    """
    time, num_pass, lat_1, lon_1, lat_2, lon_2 = 0, 0, 0, 0, 0, 0
    with st.form(key='price_estimate'):
        time = st.time_input('Time', args=tuple(range(1,24))).hour
        num_pass = st.select_slider('Number of Passengers', [1, 2, 3, 4, 5, 6, '7 or more'])
        lat_1_input, lon_1_input = st.columns(2)
        lat_2_input, lon_2_input = st.columns(2)
        with lat_1_input:
            lat_1 = st.number_input('Pickup Latitude', min_value=-90.0, max_value=90.0, value=0.0, step=0.25)
        with lon_1_input:
            lon_1 = st.number_input('Pickup Longitude', min_value=-90.0, max_value=90.0, value=0.0, step=0.25)
        with lat_2_input:
            lat_2 = st.number_input('Dropoff Latitude', min_value=-90.0, max_value=90.0, value=0.0, step=0.25)
        with lon_2_input:
            lon_2 = st.number_input('Dropoff Longitude', min_value=-90.0, max_value=90.0, value=0.0, step=0.25)
        return time, num_pass, lat_1, lon_1, lat_2, lon_2, st.form_submit_button(label='Submit')

def show_trip_map(lat_1, lon_1, lat_2, lon_2) :
    """
    Plots the given points on a map (to show a user's Uber trip)
    Parameters: lat_1 (latitude of point 1)
                lon_1 (longitude of point 1)
                lat_2 (latitude of point 2)
                lon_2 (longitude of point 2)
    Returns:    A map with 2 points plotted
    """
    COORD_TO_METERS = 111139
    lat_0 = (lat_1+lat_2)/2 if (lat_1 != 0 and lat_2 != 0) else 1
    lon_0 = (lon_1+lon_2)/2 if (lon_1 != 0 and lon_2 != 0) else 1
    width = max(abs(lon_1-lon_2), 0.1)*2*COORD_TO_METERS
    height = max(abs(lat_1-lat_2), 0.1)*2*COORD_TO_METERS
    map = Basemap(projection='lcc', lat_0=lat_0, lon_0=lon_0, width=width, height=height, resolution='f')
    f = plt.figure()
    map.drawcountries()
    map.drawstates()
    map.drawcoastlines()
    map.fillcontinents(color='lightgray')

    x_1, y_1 = map(lon_1, lat_1)
    x_2, y_2 = map(lon_2, lat_2)
    line_x = [lon_1, lon_2]
    line_y = [lat_1, lat_2]
    line_coords_x, line_coords_y = map(line_x, line_y)
    map.plot(line_coords_x, line_coords_y,marker='o',linestyle='solid',color='Blue',markersize=5,linewidth=1)
    map.plot(x_1,y_1,marker='o',color='Green',markersize=5)
    map.plot(x_2,y_2,marker='o',color='Red',markersize=5)
    plt.annotate("1", xy = (x_1,y_1), xytext=(-20,20))
    plt.annotate("2", xy = (x_2,y_2), xytext=(-20,20))
    return f

def find_nearest_dist(distance, distances) :
    """
    Finds the nearest distance in a collection of distances that is greater than
    or equal to a given distance
    Parameters: distance (the distance to be searched for)
                distances (fare vs passenger count data)
    Returns:    An estimate of the Uber trip's cost
    """
    for d in distances:
        if distance <= d:
            return d
    return distances[-1]

def calc_cost(time_fare_gb, num_pass_fare_gb, coord_fare_rounded_gb, time, num_pass, lat_1, lon_1, lat_2, lon_2) :
    """
    Estimates the cost of a user's Uber trip based on the given information
    Parameters: time_fare_gb (fare vs pickup time data)
                num_pass_fare_gb (fare vs passenger count data)
                coord_fare_rounded_gb (fare vs rounded distance data)
                time (pickup hour)
                num_pass (number of passengers)
                lat_1 (latitude of point 1)
                lon_1 (longitude of point 1)
                lat_2 (latitude of point 2)
                lon_2 (longitude of point 2)
    Returns:    An estimate of the Uber trip's cost
    """
    time_fare = time_fare_gb['fare'][time]
    num_pass_fare = num_pass_fare_gb['fare'][sort_size(num_pass)]
    coord_fare = coord_fare_rounded_gb['fare'][find_nearest_dist(round(calc_dist(lat_1, lon_1, lat_2, lon_2)), coord_fare_rounded_gb.index)]
    return round(np.mean([time_fare, num_pass_fare, coord_fare]), 2)

def display_price_estimator(time_fare_gb, num_pass_fare_gb, coord_fare_rounded_gb) :
    """
    Creates a container for the trip information form and resulting estimation
    and ride map to be displayed in
    """
    AM = 'AM'
    PM = 'PM'
    with st.expander('Price Estimator'):
        submitted = False
        time, num_pass, lat_1, lon_1, lat_2, lon_2, submitted = get_trip_data()
        if submitted:
            cost = calc_cost(time_fare_gb, num_pass_fare_gb, coord_fare_rounded_gb, time, num_pass, lat_1, lon_1, lat_2, lon_2)
            if cost < 15:
                st.success(f'Your trip will cost ${cost} for {num_pass} people travelling about {round(calc_dist(lat_1, lon_1, lat_2, lon_2))} miles at around {time%12}{AM if time < 12 else PM}')
            else:
                st.error(f'Your trip will cost ${cost} for {num_pass} people travelling about {round(calc_dist(lat_1, lon_1, lat_2, lon_2))} miles at around {time%12}{AM if time < 12 else PM}')
            st.pyplot(show_trip_map(lat_1, lon_1, lat_2, lon_2))

def main() :
    """
    Driver function for the page, loads data and creates page elements
    """
    data = load_data()
    time_vs_fare_fig, time_fare_gb = time_vs_fare(data)
    num_passengers_vs_fare_fig, num_pass_fare_gb = num_passengers_vs_fare(data)
    length_vs_fare_fig, coord_fare_rounded_gb = length_vs_fare(data)

    st.title('Uber Metrics')
    st.text('Massimo Lorenzetti - April 2022')

    st.header('Raw Data Analysis')
    display_raw_data_analysis(data)

    st.header('Data Visualizations')
    st.pyplot(time_vs_fare_fig)
    st.pyplot(num_passengers_vs_fare_fig)
    st.pyplot(length_vs_fare_fig)

    st.header('Price Estimator and Visualizer')
    display_price_estimator(time_fare_gb, num_pass_fare_gb, coord_fare_rounded_gb)

    st.sidebar.info('If you want to download the data, just click this button!')
    st.sidebar.download_button(label="Download data CSV",
                               data=data.to_csv().encode('utf-8'),
                               file_name='Uber Data CSV.csv',
                               mime='text/csv',)

main()
