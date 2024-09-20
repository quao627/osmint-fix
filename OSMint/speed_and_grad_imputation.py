import sys

import networkx as nx
import numpy as np
import pandas as pd
import requests

import osmnx as ox
import pickle

import matplotlib.pyplot as plt
import time

def impute_speed_limits(G, res_speed_limit=25, motorway_speed_limit=70):
  # Convert all speed strings to integers and integer tuples
  def convert_string_to_int(speed_entry):
    if isinstance(speed_entry, list):
      processed_list = [convert_string_to_int(item) for item in speed_entry]
      if processed_list.count(processed_list[0]) == len(processed_list):
        return processed_list[0]
      else:
        return tuple(processed_list)
    elif isinstance(speed_entry, str):
      try:
        processed_entry = int(speed_entry)
      except:
        processed_entry = int(speed_entry.replace("mph", "").strip())
      return processed_entry
    else:
      return speed_entry
    
  nodes, edges = ox.graph_to_gdfs(G)
  
  edges['maxspeed'] = edges['maxspeed'].apply(convert_string_to_int)
  edges['lanes'] = edges['lanes'].apply(convert_string_to_int)
  
    # Convert all list names to tuples (hashable)
  edges['name'] = edges['name'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
  edges['highway'] = edges['highway'].apply(lambda x: tuple(x) if isinstance(x, list) else x)

  null_speed_edges = edges[edges.maxspeed.isna()]

  # Replace all null residentials with default (default 25)
  mask = (edges.highway == 'residential') & (edges.maxspeed.isna())
  edges.loc[mask, 'maxspeed'] = res_speed_limit

  # Replace all null motorways with 70
  mask = (edges.highway == 'motorway') & (edges.maxspeed.isna())
  edges.loc[mask, 'maxspeed'] = 70
  edges[edges.highway == 'motorway']

  # Impute most consistent roads by name
  null_speed_edges = edges[edges.maxspeed.isna()]
  unique_names = null_speed_edges.name.unique()
  for name in unique_names:
    # Assume that if a street with the same street name only has one written speed limit, then that is the speed limit for the whole street
    # Get the speed limit with the maximum frequency, and if the percentage of non-null values is greater than 90%, then replace with most common speed (based on names)
    
    possible_speeds = edges[edges.name == name]['maxspeed'].value_counts()
    
    if possible_speeds.count() != 0:
      most_frequent_speed = possible_speeds.idxmax()
      frequency = possible_speeds.max()
      num_non_null = possible_speeds.sum()
      
      if frequency / num_non_null > 0.9: # Most frequent speed occurs more than 85% of the time for one street
        # Replace all null values with the unique value
        mask = (edges.name == name) & (edges.maxspeed.isna())
        edges.loc[mask, 'maxspeed'] = most_frequent_speed
    
  # Impute most consistent roads by highway type
  unique_highway_types = null_speed_edges.highway.unique()
  len(unique_highway_types)
  for highway in unique_highway_types:
    # Get the speed limit with the maximum frequency, and if the percentage of non-null values is greater than 90%, then replace with most common speed (based on highways)
    possible_speeds = edges[edges.highway == highway]['maxspeed'].value_counts()
    if possible_speeds.count() != 0:
      most_frequent_speed = possible_speeds.idxmax()
      frequency = possible_speeds.max()
      num_non_null = possible_speeds.sum()
      
      if frequency / num_non_null > 0.9: # Most frequent speed occurs more than 85% of the time for one street
        # Replace all null values with the unique value
        mask = (edges.highway == highway) & (edges.maxspeed.isna())
        edges.loc[mask, 'maxspeed'] = most_frequent_speed
        
  # Impute the all streets with lane counts using combination of highway type and street lane count
  # Start by converting lane counts to be equal to the minimum of the tuple (want to err on the side of slower instead of faster when breaking ties (tends to be more accurate))
  edges.lanes = edges.lanes.apply(lambda x: min(x) if isinstance(x, tuple) else x)
  
  for highway_type in edges.highway.unique():
    for lane_count in edges[edges.highway == highway_type].lanes.unique():
      if lane_count == None:
        continue
      
      # Get the most popular speed at this combination of speed and lanes
      speed_frequencies = edges[(edges.lanes == lane_count) & (edges.highway == highway_type)].maxspeed.value_counts()
      
      # Impute all null values with most_frequent_speed
      if speed_frequencies.count() != 0:
        most_frequent_speed = speed_frequencies.idxmax()
        if isinstance(most_frequent_speed, tuple): # ANything that is combined will get the largest speed (likely not to affect much)
          most_frequent_speed = max(most_frequent_speed)
        mask = (edges.lanes == lane_count) & (edges.highway == highway_type) & (edges.maxspeed.isna())
        edges.loc[mask, 'maxspeed'] = most_frequent_speed
  
  # Impute the rest of the streets using street names (second round without threshold)
  unique_name_types = null_speed_edges.name.unique()
  for name in unique_name_types:
    possible_speeds = edges[edges.name == name]['maxspeed'].value_counts()
    
    if possible_speeds.count() != 0:
      most_frequent_speed = possible_speeds.idxmax()
      
      mask = (edges.name == name) & (edges.maxspeed.isna())
      edges.loc[mask, 'maxspeed'] = most_frequent_speed
      
  # A few edges have tuples where they change. For the sake of simplicity, I'm replacing the tuples with the max
  edges_graph = edges.copy()
  edges_graph.maxspeed = edges_graph.maxspeed.apply(lambda x: max(x) if isinstance(x, tuple) else x)

  G_graph = ox.graph_from_gdfs(nodes, edges_graph)
  
  return G_graph

# Add node elevations from USGS

def add_node_elevations_usgs(G, pause_duration=0, precision=3):
  # elevation API endpoint ready for use
  url_template = 'http://nationalmap.gov/epqs/pqs.php?{}&units=Feet&output=json'

  # make a pandas series of all the nodes' coordinates as 'lat,lng'
  # round coordinates to 5 decimal places (approx 1 meter) to be able to fit
  # in more locations per API call
  
  node_points = pd.Series(
    # {node: f'{data["y"]:.5f},{data["x"]:.5f}' for node, data in G.nodes(data=True)}
    {node: f'x={data["x"]:.5f}&y={data["y"]:.5f}' for node, data in G.nodes(data=True)}
  )

  results = []
  for i in range(0, len(node_points)):
    location = node_points.iloc[i]
    url = url_template.format(location)

    # check if this request is already in the cache (if global use_cache=True)
    cached_response_json = ox.downloader._retrieve_from_cache(url)
    if cached_response_json is not None:
      response_json = cached_response_json
    else:
      try:
        # request the elevations from the API
        ox.utils.log(f"Requesting node elevations: {url}")
        time.sleep(pause_duration)
        response = requests.get(url)
        response_json = response.json()
        ox.downloader._save_to_cache(url, response_json, response.status_code)
      except Exception as e:
        print(e)
    
        ox.utils.log(e)
        ox.utils.log(f"Server responded with {response.status_code}: {response.reason}")

    # append these elevation results to the list of all results
    results.append(response_json["USGS_Elevation_Point_Query_Service"]["Elevation_Query"])
    print(response_json["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]["Elevation"])

  # sanity check that all our vectors have the same number of elements
  if not (len(results) == len(G) == len(node_points)):
    raise Exception(
        f"Graph has {len(G)} nodes but we received {len(results)} results from elevation API"
    )
  else:
    ox.utils.log(
        f"Graph has {len(G)} nodes and we received {len(results)} results from elevation API"
    )

  # add elevation as an attribute to the nodes
  df = pd.DataFrame(node_points, columns=["node_points"])
  df["elevation"] = [result["Elevation"] for result in results]
  df["elevation"] = df["elevation"].round(precision)
  nx.set_node_attributes(G, name="elevation", values=df["elevation"].to_dict())
  ox.utils.log("Added elevation data from USGS to all nodes.")
  
  return G

def impute_edge_grades(G):
  graph = add_node_elevations_usgs(G)
  print('node elevations calculated')
  graph = ox.add_edge_grades(graph)
  