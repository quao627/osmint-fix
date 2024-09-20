import math
import shapely
import pandas as pd
import numpy as np
import requests
import osmnx as ox
import geopandas as gpd
import scipy
from shapely.geometry import Point, LineString, Polygon
import warnings

warnings.filterwarnings('ignore')
useful_tags_path = ['name', 'lanes', 'turn:lanes:forward', 'turn:lanes:backward', 'lanes:both_ways', 'turn:lanes',
                    'maxspeed', 'highway']
# set the required information from osmnx
ox.utils.config(useful_tags_way=useful_tags_path)

turn_types = set(["slight_left", "slight_right", "right", "left", "through"])

def get_traffic_signals(city="Salt Lake City", admin_level=8, polygon=None, boundary=None):
    overpass_url = "http://overpass-api.de/api/interpreter"
    if polygon:
        coords = " ".join([x for coords in polygon.exterior.coords[:-1] for x in str(coords)[1:-1].split(", ")[::-1]])
        overpass_query = f"""
    [out:json];
    (node[highway=traffic_signals](poly:"{coords}");
     node[crossing=traffic_signals](poly:"{coords}");
     );
    out body;
    """
    else:
        overpass_query = f"""
    [out:json];
    area[name="{city}"][admin_level={admin_level}]->.a;
    (node[highway=traffic_signals](area.a);
     node[crossing=traffic_signals](area.a););
    out body;
    """

    response = requests.get(overpass_url,
                            params={'data': overpass_query})
    data = response.json()

    signals = data["elements"]
    signals = pd.DataFrame(signals)
    points = gpd.GeoSeries([Point(x, y) for x, y in zip(signals['lon'], signals['lat'])])
    signals = gpd.GeoDataFrame(signals[['id', 'lon', 'lat']], geometry=points)
    signals.crs = {'init': 'epsg:4326'}
    if boundary:
        signals = signals[signals["geometry"].apply(lambda x: x.within(boundary))]
    signals = signals.to_crs({'init': 'epsg:3395'})
    signals["x"] = signals["geometry"].apply(lambda x: x.coords[0][0])
    signals["y"] = signals["geometry"].apply(lambda x: x.coords[0][1])
    return signals.reset_index().drop("index", axis=1)


def merge(sets):
    merged = True
    while merged:
        merged = False
        results = []
        while sets:
            common, rest = sets[0], sets[1:]
            sets = []
            for x in rest:
                if x.isdisjoint(common):
                    sets.append(x)
                else:
                    merged = True
                    common |= x
            results.append(common)
        sets = results
    return sets


def find_clusters(signals, threshold=100):
    coords = signals[['x', 'y']].to_numpy()
    dist_matrix = scipy.spatial.distance.cdist(coords, coords)
    x_list, y_list = np.where(dist_matrix <= threshold)
    positions = np.stack((x_list, y_list), axis=1)
    first_col = positions[:, 0]
    second_col = positions[:, 1]
    neighbors = [set(second_col[np.where(first_col == i)]) for i in range(len(coords))]
    return merge(neighbors)


def get_turn_restrictions(city="Salt Lake City", admin_level=8, polygon=None, boundary=None, df_group=None):
    def get_geometry(data):
        if data["type"] == "way":
            geom = LineString([[x["lon"], x["lat"]] for x in data["geometry"]])
        else:
            geom = Point([data["lon"], data["lat"]])
        return geom

    overpass_url = "http://overpass-api.de/api/interpreter"
    if polygon:
        coords = " ".join([x for coords in polygon.exterior.coords[:-1] for x in str(coords)[1:-1].split(", ")[::-1]])
        overpass_query = f"""
            [out:json];
            (rel[restriction](poly:"{coords}"););
            out body geom;
            """
    else:
        overpass_query = f"""
        [out:json];
        area[name="{city}"][admin_level={admin_level}]->.a;
        (rel[restriction](area.a););
        out body geom;
        """
    response = requests.get(overpass_url,
                            params={'data': overpass_query})
    restrictions = response.json()["elements"]
    restrictions_dict = {"type": [], "id": [], "from": [], "via": [], "to": [], "from_type": [],
                         "via_type": [], "to_type": [], "from_geom": [], "via_geom": [], "to_geom": [],
                         "restriction": []}
    for d in restrictions:
        from_list = [(x["ref"], x["type"], get_geometry(x)) for x in d["members"] if x["role"] == "from"]
        via_list = [(x["ref"], x["type"], get_geometry(x)) for x in d["members"] if x["role"] == "via"]
        to_list = [(x["ref"], x["type"], get_geometry(x)) for x in d["members"] if x["role"] == "to"]
        if len(from_list) == 0:
            from_list = [(None, None, None)]
        if len(via_list) == 0:
            via_list = [(None, None, None)]
        if len(to_list) == 0:
            to_list = [(None, None, None)]
        total_len = len(from_list) * len(via_list) * len(to_list)
        for f in from_list:
            for v in via_list:
                for t in to_list:
                    restrictions_dict["from"].append(f[0])
                    restrictions_dict["via"].append(v[0])
                    restrictions_dict["to"].append(t[0])
                    restrictions_dict["from_type"].append(f[1])
                    restrictions_dict["via_type"].append(v[1])
                    restrictions_dict["to_type"].append(t[1])
                    restrictions_dict["from_geom"].append(f[2])
                    restrictions_dict["via_geom"].append(v[2])
                    restrictions_dict["to_geom"].append(t[2])
        restrictions_dict["type"].extend([d["type"]] * total_len)
        restrictions_dict["id"].extend([d["id"]] * total_len)
        restrictions_dict["restriction"].extend([d["tags"]["restriction"]] * total_len)
    restriction_df = pd.DataFrame(restrictions_dict)
    restriction_df = gpd.GeoDataFrame(restriction_df, geometry="via_geom")
    restriction_df = restriction_df[restriction_df["via_geom"].notnull()]
    if boundary:
        restriction_df = restriction_df[restriction_df["via_geom"].apply(lambda x: x.within(boundary))]
    restriction_coords = [[x, y] for x, y in
                          restriction_df["via_geom"].set_crs({"init": "epsg:4326"}).to_crs({"init": "epsg:3395"}).apply(
                              lambda x: [x.coords[0][0], x.coords[0][1]])]
    intersection_coords = [[x, y] for x, y in df_group["centroids"].apply(lambda x: [x.coords[0][0], x.coords[0][1]])]
    dists = scipy.spatial.distance.cdist(restriction_coords, intersection_coords)
    restriction_df.loc[dists.min(axis=1) < 20, "intersection"] = dists[dists.min(axis=1) < 20].argmin(axis=1)
    restriction_df = restriction_df.dropna(subset=["intersection"]).reset_index().drop("index", axis=1)
    restriction_df["intersection"] = restriction_df["intersection"].astype(int)
    restriction_df.loc[restriction_df["via_type"] == "way", "via_geom"] = restriction_df.loc[
        restriction_df["via_type"] == "way", "via_geom"].apply(lambda x: x.centroid)
    return restriction_df


def simplify_boundary(geometry):
    if type(geometry) == shapely.geometry.multipolygon.MultiPolygon:
        lon_list, lat_list = zip(*[x for geo in geometry for x in geo.exterior.coords])
    else:
        lon_list, lat_list = zip(*geometry.exterior.coords)
    top, bottom = max(lat_list), min(lat_list)
    left, right = min(lon_list), max(lon_list)
    boundary = Polygon([[left, top], [left, bottom], [right, bottom], [right, top]])
    return boundary


def get_dist(node1, node2, node_dict_m):
    pt1 = node_dict_m[node1]
    pt2 = node_dict_m[node2]
    return math.sqrt(pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2))


def find_centroid(group, signals):
    group = list(group)
    centroid = signals.iloc[group, [4, 5]].mean()
    x, y = centroid
    return Point([x, y])


def find_adjacent(group, roads, signal_dict):
    return [ID for (ID, nodes) in zip(roads["id"], roads["nodes"]) for signal in list(group) if
            signal_dict[signal] in nodes]


def countPairs(theta_dict, Gi, f=lambda x: x < 20):
    a, b = zip(*theta_dict.items())
    n = len(a)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if f(abs(a[j] - a[i])):
                pairs.append((a[j], a[i]))
    return pairs


def dist(p1, p2):
    return math.sqrt(pow(p1[0] - p2[0], 2) + pow((p1[1] - p2[1]), 2))


def get_lane_data(oneway, lanes, lanes_forward, lanes_both_ways, lanes_backward):
    if oneway == "yes":
        lane_data = int(lanes) if lanes == lanes else None
    else:
        try:
            if lanes == lanes:
                lanes = int(lanes)
                if lanes % 2 == 0:
                    if lanes_backward == lanes_backward:
                        lane_data = [lanes - int(lanes_backward), 0, int(lanes_backward)]
                    elif lanes_forward == lanes_forward:
                        lane_data = [int(lanes_forward), 0, lanes - int(lanes_forward)]
                    else:
                        lane_data = [lanes // 2, 0, lanes // 2]
                else:
                    if lanes_both_ways == lanes_both_ways:
                        lane_data = [lanes // 2, 1, lanes // 2]
                    else:
                        if lanes_forward == lanes_forward and int(lanes_forward) > lanes / 2:
                            lane_data = [int(lanes_forward), 0, lanes - int(lanes_forward)]
                        elif lanes_backward == lanes_backward and int(lanes_backward) > lanes / 2:
                            lane_data = [lanes - int(lanes_backward), 0, int(lanes_backward)]
                        else:
                            lane_data = [lanes // 2, 1, lanes // 2]
                lane_data = [int(x) for x in lane_data]
            else:
                lane_data = None
        except:
            lane_data = None
    return lane_data


def get_default_turn(lane_data):
    if pd.isnull(lane_data):
        return None
    if type(lane_data) != list:
        if lane_data == 1:
            turns = ["through;right"]
        else:
            turns = ["left;through"] + ["through"] * (lane_data - 2) + ["through;right"]
    else:
        turns = []
        _, middle, right = lane_data
        turns = middle * ["left"] + ["through"] * (right - 1) + ["through;right"]
    return turns


def get_turn_data(lane_data, oneway, turn_lanes, turn_lanes_forward):
    if not lane_data:
        return None, None
    lane_data = lane_data if type(lane_data) != list else lane_data[0]
    turns = get_default_turn(lane_data)
    if oneway == "yes":
        turn_data = turn_lanes
    else:
        turn_data = turn_lanes_forward
    if pd.isnull(turn_data):
        return turns, "default"
    if type(lane_data) == list and lane_data[1] != 0:
        turn_data = "left|" * lane_data[1] + turn_data
    for index, turn in enumerate(turn_data.split("|")):
        turn = ";".join([x for x in turn.split(";") if x in turn_types])
        if index > len(turns) - 1:
            continue
        if len(turn) > 0:
            turns[index] = turn
    return turns, "osm"


def get_pairs(intersection, f):
    def get_diff(a, b):
        return abs(a - b) if abs(a - b) < 180 else 360 - abs(a - b)

    selected = set()
    pairs = []
    diff_dict = {}
    diff_long_dict = {}
    for i in range(len(intersection)):
        for j in range(i + 1, len(intersection)):
            a, b = intersection[i]["theta"], intersection[j]["theta"]
            diff = get_diff(a, b)
            if f(diff):
                diff_dict[(i, j)] = diff
            a, b = intersection[i]["theta_long"], intersection[j]["theta_long"]
            diff = get_diff(a, b)
            if f(diff):
                diff_long_dict[(i, j)] = diff
    diff_list = sorted(diff_dict.items(), key=lambda x: x[1])
    for pair, diff in diff_list:
        if pair[0] not in selected and pair[1] not in selected:
            pairs.append(pair)
            selected.update(pair)
    diff_long_list = sorted(diff_long_dict.items(), key=lambda x: x[1])
    for pair, diff in diff_long_list:
        if pair[0] not in selected and pair[1] not in selected:
            pairs.append(pair)
            selected.update(pair)
    return pairs


def combine_same_direction(road1, road2):
    def get_average(angle1, angle2):
        if angle1 < angle2:
            angle1, angle2 = angle2, angle1
        diff = angle1 - angle2
        if diff > 180:
            avg = (angle1 + angle2) / 2 + 180
        else:
            avg = (angle1 + angle2) / 2
        while avg > 360:
            avg -= 360
        return avg

    if road1["direction"] == "out":
        road1, road2 = road2, road1
    new_road = {"road_id": [road1["road_id"], road2["road_id"]],
                "length": (road1["length"] + road2["length"]) / 2,
                "theta": get_average(road1["theta"], road2["theta"]),
                "theta_long": get_average(road1["theta_long"], road2["theta_long"]),
                "oneway": "no",
                "direction": "in",
                "lane_data": [road1["lane_data"], 0, road2["lane_data"]],
                "lane_data_source": road1["lane_data_source"] + "/" + road2["lane_data_source"],
                "turn_data": road1["turn_data"],
                "turn_data_source": road1["turn_data_source"],
                "highway": road1["highway"],
                "geometry": [road1["geometry"], road2["geometry"]]}
    return new_road


def find_theta(G, signal_node, other_node, node_dict_m):
    for u, v, a in G.edges(data=True):
        nodes = a["nodes"]
        if signal_node in nodes and other_node in nodes:
            other_node = u if signal_node == v else v
            x1, y1 = node_dict_m[other_node]
            x2, y2 = node_dict_m[signal_node]
            r = math.sqrt(pow(y1 - y2, 2) + pow(x1 - x2, 2))
            theta = math.acos((x1 - x2) / r) * 180 / math.pi
            if y1 < y2:
                theta = 360 - theta
            break
    return theta


def find_length_geometry(G, u, v):
    for _, _, a in G.edges(data=True):
        nodes = a["nodes"]
        if u in nodes and v in nodes:
            length = a["length"]
            geometry = list(a["geometry"].coords)
            break
    return length, geometry


def lane_complete(lane):
    return type(lane) == int or (type(lane) == list and all([bool(l) or l == 0 for l in lane]))


def impute_lane_data(lane1, lane2, oneway, direction1, direction2):
    if oneway == "yes":
        if type(lane1) == int:
            lane2 = lane1
        else:
            if direction2 == "in":
                lane2 = lane1[2]
            else:
                lane2 = lane1[0]
    else:
        if not lane2:
            lane2 = [None, None, None]
        if type(lane1) == int:
            if direction1 == "in" and not lane2[2]:
                lane2[2] = lane1
            elif direction1 == "out" and not lane2[0]:
                lane2[0] = lane1
        else:
            if lane1[2] and not lane2[0]:
                lane2[0] = lane1[2]
            if lane1[1] and not lane2[1]:
                lane2[1] = lane1[1]
            if lane1[0] and not lane2[2]:
                lane2[2] = lane1[0]
    return lane1, lane2


def process_turns(turns, restriction):
    if restriction == "no_left_turn":
        turns = ";".join([turn for turn in turns.split(";") if turn not in ["left", "slight_left"]])
    elif restriction == "only_straight_on":
        turns = "through"
    elif restriction == "only_right_turn":
        turns = "right"
    elif restriction == "no_straight_on":
        turns = ";".join([turn for turn in turns.split(";") if turn != "through"])
    elif restriction == "no_right_turn":
        turns = turns = ";".join([turn for turn in turns.split(";") if turn != "right"])
    else:
        turns = turns
    if not turns:
        turns = "through"
    return turns
