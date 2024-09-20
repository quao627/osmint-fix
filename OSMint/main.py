import pickle
import re
import os
import itertools
import networkx as nx
from shapely.ops import cascaded_union
import osmnx.projection as projection
from OSMint.functions import *


def get_data(city, state, out_path):
    filename = f'{out_path}/{"-".join(city.lower().split(" "))}.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
    boundary = ox.geocode_to_gdf(f'{city}, {state}, USA')
    boundary = list(boundary["geometry"])[0]
    if type(boundary) != shapely.geometry.polygon.Polygon:
        boundary = cascaded_union(list(boundary))
    polygon = simplify_boundary(boundary)

    # buffer the polygon for 1000m to get complete road network
    print("Collecting roads...")
    buffer_dist = 1000
    poly_proj, crs_utm = projection.project_geometry(boundary)
    poly_proj_buff = poly_proj.buffer(buffer_dist)
    poly_buff, _ = projection.project_geometry(poly_proj_buff, crs=crs_utm, to_latlong=True)
    network_type = "drive"
    custom_filter = None
    roads = ox.downloader._osm_network_download(poly_buff, network_type, custom_filter)

    nodes = [road for road in roads[0]["elements"] if road["type"] == "node"]
    roads = [road for road in roads[0]["elements"] if road["type"] == "way"]

    node_dict = {node['id']: [node['lon'], node['lat']] for node in nodes}
    node_dict_m = gpd.GeoDataFrame(index=node_dict.keys(), geometry=[Point(p) for p in node_dict.values()]).set_crs(
        {'init': 'epsg:4326'}).to_crs({'init': 'epsg:3395'})
    node_dict_m = {index: [row["geometry"].xy[0][0], row["geometry"].xy[1][0]] for index, row in node_dict_m.iterrows()}
    geometry = [LineString([node_dict[node] for node in road["nodes"]]) for road in roads]
    for road in roads:
        road.update(road.pop("tags"))
    useful_tags = ['name', 'type', 'id', 'nodes', 'highway', 'lanes', 'lanes:backward',
                   'lanes:both_ways', 'lanes:forward',
                   'oneway', 'turn:lanes', 'turn:lanes:backward', 'turn:lanes:both_ways',
                   'turn:lanes:forward', 'width', 'maxspeed', 'incline']
    roads = pd.DataFrame(roads)
    roads = roads[[tag for tag in roads.columns if tag in useful_tags]]
    roads = gpd.GeoDataFrame(roads, geometry=geometry)

    roads = roads.set_crs({'init': 'epsg:4326'})
    roads = roads.to_crs({'init': 'epsg:3395'})

    # build a graph containing all the nodes and their connections
    print("Constructing graph...")
    G = nx.MultiGraph()
    nodes = set()
    paths = set()
    # edge_road = defaultdict(set)
    for index, row in roads.iterrows():
        path_nodes = [group[0] for group in itertools.groupby(row["nodes"])]
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(
            [(path_nodes[i], path_nodes[i + 1], get_dist(path_nodes[i], path_nodes[i + 1], node_dict_m)) for i in
             range(len(path_nodes) - 1)], weight="length")

    for u, v, k in G.edges(keys=True):
        G[u][v][k]["nodes"] = set([u, v])
        G[u][v][k]["geometry"] = LineString([node_dict[u], node_dict[v]])

    # remove all nodes with degree equal to 2
    while any([degree == 2 and len(G[node]) > 1 for node, degree in G.degree()]):
        bi_nodes = [node for node, degree in G.degree() if degree == 2]
        for index, node in enumerate(bi_nodes):
            if node in G.nodes():
                neighbors = list(dict(G[node]).keys())
                if len(neighbors) == 2:
                    if neighbors[0] == neighbors[1]:
                        continue
                    l1, l2 = sum([G[node][neighbors[0]][k]["length"] for k in dict(G[node][neighbors[0]]).keys()]), sum(
                        [G[node][neighbors[1]][k]["length"] for k in dict(G[node][neighbors[1]]).keys()])
                    combined_nodes = [G[node][neighbors[0]][k]["nodes"] for k in dict(G[node][neighbors[0]]).keys()] + [
                        G[node][neighbors[1]][k]["nodes"] for k in dict(G[node][neighbors[1]]).keys()]
                    combined_nodes = [node for nodes in combined_nodes for node in nodes]
                    k = G.add_edge(*neighbors)
                    G[neighbors[0]][neighbors[1]][k]["length"] = l1 + l2
                    G[neighbors[0]][neighbors[1]][k]["nodes"] = set(combined_nodes)
                    G[neighbors[0]][neighbors[1]][k]["geometry"] = shapely.ops.linemerge(
                        [G[node][neighbors[0]][k]["geometry"] for k in dict(G[node][neighbors[0]]).keys()] + [
                            G[node][neighbors[1]][k]["geometry"] for k in dict(G[node][neighbors[1]]).keys()])
                    G.remove_node(node)
    G.remove_edges_from(nx.selfloop_edges(G))

    # get traffic signals
    print("Collecting traffic signals...")
    flag = False
    while not flag:
        try:
            signals = get_traffic_signals(polygon=polygon, boundary=boundary)
            flag = True
        except:
            print("Error with Overpass API. Retrying...")
    signals['keep'] = signals["id"].isin(node_dict)

    node_df = gpd.GeoDataFrame(geometry=[Point(node_dict[node]) for node in G.nodes()], index=G.nodes())
    node_df = node_df.set_crs({'init': 'epsg:4326'})
    node_df = node_df.to_crs({'init': 'epsg:3395'})
    node_df['x'] = node_df['geometry'].apply(lambda x: x.xy[0][0])
    node_df['y'] = node_df['geometry'].apply(lambda x: x.xy[1][0])
    node_df['id'] = node_df.index

    signals = signals[signals["keep"]]
    signals = signals.reset_index().drop("index", axis=1)

    groups = find_clusters(signals, threshold=80)
    signal_dict = dict(zip(signals.index, signals["id"]))
    signal_centroids = [find_centroid(group, signals) for group in groups]
    df_group = pd.DataFrame({"group": groups, "centroids": signal_centroids})
    df_group = gpd.GeoDataFrame(df_group, geometry="centroids").set_crs({"init": "epsg:3395"})
    df_group["geometry"] = df_group["centroids"].to_crs({"init": "epsg:4326"})
    df_group["nodes"] = df_group["group"].apply(lambda x: [signal_dict[index] for index in list(x)])
    G_nodes = node_df[node_df["id"].apply(lambda x: x in G.nodes)]
    nearby_roads = []
    nearby_nodes = []
    for index, node_list in enumerate(df_group["nodes"]):
        node_dist = [node_dict_m[node] for node in node_list]
        dists = scipy.spatial.distance.cdist(G_nodes[["x", "y"]], node_dist)
        adjacent_nodes = list(G_nodes.index[dists.min(axis=1) < 30].astype(int))

        nearby_nodes.append(adjacent_nodes)
        nearby_roads.append(
            [ID for (ID, nodes) in zip(roads["id"], roads["nodes"]) for node in adjacent_nodes if node in nodes])
    df_group["nearby_nodes"] = nearby_nodes
    df_group["nearby_roads"] = nearby_roads

    # get restrictions
    print("Collecting turn restrictions...")
    flag = False
    while not flag:
        try:
            restrictions = get_turn_restrictions(polygon=polygon, boundary=boundary, df_group=df_group)
            flag = True
        except:
            print("Error with Overpass API. Retrying...")
    restrictions = restrictions.dropna().reset_index().drop("index", axis=1)

    data = {"roads": roads,
            "G_nodes": G_nodes,
            "node_dict": node_dict,
            "node_dict_m": node_dict_m,
            "G": G,
            "signals": signals,
            "df_group": df_group,
            "restrictions": restrictions}
    if out_path:
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return data


def get_intersection(roads, G_nodes, node_dict_m, G, df_group, restrictions):
    def sample_lane_data(highway, one_way):
        return lanes_dict_one_way[highway] if one_way == "yes" else lanes_dict_two_way[highway]

    def complete_lane(lane, highway, oneway):
        if oneway == "yes":
            lane = sample_lane_data(highway, oneway)
        elif not lane:
            lane = sample_lane_data(highway, "no")
        else:
            if not lane[0]:
                lane[0] = sample_lane_data(highway, "yes")
            if not lane[2]:
                lane[2] = sample_lane_data(highway, "yes")
        return lane

    if "lanes:both_ways" not in roads.columns:
        roads["lanes:both_ways"] = None
    if "turn:lanes:both_ways" not in roads.columns:
        roads["turn:lanes:both_ways"] = None
    roads.loc[(roads["turn:lanes:backward"].notnull()) & (roads["lanes:backward"].isnull()), "lanes:backward"] = \
        roads.loc[
            (roads["turn:lanes:backward"].notnull()) & (roads["lanes:backward"].isnull()), "turn:lanes:backward"].apply(
            lambda x: len(x.split("|")))
    roads.loc[(roads["turn:lanes:forward"].notnull()) & (roads["lanes:forward"].isnull()), "lanes:forward"] = roads.loc[
        (roads["turn:lanes:forward"].notnull()) & (roads["lanes:forward"].isnull()), "turn:lanes:forward"].apply(
        lambda x: len(x.split("|")))
    roads.loc[(roads["turn:lanes:both_ways"].notnull()) & (roads["lanes:both_ways"].isnull()), "lanes:both_ways"] = \
        roads.loc[
            (roads["turn:lanes:both_ways"].notnull()) & (
                roads["lanes:both_ways"].isnull()), "turn:lanes:both_ways"].apply(
            lambda x: len(x.split("|")))

    roads.loc[roads["oneway"] == "-1", "nodes"] = roads[roads["oneway"] == "-1"]["nodes"].apply(lambda x: x[::-1])
    roads.loc[roads["oneway"] == "-1", "oneway"] = "yes"
    roads.loc[roads["lanes"] == "1", "oneway"] = "yes"

    roads["lane_data"] = roads.apply(
        lambda row: get_lane_data(row["oneway"], row["lanes"], row["lanes:forward"], row["lanes:both_ways"],
                                  row["lanes:backward"]), axis=1)

    # process speed limit

    print("Processing speed limit data...")
    roads["speed"] = roads["maxspeed"].apply(lambda x: int(re.findall(r'\d+', x)[0]) if not type(x) == float else None)

    # add turn restrictions:

    print("Processing turn data...")
    roads["turn_data"], roads["turn_data_source"] = zip(
        *roads.apply(lambda x: get_turn_data(x["lane_data"], x["oneway"], x["turn:lanes"], x["turn:lanes:forward"]),
                     axis=1))

    lanes_dict_two_way = dict()
    for highway in roads["highway"].unique():
        try:
            lanes_dict_two_way[highway] = \
                roads[(roads["oneway"] != "yes") & (roads["lane_data"].notnull()) & (roads["highway"] == highway)][
                    "lane_data"].value_counts().keys()[0]
        except:
            lanes_dict_two_way[highway] = [1, 0, 1]
    lanes_dict_one_way = dict()
    for highway in roads["highway"].unique():
        try:
            lanes_dict_one_way[highway] = \
                roads[(roads["oneway"] == "yes") & (roads["lane_data"].notnull()) & (roads["highway"] == highway)][
                    "lane_data"].value_counts().keys()[0]
        except:
            lanes_dict_one_way[highway] = 1

    original_degree_list = []
    intersection_dict = {}
    for group_index in range(len(df_group)):
        try:
            index = group_index
            res_df = restrictions[restrictions["intersection"] == index]
            group = list(df_group["nearby_nodes"])[index]
            geometry = list(df_group["geometry"])[index].xy
            centroid_coords = [geometry[1][0], geometry[0][0]]
            nearby_roads = list(df_group["nearby_roads"])[index]
            nearby_roads = roads[roads["id"].isin(nearby_roads)]
            nodes = G_nodes.loc[list(group)]

            if len(nodes) == 0 or len(nearby_roads) < 2:
                continue
            # keep a copy of raw data
            original_nodes = nodes.copy(deep=True)
            original_nearby_roads = nearby_roads.copy(deep=True)

            # build a graph for the intersection
            Gi = nx.Graph()
            for index, row in nearby_roads.iterrows():
                path_nodes = [group[0] for group in itertools.groupby(row["nodes"])]
                Gi.add_edges_from([(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)], label=row["id"])
            Gi = Gi.subgraph(max(nx.connected_components(Gi), key=len))
            Gi = nx.Graph(Gi)

            # remove signals with less than two neighbors
            removed_signals = set()
            for index, row in nodes.iterrows():
                signal = row["id"]
                if signal not in Gi.nodes:
                    nodes.loc[index, "keep"] = False
                    continue
                if Gi.degree[signal] == 2:
                    neighbors = list(dict(Gi[signal]).keys())
                    Gi.add_edge(*neighbors,
                                label=Gi[signal][neighbors[0]]["label"] if neighbors[0] in list(nodes["id"]) else
                                Gi[signal][neighbors[1]]["label"])
                    Gi.remove_node(signal)
                    nodes.loc[index, "keep"] = False
                else:
                    nodes.loc[index, "keep"] = True

            Gj = Gi.copy()
            for u, v in Gj.edges():
                Gj[u][v]["nodes"] = set([u, v])
            nodes_j = list(Gj.nodes)
            for node in nodes_j:
                if Gj.degree[node] == 2 and any([get_dist(node, signal, node_dict_m) < 20 for signal in nodes["id"]]):
                    neighbors = list(dict(Gj[node]).keys())
                    Gj.add_edge(*neighbors)
                    Gj[neighbors[0]][neighbors[1]]["nodes"] = Gj[node][neighbors[0]]["nodes"] | Gj[node][neighbors[1]][
                        "nodes"]
                    Gj.remove_node(node)

            nodes = nodes[nodes["keep"]]
            intersection_nodes = set(nodes["id"])
            intersection_nodes = list(intersection_nodes)
            if len(intersection_nodes) > 1:
                new_nodes = set()
                for u, v in Gj.edges():
                    if u in intersection_nodes and v in intersection_nodes:
                        new_nodes.update(Gj[u][v]["nodes"])
                for i, node1 in enumerate(intersection_nodes):
                    for node2 in intersection_nodes[i + 1:]:
                        new_nodes.update(nx.shortest_path(Gi, node1, node2))
                intersection_nodes = list(new_nodes)

            # remove those signals from roads and then cut the roads to its nearest intersection/road end
            for index, row in nearby_roads.iterrows():
                new_nodes = [node for node in row["nodes"] if node not in removed_signals]
                nearby_roads.at[index, "nodes"] = new_nodes

            theta_dict = {}
            theta_long_dict = {}
            for (u, v, a) in Gi.edges(data=True):
                if u not in intersection_nodes and v not in intersection_nodes:
                    continue
                if u in intersection_nodes and v in intersection_nodes:
                    continue
                other_node, signal_node = (u, v) if v in intersection_nodes else (v, u)
                theta_long = find_theta(G, signal_node, other_node, node_dict_m)
                road_nodes = list(nearby_roads.loc[nearby_roads["id"] == a["label"], "nodes"])[0]
                x1, y1 = node_dict_m[other_node]
                x2, y2 = node_dict_m[signal_node]
                r = math.sqrt(pow(y1 - y2, 2) + pow(x1 - x2, 2))
                theta = math.acos((x1 - x2) / r) * 180 / math.pi
                if y1 < y2:
                    theta = 360 - theta
                if road_nodes.index(signal_node) > road_nodes.index(other_node):
                    direction = "in"
                else:
                    direction = "out"
                Gi[u][v]["theta"] = theta
                Gi[u][v]["theta_long"] = theta_long
                Gi[u][v]["direction"] = direction
                theta_dict[theta] = a["label"]
                theta_long_dict[theta_long] = a["label"]

            # get length and geometry
            for (u, v) in Gi.edges:
                Gi[u][v]["length"], Gi[u][v]["geometry"] = find_length_geometry(G, u, v)

            edges = [(u, v, a) for (u, v, a) in Gi.edges(data=True) if
                    (u in intersection_nodes) + (v in intersection_nodes) == 1]

            intersection = dict()
            lines = []

            original_degree_list.append(len(edges))
            # compute angles and length
            index = 0
            for u, v, a in edges:
                if u == v:
                    continue
                label = a["label"]
                road_info = nearby_roads.loc[nearby_roads["id"] == label]
                road_info = road_info.iloc[0].to_dict()
                intersection[index] = dict()
                intersection[index]["length"] = a["length"]
                intersection[index]["geometry"] = [[coord[1], coord[0]] for coord in a["geometry"]]
                intersection[index]["road_id"] = label
                intersection[index]["theta"] = a["theta"]
                intersection[index]["theta_long"] = a["theta_long"]
                intersection[index]["direction"] = a["direction"]
                intersection[index]["lane_data"] = road_info["lane_data"]
                intersection[index]["lane_data_source"] = "osm"
                intersection[index]["turn_data"] = road_info["turn_data"]
                intersection[index]["turn_data_source"] = road_info["turn_data_source"]
                intersection[index]["oneway"] = road_info["oneway"]
                intersection[index]["highway"] = road_info["highway"]
                lines.append(intersection[index]["geometry"])
                index += 1

            # roads with opposite directions are oneway
            same_pairs = get_pairs(intersection, lambda x: x < 20)
            new_intersection = {}
            new_same_pairs = set()
            for pair in same_pairs:
                road1 = intersection[pair[0]]
                road2 = intersection[pair[1]]
                if type(road1["lane_data"]) == list or type(road2["lane_data"]) == list:
                    continue
                new_road = combine_same_direction(road1, road2)
                new_intersection[len(new_intersection)] = new_road
                new_same_pairs.add(pair)
            same_pairs = new_same_pairs
            for index, road in intersection.items():
                if all([index not in pair for pair in same_pairs]):
                    new_intersection[len(new_intersection)] = road
            intersection = new_intersection

            # ensure the directions are all "in"
            for index, road in intersection.items():
                if type(road["road_id"]) == list:
                    road_info = nearby_roads.loc[nearby_roads["id"] == road["road_id"][0]]
                else:
                    road_info = nearby_roads.loc[nearby_roads["id"] == road["road_id"]]
                road_info = road_info.iloc[0].to_dict()
                if road["direction"] == "out":
                    if road["oneway"] == "yes":
                        road["turn_data"] = []
                        continue
                    if road["lane_data"]:
                        if type(road["lane_data"]) == list:
                            road["lane_data"] = road["lane_data"][::-1]
                        road["turn_data"], road["turn_data_source"] = get_turn_data(road["lane_data"], "no",
                                                                                    road_info["turn:lanes"],
                                                                                    road_info["turn:lanes:backward"])
                    road["direction"] = "in"

            # impute roads with the same direction
            pairs = get_pairs(intersection, lambda x: 170 < x < 190)
            for pair in pairs:
                road1, road2 = intersection[pair[0]], intersection[pair[1]]
                lane1, lane2 = road1["lane_data"], road2["lane_data"]
                if lane_complete(lane1) and lane_complete(lane2):
                    continue
                if lane_complete(lane1) and not lane_complete(lane2):
                    lane1, lane2 = impute_lane_data(lane1, lane2, road2["oneway"], road1["direction"], road2["direction"])
                if lane_complete(lane2) and not lane_complete(lane1):
                    lane2, lane1 = impute_lane_data(lane2, lane1, road1["oneway"], road2["direction"], road1["direction"])
                if not lane_complete(lane1):
                    lane1 = complete_lane(lane1, road1["highway"], road1["oneway"])
                if not lane_complete(lane2):
                    lane2 = complete_lane(lane2, road2["highway"], road2["oneway"])
                road1["lane_data"] = lane1
                road2["lane_data"] = lane2

            for index, road in intersection.items():
                lane = road["lane_data"]
                if not lane_complete(lane):
                    lane = complete_lane(lane, road["highway"], road["oneway"])
                    road["lane_data"] = lane
            # impute edges that still don't have lane data
            for index, road in intersection.items():
                if not road["lane_data"]:
                    road["lane_data"] = sample_lane_data(road["highway"], road["oneway"])
                if type(road["lane_data"]) == list:
                    if not road["lane_data"][0]:
                        road["lane_data"][0] = sample_lane_data(road["highway"], "yes")
                    if not road["lane_data"][2]:
                        road["lane_data"][2] = sample_lane_data(road["highway"], "yes")

            for index, road in intersection.items():
                if not road["turn_data"]:
                    if road["direction"] != "out":
                        road["turn_data"], road["turn_data_source"] = get_turn_data(road["lane_data"], road["oneway"], None,
                                                                                    None)
                    else:
                        road["turn_data"], road["turn_data_source"] = [], "osm"

            # add turn restrictions
            for index, row in res_df.iterrows():
                res_type = row["restriction"]
                coords = row["from_geom"].coords
                x1, y1 = coords[0][0], coords[0][1]
                x2, y2 = coords[-1][0], coords[-1][1]
                r = math.sqrt(pow(y1 - y2, 2) + pow(x1 - x2, 2))
                theta = math.acos((x2 - x1) / r) * 180 / math.pi
                if y2 < y1:
                    theta = 360 - theta
                road_idx = sorted(intersection.items(), key=lambda x: abs(x[1]["theta"] - theta))[0][0]
                intersection[road_idx]["turn_data"] = [process_turns(turns, res_type) for turns in
                                                    intersection[road_idx]["turn_data"]]
            order = sorted(intersection, key=lambda x: intersection[x]["theta"], reverse=True)
            intersection = {i: intersection[road_idx] for i, road_idx in enumerate(order)}
            intersection_dict[group_index] = intersection
            intersection_dict[group_index]["geometry"] = lines
            intersection_dict[group_index]["centroid"] = centroid_coords

            print(group_index)
        except:
            print(f"Error with group {group_index}. Skipping...")
            continue
    return intersection_dict


if __name__ == "__main__":
    city = "Sacramento"
    state = "CA"
    out_path = "Example"
    data = get_data(city, state, out_path)
    intersection = get_intersection(data["roads"], data["G_nodes"], data["node_dict_m"], data["G"], data["df_group"],
                     data["restrictions"])
    