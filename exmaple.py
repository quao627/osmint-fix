from OSMint import get_data, get_intersection

def main():
    city = "Chelsea"
    state = "MA"
    out_path = "Example"
    data = get_data(city, state, out_path)
    intersection = get_intersection(data["roads"], data["G_nodes"], data["node_dict_m"], data["G"], data["df_group"],
                     data["restrictions"])    
    return {"data":data, "intersection": intersection}

if __name__ == "__main__":
    main()
    