import os
import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import sqlalchemy
import osmnx as ox
from shapely.geometry import box, LineString
from shapely.ops import substring

# Create the Flask app
def get_db_connection() -> sqlalchemy.engine.base.Engine:
    """
    Initializes a database connection using the connection string
    from the environment variables.
    """
    # This reads the DATABASE_URL you will set in the Cloud Function's configuration.
    db_uri = os.environ.get("DATABASE_URL")
    engine = sqlalchemy.create_engine(db_uri)
    return engine

def save_to_db(df_to_save):
    """
    Connects to the database and appends the given DataFrame to the
    'traffic_data' table.
    """
    print(f"Attempting to save {len(df_to_save)} rows to the database...")

    try:
        db = get_db_connection()
        with db.connect() as conn:
            # Append the DataFrame to the 'traffic_data' table
            df_to_save.to_sql(
                name='traffic_data',
                con=conn,
                if_exists='append',
                index=False
            )
        print("Successfully saved data to Neon database.")
    except Exception as e:
        print(f"Failed to save data to database: {e}")

def get_distance_matrix_data(origin, destination):
    params = {
        "origins": f"{origin[0]},{origin[1]}",
        "destinations": f"{destination[0]},{destination[1]}",
        "key": API_KEY,
        "departure_time": "now",           # for live traffic data
        "traffic_model": "best_guess"
    }
    r = requests.get(url, params=params)
    data = r.json()

    try:
        elem = data["rows"][0]["elements"][0]
        if elem["status"] == "OK":
            distance = elem["distance"]["value"]       # meters
            duration = elem["duration"]["value"]       # seconds
            duration_traffic = elem.get("duration_in_traffic", {}).get("value", duration)
            speed = distance / duration_traffic if duration_traffic > 0 else None  # m/s
            return distance, duration, duration_traffic, speed * 3.6   # convert to km/h
    except Exception as e:
        print("Error:", e)
    return None, None, None, None

def get_distance_matrix_batch(origins_list, destinations_list):
    """
    Query Distance Matrix API with multiple origins and destinations in one request.
    
    Args:
        origins_list: List of tuples [(lat1, lon1), (lat2, lon2), ...]
        destinations_list: List of tuples [(lat1, lon1), (lat2, lon2), ...]
    
    Returns:
        List of results in the same order as input pairs
    """
    # Format origins and destinations for API
    origins_str = "|".join([f"{lat},{lon}" for lat, lon in origins_list])
    destinations_str = "|".join([f"{lat},{lon}" for lat, lon in destinations_list])
    API_KEY = os.environ.get('MAPS_API_KEY')
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": origins_str,
        "destinations": destinations_str,
        "key": API_KEY,
        "departure_time": "now",
        "traffic_model": "best_guess"
    }
    
    try:
        r = requests.get(url, params=params)
        data = r.json()
        
        if data.get("status") != "OK":
            print(f"API Error: {data.get('status')}")
            return None
            
        return data
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def parse_batch_response(data, num_pairs):
    """
    Extract diagonal elements from the distance matrix response.
    Since we're batching segments where each has its own origin-destination pair,
    we only need the diagonal of the matrix (origin[i] -> destination[i]).
    
    Args:
        data: JSON response from Distance Matrix API
        num_pairs: Number of origin-destination pairs in this batch
    
    Returns:
        List of tuples: (distance, duration, duration_traffic, speed_kmph)
    """
    results = []
    
    try:
        rows = data.get("rows", [])
        
        for i in range(num_pairs):
            if i < len(rows) and i < len(rows[i]["elements"]):
                elem = rows[i]["elements"][i]  # Get diagonal element
                
                if elem["status"] == "OK":
                    distance = elem["distance"]["value"]  # meters
                    duration = elem["duration"]["value"]  # seconds
                    duration_traffic = elem.get("duration_in_traffic", {}).get("value", duration)
                    speed = (distance / duration_traffic * 3.6) if duration_traffic > 0 else None  # km/h
                    
                    results.append((distance, duration, duration_traffic, speed))
                else:
                    print(f"Element {i} status: {elem['status']}")
                    results.append((None, None, None, None))
            else:
                results.append((None, None, None, None))
    except Exception as e:
        print(f"Error parsing response: {e}")
        results = [(None, None, None, None)] * num_pairs
    
    return results


def update_edges_with_traffic(edges_df, batch_size=10):
    """
    Update the edges dataframe with traffic data using batched API requests.
    
    Args:
        edges_df: DataFrame with 'start_point' and 'end_point' columns as (lat, lon) tuples
        batch_size: Number of segments to query per API call (max 25)
    
    Returns:
        Updated DataFrame with traffic information
    """
    total_segments = len(edges_df)
    num_batches = int(np.ceil(total_segments / batch_size))
    
    print(f"Processing {total_segments} segments in {num_batches} batches...")
    
    all_results = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_segments)
        batch_df = edges_df.iloc[start_idx:end_idx]
        
        # Extract origins and destinations for this batch
        # Since start_point and end_point are already (lat, lon) tuples, just convert to list
        origins = batch_df['start_point'].tolist()
        destinations = batch_df['end_point'].tolist()
        
        print(f"Batch {batch_idx + 1}/{num_batches}: Querying {len(origins)} segments...")
        
        # Make API request
        data = get_distance_matrix_batch(origins, destinations)
        
        if data is None:
            # If request failed, append None values
            all_results.extend([(None, None, None, None)] * len(origins))
        else:
            # Parse the response
            batch_results = parse_batch_response(data, len(origins))
            all_results.extend(batch_results)
        
        # Rate limiting: sleep between requests to avoid hitting rate limits
        if batch_idx < num_batches - 1:
            time.sleep(0.1)  # Small delay between batches
    
    # Update the dataframe
    edges_df['distance_m'] = [r[0] for r in all_results]
    edges_df['duration_s'] = [r[1] for r in all_results]
    edges_df['duration_traffic_s'] = [r[2] for r in all_results]
    edges_df['speed_kmph'] = [r[3] for r in all_results]
    
    # Calculate additional metrics
    edges_df['delay_s'] = edges_df['duration_traffic_s'] - edges_df['duration_s']
    edges_df['congestion_ratio'] = edges_df['duration_traffic_s'] / edges_df['duration_s']
    
    return edges_df

def collect_and_save_data():

    #lat long
    north, south = 10.7807, 10.7733
    east, west = 76.6469, 76.6387

    #download road network graph in your bounding box, had to AI this


    #This is: (min_lon, min_lat, max_lon, max_lat)
    bbox_poly = box(76.6387, 10.7733, 76.6469, 10.7807)


    #we enter the bounding box and it returns all road segments within it (partially or fully) which are
    #driveable hence the type='drive'
    #each edge corresponds to road segment in OSM
    G = ox.graph_from_polygon(bbox_poly, network_type="drive")
    # this is a graph

    #print(type(G))

    #convert edges of the graph to a GeoDataFrame with road geometries (linestrings)
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

    #type(edges['geometry'])
    # --- Google Distance Matrix API setup ---
    API_KEY = os.environ.get('MAPS_API_KEY')
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"

    # --- Extract start and end coordinates for each edge ---
    edges["start_point"] = edges.geometry.apply(lambda geom: (geom.coords[0][1], geom.coords[0][0]))  # (lat, lon)
    edges["end_point"]   = edges.geometry.apply(lambda geom: (geom.coords[-1][1], geom.coords[-1][0]))  # (lat, lon)

    # Work on a copy
    edges_fine = []

    # Define the target segment length (in meters)
    target_length = 50

    # Ensure geometry is projected to meters (important!)
    edges_projected = edges.to_crs(epsg=3857)  # Web Mercator projection (units ~ meters)

    for idx, row in edges_projected.iterrows():
        geom = row.geometry
        length = geom.length

        if length <= target_length:
            # Keep as is
            edges_fine.append(row)
        else:
            # Split into smaller segments of length <= target_length
            n_parts = int(np.ceil(length / target_length))
            part_length = length / n_parts

            for i in range(n_parts):
                start_d = i * part_length
                end_d = min((i + 1) * part_length, length)
                new_geom = substring(geom, start_d, end_d)
                new_row = row.copy()
                new_row.geometry = new_geom
                edges_fine.append(new_row)

    # Build new GeoDataFrame
    edges_fine = gpd.GeoDataFrame(edges_fine, crs=edges_projected.crs)

    # Convert back to lat/lon for Google API calls
    edges_fine = edges_fine.to_crs(epsg=4326)


    edges_fine["start_point"] = edges_fine.geometry.apply(lambda g: (g.coords[0][1], g.coords[0][0]))  # (lat, lon)
    edges_fine["end_point"]   = edges_fine.geometry.apply(lambda g: (g.coords[-1][1], g.coords[-1][0]))  # (lat, lon)

    # Usage:
    edges_fine = update_edges_with_traffic(edges_fine, batch_size=10)

    edges_fine['start_point'] = edges_fine['start_point'].astype(str)
    edges_fine['end_point'] = edges_fine['end_point'].astype(str)


    save_to_db(edges_fine)
    
if __name__ == "__main__":
    collect_and_save_data()
