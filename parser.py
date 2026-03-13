import pandas as pd
import numpy as np
import re
import os

def parse_igra_line(line):
    try:
        def clean(val):
            # Remove non-numeric characters (flags like B, A, S)
            cleaned = re.sub(r'[^0-9\-.]', '', val)
            return float(cleaned) if cleaned and cleaned != '-9999' else np.nan

        return {
            "lvl_type": int(line[0:2]),
            "press": clean(line[9:15]),   # Pressure (Pa)
            "gph":   clean(line[16:21]),  # Altitude (m)
            "temp":  clean(line[22:27]),  # Temp (tenths of deg C)
            "rh":    clean(line[28:33]),  # Relative Humidity (tenths of %)
            "wdir":  clean(line[40:45]),  # Wind Direction (deg)
            "wspd":  clean(line[46:51])   # Wind Speed (tenths of m/s)
        }
    except:
        return None

def process_all_stations(data_dir):
    all_data_frames = []
    
    # List all files in the directory
    files = [f for f in os.listdir(data_dir) if f.endswith('-data.txt')]
    print(f"Found {len(files)} station files. Starting parse...")

    for filename in files:
        # Extract WMO ID from filename (e.g., IDM00096163-data.txt -> 96163)
        # Using regex to find the 5 digits after IDM000
        match = re.search(r'IDM000(\d{5})', filename)
        wmo_id = match.group(1) if match else "Unknown"
        
        print(f"Processing Station: {wmo_id}...")
        
        station_flights = []
        current_meta = {}
        
        with open(os.path.join(data_dir, filename), 'r') as f:
            for line in f:
                if line.startswith('#'):
                    current_meta = {
                        "wmo_id": wmo_id,
                        "year":   int(line[13:17]),
                        "month":  int(line[18:20]),
                        "day":    int(line[21:23]),
                        "hour":   int(line[24:26]),
                        "flight_id": line[13:26].replace(" ", "") # Unique ID for each launch
                    }
                else:
                    data = parse_igra_line(line)
                    if data:
                        data.update(current_meta)
                        station_flights.append(data)
        
        # Convert station list to temporary DF
        temp_df = pd.DataFrame(station_flights)
        
        # Unit conversion
        temp_df['temp'] = temp_df['temp'] / 10.0
        temp_df['rh']   = temp_df['rh'] / 10.0
        temp_df['wspd'] = temp_df['wspd'] / 10.0
        
        # Basic cleanup: Remove rows missing key training inputs
        temp_df = temp_df.dropna(subset=['press', 'gph', 'temp'])
        
        all_data_frames.append(temp_df)

    # Combine all stations into one master DataFrame
    master_df = pd.concat(all_data_frames, ignore_index=True)
    return master_df

# --- RUN THE PROCESSOR ---
data_path = './data/'  # Update this to your /data/ folder path
master_df = process_all_stations(data_path)

print(f"Final Dataset Shape: {master_df.shape}")
print(master_df['wmo_id'].value_counts()) # Shows row count per station

# Save to CSV
master_df.to_csv('data.csv', index=False)