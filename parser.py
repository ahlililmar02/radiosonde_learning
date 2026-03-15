import pandas as pd
import numpy as np
import re
import os

def parse_igra_line(line):
    try:
        # Helper function to clean and convert values
        # Per spec: -9999 is missing, -8888 is removed by quality assurance
        def clean(val):
            cleaned = re.sub(r'[^0-9\-.]', '', val)
            if not cleaned or cleaned in ['-9999', '-8888']:
                return np.nan
            return float(cleaned)

        return {
            "lvl_type1": int(line[0:1]),    # Major level type
            "lvl_type2": int(line[1:2]),    # Minor level type
            "etime":     clean(line[3:8]),  # Elapsed time (MMMSSS)
            "press":     clean(line[9:15]), # Pressure (Pa)
            "gph":       clean(line[16:21]),# Geopotential height (m)
            "temp":      clean(line[22:27]),# Temp (tenths of deg C)
            "rh":        clean(line[28:33]),# Relative Humidity (tenths of %)
            "dpdp":      clean(line[34:39]),# Dewpoint Depression (tenths of deg C)
            "wdir":      clean(line[40:45]),# Wind Direction (deg)
            "wspd":      clean(line[46:51]) # Wind Speed (tenths of m/s)
        }
    except Exception:
        return None

def process_all_stations(data_dir):
    all_data_frames = []
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return pd.DataFrame()

    files = [f for f in os.listdir(data_dir) if f.endswith('-data.txt')]
    
    for filename in files:
        print(f"Processing {filename}...")
        # IGRA ID is 11 characters starting at index 0 (e.g., IDM00096015)
        station_id = filename.split('-')[0]
        
        station_flights = []
        current_meta = {}
        
        with open(os.path.join(data_dir, filename), 'r') as f:
            for line in f:
                if line.startswith('#'):
                    # Parsing per IGRA v2.2 spec
                    raw_lat = line[55:62].strip()
                    raw_lon = line[63:71].strip()
                    
                    current_meta = {
                        "station_id": station_id,
                        "year":       int(line[13:17]),
                        "month":      int(line[18:20]),
                        "day":        int(line[21:23]),
                        "hour":       int(line[24:26]),
                        "reltime":    line[27:31].strip(), # Release time HHMM
                        "numlev":     int(line[32:36]),
                        "p_src":      line[37:45].strip(), # Pressure source
                        "np_src":     line[46:54].strip(), # Non-pressure source
                        "lat":        float(raw_lat) / 10000.0 if raw_lat not in ["-9999", ""] else np.nan,
                        "lon":        float(raw_lon) / 10000.0 if raw_lon not in ["-9999", ""] else np.nan,
                    }
                else:
                    data = parse_igra_line(line)
                    if data:
                        data.update(current_meta)
                        station_flights.append(data)

        if station_flights:
            temp_df = pd.DataFrame(station_flights)
            
            # Unit conversions per spec (tenths to units)
            temp_df['temp'] = temp_df['temp'] / 10.0
            temp_df['rh']   = temp_df['rh'] / 10.0
            temp_df['dpdp'] = temp_df['dpdp'] / 10.0
            temp_df['wspd'] = temp_df['wspd'] / 10.0
            
            # Calculate actual pressure in hPa/mb for easier reading (Pa / 100)
            temp_df['press_hpa'] = temp_df['press'] / 100.0
            
            all_data_frames.append(temp_df)

    if not all_data_frames:
        return pd.DataFrame()

    master_df = pd.concat(all_data_frames, ignore_index=True)
    print(f"Total records processed: {len(master_df)}")
    return master_df

# --- EXECUTION ---
data_path = './data/' 
master_df = process_all_stations(data_path)
if not master_df.empty:
    master_df.to_csv('igra_v2_data.csv', index=False)
    print("File saved as igra_v2_data.csv")