import pandas as pd
import numpy as np
import re
import os

def parse_igra_line(line):
    try:
        # Helper function to clean and convert values, treating -9999 as NaN
        def clean(val):
            cleaned = re.sub(r'[^0-9\-.]', '', val)
            return float(cleaned) if cleaned and cleaned != '-9999' else np.nan

        return {
            "lvl_type": int(line[0:2]),
            "press": clean(line[9:15]),
            "gph":   clean(line[16:21]),
            "temp":  clean(line[22:27]),
            "rh":    clean(line[28:33]),
            "wdir":  clean(line[40:45]),
            "wspd":  clean(line[46:51])
        }
    except:
        return None

# Main processing function
def process_all_stations(data_dir):
    all_data_frames = []
    files = [f for f in os.listdir(data_dir) if f.endswith('-data.txt')]
    
    # Extract WMO ID and parse each file
    for filename in files:
        print(f"Processing {filename}...")
        match = re.search(r'IDM000(\d{5})', filename)
        wmo_id = match.group(1) if match else "Unknown"
        
        station_flights = []
        current_meta = {}
        
        # Read file and parse lines
        with open(os.path.join(data_dir, filename), 'r') as f:
            for line in f:
                if line.startswith('#'):
                    # Parsing based on IGRA format assumptions
                    raw_lat = line[55:62].strip()
                    raw_lon = line[63:72].strip()
                    
                    current_meta = {
                        "wmo_id":   wmo_id,
                        "year":     int(line[13:17]),
                        "month":    int(line[18:20]),
                        "day":      int(line[21:23]),
                        "hour":     int(line[24:26]),
                        "source":   line[31:51].strip(), 
                        "lat":      float(raw_lat) / 10000.0 if raw_lat != "-9999" else np.nan,
                        "lon":      float(raw_lon) / 10000.0 if raw_lon != "-9999" else np.nan,
                        "flight_id": line[13:26].replace(" ", "") 
                    }
                else:
                    data = parse_igra_line(line)
                    if data:
                        data.update(current_meta)
                        station_flights.append(data)

            print(f"  Parsed {len(station_flights)} flights for station {wmo_id}")
        
        if not station_flights: continue

        temp_df = pd.DataFrame(station_flights)
        
        # Unit conversion
        temp_df['temp'] = temp_df['temp'] / 10.0
        temp_df['rh']   = temp_df['rh'] / 10.0
        temp_df['wspd'] = temp_df['wspd'] / 10.0
        
        
        all_data_frames.append(temp_df)

    master_df = pd.concat(all_data_frames, ignore_index=True)
    print(f"Total records processed: {len(master_df)}")
    return master_df

# Run and Save
data_path = './data/' 
master_df = process_all_stations(data_path)
master_df.to_csv('data.csv', index=False)