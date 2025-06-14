#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#location extraction using latitude and longitude
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Initialize the geolocator
geolocator = Nominatim(user_agent="geo_locator")  # Replace with a descriptive user agent

def reverse_geocode(lat, lon, retries=3):
    """
    Reverse geocodes coordinates to an address.  Handles potential errors and retries.

    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        retries (int): Number of retries in case of failure.

    Returns:
        str: Address, or None if geocoding fails after multiple retries.
    """
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, timeout=10) # increased timeout
        if location:
            return location.address
        else:
            return None
    except GeocoderTimedOut:
        if retries > 0:
            print(f"Timeout error. Retrying... (Retries left: {retries})")
            return reverse_geocode(lat, lon, retries - 1)
        else:
            print("Max retries reached. Geocoding failed due to timeout.")
            return None
    except GeocoderServiceError as e:
        print(f"Geocoding service error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


# Input and output file paths
input_csv_file = "E:\\road_project_all_final\\road_project_output_darewadi\\detection_with_gps.csv"  # Replace with your input file name
output_csv_file = "E:\\road_project_all_final\\road_project_output_darewadi\\final.csv"  # Replace with your desired output file name

# Read the CSV file into a pandas DataFrame
try:
    df = pd.read_csv(input_csv_file)
except FileNotFoundError:
    print(f"Error: The file '{input_csv_file}' was not found.")
    exit()
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    exit()


# Create a new column for the address
df['Address'] = None  # Initialize the column

# Iterate through the DataFrame and perform reverse geocoding
for index, row in df.iterrows():
    latitude = row['Latitude']
    longitude = row['Longitude']

    # Check for missing or invalid latitude/longitude values
    if pd.isna(latitude) or pd.isna(longitude):
        print(f"Warning: Missing latitude or longitude at row {index}. Skipping.")
        df.loc[index, 'Address'] = "Missing Coordinates"  # Or some other placeholder
        continue

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except ValueError:
        print(f"Warning: Invalid latitude or longitude format at row {index}. Skipping.")
        df.loc[index, 'Address'] = "Invalid Coordinates"
        continue

    address = reverse_geocode(latitude, longitude)

    if address:
        df.loc[index, 'Address'] = address
        print(f"Row {index}: Geocoded to {address}")
    else:
        df.loc[index, 'Address'] = "Geocoding Failed"
        print(f"Row {index}: Geocoding failed.")


# Save the updated DataFrame to a new CSV file
try:
    df.to_csv(output_csv_file, index=False)
    print(f"Successfully saved the updated data to '{output_csv_file}'")
except Exception as e:
    print(f"Error saving the CSV file: {e}")

print("Script completed.")

