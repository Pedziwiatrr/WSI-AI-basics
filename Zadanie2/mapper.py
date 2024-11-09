import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import time


def get_city_coordinates(cities):
    geolocator = Nominatim(user_agent="city_mapper")
    coordinates = {}
    for city in cities:
        location = geolocator.geocode(city + ', Poland')
        if location:
            coordinates[city] = (location.longitude, location.latitude)
        time.sleep(1)
    return coordinates


def plot_city_route(cities, coordinates):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Route through selected Polish cities")

    lons = [coordinates[city][0] for city in cities if city in coordinates]
    lats = [coordinates[city][1] for city in cities if city in coordinates]

    ax.plot(lons, lats, marker='o', color='blue', markersize=5, linestyle='-', linewidth=1)

    for city, (lon, lat) in coordinates.items():
        if city in cities:
            ax.text(lon, lat, city, fontsize=8, ha='right')

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid()
    plt.show()


def create_map(cities):
    coordinates = get_city_coordinates(cities)
    plot_city_route(cities, coordinates)
