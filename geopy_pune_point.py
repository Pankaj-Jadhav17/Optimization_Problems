from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="PRJ_App")

pt = geolocator.geocode("Pune, Maharashtra, India").point
print(pt[0], pt[1])