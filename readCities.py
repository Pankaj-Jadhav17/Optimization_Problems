from geopy.geocoders import Nominatim

############################################################
############################################################

def readCities():
    P = []      # coordinates of cities
    j = 0

    # Get coordinates of cities
    geolocator = Nominatim(user_agent="BWG_App")

    with open("./India_cities.txt") as file:
        for line in file:
            city = line.rstrip('\n')

            if city == "":
                break

            city += ", India"

            pt = geolocator.geocode(city, timeout=10000)

            print("City =", city, pt.latitude, pt.longitude)

            P.insert(j, [pt.latitude, pt.longitude])

            j += 1

    # P = array(P)
    return P

############################################################