import requests

def data_on_crimes():

    url = "https://data.cityofchicago.org/resource/6zsd-86xi.json?" + \
          "$where=year = 2017 OR year = 2018"
    session = requests.session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    request = session.get(url)

    return request.json()

