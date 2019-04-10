import requests

def data_on_crimes():
    '''
    '''

    url = "https://data.cityofchicago.org/resource/6zsd-86xi.json?" + \
          "$where=year = 2017 OR year = 2018&" + \
          "$limit=600000"
    session = requests.session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    request = session.get(url)

    return request.json()

def data_on_blocks():
    '''
    '''

    url = 'https://api.census.gov/data/2017/acs/acs5?get=C02003_001E,' + \
          'C02003_004E,B19013_001E,B25010_001E,NAME&for=block%20group:*' + \
          '&in=state:17&in=county:031&in=tract:*'

    session = requests.session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    request = session.get(url)

    return request.json()
