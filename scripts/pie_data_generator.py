"""
Author: Sreetama Sarkar
Date: 8/10/2020
"""
import requests
import lxml.html as lh
import pandas as pd
from predictCO2.preprocessing.co2_percent_dict import co2_percentage

url='https://www.worldometers.info/world-population/population-by-country/'
#Create a handle, page, to handle the contents of the website
page = requests.get(url)
#Store the contents of the website under doc
doc = lh.fromstring(page.content)
#Parse data that are stored between <tr>..</tr> of HTML
tr_elements = doc.xpath('//tr')

#Check the length of the first 12 rows
print([len(T) for T in tr_elements[:12]])

#Create empty list
col=[]
i=0
#For each row, store each first element (header) and an empty list
#Each header is appended to a tuple along with an empty list
for t in tr_elements[0]:
    i+=1
    name=t.text_content()
    print('%d:"%s"'%(i,name))
    col.append((name,[]))

# Since out first row is the header, data is stored on the second row onwards
for j in range(1, len(tr_elements)):
    # T is our j'th row
    T = tr_elements[j]

    # i is the index of our column
    i = 0

    # Iterate through each element of the row
    for t in T.iterchildren():
        data = t.text_content()
        # Check if row is empty
        if i > 0:
            # Convert any numerical value to integers
            try:
                data = int(data)
            except:
                pass
        # Append the data to the empty list of the i'th column
        col[i][1].append(data)
        # Increment i for the next column
        i += 1

Dict={title:column for (title,column) in col if title == 'Population (2020)' or title == 'Country (or dependency)' or
      title == 'Land Area (Km²)' or title == 'Density (P/Km²)'}
df=pd.DataFrame(Dict)
print(df.head())
df.rename(columns={'Country (or dependency)': 'Country', 'Population (2020)': 'Population', 'Land Area (Km²)': 'Area', 'Density (P/Km²)': 'Population Density'}, inplace=True)
df['Population'] = df['Population'].str.replace(',','')
# df['Population Density'] = df['Population Density'].str.replace(',','')
df['Area'] = df['Area'].str.replace(',','')
df.dropna(inplace=True)
df['Population'] = df['Population'].astype('int64')
# df['Population Density'] = df['Population Density'].astype('float64')
df['Area'] = df['Area'].astype('float64')
df['Country'] = df['Country'].astype('str')
df['co2_percent'] = 0
for index, row in df.iterrows():
    if row['Country'] == 'Czech Republic (Czechia)':
        df.loc[index, 'Country'] = 'Czech Republic'

for index, row in df.iterrows():
    if row['Country'] in co2_percentage.keys():
        df.loc[index, 'co2_percent'] = co2_percentage[row['Country']]
    else:
        df.drop(index, inplace=True)

# cc_codes = pd.read_csv('https://gist.githubusercontent.com/tadast/8827699/raw/3cd639fa34eec5067080a61c69e3ae25e3076abb/countries_codes_and_coordinates.csv')
cc_codes = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
for index, row in cc_codes.iterrows():
    if row['COUNTRY'] == "Korea, South":
        cc_codes.loc[index, 'COUNTRY'] = 'South Korea'

# country_code = []
# for country in df['Country'].values:
#      country_code.append(country[:3])

df['CODE'] = 'aaa'
for index, row in df.iterrows():
    if row['Country'] in cc_codes['COUNTRY'].values:
        df.loc[index, 'CODE'] = cc_codes[cc_codes['COUNTRY'] == row['Country']]['CODE'].values[0]
    else:
        print(row['Country'])

df.to_pickle('dataset/Country_population_co2.pkl')

