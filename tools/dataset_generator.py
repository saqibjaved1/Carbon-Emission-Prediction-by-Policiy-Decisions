# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 01:38:32 2020

@author: Toaha
"""
import numpy as np
import pandas as pd
import country_converter as coco
import sys

# Initialize the Country Converter
cc = coco.CountryConverter()

def read_emissions(emission_file_name):
    # Import the emissions file into a Pandas dataframe
    emissions = pd.ExcelFile(emission_file_name)
    country_emissions = np.array(emissions.sheet_names)
    emi_dict = pd.read_excel(emission_file_name, sheet_name=None)
    values = emi_dict.values()

    # Use the country converter for a uniform notation
    names_init = list(country_emissions)
    names_gcp = cc.convert(names=names_init, to='name_short')
    # Generate a new emission dictionary with the modified country names
    emi_dict_new = dict(zip(names_gcp, values))
    return names_gcp, emi_dict_new

def read_stringency(stringency_file_name):
    # Read the stringency file
    stringency = pd.read_csv(stringency_file_name)
    country_names = stringency.CountryName.values
    names_stringency = list(np.unique(country_names))  # Get the unique country names
    return stringency, names_stringency


def generate_emission_and_stringency_dicts (names_gcp, names_stringency, emi_dict_new, stringency):
    # Find the common countries between the two datasets
    common_countries = list(set(names_gcp) & set(names_stringency))
    # Separate the emission dict based on common countries
    emit_mod = {key: emi_dict_new[key] for key in common_countries}
    # Remove the timestamps from the date
    for key in common_countries:
        emit_mod[key].DATE = pd.to_datetime(emit_mod[key].DATE, errors='coerce').dt.date

    # Separate the stringency countries into separate dictionary
    i = 0
    stringency_dict = {}
    while i < len(common_countries):
        key = common_countries[i]
        value = stringency[stringency.CountryName == key]
        stringency_dict[key] = value
        i += 1

    return emit_mod, stringency_dict, common_countries

def file_generator (dict_name, filename, common_countries):

    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    i=0
    while i < len(common_countries):
        key = common_countries[i]
        dict_name[key].to_excel(writer, sheet_name=key)
        i += 1

    writer.save()

def main():

    emission_file_name = sys.argv[1]
    stringency_file_name = sys.argv[2]
    names_gcp, emi_dict_new = read_emissions(emission_file_name)
    stringency, names_stringency = read_stringency(stringency_file_name)
    emit_mod, stringency_dict, common_countries = generate_emission_and_stringency_dicts(names_gcp, names_stringency, emi_dict_new, stringency)
    file_generator(emit_mod, 'Modified_Emission_Data.xlsx', common_countries)
    file_generator(stringency_dict, 'Modified_Stringency_Data.xlsx', common_countries)



if __name__ == "__main__":
    main()


