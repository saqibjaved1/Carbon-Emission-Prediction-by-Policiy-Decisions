"""Created by: Tapan Sharma
Date: 20/06/20
"""
import Globals
import pandas as pd
import yaml


def conform_date(date_str):
    """
    Method conforms the date in Global Carbon project to match with the date of OxCGRT i.e. "YYYYMMDD" e.g. 20200101
    :param date_str: Date string from Global Carbon project (of form "DD/MM/YYYY")
    :return: date in form "YYYYMMDD"
    """
    if isinstance(date_str, pd.Timestamp):
        return date_str.strftime(('%Y%m%d'))
    else:
        date_parts = None

        if '/' in date_str:
            date_parts = date_str.split('/')
        if '-' in date_str:
            date_parts = date_str.split('-')
        if '.' in date_str:
            date_parts = date_str.split('.')

        if date_parts:
            year = date_parts[2]
            month = date_parts[1]
            day = date_parts[0]
            if len(day) == 1:
                day = '0' + day
            if len(month) == 1:
                month = '0' + month
            conformed_date = year + month + day
            return conformed_date
        else:
            raise ValueError("Failed to conform date in YYYYMMDD format for received: {}".format(date_str))


