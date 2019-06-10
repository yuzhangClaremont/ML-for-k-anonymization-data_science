"""
run mondrian with given parameters
Authour: Yun "Max" Zhang
"""

# !/usr/bin/env python
# coding=utf-8
# from mondrian import mondrian
# from utils.read_adult_data import read_data as read_adult
# from utils.read_informs_data import read_data as read_informs
import sys, copy, random, csv

# DATA_SELECT = 'a'
# RELAX = False
# INTUITIVE_ORDER = None


def data_to_csv(input):
    """
    read .data file, output csv file
    """
    data_file = open(input, 'rU')
    name = input.split('.')[0]
    print(name+'!!!!')
    with open(name+'.csv', mode='w') as output:
        output_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in data_file:

            line = line.strip().split(',')
            print(line)
            output_writer.writerow(line)
    # print(len(data_file) )

read_data('adult.data')