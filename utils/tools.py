"""
Useful tools 
Authour: Yun "Max" Zhang
"""

def data_to_csv(input):
    """
    read .data file, output csv file
    """
    data_file = open(input, 'rU') # univseral new lines https://docs.python.org/3.5/library/functions.html#open
    name = input.split('.')[0]

    with open(name+'.csv', mode='w') as output:
        output_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in data_file:

            line = line.strip().split(',')
            output_writer.writerow(line)