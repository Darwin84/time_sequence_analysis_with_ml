import numpy as np
from scipy.signal import argrelextrema, find_peaks, find_peaks_cwt


def read_file(file_path, start_time, end_time):

    with open(file_path, "r") as fp:

        line_info = []
        while True:
            line = fp.readline()
            if not line:
                break
            line = line.strip()
            line = line.split(",")
            date_time = line[0]
            start_time = start_time.replace("-","")
            end_time = end_time.replace("-","")
            if start_time < date_time < end_time:
                line_h = line[2]
                line_l = line[3]
                line_o = line[1]
                line_c = line[4]
                line_info.append([float(line_o),
                                  float(line_h),
                                  float(line_l),
                                  float(line_c)])
        
    return line_info


def write_file(file_path, data_info):
    
    with open(file_path, "w") as fp:
        for data in data_info:
            fp.write(str(data)+"\n")

             
# def construct_sequence(data_info):

#     training_data = []
#     for i in range(60,len(data_info)-10):
#         input = data_info[i-50:i+1]
#         output = data_info[i+1:i+5]
#         tmp_list = []
#         tmp_list.extend(input)
#         tmp_list.extend(output)
#         training_data.append(tmp_list)

#     return training_data


def construct_sequence(data_info):

    training_data = []
    for i in range(21,len(data_info)-10):
        input = data_info[i-20:i]
        # output = data_info[i+1][-1] - data_info[i+1][0]
        output = data_info[i]
        tmp_list = []
        for ii in input:
            tmp_list.extend(ii)
        # print("input: ",input)
        # tmp_list.extend(input)
        tmp_list.extend(output)
        training_data.append(tmp_list)

    return training_data
        

if __name__ == "__main__":
    date_info = {
        "2009": ["2020-04-15", "2020-08-15"],
        "2010": ["2020-04-15", "2020-08-15"],
        "2005": ["2020-01-01", "2020-04-15"],
        "2001": ["2019-08-15", "2019-12-30"],
        "1909": ["2019-04-15", "2019-08-15"],
        "1910": ["2019-04-15", "2019-08-15"],
        "1905": ["2019-01-01", "2019-04-15"],
        "1901": ["2018-08-15", "2018-12-30"],
        "1810": ["2018-04-15", "2018-08-15"],
        "1809": ["2018-04-15", "2018-08-15"],
        "1805": ["2018-01-01", "2018-04-15"],
        "1801": ["2017-08-15", "2017-12-30"],
        "1710": ["2017-04-15", "2017-08-15"],
        "1709": ["2017-04-15", "2017-08-15"],
        "1705": ["2017-01-01", "2017-04-15"],
        "1701": ["2016-08-15", "2016-12-30"],
        "1609": ["2016-04-15", "2016-08-15"],
        "1605": ["2016-01-01", "2016-04-15"],
        "2012": ["2020-05-01", "2020-12-30"],
        "2006": ["2020-01-01", "2020-05-01"],
        "1912": ["2019-05-01", "2019-12-30"],
        "1906": ["2019-01-01", "2019-05-01"],
        "1812": ["2018-05-01", "2018-12-30"],
        "1806": ["2018-01-01", "2018-05-01"]
    }
    line_info = read_file("./fifteen_p2001.txt", date_info["2001"][0], date_info['2001'][1])
    training_data = construct_sequence(line_info)
    write_file("./training_data.txt", training_data)
