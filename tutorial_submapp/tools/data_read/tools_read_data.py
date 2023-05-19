import datetime

def unix_time_seconds(year,month,day):
#  Ref:
#    https://stackoverflow.com/questions/6999726/how-can-i-convert-a-datetime-object-to-milliseconds-since-epoch-unix-time-in-p
    dt = datetime.datetime(year,month,day)
    epoch = datetime.datetime.utcfromtimestamp(0) # total seconds since UTC 1970.1.1:00:00:00
    return int((dt - epoch).total_seconds())
