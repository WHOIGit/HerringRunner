import os, sys
import pandas as pd
import datetime as dt
from tqdm import tqdm


print('LOADING WEATHER', flush=True)

weather = pd.read_csv('weather-hourly_2017-06-01_2019-12-31.csv',index_col=0, parse_dates=True)
print(weather.head(), flush=True)

weather_daily = pd.read_csv('weather-daily_2017-06-01_2019-12-31.csv',index_col=0, parse_dates=['datetime','sunrise','sunset'])
print(weather_daily.head(), flush=True)


def frame2video(f):
    return "'"+f.rsplit('_',1)[0]

def frame2ts(f):
    video_id,ms = f.rsplit('_',1) 
    # example video_id: 20170701145052891 --> 2017-07-01 14:50:52.891
    ts = dt.datetime.strptime(video_id+'000','%Y%m%d%H%M%S%f')
    ts = ts + dt.timedelta(milliseconds=int(ms))
    return ts

def ts2tsr(ts):  # temperature, solar radiation, rain
    ts = ts - dt.timedelta(minutes=ts.minute, seconds=ts.second, microseconds=ts.microsecond)
    try:
        temp = weather.at[ts,'temp']
        solar = weather.at[ts,'solarradiation']
        precip = weather.at[ts,'precip']
    except KeyError:
        print('tsr KeyError:',ts)
        return '','',''
    
    try: 
        temp = int(temp)
        solar = int(solar)
        precip = int(precip)
    except TypeError: 
        print('tsr TypeError: multiple weather entries for:', ts)
        print(weather.loc[ts])
        print('using first instance...')
        temp = int(temp.iloc[0])
        solar = int(solar.iloc[0])
        precip = int(precip.iloc[0])
        
    return temp,solar,precip 


daycats = ['   morning','  late-morning',' noon','afternoon','late-afternoon']
def ts2daycat(ts):
    date = ts.strftime('%Y-%m-%d')
    sunrise = weather_daily.loc[date].sunrise
    sunset = weather_daily.loc[date].sunset
    if isinstance(sunrise,pd.Series):
        sunrise = sunrise.iloc[0]
        sunset = sunset.iloc[0]
    sundiff = sunset-sunrise
    sunseg = sundiff/len(daycats)
    cats = ['night']+daycats
    for i in range(len(cats)):
        if ts<sunrise+i*sunseg:
            return cats[i]
    return 'night'


def cloudcat(val):
    val = val/100
    if val<0.33:
        return 'clear'
    elif val<0.66:
        return 'partly-cloudy'
    elif val<0.88:
        return 'mostly-cloudy'
    else:
        return 'overcast'

def ts2cloudcat(ts):
    ts = ts - dt.timedelta(minutes=ts.minute, 
                           seconds=ts.second, 
                           microseconds=ts.microsecond)
    val = weather.loc[ts].cloudcover
    if isinstance(val,pd.Series): val = val.iloc[0]
    return cloudcat(val)


def mooncat(val):
    val = val*100
    if val<12.5: return 'new'
    elif val<12.5*2: return 'crescent'
    elif val<12.5*3: return 'quarter'
    elif val<12.5*4: return 'gibbous'
    elif val<12.5*5: return 'full'
    elif val<12.5*6: return 'gibbous'
    elif val<12.5*7: return 'quarter'
    else: return 'crescent'


def ts2mooncat(ts):
    date = ts.strftime('%Y-%m-%d')
    val = weather_daily.loc[date].moonphase
    if isinstance(val,pd.Series): val = val.iloc[0]
    return mooncat(val)

def ts2all(ts):
    results = [ts.date()]

    for x in 'year month day hour minute'.split():
        results.append(getattr(ts,x))

    t,s,r = ts2tsr(ts)
    results.extend([t,s,r])

    c = ts2cloudcat(ts)
    d = ts2daycat(ts)
    m = ts2mooncat(ts)
    results.extend([c,d,m])
    return results
all_other = 'date year month day hour minute temperature solarradiation precipitation cloudcat daycat mooncat'


print('LOADING DATA', flush=True)

fin_str =  sys.argv[0] # 'full_export.csv'
fout_str = sys.argv[1] # 'full_export_plus.csv'

with open(fin_str,'r') as fin, open(fout_str,'a') as fout:
    header = next(fin).rstrip()  # 'frame,count,video,ts'
    header += ',date,year,month,day,hour,minute,temperature,solarradiation,precipitation,cloudcat,daycat,mooncat\n'
    print(header,flush=True)
    fout.write(header)
    for line in tqdm(fin):
        line = line.rstrip()
        ts = dt.datetime.strptime(line.split(',')[-1],'%Y-%m-%d %H:%M:%S.%f')
        new_params = ts2all(ts)
        new_params = map(str,new_params)
        line += ','+','.join(new_params)+'\n'
        fout.write(line)
            

print('DONE!')
