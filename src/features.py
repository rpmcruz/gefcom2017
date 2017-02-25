import numpy as np
import pandas as pd
import datetime
import holidays
pd.options.mode.chained_assignment = None
from sklearn.preprocessing import OneHotEncoder

def angular_diff(diff, revolution):
    return np.abs((diff+revolution//2) % revolution - revolution//2)

def extract_date_features(dates, year):
    df = pd.DataFrame()
    df['date'] = dates  # for debugging purposes

    day = pd.Series([d.timetuple().tm_yday for d in dates])
    df['day'] = day

    df['sin.day'] = np.sin(day*2*np.pi/365 + np.pi/4)
    df['cos.day'] = np.cos(day*2*np.pi/365 + np.pi/4)

    # on average, 191 = datetime.date(2016, 7, 9) is the hottest day
    #df['day_solstice'] = np.min((
    #    np.abs(df['day'] - datetime.date(year-1, 7, 9).timetuple().tm_yday),
    #    np.abs(df['day'] - datetime.date(year, 7, 9).timetuple().tm_yday),
    #    np.abs(df['day'] - datetime.date(year+1, 7, 9).timetuple().tm_yday),
    #))
    
    date1 = datetime.datetime(year-1, 7, 9)
    date2 = datetime.datetime(year, 7, 9)
    date3 = datetime.datetime(year+11, 7, 9)
    daydistance1 = [np.abs(d - date1).days for d in dates]
    daydistance2 = [np.abs(d - date2).days for d in dates]
    daydistance3 = [np.abs(d - date3).days for d in dates]
    df['day_solstice'] = np.amin([daydistance1, daydistance2, daydistance3], axis=0)    
    
    date1 = datetime.datetime(year-1, 2, 15)
    date2 = datetime.datetime(year, 2, 15)
    date3 = datetime.datetime(year+1, 2, 15)
    daydistance1 = [np.abs(d - date1).days for d in dates]
    daydistance2 = [np.abs(d - date2).days for d in dates]
    daydistance3 = [np.abs(d - date3).days for d in dates]
    df['day_solstice_2'] = np.amin([daydistance1, daydistance2, daydistance3], axis=0)
    
    date1 = datetime.datetime(year-1, 4, 25)
    date2 = datetime.datetime(year, 4, 25)
    date3 = datetime.datetime(year+1, 4, 25)
    daydistance1 = [np.abs(d - date1).days for d in dates]
    daydistance2 = [np.abs(d - date2).days for d in dates]
    daydistance3 = [np.abs(d - date3).days for d in dates]
    df['dist_25_04'] = np.amin([daydistance1, daydistance2, daydistance3], axis=0)
    
    date1 = datetime.datetime(year-1, 7, 20)
    date2 = datetime.datetime(year, 7, 20)
    date3 = datetime.datetime(year+1, 7, 20)
    daydistance1 = [np.abs(d - date1).days for d in dates]
    daydistance2 = [np.abs(d - date2).days for d in dates]
    daydistance3 = [np.abs(d - date3).days for d in dates]
    df['dist_20_07'] = np.amin([daydistance1, daydistance2, daydistance3], axis=0)    
    
    date1 = datetime.datetime(year-1, 10, 31)
    date2 = datetime.datetime(year, 10, 31)
    date3 = datetime.datetime(year+1, 10, 31)
    daydistance1 = [np.abs(d - date1).days for d in dates]
    daydistance2 = [np.abs(d - date2).days for d in dates]
    daydistance3 = [np.abs(d - date3).days for d in dates]
    df['dist_31_10'] = np.amin([daydistance1, daydistance2, daydistance3], axis=0)
    
    date1 = list(holidays.US(years=year-1).keys())[list(holidays.US(years=year-1).values()).index('Thanksgiving')]
    date2 = list(holidays.US(years=year).keys())[list(holidays.US(years=year).values()).index('Thanksgiving')]
    date3 = list(holidays.US(years=year+1).keys())[list(holidays.US(years=year+1).values()).index('Thanksgiving')]
    daydistance1 = [np.abs(d.date() - date1).days for d in dates]
    daydistance2 = [np.abs(d.date() - date2).days for d in dates]
    daydistance3 = [np.abs(d.date() - date3).days for d in dates]
    df['distance_Thanksgiven'] = np.amin([daydistance1, daydistance2, daydistance3], axis=0)    
    
    startdate = datetime.datetime(2003, 1, 1)
    daydistance = [(d - startdate).days for d in dates]
    df['daydistance'] = daydistance

    month = [[d.timetuple().tm_mon-1] for d in dates]
    month_bin = OneHotEncoder(12, dtype=int, sparse=False).fit_transform(month)
    for i in range(12):
        df['month_%d' % i] = month_bin[:, i]
        
    hour = [[d.timetuple().tm_hour] for d in dates]
    hour_bin = OneHotEncoder(24, dtype=int, sparse=False).fit_transform(hour)
    for i in range(24):
        df['hour_%d' % i] = hour_bin[:, i]

    hour = pd.Series([d.timetuple().tm_hour for d in dates])
    df['hour'] = hour
    df['sin.hour'] = np.sin(hour*2*np.pi/24)
    df['cos.hour'] = np.cos(hour*2*np.pi/24)
    df['night_hours'] = [1 if 0 <= h <= 6 else 0 for h in hour]

    weekdays = [d.weekday() for d in dates]
    df['weekend'] = [1 if d >= 5 else 0 for d in weekdays]
    for i, s in enumerate([
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday',
            'sunday']):
        df[s] = [1 if d == i else 0 for d in weekdays]

    hs = holidays.US(years=year)
    df['holiday'] = [1 if d in hs else 0 for d in dates]
    weekday_holiday = [7 if d in hs else d.weekday() for d in dates]
    df['weekday_holiday'] = weekday_holiday
    df['workday'] = [1 if d <= 4 else 0 for d in weekday_holiday]
    
    diff = datetime.timedelta(days=1)
    special_holidays = [
        'Independence Day', 'New Year\'s Day', 'Christmas Day',
        'Thanksgiving']
    for i, holiday in enumerate(special_holidays):
        df['special_holiday.%d' % i] = [
            1 if hs.get(d) == holiday or
            hs.get(d+diff) == holiday or
            hs.get(d-diff) == holiday else 0 for d in dates]

    df['holiday_neighbor'] = [
        1 if d+diff in hs or d-diff in hs else 0 for d in dates]
    
    # dist holidays
    hs = holidays.US(years=[year-1, year, year+1])
    dist = [(date.date() - min(hs, key=lambda d: abs(d - date.date()))).days
            for date in dates]
    df['dist.holiday'] = dist

    for m in range(1, 12+1):
        ref = datetime.datetime(year, m, 15)
        df['dist.day15.%d' % m] = [
        angular_diff((d-ref).days, 365) for d in dates]
        
    for m in range(1, 12+1):
        ref = datetime.datetime(year, m, 1)
        df['dist.day1.%d' % m] = [
        angular_diff((d-ref).days, 365) for d in dates]
        
    for m in range(1, 12+1):
        ref = datetime.datetime(year, m, 25)
        df['dist.day25.%d' % m] = [
        angular_diff((d-ref).days, 365) for d in dates]  
        
    start = datetime.datetime(year, 3, 1)
    start += datetime.timedelta(days=13-start.weekday())
    end = datetime.datetime(year, 11, 1)
    end += datetime.timedelta(days=6-end.weekday())
    df['is-dst'] = [1 if start <= d <= end else 0 for d in dates]
    

    return df
'''
regions = ['CT', 'ME', 'NEMASSBOOST', 'NH', 'RI', 'SEMASS', 'VT', 'WCMASS',
           'MASS', 'TOTAL']
'''

def get_year(region, year):
    if region == 'TOTAL':
        if year <= 2004:
            region = 'NEPOOL'
        elif year == 2016:
            region = 'ISO NE CA'
        else:
            region = 'ISONE CA'
    elif region == 'SEMASS':
        if year == 2016:
            region = 'SEMA'
    elif region == 'WCMASS':
        if year == 2016:
            region = 'WCMA'
    elif region == 'NEMASSBOST':
        if year == 2016:
            region = 'NEMA'

    print('load %s %d' % (region, year))
    filename = '../data/%d_smd_hourly.xls' % year
    ws = pd.read_excel(filename, region)

    days = ws['Date']
    if 'Hour' in ws:
        hours = ws['Hour']-1
    else:
        hours = ws['Hr_End']-1
    dates = [
        d.to_pydatetime() + datetime.timedelta(hours=int(h))
        for d, h in zip(days, hours)]

    df = extract_date_features(dates, year)
    if 'DEMAND' in ws:
        df['DEMAND'] = ws['DEMAND']
    else:
        df['DEMAND'] = ws['RT_Demand']
    # replace demand by neighbor average when 0 (NA)
    for i in df.index[df['DEMAND'] == 0]:
        df['DEMAND'].iloc[i] = df['DEMAND'].iloc[[i-1, i+1]].mean()

    if year == 2016:
        df['DryBulb'] = ws['Dry_Bulb']
    else:
        df['DryBulb'] = ws['DryBulb']
    return df


def add_past_features(X):
    week = 7*24
    df['avg2weeks'] = 0
    for i in range(3*week, len(df), week):
        df['avg2weeks'].iloc[i:i+week] = \
            df['DEMAND'].iloc[i-3*week:i-2*week].mean()
    return df.iloc[3*week:]


if __name__ == '__main__':
    regions = ['ME', 'NH', 'VT', 'CT', 'RI', 'TOTAL', 'SEMASS', 'WCMASS',
               'NEMASSBOST']
    for region in regions:
        df = pd.DataFrame()
        for year in range(2003, 2016+1):
            df = df.append(get_year(region, year))
        df = add_past_features(df)
        print('saving...')
        df.to_csv('../out/data/%s.csv' % region, index=False)

df0 = pd.read_csv('../out/data/SEMASS.csv')
df1 = pd.read_csv('../out/data/WCMASS.csv')
df2 = pd.read_csv('../out/data/NEMASSBOST.csv')
df0['DEMAND'] = df0['DEMAND'] + df1['DEMAND'] + df2['DEMAND']
df0['DryBulb'] = df0['DryBulb'] + df1['DryBulb'] + df2['DryBulb']
df0.to_csv('../out/data/MASS.csv', index=False)
