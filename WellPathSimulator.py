import socket
import json
import time
import pandas as pd
import keyboard
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = "192.168.1.4"
port = 8000
buffer_size = 8192
s.connect((host, port))
print("Listening on %s:%s..." % (host, str(port)))

df = pd.DataFrame(columns = ['btimestamp', 'bx', 'by', 'bz'])

dfsrv = pd.DataFrame()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('EW (ft)')
ax.set_ylabel('NS (ft)')
ax.set_zlabel('TVD (ft)')

def json_rec_to_df(df, mylist):
    for rec in mylist:
        jdata = json.loads(rec)
        df = pd.DataFrame([[jdata['magneticField']['timestamp'], 
                            jdata['magneticField']['value'][0]*1000,
                            jdata['magneticField']['value'][1]*1000,
                            jdata['magneticField']['value'][2]*1000,
                            jdata['accelerometer']['timestamp'],
                            jdata['accelerometer']['value'][0]*(1000/9.80665),
                            jdata['accelerometer']['value'][1]*(1000/9.80665),
                            jdata['accelerometer']['value'][2]*(1000/9.80665)]],
                            columns=['btimestamp', 'bx', 'by', 'bz',
                                     'gtimestamp', 'gx', 'gy', 'gz'])
    return df

def survey_calc(df):
    df['G'] = (df['gx']**2 + df['gy']**2 + df['gz']**2)**0.5
    df['B'] = (df['bx']**2 + df['by']**2 + df['bz']**2)**0.5
    df['Dip'] = 90 - np.degrees(np.arccos((df.gx*df.bx + df.gy*df.by + df.gz*df.bz) / \
                                          (df.G*df.B)))
    df['INC'] = np.degrees(np.arccos(df.gx / df.G))
    df['INC'] = abs(df['INC'] - 180) #Flip the axis so it lines up with the azimuth 
    df['AZI'] = np.degrees(np.arctan2(df.G*(df.gy*df.bz - df.gz*df.by),
                                      df.bx*df.G**2 - (df.gx*df.bx + df.gy*df.by + \
                                                       df.gz*df.bz)*df.gx)) \
              + np.where((df.gy*df.bz - df.gz*df.by) < 0, 360, 0)
    df['AZI'] = abs(df['AZI'] - 360) #Reverse the values
    return df

def min_curve_calc(MD, I1, I2, A1, A2):
    I1 = np.radians(I1)
    I2 = np.radians(I2)
    A1 = np.radians(A1)
    A2 = np.radians(A2)
    DLS = np.arccos(np.cos(I2 - I1) - (np.sin(I1)*np.sin(I2)*(1-np.cos(A2-A1))))
    RF = (2/DLS) * np.tan(DLS/2)
    NS = (MD / 2) * (np.sin(I1) * np.cos(A1) + np.sin(I2) * np.cos(A2)) * RF
    EW = (MD / 2) * (np.sin(I1) * np.sin(A1) + np.sin(I2) * np.sin(A2)) * RF
    TVD = (MD / 2) * (np.cos(I1) + np.cos(I2)) * RF
    return NS, EW, TVD, DLS, RF

while True:
    data = s.recv(buffer_size)
    mylist = data.decode('utf-8').split('\n')
    mylist = filter(None, mylist) #mylist is now a filter object
    
    df = json_rec_to_df(df, mylist)
    df = survey_calc(df)

    try:
        if keyboard.is_pressed('space'):
            time.sleep(0.5)
            dfsrv = dfsrv.append(df[-1:])
            dfsrv['MD'] = 100
            print(dfsrv[['MD', 'INC', 'AZI', 'gx', 'gy', 'gz', 'bx', 'by', 'bz']][-1:])
            dfsrv['I1'] = dfsrv['INC'].shift(1)
            dfsrv['A1'] = dfsrv['AZI'].shift(1)
            resdf = dfsrv.apply(lambda x: min_curve_calc(x['MD'], 
                                                         x['I1'],
                                                         x['INC'],
                                                         x['A1'],
                                                         x['AZI']), 
                                                         axis=1,
                                                         result_type='expand')
            resdf.columns = ['NS', 'EW', 'TVD', 'DLS', 'RF']
            resdf['TVD'] = -resdf['TVD']  
            df_final = pd.concat([dfsrv, resdf], axis=1)
            df_final['NS'] = df_final['NS'].cumsum() 
            df_final['EW'] = df_final['EW'].cumsum()
            df_final['TVD'] = df_final['TVD'].cumsum()
            df_final['DLS'] = np.degrees(df_final['DLS'])
            df_final['MD'] = df_final['MD'].cumsum()
            ax.scatter(df_final['EW'], df_final['NS'], df_final['TVD'])
            plt.pause(0.5)
    except:
        pass

plt.show()