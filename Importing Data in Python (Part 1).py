# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 18:30:13 2019

@author: Jie
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


###############################################################################
# Assign filename: file
file = 'F:/0 - PhD at UTD/2019 Spring/DataCamp/Importing Data in Python (Part 1)/seaslug.txt'

# Import file: data
data = np.loadtxt(file, delimiter='\t', dtype=str)

# Print the first element of data
print(data[0])

# Import data as floats and skip the first row: data_float
data_float = np.loadtxt(file, delimiter='\t', dtype=float, skiprows=1)

# Print the 10th element of data_float
print(data_float[9])

# Plot a scatterplot of the data
plt.scatter(data_float[:, 0], data_float[:, 1])
plt.xlabel('time (min.)')
plt.ylabel('percentage of larvae')
plt.show()
###############################################################################
data=np.genfromtxt('F:/0 - PhD at UTD/2019 Spring/DataCamp/Importing Data in Python (Part 1)/titanic_sub.csv', delimiter=',', names=True, dtype=None)
print(data)
print(data['Survived'])

###############################################################################
# Import pandas
import pandas as pd

# Assign spreadsheet filename: file
file = 'F:/0 - PhD at UTD/2019 Spring/DataCamp/Importing Data in Python (Part 1)/battledeath.xlsx'

# Load spreadsheet: xl
xl = pd.ExcelFile(file)

# Print sheet names
print(xl.sheet_names)

###############################################################################
# Load a sheet into a DataFrame by name: df1
df1 = xl.parse('2004')

# Print the head of the DataFrame df1
print(df1.head())

# Load a sheet into a DataFrame by index: df2
df2 = xl.parse('2002')

# Print the head of the DataFrame df2
print(df2.head())
###############################################################################
# Parse the first sheet and rename the columns: df1
df11 = xl.parse(0, skiprows=[0], names=['Country', 'AAM due to War (2002)'])

# Print the head of the DataFrame df1
print(df11.head())

# Parse the first column of the second sheet and rename the column: df2
df22 = xl.parse('2002', skiprows=[0], parse_cols=[0], names=['Country'])

# Print the head of the DataFrame df2
print(df22.head())

###############################################################################
# Import sas7bdat package
from sas7bdat import SAS7BDAT

# Save file to a DataFrame: df_sas
with SAS7BDAT('F:/0 - PhD at UTD/2019 Spring/DataCamp/Importing Data in Python (Part 1)/sales.sas7bdat') as file:
    df_sas=file.to_data_frame()

# Print head of DataFrame
print(df_sas.head())

# Plot histogram of DataFrame features (pandas and pyplot already imported)
pd.DataFrame.hist(df_sas[['P']])
plt.ylabel('count')
plt.show()
###############################################################################
# Import pandas
import pandas as pd

# Load Stata file into a pandas DataFrame: df
df=pd.read_stata('F:/0 - PhD at UTD/2019 Spring/DataCamp/Importing Data in Python (Part 1)/disarea.dta')

# Print the head of the DataFrame df
print(df.head())

# Plot histogram of one column of the DataFrame
pd.DataFrame.hist(df[['disa10']])
plt.xlabel('Extent of disease')
plt.ylabel('Number of countries')
plt.show()
###############################################################################
# Import packages
import numpy as np
import h5py

# Assign filename: file
f='F:/0 - PhD at UTD/2019 Spring/DataCamp/Importing Data in Python (Part 1)/L-L1_LOSC_4_V1-1126259446-32.hdf5'

# Load file: data
data = h5py.File(f, 'r')

# Print the datatype of the loaded file
print(type(data))

# Print the keys of the file
for key in data.keys():
    print(key)

###############################################################################
# Get the HDF5 group: group
group=data['strain']

# Check out keys of group
for key in group.keys():
    print(key)

# Set variable equal to time series data: strain
strain=group['Strain'].value

# Set number of time points to sample: num_samples
num_samples=10000

# Set time vector
time = np.arange(0, 1, 1/num_samples)

# Plot data
plt.plot(time, strain[:num_samples])
plt.xlabel('GPS Time (s)')
plt.ylabel('strain')
plt.show()
###############################################################################
# Import package
import scipy.io

# Load MATLAB file: mat
mat=scipy.io.loadmat('F:/0 - PhD at UTD/2019 Spring/DataCamp/Importing Data in Python (Part 1)/ja_data2.mat')

# Print the datatype type of mat
print(type(mat))

###############################################################################
# Print the keys of the MATLAB dictionary
print(mat.keys())

# Print the type of the value corresponding to the key 'CYratioCyt'
print(type(mat['CYratioCyt']))

# Print the shape of the value corresponding to the key 'CYratioCyt'
print(np.shape(mat['CYratioCyt']))

# Subset the array and plot it
data = mat['CYratioCyt'][25, 5:]
fig = plt.figure()
plt.plot(data)
plt.xlabel('time (min.)')
plt.ylabel('normalized fluorescence (measure of expression)')
plt.show()
###############################################################################
# Import necessary module
from sqlalchemy import create_engine

# Create engine: engine
engine=create_engine('sqlite:///F:/0 - PhD at UTD/2019 Spring/DataCamp/Importing Data in Python (Part 1)/Chinook.sqlite')

# Save the table names to a list: table_names
table_names=engine.table_names()

# Print the table names to the shell
print(table_names)

###############################################################################
# Open engine connection: con
con=engine.connect()

# Perform query: rs
rs = con.execute('SELECT * from Album')

# Save results of the query to DataFrame: df
df = pd.DataFrame(rs.fetchall())

# Close connection
con.close()

print(df.head())
###############################################################################
# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute('SELECT LastName, Title FROM Employee')
    df = pd.DataFrame(rs.fetchmany(size=3))
    df.columns = rs.keys()

# Print the length of the DataFrame df
print('Length of the DataFrame is: ', len(df), '\n')

# Print the head of the DataFrame df
print(df.head())
###############################################################################
with engine.connect() as con:
    rs = con.execute('SELECT * from Employee WHERE EmployeeId >= 6')
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()

# Print the head of the DataFrame df
print(df.head())
###############################################################################
# Open engine in context manager
with engine.connect() as con:
    rs = con.execute('SELECT * FROM Employee ORDER BY BirthDate')
    df = pd.DataFrame(rs.fetchall())

    # Set the DataFrame's column names
    df.columns = rs.keys()

# Print head of DataFrame
print(df.head())
###############################################################################
# Execute query and store records in DataFrame: df
df = pd.read_sql_query('SELECT * FROM Album', engine)

# Print head of DataFrame
print(df.head())

# Open engine in context manager and store query result in df1
with engine.connect() as con:
    rs = con.execute("SELECT * FROM Album")
    df1 = pd.DataFrame(rs.fetchall())
    df1.columns = rs.keys()

# Confirm that both methods yield the same result
print(df.equals(df1))
###############################################################################
# Import packages
from sqlalchemy import create_engine
import pandas as pd

# Create engine: engine
engine=create_engine('sqlite:///F:/0 - PhD at UTD/2019 Spring/DataCamp/Importing Data in Python (Part 1)/Chinook.sqlite')

# Execute query and store records in DataFrame: df
con=engine.connect()
df=pd.read_sql_query('SELECT * FROM Employee WHERE EmployeeId >= 6 ORDER BY BirthDate', con)
con.close()

# Print head of DataFrame
print(df.head())
###############################################################################
# Import packages
from sqlalchemy import create_engine
import pandas as pd

# Create engine: engine
engine=create_engine('sqlite:///F:/0 - PhD at UTD/2019 Spring/DataCamp/Importing Data in Python (Part 1)/Chinook.sqlite')

# Execute query and store records in DataFrame: df
with engine.connect() as con:
    df=pd.read_sql_query('SELECT * FROM Employee WHERE EmployeeId >= 6 ORDER BY BirthDate', con)

# Print head of DataFrame
print(df.head())
###############################################################################
with engine.connect() as con:
    rs = con.execute('SELECT Title, Name FROM Album INNER JOIN Artist ON Album.ArtistID = Artist.ArtistID')
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()

# Print head of DataFrame df
print(df.head())
###############################################################################
with engine.connect() as con:
    df=pd.read_sql_query('SELECT * FROM PlaylistTrack INNER JOIN Track ON PlaylistTrack.TrackId = Track.TrackId WHERE Milliseconds < 250000', con)

# Print head of DataFrame
print(df.head())
###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

