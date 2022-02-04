#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# code for this file comes from : https://www.kaggle.com/sobhanmoosavi/us-accidents


# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("US_Accidents_Dec20.csv")


# In[3]:


data.columns


# In[ ]:


# I think it would be interesting to look at the correlation between accidents that happen between the different times of day 
# Along with weather, and the severity of the accident 


# In[ ]:


data.describe()


# In[ ]:


len(data)


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[4]:


cols = ["ID", "State", "Weather_Condition", "Sunrise_Sunset", "Wind_Speed(mph)","Precipitation(in)", "Visibility(mi)", "Severity", "TMC", "Distance(mi)"]
accident_Data = data[cols]
accident_Data = accident_Data[accident_Data["Sunrise_Sunset"].notna()]
accident_Data = accident_Data[accident_Data["Wind_Speed(mph)"].notna()]
accident_Data["Precipitation(in)"].fillna(0.0, inplace = True)
accident_Data = accident_Data[accident_Data["TMC"].notna()]

accident_Data


# In[ ]:





# In[5]:


accident_Data.set_index("ID", inplace = True)


# In[6]:


accident_Data["Weather_Condition"].value_counts()


# In[ ]:


##The majority of accidents have reported Fair weahter conditions (??) 


# In[7]:


wd = accident_Data.groupby(["Weather_Condition"])
wd.first()


# In[8]:


accident_Data["Weather_Condition"].nunique()


# In[9]:


## changing similar values in the Weather_Condition column to one value

## Light Rain 
accident_Data = accident_Data.replace(to_replace = ["Light Rain", "Light Drizzle", "Drizzle and Fog", "Drizzle", "Heavy Drizzle", "Light Rain Showers", "Light Thunderstorms and Rain", "Light Rain Shower", "Light Rain with Thunder", "Light Rain / Windy", "Mist", "Light Drizzle / Windy", "Drizzle / Windy", "Light Thunderstorm"], value = "Light Rain")

## Rain
accident_Data = accident_Data.replace(to_replace = ["Rain", "Rain / Windy", "Rain Showers", "T-Storm", "T-Storm / Windy", "Thunder", "Thunder / Windy", "Thunder in the Vicinity", "Thunderstorm", "Thunderstorms and Rain", "Rain Shower", "Showers in the Vicinity"], value = "Rain")

##Heavy Rain
accident_Data = accident_Data.replace(to_replace = ["Heavy Rain", "Heavy Rain / Windy", "Heavy T-Storm", "Heavy T-Storm / Windy", "Heavy Thunderstorms and Rain", "Heavy Rain Showers"], value= "Heavy Rain")

## Light Snow 
accident_Data = accident_Data.replace(to_replace = ["Light Snow", "Light Blowing Snow", "Light Snow Shower", "Light Snow / Windy", "Light Thunderstorms and Snow", "Light Snow Showers", "Light Snow with Thunder", "Light Rain Shower / Windy", "Light Snow Grains"], value="Light Snow")

##Snow
accident_Data = accident_Data.replace(to_replace = ["Wintry Mix", "Snow", "Blowing Snow", "Blowing Snow / Windy", "Thunderstorms and Snow", "Squalls", "Squalls / Windy", "Snow / Windy", "Snow Grains", "Low Drifting Snow", "Snow Showers", "Snow and Thunder"], value = "Snow")

## Heavy Snow
accident_Data = accident_Data.replace(to_replace = ["Heavy Snow", "Heavy Snow / Windy", "Heavy Thunderstorms and Snow", "Heavy Blowing Snow", "Heavy Snow with Thunder"], value="Heavy Snow") 

##Fog // Haze 
accident_Data = accident_Data.replace(to_replace = ["Fog / Windy", "Fog", "Haze", "Haze / Windy", "Shallow Fog", "Patches of Fog", "Smoke", "Patches of Fog / Windy", "Light Fog", "Light Haze", "Smoke / Windy", "Partial Fog", "Partial Fog / Windy"], value = "Fog/Haze/Smoke")

##Icy Conditions 
accident_Data = accident_Data.replace(to_replace = ["Light Freezing Drizzle", "Light Hail", "Light Freezing Fog", "Light Freezing Rain", "Wintry Mix / Windy", "Light Snow and Sleet", "Light Ice Pellets", "Heavy Sleet", "Light Ice Pellets", "Snow and Sleet", "Hail", "Sleet", "Light Sleet", "Ice Pellets", "Freezing Rain", "Snow and Sleet / Windy", "Small Hail", "Heavy Thunderstorms with Small Hail", "Light Freezing Rain / Windy", "Thunder / Wintry Mix", "Light Snow and Sleet / Windy", "Heavy Freezing Rain", "Rain and Sleet", "Thunder / Wintry Mix / Windy", "Heavy Ice Pellets", "Freezing Drizzle", "Heavy Freezing Drizzle"], value = "Icy Conditions")

##Fair Weather 
accident_Data = accident_Data.replace(to_replace = ["Clear", "Fair", "Cloudy", "Cloudy / Windy", "Fair / Windy", "Mostly Cloudy", "Overcast", "Partly Cloudy", "Partly Cloudy / Windy", "Scattered Clouds", "Mostly Cloudy / Windy", "N/A Precipitation"], value="Fair Weather") 

##Extreme Winds
accident_Data = accident_Data.replace(to_replace = ["Tornado", "Blowing Dust", "Blowing Dust / Windy", "Sand", "Widespread Dust", "Widespread Dust / Windy", "Blowing Sand", "Sand / Dust Whirlwinds", "Funnel Cloud", "Sand / Dust Whirlwinds / Windy"], value = "Extreme Winds")


# In[10]:


accident_Data["Weather_Condition"].nunique()


# In[11]:


accident_Data["Weather_Condition"].value_counts()


# In[12]:


accident_Data["State"].value_counts(normalize=True).nlargest(10).sort_values().plot(kind = "barh")


# In[13]:


states_Data = accident_Data.groupby(["State"])


# In[14]:


sc_Data = states_Data.get_group("SC")
##Distance(mi) and TMC was not showing valuable information 
sc_Data.drop(["TMC", "Distance(mi)", "Precipitation(in)"], inplace=True, axis=1) 


# In[15]:


sc_Data["Weather_Condition"].value_counts()


# In[16]:


sc_Data["Weather_Condition"].value_counts(normalize=True).sort_values().plot(kind = "barh")


# In[58]:



sc_Data = sc_Data.groupby(["Weather_Condition"]).mean()
sc_Data.reset_index()
fig=plt.figure(figsize=(14,8), dpi= 100, facecolor='w', edgecolor='k')
plt.title("Accidents based on Weather Conditions in South Carolina")
plt.xlabel("Weather Condition")
plt.plot(sc_Data)
plt.legend(["Wind_Speed(mph)", "Visibility(mi)", "Severity"], ncol=1, loc = "upper right");


# In[14]:


sc_Data 


# In[17]:


ny_Data = states_Data.get_group("NY")
ny_Data.drop(["TMC", "Distance(mi)", "Precipitation(in)"], inplace=True, axis=1) 


# In[18]:


ny_Data["Weather_Condition"].value_counts()


# In[19]:


ny_Data["Weather_Condition"].value_counts(normalize=True).sort_values().plot(kind = "barh")


# In[42]:


ny_Data = ny_Data.groupby(["Weather_Condition"]).mean()
ny_Data.reset_index()
fig=plt.figure(figsize=(14,8), dpi= 100, facecolor='w', edgecolor='k')
plt.title("Accidents based on Weather Conditions in New York")
plt.xlabel("Weather Condition")
plt.plot(ny_Data)
plt.legend(["Wind_Speed(mph)", "Visibility(mi)", "Severity"], ncol=1, loc = "upper right");


# In[15]:


ny_Data = ny_Data.groupby("Weather_Condition").agg({"Severity": ["mean"]})


# In[16]:


ny_Data


# In[20]:


tx_Data = states_Data.get_group("TX")
tx_Data.drop(["TMC", "Distance(mi)", "Precipitation(in)"], inplace=True, axis=1)


# In[21]:


tx_Data["Weather_Condition"].value_counts()


# In[22]:


tx_Data["Weather_Condition"].value_counts(normalize=True).sort_values().plot(kind = "barh")


# In[53]:


tx_Data = tx_Data.groupby(["Weather_Condition"]).mean()
tx_Data.reset_index()
fig=plt.figure(figsize=(14,8), dpi= 100, facecolor='w', edgecolor='k')
plt.title("Accidents based on Weather Conditions in Texas")
plt.xlabel("Weather Condition")
plt.plot(tx_Data)
plt.legend(["Wind_Speed(mph)", "Visibility(mi)", "Severity"], ncol=1, loc = "upper right");


# In[29]:


tx_Data = tx_Data.groupby("Weather_Condition").agg({"Severity": ["mean"]})


# In[30]:


tx_Data


# In[ ]:




