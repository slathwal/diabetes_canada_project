import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/shefalilathwal/Documents/Hands_on_machine_learning/projects/data/pumf_cchs.csv")  # Update path as needed
    df = df[df["CCC_095"].isin([1.0, 2.0])] # only include people who have diabetes
    df = df[~(df["DHHGAGE"] == 1)] # only include adults over the age of 18

    geo_mapping = dict([("NEWFOUNDLAND AND LABRADOR", 10),
    ("PRINCE EDWARD ISLAND", 11),
    ("NOVA SCOTIA", 12),
    ("NEW BRUNSWICK", 13),
    ("QUEBEC", 24),
    ("ONTARIO", 35),
    ("MANITOBA" , 46),
    ("SASKATCHEWAN" ,47),
    ("ALBERTA", 48),
    ("BRITISH COLUMBIA" , 59),
    ("YUKON/NORTHWEST/NUNAVUT TERRITORIES",60)]
    )
    geo_mapping_reverse = {v: k for k,v in geo_mapping.items()}
    df["province"] = [geo_mapping_reverse[num] for num in df["GEOGPRV"]]

    # change the values in the diabetes columns (0 = no, 1 = yes)
    df["diabetes"] =  df['CCC_095'].replace({2.0: 0.0})

    # Change the values in sex column
    df["sex"] = df["DHH_SEX"].replace({1.0: "Male", 2.0: "Female"})

    # Change the value in age column
    age_mapping = {"12-17": 1, "18-34": 2, "35-49": 3, "50-64": 4, "65+": 5}
    age_mapping_reverse = {v: k for k, v in age_mapping.items()}
    df["age"] = df["DHHGAGE"].replace(age_mapping_reverse)

    # Select the column to use in the dashboard
    df = df[["GEOGPRV", "GEODGHR4", "DHHGAGE", "DHH_SEX", "CCC_095", "age", "sex", "province", "diabetes"]]

    gdf = gpd.read_file("/Users/shefalilathwal/Documents/Hands_on_machine_learning/projects/lpr_000b16a_f/lpr_000b16a_f.shp")  # .shp file for provinces
    
    
    return df, gdf

df, gdf = load_data()
#print(gdf)

st.set_page_config(page_title="Canadian Diabetes Dashboard", layout="wide")

# ======================
# Sidebar Filters (Optional)
# ======================
st.sidebar.header("Filter Data")
selected_sex = st.sidebar.multiselect("Sex", df["sex"].unique(), default=df["sex"].unique())
selected_provinces = st.sidebar.multiselect("Province", df["province"].unique(), default=df["province"].unique())

filtered_df = df[df["sex"].isin(selected_sex) & df["province"].isin(selected_provinces)]

# ======================
# Key Metrics
# ======================
st.title("🩺 Canadian Diabetes Dashboard (2019-2020)")

#col1, col2, col3 = st.columns(3)
#col4, col5 = st.columns(2)
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("Total Respondents")
    st.markdown(f"# {len(filtered_df)}")

with col2:
    st.markdown("Total with Diabetes")
    st.markdown(f'# {int(filtered_df["diabetes"].sum())}')

with col3:
    st.markdown("Provinces and Territories Covered")
    prov_counts = filtered_df.groupby("province")["diabetes"].count()
    st.markdown(f"# {len(prov_counts)}")

with col4:
    st.markdown("Respondents by Sex")
    sex_counts = filtered_df["sex"].value_counts()
    sex_display = "<br>".join([f"<b>{sex}</b>: {count}" for sex, count in sex_counts.items()])
    st.markdown(sex_display, unsafe_allow_html=True)
    #st.metric("Respondents by Sex", md)

with col5:
    #st.metric("Respondents by Age Groups", ", ".join(f"{age}: {count}" for age, count in filtered_df['age'].value_counts().sort_index().items()))
    st.markdown("Respondents by Age")
    age_counts = filtered_df["age"].value_counts().sort_index()
    age_display = "<br>".join([f"<b>{age}</b>: {count}" for age, count in age_counts.items()])
    st.markdown(age_display, unsafe_allow_html=True)

# ======================
# Map: Diabetes % by Province
# ======================
st.subheader("📍 Diabetes Prevalence by Province")

diabetes_by_province = filtered_df.groupby("GEOGPRV")["diabetes"].mean().reset_index()
diabetes_by_province.columns = ["GEOGPRV", "diabetes_rate"]
geo_mapping = dict([("NEWFOUNDLAND AND LABRADOR", 10),
    ("PRINCE EDWARD ISLAND", 11),
    ("NOVA SCOTIA", 12),
    ("NEW BRUNSWICK", 13),
    ("QUEBEC", 24),
    ("ONTARIO", 35),
    ("MANITOBA" , 46),
    ("SASKATCHEWAN" ,47),
    ("ALBERTA", 48),
    ("BRITISH COLUMBIA" , 59),
    ("YUKON/NORTHWEST/NUNAVUT TERRITORIES",60)]
    )
geo_mapping_reverse = {v: k for k,v in geo_mapping.items()}
diabetes_by_province["province"] = diabetes_by_province["GEOGPRV"].replace(geo_mapping_reverse)
geo_merged = gdf.merge(diabetes_by_province, left_on = pd.to_numeric(gdf["PRIDU"]), right_on="GEOGPRV", how = "outer")
territory_value = geo_merged[geo_merged["PRIDU"]=="60"]["diabetes_rate"].values[0]
geo_merged.loc[geo_merged["PRIDU"].isin(["61", "62"]),"diabetes_rate"] = territory_value
geo_merged["diabetes_rate"] = geo_merged["diabetes_rate"]*100
fig, ax = plt.subplots(figsize=(12, 8))
geo_merged.plot(column='diabetes_rate', legend=True, cmap='bwr', ax = ax)
ax.set_title('Percentage of survey respondents with diabetes')
ax.set_axis_off()
st.pyplot(fig)

# ======================
# Diabetes by Sex in Province
# ======================
st.subheader("👥 Diabetes Prevalence by Sex in Each Province")

# Helper to get custom colors from colormap
def get_cmap_colors(cmap_name, n):
    cmap = cm.get_cmap(cmap_name)
    return [cmap(i / (n - 1)) for i in range(n)]



sex_group = filtered_df.groupby(["province", "sex"])["diabetes"].mean().reset_index()
sex_group["diabetes"] = sex_group["diabetes"]*100
if "Male" in sex_group["sex"].unique():
    sex_group = sex_group.pivot(index = "province", columns = "sex", values = "diabetes").sort_values(by = "Male")
else:
    sex_group = sex_group.pivot(index = "province", columns = "sex", values = "diabetes").sort_values(by = "Female")


print(sex_group)
# Colors from 'bwr'
if len(sex_group.columns) < 2:
    sex_colors = get_cmap_colors('bwr', 2)[0]
else:
    sex_colors = get_cmap_colors('bwr', len(sex_group.columns))

fig, ax = plt.subplots(figsize=(12, 8))
sex_group.plot.barh(ax = ax, color = sex_colors)
ax.set_xlabel("Diabetes Prevalence")
ax.set_title("Diabetes Prevalence by Sex in Each Province")
ax.legend(title="Sex")
st.pyplot(fig)

# ======================
# Diabetes by Age Group
# ======================
st.subheader("📊 Diabetes Prevalence by Age Group")

age_group = filtered_df.groupby("age")["diabetes"].mean().reset_index()
age_group["diabetes"] = age_group["diabetes"]*100
age_colors = get_cmap_colors('bwr', len(age_group))

fig, ax = plt.subplots(figsize=(10, 5))
age_group.plot.barh(y = "diabetes", x = "age", ax=ax, color = age_colors[0])
ax.set_xlabel("Diabetes Prevalence")
ax.set_title("Diabetes Prevalence by Age Group")
st.pyplot(fig)
