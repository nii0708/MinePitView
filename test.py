import streamlit as st
import pandas as pd
import utils
from streamlit_folium import folium_static
from shapely.geometry import LineString
import numpy as np
import geopandas as gpd

st.set_page_config(layout="wide")

if 'button' not in st.session_state:
    st.session_state.button = False

def click_button():
    st.session_state.button = not st.session_state.button

st.title("Display Block Model")

uploaded_block_model = st.file_uploader("Upload BlockModel CSV", type=["csv"])
uploaded_elevation = st.file_uploader("Upload TopoLidar str", type=["str"])

projection = st.text_input("Projection System")

if uploaded_block_model is not None and uploaded_elevation is not None:
    try:
        df = pd.read_csv(uploaded_block_model, skiprows=[1,2,3])
        df = df[['centroid_x','centroid_y','centroid_z','dim_x','dim_y','dim_z','rock']]

        df_elevation = pd.read_csv(uploaded_elevation,skiprows=[1],names=['name','x','y','z','end'])
        df_elevation = df_elevation.drop(columns=['name','end']).dropna().iloc[:-2]
        df_elevation['dim_y'] = df['dim_y'].iloc[0]
        df_elevation['dim_x'] = df['dim_x'].iloc[0]
        st.button('Show Topdown View', on_click=click_button)
        
        if st.session_state.button:
            gdf = utils.create_polygon_from_coords_and_dims(df, 'centroid_x', 'centroid_y', 'dim_x', 'dim_y')
            gdf = gdf.set_crs(projection,allow_override=True)
            # gdf.drop_duplicates('geometry').explore(tiles='Esri.WorldImagery')
            top_down_view  = gdf.drop_duplicates('geometry').explore()
            folium_static(top_down_view)
            if top_down_view is not None:
                lon_start = st.number_input("start point longitude")
                lat_start = st.number_input("start point latitude")
                lon_end = st.number_input("end point longitude")
                lat_end = st.number_input("end point latitude")
                lower_point = [lon_start,lat_start]
                upper_point = [lon_end,lat_end]
                line = LineString([lower_point,upper_point])
                if st.button("Display CrossSection View"):
                    gdf_intersections,line_point = utils.intersect_polygons_with_line(gdf, line, lower_point)
                    line_point = np.array(utils.remove_duplicate_sublists_set(line_point))
                    z_dim = df['dim_x'].iloc[0]
                    gdf_intersections['geometry'] = gdf_intersections.apply(lambda x: utils.create_grid_polygon(x['geometry'], x['z_centroid'], z_dim = 1),axis=1)
                    x_line,y_line,z_section = utils.get_elevation(df_elevation,line_point)
                    line_point = np.sort(np.array([x_line,y_line]).T, axis=0, kind=None, order=None)
                    lon_elevation = utils.get_coordinates(utils.rotate_linestring_to_vertical(LineString([tuple(coord) for coord in line_point]),lower_point))[:,0]
                    line_elevation = []
                    for index in range(1,len(line_point)):
                        line_elevation.append(LineString(
                            [
                                [lon_elevation[index-1],z_section[index-1]],
                                [lon_elevation[index],z_section[index]]
                            ]))
                    gdf_intersections = gdf_intersections.set_crs("EPSG:32652",allow_override=True)
                    gdf_elevation = gpd.GeoDataFrame(geometry=line_elevation)
                    gdf_elevation = gdf_elevation.set_crs("EPSG:32652",allow_override=True)

                    cross_view_block = gdf_intersections.explore(column="rock", cmap="seismic",tiles=None)

                    cross_view = gdf_elevation.explore(m=cross_view_block,tiles=None)
                    folium_static(cross_view)
        else:
            st.info("Input your projection.")

    except Exception as e:
        st.error(f"Error reading CSV: {e}")

else:
    st.info("Please upload a your file.")