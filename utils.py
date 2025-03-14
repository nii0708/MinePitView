import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point, GeometryCollection
import numpy as np
from shapely import get_coordinates
import math
from scipy.spatial import cKDTree
from scipy.interpolate import griddata


def create_polygon_from_coords_and_dims(df, long_col, lat_col, width_col, length_col):
    """
    Creates a GeoDataFrame with polygons from a pandas DataFrame containing coordinates and dimensions.

    Args:
        df (pd.DataFrame): DataFrame with columns for longitude, latitude, width, and length.
        long_col (str): Name of the longitude column.
        lat_col (str): Name of the latitude column.
        width_col (str): Name of the width column.
        length_col (str): Name of the length column.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with polygons.
    """

    polygons = []
    for index, row in df.iterrows():
        lon = row[long_col]
        lat = row[lat_col]
        width = row[width_col]
        length = row[length_col]

        # Calculate the corner points of the rectangle
        # Simple rectangle calculation (assuming orientation is aligned with axes)
        # This is a simplification. For rotated rectangles, you'd need more complex calculations.
        half_width = width / 2
        half_length = length / 2

        p1 = (lon - half_width, lat - half_length)
        p2 = (lon + half_width, lat - half_length)
        p3 = (lon + half_width, lat + half_length)
        p4 = (lon - half_width, lat + half_length)

        polygon = Polygon([p1, p2, p3, p4])
        polygons.append(polygon)

    gdf = gpd.GeoDataFrame(df, geometry=polygons)
    return gdf

def rotate_linestring_to_vertical(line, rotation_point):
    """
    Rotates a Shapely LineString around a custom point to be vertical.

    Args:
        line (shapely.geometry.LineString): The input LineString.
        rotation_point (tuple): The (x, y) coordinates of the rotation point.

    Returns:
        shapely.geometry.LineString: The rotated LineString.
    """

    if not isinstance(line, LineString):
        raise ValueError("Input must be a Shapely LineString.")

    if line.is_empty:
        return line

    coords = list(line.coords)

    if len(coords) < 2:
        return line #no rotation needed.

    rp_x, rp_y = rotation_point

    # Calculate the angle of the line
    x1, y1 = coords[0]
    x2, y2 = coords[-1]
    angle_radians = math.atan2(y2 - y1, x2 - x1)

    # Calculate the rotation angle needed to make it vertical
    rotation_angle_radians = -angle_radians

    rotated_coords = []
    for x, y in coords:
        # Translate to origin (rotation point)
        x_translated = x - rp_x
        y_translated = y - rp_y

        # Rotate
        x_rotated = x_translated * math.cos(rotation_angle_radians) - y_translated * math.sin(rotation_angle_radians)
        y_rotated = x_translated * math.sin(rotation_angle_radians) + y_translated * math.cos(rotation_angle_radians)

        # Translate back
        x_final = x_rotated + rp_x
        y_final = y_rotated + rp_y

        rotated_coords.append((x_final, y_final))

    return LineString(rotated_coords)

def intersect_polygons_with_line(gdf_polygons, line, lower_point, proj = "EPSG:32652"):
    """
    Finds the intersection of polygons in a GeoDataFrame with a Shapely LineString.

    Args:
        gdf_polygons (gpd.GeoDataFrame): GeoDataFrame containing polygons.
        line (shapely.geometry.LineString): Shapely LineString to intersect with.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the intersection geometries.
        If no intersections are found, returns an empty geodataframe.
    """

    intersections   = []
    z_centroids     = []
    rocks           = []
    line_point      = []
    for index,row in gdf_polygons.iterrows():
        polygon     = row['geometry'] 
        rock        = row['rock']
        z_centroid  = row['centroid_z']
        intersection = line.intersection(polygon)
        if not intersection.is_empty and not isinstance(intersection, GeometryCollection): #Geometry collection with no geometries is considered empty
            # print(get_coordinates(intersection))
            line_point += get_coordinates(intersection).tolist()
            intersections.append(rotate_linestring_to_vertical(intersection,lower_point))
            # intersections.append(intersection)
            z_centroids.append(z_centroid)
            rocks.append(rock)

    if intersections:
        gdf_intersections = gpd.GeoDataFrame({'z_centroid'  :z_centroids,
                                              'rock'        :rocks}
                                             ,geometry=intersections)
        # gdf_intersections = gdf_intersections.set_crs(proj,allow_override=True)
        return gdf_intersections,line_point
    else:
        return gpd.GeoDataFrame(geometry=[]),None #Return empty geodataframe if no intersections.
    
def remove_duplicate_sublists_set(list_of_lists):
    """Removes duplicate sublists using a set (order not guaranteed)."""
    unique_tuples = set(tuple(sublist) for sublist in list_of_lists)
    return [list(t) for t in unique_tuples]

def create_grid_polygon(line, z_centroid, z_dim):
    """
    Creates a Shapely Polygon (grid cell) from a horizontal LineString.

    Args:
        line (shapely.geometry.LineString): The horizontal LineString representing the length.
        z_centroid (float): The z-coordinate (center) of the grid cell.
        z_dim (float): The height of the grid cell.

    Returns:
        shapely.geometry.Polygon: The grid cell Polygon.
    """

    if not isinstance(line, LineString):
        raise ValueError("Input must be a Shapely LineString.")

    if line.is_empty:
      return None

    coords = list(line.coords)
    if len(coords) < 2:
      return Polygon(Point(coords[0]).buffer(z_dim/2)) #create circle if only one point.

    x1, y1 = coords[0]
    x2, y2 = coords[-1]

    # Calculate the half-height for the polygon
    half_height = z_dim #/ 2.0

    # Create the polygon coordinates
    polygon_coords = [
        (x1, y1, z_centroid - half_height),
        (x2, y2, z_centroid - half_height),
        (x2, y2, z_centroid + half_height),
        (x1, y1, z_centroid + half_height),
        (x1, y1, z_centroid - half_height),  # Close the polygon
    ]
    # print(z_dim)
    # print(half_height)
    # Project the 3D coordinates to 2D for Shapely Polygon creation
    polygon_2d_coords = [(x, z) for x, y, z in polygon_coords]

    return Polygon(polygon_2d_coords)

def get_elevation(df,line_point):
    # Load CSV data (Assuming CSV columns: X, Y, Z)
    # csv_file = "points.csv"  # Change this to your file path

    # Extract X, Y, Z
    x = df['y'].values
    y = df['x'].values
    z = df['z'].values

    # Define the cross-section line (start & end points)
    # p1 = lower_point  # (X, Y) start
    # p2 = upper_point  # (X, Y) end

    p1 = (line_point[:,0].min(), line_point[:,1].min())  # (X, Y) start
    p2 = (line_point[:,0].max(), line_point[:,1].max())  # (X, Y) end

    # Generate points along the section line
    # Number of points to interpolate along the line
    # x_line = line_point[:,0]#
    x_line = np.linspace(p1[0], p2[0], 100)
    # y_line = line_point[:,1]#
    y_line = np.linspace(p1[1], p2[1], 100)
    num_samples = len(x_line)
    # num_samples = 200

    # Find nearest points from the dataset to the cross-section line
    tree = cKDTree(np.column_stack((x, y)))
    _, idx = tree.query(np.column_stack((x_line, y_line)), k=1)

    # # Get the corresponding Z values
    z_section = z[idx]

    # Interpolation for a smoother profile
    # z_interp = griddata((x, y), z, (x_line, y_line), method='linear')
    return x_line,y_line,z_section

