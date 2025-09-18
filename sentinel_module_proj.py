from geopy.geocoders import Nominatim
import geopy.distance as distance
from shapely import geometry

from shapely.ops import transform

from sentinelsat import SentinelAPI
from datetime import date

import utm

import zipfile
import re

from pathlib import Path

import rasterio
from rasterio.enums import Resampling
from rasterio import mask

import numpy as np

from matplotlib import pyplot as plt

from PIL import Image
from PIL.ImageOps import equalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


def area_of_interest(city, country, user_agent):
    geolocator = Nominatim(user_agent=f"{user_agent}")
    location = geolocator.geocode(f"{city}, {country}")
    return location


def distance_direction_of_area(location, Km, shape):
    d = distance.distance(kilometers=Km)
     
    # Use 31 bearings for a smoother circle, 31 for close the circle
    if shape == "circle":
        bearings = [i * 360 / 30 for i in range(31)]  
    elif shape == "rectangle":
        bearings = [45, 135, 225, 315, 405]
    elif shape == "triangle":
        bearings = [0, 120, 240]
    else:
        raise ValueError("Invalid shape specified.")

    # create vertex
    vertexes = [d.destination(point=location.point, bearing=x) for x in bearings]

    # create polygon object
    poly = geometry.Polygon(vertexes)
    
    return bearings, vertexes, poly


def reverse_lat_lon(poly):
    tmp_fun = lambda x, y, z: (y, x)
    poly1 = transform(tmp_fun, poly)
    return poly1


def select_the_product(user, password, start_date, end_date,
                       max_cloud, platformname, poly):
    
    api = SentinelAPI(user, password)

    date_range = (start_date, end_date)
    cloud_range = (0, max_cloud)

    products = api.query(poly, date=date_range,
                         cloudcoverpercentage=cloud_range, platformname=platformname)

    return api, products


def show_the_products(products):
    for key, val in products.items():
        filename = val.get('filename')
        cloudcover = val.get('cloudcoverpercentage')
        uuid = val.get('uuid')
        
        print("Filename:", filename)
        print("Cloudcover Percentage:", cloudcover)
        print("UUID:", uuid)
        print('\n')


def download_the_image(api, uuid, directory_path):
    api.download(uuid, directory_path=directory_path)

    
def get_bandlist_from_zip(path_of_zip):
    with zipfile.ZipFile(path_of_zip) as zp:
        file_list = zp.filelist

    return file_list


def filter_file_list(file_list):
    bands, bands20, bands60 = {}, {}, {} 
    
    # iterate over all files and access to the name
    for file in file_list:
        try:
            name = file.filename
        except:
            name = file.name
        # if filename matches the expression, extract the band num and resolution
        if re.search(r'B.{2}_\d\dm.jp2$', name): 
            name = name.split('/')[-1]

            band_no, resol = re.search(r'(B.{2})_(\d\d)m',name).groups()

            # we don't want: 1, 9, 10 -> skip; otherwise we add in the bands dict
            if band_no in ('B01', 'B09', 'B10'):
                continue

            if resol == '10':
                bands[band_no] = file
            elif resol == '20':
                bands20[band_no] = file
            else:
                bands60[band_no] = file
    
    # iterate over 20, 60 and add only the bands not already in dict           
    for data in (bands20, bands60):
        for key, val in data.items():
            if key not in bands.keys():
                bands[key] = val
            
    return bands


def extract_bands(path_of_zip, out_path, bands_dict):

    with zipfile.ZipFile(path_of_zip) as zp:
        for item in bands_dict.values():
            zp.extract(item, out_path)

            
def transform_polygon(poly):
            
    x, y = [], []
    # iterate over the the coordinates of the polygon and for each pair of lat and long
    # call the utm... to convert to UTM coordinates then take only 2 values
    for (lat, lon) in zip(*poly.boundary.coords.xy):
        a,b,c,d = utm.from_latlon(lat, lon)
        x.append(a)
        y.append(b)
    
    # return the tuple of x and y lists
    return x, y


def save_polygon_in_utm(x, y, out_path):
    
    # create 2 lists, round the values, converting in strings and zip lists together (tuple)
    x1 = [str(round(a, 2)) for a in x] 
    y1 = [str(round(a, 2)) for a in y]
    xy = list(zip(x1, y1)) 
    # create a list with a comma separater
    xy = [','.join(it) for it in xy]
    
    # write the strings in xy list in the file
    with open(out_path, 'w') as f:
        f.write('\n'.join(xy))
        
               
def read_polygon(in_path):
    
    with open(in_path) as f:
        poly = f.read()
    
    # splits poly strings into a list of lines
    poly = [line.split(',') for line in poly.splitlines()]
    # iterate over the lists in the poly and convert 2 elements in float and create a tuple that contains it
    poly = [(float(it[0]), float(it[1])) for it in poly]
    # create a poly obj using the constructor
    poly = geometry.Polygon(poly)
    return poly


def get_file_list(in_path):
    
    path = Path(in_path)
    # call glob on the path obj (iterator over all files) then converted to list
    file_list = list(path.glob('**/*'))
    return file_list


def get_profile(bands, band_name):
    
    with rasterio.open(bands[band_name]) as src:
        profile = src.profile

    return profile


def resample_all_image(bands, profile):

    # iterate over dictionary and find the file (20m) -> create a new file with extension .tiff and name 10m
    for key, file in bands.items():
        if re.search(r'(20m)', file.name):
            new = re.sub(r'(20m)', '10m', str(file))
            new = Path(new).with_suffix('.tiff')

            with rasterio.open(file) as src:
                
                # open the file and reads the 1st band and set the parameters
                # the resampled data will have the same shape of the target obj
                ar = src.read(1,
                              out_shape=(profile['count'],
                                         profile['height'],
                                         profile['width']),
                              resampling=Resampling.nearest)
            # resampling by the nearest-neighbor (interpolation)
            
            # open the file in w, and writes the ar in the 1st band
            # profile defined before
            with rasterio.open(new, 'w', **profile) as dst:
                dst.write(ar, 1)
            
            print(f'Saved: {new.name}')
            
            # ar: After the src.read() operation, ar will contain the pixel values
            # of the specified band from the raster file
            
            # The ar variable holds the pixel values for a single band extracted
            # from the raster file using the src.read() method -> specific aspect 
            
    return new


def get_metadata(new, poly):
    with rasterio.open(new) as src:
        # return a masked version of raster data!
        out_image, out_transform = mask.mask(src, [poly], crop=True)
        # withdraw the meta
        meta = src.meta
        return out_image, out_transform, meta


def order_bands(bands):

    band_dict = {}

    # for each band (in list) use the re.search to extract band name... if B8A -> 81
    # otherwise set key to * 10
    for b in bands:
        band_name = re.search(r'(B.{2})', b.name)[0]
        if band_name == 'B8A':
            key = 81
        else:
            #number multiply by 10 (2nd and 3rd char of the b_name)
            key = int(band_name[1:])*10

        # adds key-value to the dict, value = tuple (band_name and band obj -path)
        band_dict[key] = (band_name,b)

    return band_dict


def write_image(sent_path, band_dict, poly, meta):
    
    # open the image and meta dict to w and iterate by the dictionary of the bands (sorted) 
    
    with rasterio.open(sent_path, 'w', **meta) as dst:
         # iterate over band_dict using enum for sorted the keys
        for i, key in enumerate(sorted(band_dict), 1):
            # retrieve tha path of current band
            im_path = band_dict[key][1]

            # open it, create a masked version of the raster
            with rasterio.open(im_path) as src:
                out_image, out_transform = mask.mask(src, [poly], crop = True)
            
            # retrieve the band name
            band_name = band_dict[key][0]
            
            # write the masked data to the out in the appropiate channel (i)
            dst.write(out_image[0, ...], i);
            
            # the direct write like descriptions not works! set the value of name
            dst.update_tags(i, name=band_name)
            

def open_the_image_and_print_description(path):
    
    with rasterio.open(path) as src:
        # read image data into an array
        ar = src.read()
        
        # Retrieve band descriptions from tags iterate to count (num bands)
        band_descriptions = []
        for i in range(src.count +1):
            tags = src.tags(i)
            if 'name' in tags:
                band_descriptions.append(tags['name'])

        print(band_descriptions)
        # Sequence of the band names
        
    return ar


def rearrange_dimension(sent_path):
            
    with rasterio.open(sent_path) as src:
        ar = src.read()

    ar1 = np.transpose(ar,(1,2,0))
    return ar1


def min_max_scale(ar, min_val = 0, max_val = 1, uint8=False):
    # create a new array with shape of ar
    res = np.zeros_like(ar, dtype=np.float32)
    
    # iterate over the last dimension of ar (bands) and scales the values
    for i in range(ar.shape[-1]):
        band = ar[..., i].copy()
        scale = (band - band.min()) / (band.max() - band.min())
        # stored the scaled values
        res[..., i] = scale
    
    # transformed in spec range
    res = res * (max_val - min_val) + min_val
    
    if uint8:
        # scaling on 0-255
        res = (res * 255).astype(np.uint8);
    
    return res


def display_image_bands(ars, bands):
    image = Image.fromarray(ars[:, :, bands])
    # histogram equalization to enhance its contrast
    image = equalize(image)
    return image


def reshape_img_to_table(ar):
    # reshape from 3D to 2D (first 2D flatten in 1D)
    # size of 1D aut calc based on the size of the input array (2nd D)
    data = np.reshape(ar, (-1, ar.shape[-1]))
    return data


def reshape_table_to_img(ar, labels):
    # reshape the labels to have the same first 2D of ar
    res = labels.reshape(*ar.shape[:2])
    return res


def rescale(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    MinMaxScaler()
    # rescale each feature individually
    data_scaled= scaler.transform(data)

    return data_scaled


def kmeans(data, k, seed=None, max_iterations=100):
    
    # Initialize k centroids
    rng = np.random.default_rng(seed)
    centroids = data[rng.choice(range(data.shape[0]), size=k, replace=False)]

    # Assign data points to clusters
    for i in range(max_iterations):
        
        # calculate the distance between each datapoint and each centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=-1)
        # find the index of the closest centroid 
        labels = np.argmin(distances, axis=-1)

        # Update centroids (mean of all data points assigned) -> with label equal to cluster index
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # Check convergence -> if centroids stays still -> stop algorithm
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


def kmeans_plus_plus(data, k, seed=None, max_iterations=100):
    
    # Choose randomly 1 centroid
    rng = np.random.default_rng(seed)
    centroids = [data[rng.choice(range(data.shape[0]))]]
    
    # find the other centroids based on the square distance from data-points
    for _ in range(k - 1):
        # distance from each data point to its nearest centroid then find the minimum
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=-1)
        min_distances = np.min(distances, axis=-1)
        # points far away from there nearest centroid -> higher probability to be choosen
        probabilities = min_distances / np.sum(min_distances)
        new_centroid = data[rng.choice(range(data.shape[0]), p=probabilities)]
        centroids.append(new_centroid)
    centroids = np.array(centroids)

    for i in range(max_iterations):
        # Assign data points to clusters (based on distance), take the index of centroids and stored it in labels
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=-1)
        labels = np.argmin(distances, axis=-1)

        # Update centroids (calculate the mean of all data points assigned)
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # Check convergence -> if centroids stays still -> stop algorithm
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


def kmeans_sklearn(data, n_clusters, n_init):
    kmn = KMeans(n_clusters=n_clusters, n_init=n_init).fit(data)
    return kmn


# searching for a the rate where descrese in inertia slows down
def elbow(c, data, verb= False):
    res = []
    # for each k, the function creates an istance of KMeans with K clusters 
    # -> individually test 
    for k in c:
        kmeans = KMeans(n_clusters=k, n_init =1, random_state=0).fit(data)
        # sum of square distances of points to closest center
        res.append((k, kmeans.inertia_))
        
    return res
    

def save_rgb(ars, out_path):
    rgb = Image.fromarray(ars[:, :, [2, 1, 0]])
    rgb.save(out_path)
    return rgb


def open_image(ad1):
    im = Image.open(ad1)
    return im

def img_to_array(im_label):
    ar = np.array(im_label)
    return ar


def random_forest(ar_img, ar_label, n_estimators=10, ignore_background=False):
    # Create an instance of RandomForestClassifier
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators)

    if ignore_background:
        # Create a mask to select only the labeled pixels that are not in the background class
        mask = ar_label != 0

        # Reshape the labeled data and the corresponding labels
        X_labeled = ar_img[mask].reshape(-1, ar_img.shape[-1])
        y_labeled = ar_label[mask].reshape(-1)
    else:
        # Reshape the labeled data and the corresponding labels into 2D (pixel, channel)
        X_labeled = ar_img.reshape(-1, ar_img.shape[-1])
        # into 1D with shape num_pixel (class assigned)
        y_labeled = ar_label.reshape(-1)

    # Fit the Random Forest classifier
    rf_classifier.fit(X_labeled, y_labeled)

    # Reshape the image data to (num_pixels, num_channels)
    X_unlabeled = ar_img.reshape(-1, ar_img.shape[-1])

    # Predict labels for the unlabeled data
    y_pred = rf_classifier.predict(X_unlabeled)

    # Reshape the predicted labels to match the dimensions of the image
    predicted_labels = y_pred.reshape(ar_label.shape)
    
    return predicted_labels


