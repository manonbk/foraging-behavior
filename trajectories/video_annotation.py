'''
Annotation of trajectory videos
'''
from movie import *
import imageio
from skimage import color
from skimage.draw import circle_perimeter, ellipse_perimeter, line, ellipse
import pandas as pd
import numpy as np
import hashlib
from PIL import Image, ImageFont, ImageDraw
from matplotlib import cm

def number_to_rgb(value, cmap='seismic'):
    '''
    Convert a number between 0 and 1 to a color, which distinguishes neatly between values lower and greater than 0.5.
    '''
    # Get the colormap
    colormap = cm.get_cmap(cmap)

    # Get the RGB color from the colormap
    rgb = colormap(value)[:3]  # Ignore the alpha value
    return rgb

def id_to_rgb(id_number):
    # Convert the ID to a string and hash it using SHA-256
    id_str = str(id_number)
    hash_object = hashlib.sha256(id_str.encode())

    # Get the first 6 characters of the hex digest (3 bytes, 2 characters each)
    hex_color = hash_object.hexdigest()[:6]

    # Convert the hex string to RGB values
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)

def circle_cells(image, table, pixel_size=1, radius=20, normalize=True):
    '''
    Adds a colored circle in the image around cells tracked in the table.
    '''
    radius = int(radius / pixel_size)
    if normalize:
        image -= image.min()
        image = ((image.astype(np.uint16)*255)//image.max()).astype(np.uint8)
    rgb_image = color.gray2rgb(image)
    for row in table.itertuples():
        xc, yc = row.x, row.y
        xc, yc = int(xc/pixel_size), int(yc/pixel_size)
        rr, cc = circle_perimeter(yc, xc, radius, shape=rgb_image.shape)
        rgb_image[rr, cc] = [255, 0, 0] #id_to_rgb(row.id) #[255, 0, 0]
    return rgb_image

def ellipse_cells(image, table, pixel_size=1, normalize=True, filled=False, unique_color=False):
    '''
    Adds an ellipse in the image around cells tracked in the table.

    TODO:
    - random color depending on id (optional)
    '''
    if normalize:
        image -= image.min()
        image = ((image.astype(np.uint16)*255)//image.max()).astype(np.uint8)
    if len(image.shape)==2:
        rgb_image = color.gray2rgb(image)
    else:
        rgb_image = image
    for row in table.itertuples():
        xc, yc, length, width, theta = row.x, row.y, row.length, row.width, row.angle
        xc, yc, length, width = int(xc/pixel_size), int(yc/pixel_size), int(length/pixel_size), int(width/pixel_size)
        if filled:
            rr, cc = ellipse(yc, xc, int(width/2), int(length/2), rotation=-theta, shape=rgb_image.shape)
        else:
            rr, cc = ellipse_perimeter(yc, xc, int(width/2), int(length/2), orientation=theta, shape=rgb_image.shape)
        if 'spin' in table.columns:
            if row.spin>0.5:
                rgb_image[rr, cc] = [255, 0, 0]
            else:
                rgb_image[rr, cc] = [0, 255, 0]
        else:
            if unique_color:
                rgb_image[rr, cc] = id_to_rgb(row.id) #[255, 0, 0]
            else:
                rgb_image[rr, cc] = [255, 0, 0]
        # Direction vector
        end_x = int(xc + length/2 * np.cos(theta))
        end_y = int(yc + length/2 * np.sin(theta))
        rr, cc = line(yc, xc, end_y, end_x)
        rr = np.clip(rr, 0, image.shape[0]-1)
        cc = np.clip(cc, 0, image.shape[1]-1)
        rgb_image[rr, cc] = [255, 0, 0] #id_to_rgb(row.id) #[255, 0, 0]

    return rgb_image

def write_movie_with_tracking(movie, filename, table, pixel_size=1, crop=False, quality=5, radius=20,
                              invert=False, ellipse=False):
    '''
    Adds a colored ellipse around cells tracked in the table.

    if `crop` is True, show only the frame interval where there are cells, and crop the region where there are cells.
    '''
    writer = imageio.get_writer(filename, fps=movie.fps, quality=quality)
    #if 'z' in table.columns:  # write z information
    font = ImageFont.truetype("Arial.ttf", 30)
    if crop:
        # Select frames
        f1, f2 = table['frame'].min(), table['frame'].max()
        if not(isinstance(crop, bool)): # Assume it's a number of frames
            f1 -= crop
            f2 += crop
        movie.start = max(0, int(f1))
        movie.end = int(f2)+1
        movie.rewind()
        x1, x2 = int((table['x'].min()-radius)/pixel_size), int((table['x'].max()+radius)/pixel_size)
        y1, y2 = int((table['y'].min()-radius)/pixel_size), int((table['y'].max()+radius)/pixel_size)
        #w, h = x2-x1, y2-y1
        #x1 -= int(w/3)
        #x2 += int(w/3)
        #y1 -= int(h/3)
        #y2 += int(h/3)
        #x1 = np.clip(x1, 0, None)
        #y1 = np.clip(y1, 0, None)
        if (x1<0):
            x1 = 0
        if (y1<0):
            y1 = 0
        x_slice = slice(x1,x2+1)
        y_slice = slice(y1, y2 + 1)
    else:
        x_slice = y_slice = slice(0, -1)
    for frame in movie.frames():
        if invert:
            frame = 255-frame
        n = movie.position-1
        #image = circle_cells(frame, table[table['frame'] == n], pixel_size=pixel_size, radius=radius)
        if ellipse:
            image = ellipse_cells(frame, table[table['frame'] == n], pixel_size=pixel_size)
        else:
            image = color.gray2rgb(frame)
        if 'z' in table.columns: # write z information
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)
            for row in table[table['frame'] == n].itertuples():
                draw.text((int((row.x + radius)//pixel_size),
                           int((row.y/pixel_size))), str(int(row.z)), fill="red", font=font)
            # Convert back to a NumPy array
            image = np.array(pil_image)
        else:
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)
            for row in table[table['frame'] == n].itertuples():
                draw.text((int((row.x + radius)//pixel_size),
                           int((row.y/pixel_size))), str(int(row.id)), fill="red", font=font)
            # Convert back to a NumPy array
            image = np.array(pil_image)

        writer.append_data(image[y_slice, x_slice])
    writer.close()

def write_tracking_movie(filename, table, pixel_size=1, fps=10., quality=5, radius=100, filled=False):
    '''
    Plots a colored ellipse around cells tracked in the table.
    '''
    writer = imageio.get_writer(filename, fps=fps, quality=quality)
    if 'z' in table.columns:  # write z information
        font = ImageFont.truetype("Arial.ttf", 30)
    nframes = int(table['frame'].max())
    width = int((table['x'].max() + radius)/pixel_size)
    height = int((table['y'].max() + radius)/pixel_size)
    for n in range(int(table['frame'].min()), nframes+1):
        if (n % 100) == 0:
            print(n, "/", nframes+1)
        frame = np.ones((height, width), dtype=np.uint8)*255
        image = ellipse_cells(frame, table[table['frame'] == n], pixel_size=pixel_size, normalize=False, filled=filled)
        if 'z' in table.columns: # write z information
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)
            for row in table[table['frame'] == n].itertuples():
                draw.text((int((row.x + radius)//pixel_size),
                           int((row.y/pixel_size))), str(int(row.z)), fill="red", font=font)
            # Convert back to a NumPy array
            image = np.array(pil_image)

        writer.append_data(image)
    writer.close()
