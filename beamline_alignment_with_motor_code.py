from scipy.ndimage import median_filter
import time
import epics as PyEpics
import plotly.express as px
import math
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import cv2
import os
import torch
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import plotly.graph_objs as go
from scipy.optimize import curve_fit
from PIL import Image
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

#The sam_checkpoint will need to be changed 
sam_checkpoint = os.path.expanduser('~/opt/auto_beamline_alignment_tomo/model/sam_vit_h_4b8939.pth')
model_type = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


####################################################
# all_image_masks hold masks for each image, 
# edges hold the left edge and right edge of the beam selected,
# directory_input, angle_rotation, and time_exposure hold the info that the user typed it on the web
#           -directory_input is used to find the directory that holds the images, 
#           -angle_rotation is how much the object will be rotated by
#           -time_exposure is how much the camera will sleep for when it takes the pic
# normalized_images holds all the normalized images
# height and width hold the dimensions of the image used for sam gridpoint
# answer_normalization holds the answer on whether the user selected normalization
# image_0 will hold the image with just the beam and no object
# all the clicks are to keep track of how many times the user clicks the buttons
# style is just to change the style of text and fig is to change the background to black of the images
#pixel_size is used for calculations when the start theta, offset, and radius given
#camera_type, file_type, and cam_name are all names used for the motors that the user types in 
#pname and froot are variables used for the file names for the motors
#mtr_samXE , mtr_samYE , mtr_aeroXE , mtr_samOme are names for the motors that the user types in 
#off_set, rad_ius, and start_theta hold the values that the func_fitting function calculates 
#x_universal and y_universal are the coordinates that the user clicked on to pinpoint the object in the image

all_image_masks = []
edges = []
directory_input = ''
angle_rotation = 0.0
time_exposure = 0.0
normalized_images = []
height = None
width = None
answer_normalization = None
clicks_tracker = 0
click_counter = 0
n_clicks_tracker = 0
im_0 = None
restart_clicks_tracker = 0
pixel_size = 0.0
camera_type = ''
file_type = ''
cam_name = ''
cam = ''
pname = None
froot = None
mtr_samXE = None
mtr_samYE = None
mtr_aeroXE = None
mtr_samOme = None
off_set = None
rad_ius = None
start_theta = None
x_univesal = None
y_universal = None


style = {'color' : '#7FDBFF', 'fontSize': 15, 'fontFamily' : 'OCR A Std, monospace'}
fig = {'layout': {
         'plot_bgcolor': 'rgb(40, 40, 40)','paper_bgcolor': 'rgb(40, 40, 40)','xaxis': {'gridcolor': 'white'},       
         'yaxis': {'gridcolor': 'white'}}} 

####################################################
#This function takes it normalize which is the answer to whether the user selects normalization and 
# the image that the normalization will happen on. If user selects no normalization, then the image 
# will be divided by 0. A cv2 color map is added as well to help SAM recognize objects better. The 
# normalized image is returned 
def normalization(normalize, image ,image_0):
    if normalize:
        first_ch0 = np.array(image_0).astype(np.float32)
    else:
            first_ch0 = np.ones(np.array(image_0).shape).astype(np.float32)       
    image_ch0 = np.array(image).astype(np.float32)
    norm_image = image_ch0/first_ch0
    norm_image = median_filter(norm_image,3)
    norm_image = cv2.normalize(norm_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    norm_image = norm_image.astype(np.uint8)
    norm_image = cv2.cvtColor(norm_image, cv2.COLOR_GRAY2RGB)
    norm_image = cv2.applyColorMap(norm_image, cv2.COLORMAP_JET)
    return norm_image

####################################################
#This function is to calculate the trend line
def func_fitting(x,oft,rad,st):
    return oft + rad*np.sin((st+x)*np.pi/180)

####################################################
#This function combines the masks for one image, so it's easier to display
def combine_masks(masks_combo):
    merge_mask = np.zeros_like(masks_combo[0]['segmentation'])
    for mask_combo in masks_combo:
        data_mask = mask_combo['segmentation']
        merge_mask += data_mask
        merge_mask = np.clip(merge_mask, 0, 255).astype(np.uint8)
    return merge_mask

####################################################
#This function takes in the x and y points that the user clicked on then goes through all the masks in the one image to find the 
# mask that contains those coordinates. Once the mask is found, it determines the left and right x - edge coordinates based on 
# the y. It then calculates the midpoint between the left and right x - coords. It returns the midpoint, the left x, and right x.
def get_mid_point(masks_example, y,x):
    pixel = None
    for i, mask_example in enumerate(masks_example):
        segmentation_mask = mask_example['segmentation']
        if segmentation_mask[y, x]:
            pixel = i
        else:
            pass
    row_mask = masks_example[pixel]['segmentation'][y,:]
    object_indices = np.where(row_mask)[0]
    if len(object_indices) > 0:     
        min_x = np.min(object_indices)     
        max_x = np.max(object_indices)     
        mid_point = calculate_mid(max_x,min_x)
        return int(mid_point), int(min_x), int(max_x)
    else:     
        return 'Step point might be too large' 

def calculate_mid(x1, x2):
    return (x1 + x2) / 2    

def calculate_difference(x1,x2):
    return abs(x1-x2)

####################################################
#This function first calculates the gridpoints by using the calculate_point functions which divides the 
# points by the width or height of the image. Then defines the mask generator with those points so when 
# the mask generator is called, SAM looks for those specific points in the images and makes the masks 
# based off those points. Then the midpoint is calculated for that mask and compared to the edges to
# see whether the midpoint is close to the edges. The difference between the edges of the object and 
# the edges of the beam is returned as well as the midpoint. 
def generate_sam_and_find_edge(mid_mask,y_coord, wdth, hght, input_edges, norm_img):
    x_gridpoint, y_gridpoint = calculate_point(mid_mask,y_coord,wdth,hght)
    x_gridpoint_edge, y_gridpoint_edge = calculate_point(input_edges[0], input_edges[1], wdth,hght)
    points = [np.array([[x_gridpoint,y_gridpoint],[x_gridpoint_edge,y_gridpoint_edge]])]
    mask_generator_pts = SamAutomaticMaskGenerator(sam,points_per_side = None, point_grids = points)
    masks_all = mask_generator_pts.generate(norm_img)
    mp , left, right = get_mid_point(masks_all, y_coord, mid_mask)
    diff_right = calculate_difference(right, edges[1])
    diff_left = calculate_difference(left, edges[0])            
    return diff_right, diff_left, mp

def calculate_point(x,y,image_size_x, image_size_y):
    point_x = x/image_size_x
    point_y = y/image_size_y
    return point_x, point_y

####################################################
#This function works similarly to the get midpoint function but just returns the edges coordinates. 
def edge_detection(mask_edge, y_edge, x_edge):
    pixel_num = 0
    for i, m_edge in enumerate(mask_edge):
        segmentation = m_edge['segmentation']
        if segmentation[y_edge, x_edge]:
            pixel_num = i
        else:
            pass
    row = mask_edge[pixel_num]['segmentation'][y_edge,:]
    indices = np.where(row)[0]
    if len(indices) > 0: 
        min_x = np.min(indices) 
        max_x = np.max(indices)
        return min_x, max_x
    else: return 'Step point might be too large'

####################################################
#This function will move the pin out of the way to take a normalized image
def move_motors_normalize(time_norm):
    mtr_samXE.move(-2.0, relative=True, wait=True)
    pfname = move_motor(0, time_norm)
    
    #Set image_0 to the image taken without the pin which will be used for normalization
    image_norm = Image.open(pfname)
    
    #Move object back into frame
    mtr_samXE.move(2.0, relative=True, wait=True)
    
    #Move motor to angle 0, this will be displayed to the user 
    image_path = move_motor(0, time_norm)
    
    im = Image.open(image_path)
    width_norm, height_norm = im.size
    
    return width_norm, height_norm, im, image_norm
    
####################################################
#This function will move the motors, take a picture, normalize it if the user chooses to, find the midpoint for each image 
# and then track if the midpoint gets too close to the edges. If it does, then the motors will begin moving the other way 
# and continue until the object gets too close to the other edge. Once it does, the scatter plot of the midpoints and angles
# is created, a curve fit is applied, and the params are found. The original image with SAM with gridpoints used on it, the 
# scatter, and the params are returned. 
def graph_scatter(first_midpoint, rots, y_coord):
    global im_0
    coords = np.array([])
    theta = np.array([])
    midpoint = first_midpoint
    mid_for_mask_reverse = first_midpoint
    mid_for_mask = first_midpoint
    coords = np.append(coords, midpoint)
    theta = np.append(theta, 0)
    th = 0
    th_reverse = 0
    num_of_rotations = 180/rots


    for i in range(1,int(num_of_rotations)):
        
        image_file = move_motor(i * angle_rotation, time_exposure)
        im  = Image.open(image_file)
        if answer_normalization == 1:
            norm_im = normalization(1, im, im_0)
        else:
            norm_im = normalization(0, im, im_0)
        
        dif_right, dif_left, mid_point = generate_sam_and_find_edge(mid_for_mask, y_coord, width, height,edges, norm_im)
        if dif_left < 50 or dif_right < 50:
            #Move motor back to angle 0 so the pin can move the other direction
            for j in range(1,int(num_of_rotations)):
                image_file_reverse = move_motor(-j * angle_rotation, time_exposure)
                im_reverse  = Image.open(image_file_reverse)
                if answer_normalization == 1:
                    norm_im_reverse = normalization(1, im_reverse, im_0)
                else:
                    norm_im_reverse = normalization(0, im_reverse, im_0)
                    
                diff_right_reverse, diff_left_reverse, mid_reverse = generate_sam_and_find_edge(mid_for_mask_reverse, y_coord, width, height,edges,norm_im_reverse)
                
                if diff_left_reverse < 50 or diff_right_reverse < 50:
                    break
                else:
                    th_reverse -= angle_rotation
                    mid_for_mask_reverse = mid_reverse
                    theta = np.append(theta, th_reverse)
                    coords = np.append(coords, mid_for_mask_reverse)
                    print(f'Midpoint for reverse {mid_for_mask_reverse}')
            break
        else:
            th += angle_rotation
            theta = np.append(theta, th)
            mid_for_mask = mid_point
            print(f'Midpoint is now {mid_for_mask}')
            coords = np.append(coords, mid_for_mask)
        
    print(f'coords {coords}')
    print(f'theta {theta}')
    max_ = np.max(coords)
    min_ = np.min(coords)
    rad = (max_ - min_) / 2
    off = (max_ + min_) / 2
    df = pd.DataFrame({'x': theta, 'y': coords})
    bounds = ([-3000, 0, -180], [3000, 2000, 180])
    p0 = [off, rad, 0]
    params, params_cov = curve_fit(func_fitting, theta, coords, p0=p0, bounds=bounds)
    params[2] *= -1
    df_fit = pd.DataFrame(
        {'x': theta, 'y': [math.ceil(value) for value in func_fitting(theta, params[0], params[1], params[2])]})

    scatter = go.Figure()
    scatter.add_trace(go.Scatter(x=df['x'], y=df['y']))
    scatter.add_trace(go.Scatter(x=df_fit['x'], y=df_fit['y']))
    scatter.update_layout(xaxis_title='Angle (Degrees)', yaxis_title='Midpoint', plot_bgcolor='black')
    scatter.update_traces(mode='markers')
    
    display = combine_masks(all_image_masks[0])
    reg = px.imshow(display, color_continuous_scale='gray')
    
    return reg, scatter, params

####################################################
#This function moves the motors and returned the file name
def move_motor(angle,time_needed):
    global cam_name, file_type, camera_type, pname
    t0 = time.time()
    mtr_samOme.move(angle, wait = True)
    ############# VERIFY!!!!
    PyEpics.caput(cam_name + ':' + camera_type + ':ImageMode', 'Single', wait=True)
    PyEpics.caput(cam_name + ':' + camera_type + ':AcquireTime', time_needed, wait=True)
    PyEpics.caput(cam_name + ':' + file_type +':AutoSave', 'Yes', wait=True)
    time.sleep(0.05)
    PyEpics.caput(cam_name + ':' + camera_type + ':Acquire', 1, wait=True)
    time.sleep(0.05)
    PyEpics.caput(cam_name + ':' + file_type + ':AutoSave', 'No', wait=True)
    time.sleep(0.05)
    fname=PyEpics.caget(cam_name + ':' + file_type + ':FileName_RBV', 'str') + "_%06d"%(PyEpics.caget(cam_name + ':' + file_type + ':FileNumber_RBV')-1) + '.tif'
    pfname=os.path.join(pname, fname)
    return pfname

app.layout = dbc.Container([
    dbc.Row([
        html.Br()
    ]),
    dbc.Row([
        dbc.Col([
            html.H1('Automated Beamline Alignment ', style = {'color' : '#7FDBFF', 'textAlign': 'center', 'fontSize': 30, 'fontFamily' : 'OCR A Std, monospace'})
        ])
    ]),
    dbc.Row([
        html.Br()
    ]),
    dbc.Row([
	    dbc.Col([
            dbc.Row([
                html.H2(['Instructions'], style = {'color' : '#7FDBFF', 'textAlign': 'left', 'fontSize': 20, 'fontFamily' : 'OCR A Std, monospace', 'fontWeight': 'bold'})
            ]),
            dbc.Row([
                html.Div(['1. Enter the directory, exposure time, and rotation step. Select or unselect the "Normalization" and then click "Run first image"'], style = style),
                html.Div(['****  Optional: Enter Epics PV names for samXE, samYE, aeroX, aero, pixel size, file type, camera type, and camera name. ****'], style = style),
            ]),
            dbc.Row([
                html.Br()
            ]),
            dbc.Row([
                html.Div(['2. Select the "Draw rectangle" tool in the upper right hand corner of the "SAM" image. Select the beam in the SAM image and click "Submit beam".'], style = style)
            ]),
            dbc.Row([
                html.Br()
            ]),
            dbc.Row([
                html.Div(['3. Select the "Zoom" tool in the upper right hand corner of the "SAM" image. Click the middle of the object in the SAM image and click "Run all images".'], style = style)
            ]),
            dbc.Row([
                html.Br()
            ]),
            dbc.Row([
                html.Div(['4. Once offset, radius, and start theta are displayed, click the "Center pin and verify" button to center the beam.'], style = style)
            ]),
        ])
    ]),
    dbc.Row([
        html.Br()
    ]),
    dbc.Row([
        html.Br()
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                html.Div(['Directory: '], style=style) 
            ]),
            dbc.Row([
                dcc.Input(id='directory-input', type='text', style={'width': '200px'})
            ])
        ]),
        dbc.Col([
            dbc.Row([
                html.Div(['Exposure Time (seconds): '], style=style) 
            ]),
            dbc.Row([
                dcc.Input(id = 'time-input', value = 0.3, max = 10, type = 'number', style={'width': '200px'})  # Adjust style as needed
            ])
        ]),
        dbc.Col([
            dbc.Row([
                html.Div(['Rotation Step (degrees): '], style=style) 
            ]),
            dbc.Row([
                dcc.Input(id = 'angle-input',value = 5, max = 10, type = 'number', style={'width': '200px'})  # Adjust style as needed
            ])
        ])
    ]),
    dbc.Row([
        html.Br()
    ]), 
    dbc.Row([
        html.Br()
    ]), 
    dbc.Row([
        dbc.Col([
            dbc.Row([
                html.Div(['samXE (optional): '], style=style) 
            ]),
            dbc.Row([
                dcc.Input(id='samXE-input', value = '1ide1:m34',type='text', style={'width': '200px'})
            ])
        ]),
        dbc.Col([
            dbc.Row([
                html.Div(['samZE (optional): '], style=style)
            ]),
            dbc.Row([
                dcc.Input(id='samYE-input', value = '1ide1:m36',type='text', style={'width': '200px'})
            ])
        ]),
        dbc.Col([
            dbc.Row([
                html.Div(['aeroXE (optional): '], style=style) 
            ]),
            dbc.Row([
                dcc.Input(id='aeroXE-input', value = '1ide1:m101',type='text', style={'width': '200px'})
            ])
        ]),
        dbc.Col([
            dbc.Row([
                html.Div(['Rot (optional): '], style=style)
            ]),
            dbc.Row([
                dcc.Input(id='aero-input', value = '1ide:m9',type='text', style={'width': '200px'})
            ])
        ]),
    
    ]),
    dbc.Row([
        html.Br()
    ]), 
    dbc.Row([
        dbc.Col([
            dbc.Row([
                html.Div(['Pixel size (microns)(optional): '], style=style) 
            ]),
            dbc.Row([
                dcc.Input(id='pixel-input', value = 1.172,type='number', style={'width': '200px'})
            ])
        ]),
        dbc.Col([
            dbc.Row([
                html.Div(['File type (optional): '], style = style)
            ]),
            dbc.Row([
                dcc.Input(id='filetype-input', value = 'TIFF1',type='text', style={'width': '200px'})
            ])
        ]),
        dbc.Col([
            dbc.Row([
                html.Div(['Camera type (optional): '], style = style)
            ]),
            dbc.Row([
                dcc.Input(id='camera-input', value = 'cam1',type='text', style={'width': '200px'})
            ])
        ]),
        dbc.Col([
            dbc.Row([
                html.Div(['Camera name (optional): '], style = style)
            ]),
            dbc.Row([
                dcc.Input(id='cameraname-input', value = '1idPG1',type='text', style={'width': '200px'})
            ])
        ])
    ]), 
    dbc.Row([
        html.Br()
    ]), 
    dbc.Row([
        html.Br()
    ]), 
    dbc.Row([
       dbc.Col([
           dcc.Checklist(id = 'checklist-id', options = [{'label': 'Add normalization (strongly recommended)', 'value': 'norm'}],style = style)
       ]) 
    ]),
    dbc.Row([
	    html.Br()
    ]), 
    dbc.Row([
        dbc.Col([
        html.Button('Run first image', id='button', style={'background-color': 'black', 'width': '200px', 'color' : '#7FDBFF', 'fontSize': 15, 'fontFamily' : 'OCR A Std, monospace'})
        ]),
    ]),
    dbc.Row([
	html.Br()
    ]),
    dbc.Row([
	    html.Br()
    ]),
    dbc.Row([
	    dbc.Col([
            dbc.Row([
                html.Div(['Original'], style = style)
             ]),
            dbc.Row([
                dcc.Graph(id='og-img',figure = fig)
            ])
        ]),
        dbc.Col([
            dbc.Row([
                html.Div(['Normalized if selected, else original with color map.'], style = style)
            ]),
            dbc.Row([
                dcc.Graph(id='nl-img',figure = fig)
            ])
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                html.Div(['SAM'], style = style),
            ]),
            dbc.Row([
                dcc.Graph(id='sam-img',figure = fig, config={'modeBarButtonsToAdd': ['drawrect', 'eraseshape']})
            ]),
        dbc.Row([
	        html.Br()
        ]),
        dbc.Col([
                html.Button('Submit beam', id='button-edge', style={'background-color': 'black', 'width': '200px', 'color' : '#7FDBFF', 'fontSize': 15, 'fontFamily' : 'OCR A Std, monospace'})
            ]),
        ]),
        dbc.Row([
	        html.Br()
        ]),
        dbc.Row([
            html.Div(id='beam', style=style)
        ]),
        dbc.Row([
	        html.Br()
        ]),
        dbc.Col([
            dbc.Row([
                html.Div(['2. Select the "Draw rectangle" tool in the upper right hand corner of the "SAM" image. Select the beam in the SAM image and click "Submit beam".'], style = style)
            ]),
            dbc.Row([
                html.Br()
            ]),
            dbc.Row([
                html.Div(['3. Select the "Zoom" tool in the upper right hand corner of the "SAM" image. Select the middle of the object in the SAM image and click "Run all images".'], style = style)
            ]),
            dbc.Row([
                html.Br()
            ]),
            dbc.Row([
                html.Div(['4. Once offset, radius, and start theta are displayed, click the "Center pin and verify" button to center the beam.'], style = style)
            ]),
        ]),
        dbc.Row([
	        html.Br()
        ]),
        dbc.Col([
            html.Button('Run all images', id='button-two',
                        style={'background-color': 'black', 'width': '200px', 'color': '#7FDBFF', 'fontSize': 15,
                               'fontFamily': 'OCR A Std, monospace'})
        ]),
        dbc.Row([
	        html.Br()
        ]),
        dbc.Col([
            dbc.Row([
                html.Div(['SAM with Grid Point'], style = style)
            ]),
            dbc.Row([
                dcc.Graph(id='sam-grid',figure = fig)
            ])
        ]),
    ]),
    dbc.Row([
            dbc.Col([
            dbc.Row([
                html.Div(['Graph'], style = style)
            ]),
            dbc.Row([
                dcc.Graph(id='my-graph', figure =fig, style = {'float' : 'center'})
            ]),
            dbc.Row([
                html.Br()
            ]),
            dbc.Row([
                html.Div(id = 'output-data', style = style)
            ])
        ])
    ]),
    dbc.Row([
        html.Br()
    ]),
    dbc.Col([
        html.Button('Center pin and verify', id='button-restart',
                    style={'background-color': 'black', 'width': '200px', 'color': '#7FDBFF', 'fontSize': 15,
                           'fontFamily': 'OCR A Std, monospace'})
    ]),
    dbc.Row([
        html.Br()
    ]),
    dbc.Row([
        html.Br()
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                html.Div(['New Image at Center of Axis'], style = style)
            ]),
            dbc.Row([
                dcc.Graph(id='new-image',figure = fig)
            ])
        ]),
        dbc.Col([
            dbc.Row([
                html.Div(['Graph for new Image'], style = style)
            ]),
            dbc.Row([
                dcc.Graph(id='new-graph', figure =fig, style = {'float' : 'center'})
            ]),
            dbc.Row([
                html.Div(id = 'output-new-data', style = style)
            ])
        ])
    ]),
])


####################################################
#This callback returns the original image, the normalized or not normalized image, and the image with the
# regular SAM used on it once the button has been clicked and the directory, angle, and time been entered,
@app.callback(
    Output('og-img', 'figure'),
    Output('nl-img', 'figure'),
    Output('sam-img', 'figure'),
    Input('directory-input', 'value'),
    Input('time-input', 'value'),
    Input('angle-input', 'value'),
    Input('button', 'n_clicks'),
    Input('checklist-id', 'value'),
    Input('samXE-input', 'value'),
    Input('samYE-input', 'value'),
    Input('aeroXE-input', 'value'),
    Input('aero-input', 'value'),
    Input('pixel-input', 'value'),
    Input('filetype-input', 'value'),
    Input('camera-input', 'value'),
    Input('cameraname-input', 'value'),
)
def update_imgs(dt, time_input, angle_input, clicks, checklist, xe, ye, aeroxe, aero_input, pixel_input, file, cam_input, camname):
    global clicks_tracker
    global height, width, directory_input, time_exposure, angle_rotation, answer_normalization,pixel_size, camera_type, file_type,cam_name,off_set,rad_ius, mtr_samXE, mtr_samYE, mtr_samOme, mtr_aeroXE, pname, froot, im_0
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    if clicks is not None: 
        if clicks > clicks_tracker:
            clicks_tracker = clicks
            if dt is not None and time_input is not None and angle_input is not None:
                pname = dt
                time_exposure = time_input
                angle_rotation = angle_input
                pixel_size = pixel_input
                file_type = file
                cam_name = camname
                camera_type = cam_input
                froot = 'pin_alignment'
                # PyEpics.caput(cam_name + ':' + file_type + ':FilePath', pname, wait=True)
                PyEpics.caput(cam_name + ':' + file_type + ':FileName', froot, wait=True)
                mtr_samXE = PyEpics.Motor(xe)
                mtr_samYE = PyEpics.Motor(ye)
                mtr_samOme = PyEpics.Motor(aero_input)
                mtr_aeroXE = PyEpics.Motor(aeroxe)

                print(f'******directory input: {pname} **********')
                print(f'******froot input: {froot} **********')

                
                width, height, im_1, image0 = move_motors_normalize(time_exposure)
                im_0 = image0
                if checklist is None:
                    answer_normalization = 0
                    image_norm = normalization(0, im_1, image0)
                elif 'norm' in checklist:
                    image_norm = normalization(1, im_1, image0)
                    answer_normalization = 1
                else:
                    image_norm = normalization(0, im_1, image0)
                    answer_normalization = 0
                    
                mask_image_norm = mask_generator.generate(image_norm)
                display_mask = combine_masks(mask_image_norm)
                
                all_image_masks.append(mask_image_norm)
                og_img = px.imshow(im_1, color_continuous_scale='gray', template = 'plotly_dark')
                nl_img = px.imshow(image_norm, color_continuous_scale='gray', template = 'plotly_dark')
                sam_img = px.imshow(display_mask, color_continuous_scale='jet', template = 'plotly_dark')
                
                return og_img, nl_img, sam_img
            else:
                return px.imshow([]), px.imshow([]), px.imshow([])
    else:
        return px.imshow([]), px.imshow([]), px.imshow([])
        
@app.callback(
    Output('new-image', 'figure'),
    Output('output-new-data', 'children'),
    Output('new-graph', 'figure'),
    Input('button-restart', 'n_clicks')
) 
def new_image(clicks_restart):
    global restart_clicks_tracker, pixel_size, rad_ius, off_set, start_theta, im_0, width, height, answer_normalization, y_universal, angle_rotation, im_0
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    if clicks_restart is not None:
        if clicks_restart > restart_clicks_tracker:
            restart_clicks_tracker = clicks_restart
            beam_center = edges[1] - edges[0]
            move_aero_x = ((off_set - (beam_center/2)) * pixel_size)/1000
            move_sam_x = (rad_ius * (np.sin(start_theta*np.pi/180))*pixel_size)/1000
            move_sam_y = (-rad_ius * (np.cos(start_theta*np.pi/180))*pixel_size)/1000
            print(move_aero_x,move_sam_x,move_sam_y)
            
            mtr_samXE.move(move_sam_x,relative=True, wait=True)
            mtr_samYE.move(move_sam_y,relative=True, wait=True)
            mtr_aeroXE.move(move_aero_x, relative=True, wait=True) #Is this the right mtr??
            
            width, height, im_1, img0 = move_motors_normalize(time_exposure)
            im_0 = img0
            if answer_normalization == 0:
                image_norm = normalization(0, im_1, img0)
            else:
                image_norm = normalization(1, im_1, img0)
                
                    
            mask_image_norm = mask_generator.generate(image_norm)
            display_mask = combine_masks(mask_image_norm)
            sam_img = px.imshow(display_mask, color_continuous_scale='gray', template = 'plotly_dark')
            
            first_midpoint, no_x, no_y = get_mid_point(mask_image_norm, y_universal, x_univesal)
            reg1, scatter1, params1 = graph_scatter(first_midpoint,angle_rotation, y_universal)
            
            return sam_img, f'Rotation axis position (pixels): {params1[0]}, Pin offset from rotation axis (pixels): {params1[1]}, Pin offset angle (degrees): {params1[2]}', scatter1
        
        else: 
            return px.imshow([]), px.imshow([]), px.imshow([]), px.imshow([]), go.Figure(), None
    else: 
        return px.imshow([]), px.imshow([]), px.imshow([]), px.imshow([]), go.Figure(), None
            
####################################################
#This call back returns the beam edge coordinates once the beam has been selected
@app.callback(
    Output('beam', 'children'),
    Input('sam-img', 'relayoutData'),
    Input('button-edge', 'n_clicks'),
)
def update_edge(relayout_data, click):
    global click_counter
    if click is not None:
        if click > click_counter:
            click_counter = click
            if "shapes" in relayout_data:
                x_coordinate = relayout_data['shapes'][0]['x0']
                y_coordinate = relayout_data['shapes'][0]['y0']
                minimum_x, maximum_x = edge_detection(all_image_masks[0], int(y_coordinate), int(x_coordinate))
                edges.append(minimum_x)
                edges.append(maximum_x)
                print(f'Max: {maximum_x} and Min: {minimum_x}')
                return f'Beam has been selected. Edges of beam are {minimum_x} and {maximum_x}'
            
####################################################
#This callback returns the image with SAM gridpoints used on it as well as the graph with specific offset, 
# radius, and start theta calculated once the button has been clicked and the user selects the object.
@app.callback(
    Output('sam-grid', 'figure'),
    Output('my-graph', 'figure'),
    Output('output-data', 'children'),
    Input('button-two', 'n_clicks'),
    Input('sam-img', 'clickData'),  
)
def update_sam_grid(n_clicks, clickData):
    global width, height,directory_input,  angle_rotation, answer_normalization, n_clicks_tracker, time_exposure, off_set, rad_ius, start_theta, x_univesal, y_universal
    
    if n_clicks is not None:
        if n_clicks > n_clicks_tracker:
            n_clicks_tracker = n_clicks
            x_coord = clickData['points'][0]['x']
            x_univesal = x_coord
            y_coord = clickData['points'][0]['y']
            y_universal = y_coord
            first_midpoint, no_x, no_y = get_mid_point(all_image_masks[0], y_coord, x_coord)
            reg, scatter, params = graph_scatter(first_midpoint, angle_rotation, y_coord)
            
            off_set = params[0]
            
            rad_ius = params[1]
            
            start_theta = params[2]
            
            return reg, scatter, f'Rotation axis position (pixels): {params[0]}, Pin offset from rotation axis (pixels): {params[1]}, Pin offset angle (degrees): {params[2]}'

        else:
            return px.imshow([]), go.Figure(), f''
    else:
        return px.imshow([]), go.Figure(), f''


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug=False, port = 8055)
