"""
*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 5A of Geo Guide (GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
******************************************************************************************
"""

"""

* Team Id : 1267

* Author List : Adithya G Jayanth, Ananya Sharma, Ankith L, Ilaa S Chenjeri

* Filename: GG_1267.py

* Theme: Geo Guide

* Functions: 
[
    aruco_details,
    euclidean_distance,
    nearest,
    write_into_csv,
    all_aruco,
    tracker,
    read_csv,
    check,
    set_resolution,
    event_identification,
    update_events,
    filter,
    find_max_count_number,
    get_max,
    classify_event,
    classification,
    detect_ArUco_details,
    mark_ArUco_image,
    task_4a_return,
    filter_list
]

* Global Variables: 
[
    deque_A,
    deque_B,
    deque_C,
    deque_D,
    deque_E,
    win_counter,
    is_sent,
    prev_is_sent,
    default_is_sent,
    reset_counter,
    global_points,
    device,
    model,
    detected,
    count,
    final_pts,
    events,
    dict,
    final_list,
    counter,
    combat,
    human_aid_rehabilitation,
    military_vehicles,
    fire,
    destroyed_buildings,
    boxes,
    final_pts,
    last_recorded_cord,
    previous_recorded_cord,
    near,
    init_flag_static,
    blank_score,
    near_coords,
    details_aruco,
]

"""


####################### IMPORT MODULES #######################

from path import path_gen
import math
import csv
import numpy as np
import cv2
import time
import torch
from torchvision import transforms
from PIL import Image
import cv2
from collections import Counter
import socket
from time import sleep
from collections import deque
import select
import json


###################### GLOBAL VARIABLES #########################

deque_A = deque(maxlen=10)
deque_B = deque(maxlen=10)
deque_C = deque(maxlen=10)
deque_D = deque(maxlen=10)
deque_E = deque(maxlen=10)
win_counter = 0
is_sent = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, None: 1, "F": 0}
prev_is_sent = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, None: 1, "F": 0}
default_is_sent = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, None: 1, "F": 0}
reset_counter = -1
global_points = np.array([[870.0, 73.0], [215.0, 30.0], [845.0, 697.0], [194.0, 704.0]])
device = "cpu"
model = None  # model is made global to prevent loading into gpu repeatedly
detected = [0, 0, 0, 0, 0]
count = 0
final_pts = None
events = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
dict = {5: "A", 4: "B", 3: "D", 2: "C", 1: "E"}
final_list = []
counter = 1
combat = "combat"
human_aid_rehabilitation = "human_aid_rehabilitation"
military_vehicles = "military_vehicles"
fire = "fire"
destroyed_buildings = "destroyed_buildings"
boxes = [[], [], [], [], []]
final_pts = None
last_recorded_cord = None
previous_recorded_cord = None
near = 23
init_flag_static = 1
blank_score = [0, 0, 0, 0, 0, 0]
near_coords = [39.6128542, -74.3629792]
details_aruco = {
    21: [[322, 882], 0],
    34: [[315, 466], 3],
    24: [[52, 774], 0],
    16: [[816, 877], 0],
    33: [[406, 469], 0],
    41: [[240, 309], 0],
    32: [[563, 475], 0],
    36: [[812, 332], 0],
    17: [[723, 877], -1],
    52: [[44, 147], 0],
    38: [[658, 326], 1],
    40: [[327, 312], 1],
    22: [[52, 710], 0],
    39: [[405, 318], 0],
    48: [[333, 147], 0],
    50: [[43, 376], 0],
    35: [[570, 323], -1],
    31: [[632, 475], 1],
    14: [[955, 792], 0],
    30: [[810, 481], 0],
    37: [[739, 329], 0],
    47: [[411, 148], 0],
    42: [[160, 305], 0],
    20: [[407, 879], 1],
    54: [[173, 40], 0],
    23: [[47, 842], 0],
    15: [[928, 872], 0],
    12: [[947, 437], 0],
    13: [[956, 721], 0],
    9: [[944, 504], 1],
    44: [[697, 159], -1],
    45: [[599, 156], 1],
    28: [[595, 676], 0],
    25: [[182, 690], 0],
    29: [[821, 675], 0],
    26: [[292, 681], -1],
    43: [[776, 169], 0],
    18: [[611, 876], 1],
    51: [[44, 260], 0],
    10: [[948, 240], 1],
    19: [[506, 877], 0],
    53: [[79, 61], 0],
    11: [[950, 567], 1],
    46: [[503, 151], 1],
    8: [[949, 365], 1],
    49: [[39, 489], 0],
    27: [[405, 673], 1],
}
#################################################################


"""

    * Function Name: aruco_details
    * Input: image (numpy array) - Input CV2 image from the webcam,
             lat_lon (List) - Latitude and longitude information of the aruco in QGIS
    * Output: ArUco_details_dict (dict) - Dictionary containing ArUco marker details,
              ArUco_corners (dict) - Dictionary containing ArUco marker corners
    * Logic: 
        1) In this code the corner arucos are detected first. The detectMarkers return 
        the value of the center coordinates and orientation.
        2) Then those coordinates are used for perspective transformation of the video 
        feed. The coordinates are stored in a global variable.
        3) The all_aruco function is called and the details of the aruco is returned.
        4) Nearest and Check functions are called which returns the nearest aruco and 
        checks if the bot is traversing through the imaginary nodes.
        5) Tracker function is called to update the QGIS location.
        6) The final CV2 image is showed using imshow.
    * Example Call: aruco_details(image, lat_lon)

"""


def aruco_details(image, lat_lon):
    global detected, count, final_pts, global_points
    ArUco_details_dict = {}
    ArUco_corners = {}
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(image)
    desired_ids = [5, 7, 6, 4]
    marker_corners = []
    marker_id_order = []
    if ids is not None:
        for i in range(len(ids)):
            if ids[i][0] in desired_ids:
                marker_corners.append(corners[i][0])
                marker_id = int(ids[i][0])
                marker_id_order.append(marker_id)
                corners_list = corners[i][0]
                center_x = int(np.mean(corners_list[:, 0]))
                center_y = int(np.mean(corners_list[:, 1]))
                dx = corners_list[1][0] - corners_list[0][0]
                dy = corners_list[1][1] - corners_list[0][1]
                marker_orientation = int((180.0 / math.pi) * math.atan2(dy, dx))
                ArUco_details_dict[marker_id] = [
                    [center_x, center_y],
                    marker_orientation,
                ]
                ArUco_corners[marker_id] = corners_list.tolist()
        if len(marker_corners) >= 4:
            marker_centers = np.mean(np.array(marker_corners), axis=1).astype(int)
            combined_lists = list(zip(marker_id_order, marker_centers))
            sorted_combined_lists = sorted(combined_lists, key=lambda x: x[0])
            if count < 15:
                sorted_list1, final_pts = zip(*sorted_combined_lists)
                count += 1

            paper = image
            pts1 = np.float32(final_pts)
            pts2 = np.float32([[1000, 0], [0, 0], [1000, 1000], [0, 1000]])
            if pts1 is not None:
                global_points = pts1.copy()
            M = cv2.getPerspectiveTransform(global_points, pts2)
            dst = cv2.warpPerspective(paper, M, (1000, 1000))
            details_aruco_with_100 = all_aruco(dst)
            n_id, n_loc = nearest(details_aruco_with_100)
            which_node = check(details_aruco_with_100)
            is_sent[which_node] = 1
            tracker(n_id, lat_lon)

            ## RED AND BLUE DOT ON THE BOT AND NEAREST ARUCO
            # red_color = (0, 0, 255)
            # radius = 5
            # z = int(n_loc[0])
            # x = int(n_loc[1])
            # cv2.circle(dst, (z, x), radius, red_color, -1)
            # blue_color = (255,0, 0)
            # # print(details_aruco_with_100)
            # # print(details_aruco_with_100[100][0])
            # if 100 in details_aruco_with_100:
            #     temp = details_aruco_with_100[100][0]
            #     center_x = temp[0]
            #     center_y = temp[1]
            #     cv2.circle(dst, (center_x, center_y), radius, blue_color, -1)
            # else:
            #     # print("Bot aruco not detected")
            #     pass

            cv2.imshow("img", dst)
            cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            key = cv2.waitKey(1) & 0xFF
        else:
            paper = image
            pts2 = np.float32([[1000, 0], [0, 0], [1000, 1000], [0, 1000]])
            M = cv2.getPerspectiveTransform(global_points, pts2)
            dst = cv2.warpPerspective(paper, M, (1000, 1000))
            details_aruco_with_100 = all_aruco(dst)
            n_id, n_loc = nearest(details_aruco_with_100)
            which_node = check(details_aruco_with_100)
            is_sent[which_node] = 1
            tracker(n_id, lat_lon)

            ## RED AND BLUE DOT ON THE BOT AND NEAREST ARUCO
            # red_color = (0, 0, 255)
            # radius = 5
            # z = int(n_loc[0])
            # x = int(n_loc[1])
            # cv2.circle(dst, (z, x), radius, red_color, -1)
            # # print(details_aruco_with_100)
            # # print(details_aruco_with_100[100][0])
            # blue_color = (255,0, 0)
            # if 100 in details_aruco_with_100:
            #     temp = details_aruco_with_100[100][0]
            #     center_x = temp[0]
            #     center_y = temp[1]
            #     cv2.circle(dst, (center_x, center_y), radius, blue_color, -1)
            # else:
            #     print("Bot aruco not detected")

            cv2.imshow("img", dst)
            cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            key = cv2.waitKey(1) & 0xFF

    return ArUco_details_dict, ArUco_corners


"""

    * Function Name: euclidean_distance
    * Input: point1 (list) - Coordinates of the first point (x1, y1)
             point2 (list) - Coordinates of the second point (x2, y2)
    * Output: float - Euclidean distance between the two points
    * Logic: 
        1)Calculates the Euclidean distance between two points using the formula:
              
              distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
    
    * Example Call: distance = euclidean_distance((0, 0), (3, 4))

"""


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


"""

    * Function Name: nearest
    * Input: details (dict) - Dictionary containing details of Aruco markers
    * Output: nearest_point_id (list), nearest_point_coords (list)
    * Logic: 
        1) Find the nearest Aruco marker to an ID 100 Aruco marker.
        2) Euclidean Distance function is called for each point and 100 ID coordinates.
        3) The smallest distance is termed as nearest point. 
        4) If the nearest marker is further than a threshold distance of 75 units, 
        it returns the previous nearest marker.
        5) The nearest ID and its respective coordinates are globalized 
    * Example Call: nearest_id, nearest_coords = nearest(details)
    
"""


def nearest(details):
    global near, near_coords
    if 100 in details:
        hun = details[100][0]
        nearest_point_id = min(
            details_aruco.keys(),
            key=lambda x: euclidean_distance(hun, details_aruco[x][0]),
        )
        # print("Nearest point ID:", nearest_point_id)
        distance = euclidean_distance(hun, details_aruco[nearest_point_id][0])
        # print("Distance from nearest point : ", round(distance))
        if distance > 75:
            return near, near_coords
        near = nearest_point_id
        near_coords = details_aruco[nearest_point_id][0]
        return near, near_coords
    else:
        # print("Bot not recognised")
        return near, near_coords


"""

    * Function Name: write_into_csv
    * Input: id (int) - ID of the Aruco marker
             cord (list) - Coordinates of the Aruco marker (x, y)
    * Output: None
    * Logic: 
        1) Writes the ID and coordinates of an Aruco marker into a CSV file.
        if the coordinates are different from the last two recorded coordinates.
        2) Writes the path taken by the bot in the arena, the ID and its respective
        coordinates are written.
    * Example Call: write_into_csv(1, (100, 200))

"""


def write_into_csv(id, cord):
    global last_recorded_cord, previous_recorded_cord
    if cord != last_recorded_cord and cord != previous_recorded_cord:
        with open("path.csv", mode="a", newline="") as write_file:
            writer = csv.writer(write_file, delimiter=",")
            writer.writerow([id, str(cord[0]), str(cord[1])])
        previous_recorded_cord = last_recorded_cord
        last_recorded_cord = cord


"""

    * Function Name: all_aruco
    * Input: image (numpy.ndarray) - input image from CV2 
    * Output: ArUco_details_dict (dict) - dictionary containing ArUco marker 
        IDs as keys and their details (center coordinates and orientation) as values
    * Logic: 
        1) Detects ArUco markers in the input image and extracts their center coordinates
        and their orientation in degrees. 
        2) Returns a dictionary containing this information.
    * Example Call: ArUco_details = all_aruco(image)

"""


def all_aruco(image):
    ArUco_details_dict = {}
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(image)
    if ids is not None:
        for i in range(len(ids)):
            marker_id = int(ids[i][0])
            corners_list = corners[i][0]
            center_x = int(np.mean(corners_list[:, 0]))
            center_y = int(np.mean(corners_list[:, 1]))
            dx = corners_list[1][0] - corners_list[0][0]
            dy = corners_list[1][1] - corners_list[0][1]
            marker_orientation = int((180.0 / math.pi) * math.atan2(dy, dx) + 360) % 360

            ArUco_details_dict[marker_id] = [[center_x, center_y], marker_orientation]

    return ArUco_details_dict


"""

    * Function Name: tracker
    * Input: ar_id (int) - AR marker ID, 
             lat_lon (dict) - dictionary mapping AR marker IDs to (latitude, longitude) of  QGIS 
    * Output: None
    * Logic: 
        1) Updates the CSV file "live_data.csv" with the latest latitude and longitude 
        information for the nearest aruco ID and coordinates.
        2) It is used to update Live_data.csv as the QGIS software is retrieving 
        the coordinates data from that file.
        3) Only one ID and coordinates is written into the file.
    * Example Call: tracker(1, {1: (latitude, longitude)})

"""


def tracker(ar_id, lat_lon):
    ar_id = str(ar_id)
    lat, lon = lat_lon.get(ar_id, ("None", "None"))
    data = []
    file = open("live_data.csv", mode="r", newline="")
    reader = csv.reader(file)
    headers = next(reader)
    data = list(reader)
    if data and all(cell == "None" for cell in data[0]):
        data[0] = [near_coords[0], near_coords[1]]
    else:
        if data[0][0] != "None" and data[0][1] != "None":
            data[0][0] = lat
            data[0][1] = lon
        else:
            data[0][0] = near_coords[0]
            data[0][1] = near_coords[1]
    file.close()
    file = open("live_data.csv", mode="w", newline="")
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(data)
    file.close()
    ar_id_str = str(ar_id)
    if ar_id_str in lat_lon:
        write_into_csv(ar_id, lat_lon[ar_id_str])


"""

    * Function Name: read_csv
    * Input: No input parameters
    * Output: Dictionary (lat_lon) with:
        * Keys: Strings representing IDs from the CSV file
        * Values: Lists containing two elements: latitude and longitude
    * Logic:
        1. Opens the "lat_long1.csv" file in read mode.
        2. Creates a CSV reader object.
        3. Iterates through each row in the CSV file.
        4. For each row:
            - Extracts ID, latitude, and longitude values.
            - Adds the ID as a key and a list containing latitude and longitude as the value to the lat_lon dictionary.
        5. Closes the file.
        6. Returns the lat_lon dictionary.
    * Example Call:
        my_data = read_csv()
        print(my_data["ID123"])  # Print the latitude and longitude for ID "ID123"

"""


def read_csv():
    lat_lon = {}
    file = open("lat_long1.csv", mode="r")
    reader = csv.reader(file)
    for i in reader:
        id = i[0]
        lat = i[1]
        lon = i[2]
        lat_lon[id] = [lat, lon]
    file.close()
    return lat_lon


"""

    * Function Name: check
    * Input:
        * details: A dictionary containing various data points with unknown structure.
    * Output:
        * String: 
            * "A" if details[100][0] is within certain longitude and latitude ranges.
            * "B", "C", "D", "E", or "F" based on additional similar conditions.
            * None if details[100] doesn't exist or any condition isn't met.
    * Logic:
        1. Checks if the key "100" exists in the `details` dictionary.
        2. If it exists, retrieves the coordinates of that key and stores it in `hun`.
        3.  * Checks if the first element (`hun[0]`) falls within specific longitude ranges.
            * Checks if the second element (`hun[1]`) falls within specific latitude ranges.
            * Returns "A", "B", "C", "D", "E", or "F" depending on the matched ranges.
        4. If any checks fail or `hun` doesn't exist, returns None.
    * Example Call:
        result = check({"100": [[210, 800], [700, 600]]})
        print(result)  # Output: "A" (assuming ranges match)

"""


def check(details):
    if 100 in details:
        hun = details[100][0]
        if hun[0] > 208 and hun[0] < 259 and hun[1] > 760 and hun[1] < 870:
            return "A"
        elif hun[0] > 684 and hun[0] < 751 and hun[1] > 563 and hun[1] < 672:
            return "B"
        elif hun[0] > 690 and hun[0] < 755 and hun[1] > 380 and hun[1] < 474:
            return "C"
        elif hun[0] > 180 and hun[0] < 257 and hun[1] > 352 and hun[1] < 470:
            return "D"
        elif hun[0] > 203 and hun[0] < 275 and hun[1] > 45 and hun[1] < 150:
            return "E"
        elif hun[0] > 691 and hun[0] < 765 and hun[1] > 56 and hun[1] < 146:
            return "F"


"""

    * Function Name: set_resolution
    * Input:
        * cap: OpenCV VideoCapture object representing a video source.
        * width: Integer representing the desired frame width.
        * height: Integer representing the desired frame height.
    * Output:
        * None. The function modifies the `cap` object to use the given resolution.
    * Logic:
        1. Uses the `cap.set` method to set the `CAP_PROP_FRAME_WIDTH` property to the given `width`.
        2. Uses the `cap.set` method to set the `CAP_PROP_FRAME_HEIGHT` property to the given `height`.
        3. These calls attempt to change the resolution of the captured frames from the video source represented by `cap`.
    * Example Call:
        # Open a video capture object
        cap = cv2.VideoCapture(0)
        # Set resolution to 640x480 (might not be exact)
        set_resolution(cap, 640, 480)

"""


def set_resolution(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


"""

    * Function Name: event_identification
    * Input:
        * img: NumPy array representing an image.
    * Output:
        * list: A list of cropped sub-images from the input image, potentially associated with different events.
    * Logic:
        1. Defines five constant slices based on coordinates to extract sub-images from specific regions of the input image.
        2. Extracts sub-images for each defined region:
            * `im1` from top-left region (E).
            * `im2` from bottom-right region (C).
            * `im3` from bottom-left region (D).
            * `im4` from bottom-right corner region (B).
            * `im5` from top-left corner region (A).
        3. **Commented out code:** Shows examples of displaying or saving sub-images for debugging purposes.
        4. Adds the extracted sub-images (`im1`, `im4`, `im2`, `im3`, `im5`) to an `event_list`.
    * Example Call:
        # Assuming you have an image loaded as `img`
        event_list = event_identification(img)
    
"""


def event_identification(img):
    event_list = []
    im1 = img[132:218, 200:287]  # E
    im2 = img[478:556, 680:761]  # C
    im3 = img[471:549, 191:266]  # D
    im4 = img[680:766, 672:757]  # B
    im5 = img[882:963, 200:285]  # A

    # cv2.imshow("im1", im1)
    # cv2.imshow("im2", im2)
    # cv2.imwrite("img2.png", im2)
    # cv2.imshow("im3", im3)
    # cv2.imshow("im4", im4)
    # cv2.imshow("im5", im5)
    # key = cv2.waitKey(1) & 0xFF

    event_list.append(im5)
    event_list.append(im4)
    event_list.append(im2)
    event_list.append(im3)
    event_list.append(im1)
    cv2.waitKey(1)
    return event_list


"""

    * Function Name: update_events
    * Input:
        * non_zero_pixel_count: An integer representing the number of non-zero pixels in an image.
    * Output:
        * integer:
            * 1 if the `non_zero_pixel_count` is less than 70, indicating presence of an event.
            * 0 otherwise, indicating absence of an event.
    * Logic:
        1. Checks if the `non_zero_pixel_count` is less than 70. 
        2. If the count is less than 70, returns 1, signaling event detection.
        3. Otherwise, returns 0, suggesting absence of an event.
    * Example Call:
    #calct
        event_status = update_events(non_zero_pixel_count)
        if event_status == 1:
            # Handle presence of event
            print("Event detected!")
        else:
            # Handle absence of event
            print("No event placed.")

"""


def update_events(non_zero_pixel_count):
    if non_zero_pixel_count < 70:
        return 1
    else:
        return 0


"""
    * Function Name: filter
    * Input:
        * `img`: An OpenCV image object representing the input image.
    * Output:
        * None (void function). Updates the global `events` dictionary with 1 if the image has
          less number of pixels which lie in the range of mask values, which means an event is 
          present, otherwise a 0 is appended to depict a blank event.
    * Logic:
        1. Crops five regions from the input image based on fixed coordinates, which are the 
        coordinates of a fixed area square, from the top right part of the image.
        2. Saves each cropped part as a separate image: `img1.jpg`, `img2.jpg`, ..., `img5.jpg`.
        3. Loops through each saved image:
            * Reads the image.
            * Converts it to HSV color space.
            * Defines a specific color range based on HSV values.
            * Creates a mask to identify pixels within that color range.
            * Calculates the number of non-zero pixels in the masked image.
            * Uses the `update_events` function to interpret the pixel count and update the `events` 
            dictionary with a value (1 for presence of event, 0 for absence).
            * Appends the values 1 or 0 to a  deque.
            * Updates the corresponding letter's value in the `events` dictionary with the maximum 
            value in its deque.
    * Example Call:
        filter(img)
        print(events["A"])  # prints 1 if event is present, and 0 if it is absent
        print(events["B"]) 

    """


def filter(img):
    global events
    im1 = img[135:216, 206:287]
    im2 = img[480:557, 688:768]
    im3 = img[471:549, 191:266]
    im4 = img[681:765, 673:759]
    im5 = img[882:963, 200:285]

    img1 = im1[0:25, 50:75]
    img2 = im2[0:35, 45:80]
    img3 = im3[0:35, 35:75]
    img4 = im4[0:35, 45:80]
    img5 = im5[0:35, 40:75]

    cv2.imwrite("img1.jpg", img1)
    cv2.imwrite("img2.jpg", img2)
    cv2.imwrite("img3.jpg", img3)
    cv2.imwrite("img4.jpg", img4)
    cv2.imwrite("img5.jpg", img5)

    for i in range(1, 6):
        img = cv2.imread(f"img{i}.jpg")
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_limit = np.array([20, 30, 20])
        upper_limit = np.array([28, 255, 200])
        mask = cv2.inRange(hsv_img, lower_limit, upper_limit)
        res = cv2.bitwise_and(img, img, mask=mask)
        non_zero_pixel_count = cv2.countNonZero(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY))
        letter = dict[i]
        value = update_events(non_zero_pixel_count)
        current_deque = globals()[f"deque_{letter}"]
        current_deque.append(value)
        if current_deque:
            events[letter] = max(current_deque)
        else:
            events[letter] = 0


"""

    * Function Name: find_max_count_number
    * Input:
        * lst: A list containing numbers or other countable elements.
    * Output:
        * max_count_number which is the value of the most occuring number in a list.
    * Logic:
        1. Creates a Counter object `counts` from the input list, which stores the counts 
        of each distinct element.
        2. Finds the element with the maximum count using the `max` function with a 
        key argument that specifies `counts.get` to retrieve the count for each element.
        3. Returns the element with the maximum count.
    * Example Call:
        my_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        most_frequent_number = find_max_count_number(my_list)
        print("The most frequent number is:", most_frequent_number)  # Output: 3
    
"""


def find_max_count_number(lst):
    counts = Counter(lst)
    max_count_number = max(counts, key=counts.get)
    return max_count_number


"""

    * Function Name: get_max
    * Input:
        * target_image_pred_label: The predicted label for a specific image.
        * img_index: The index of the image in the global `boxes` list.
    * Output:
        * integer: The most frequent predicted label within the last 60 predictions for the given image.
    * Logic:
        1) This function is used to store the predictions of the model in a dictionary called boxes.
        2) find_max_count_number function is called which gets max of the boxes list.
        3) The keys of boxes and is values are updated with max occuring number.
        4) This is done to reduce flickering of the model.
        5) The score is returned .
    * Example Call:
        most_frequent_label = get_max(predicted_label, image_index)
    
"""


def get_max(target_image_pred_label, img_index):
    global boxes
    l = boxes[img_index]
    l.append(target_image_pred_label)
    if len(l) == 60:
        l.pop(0)
    score = find_max_count_number(l)
    boxes[img_index] = l
    return score


"""

    * Function Name: classify_event
    * Input:
        * image: A NumPy array representing an image.
        * img_index: The index of the image in a global list.
    * Output:
        * string: The predicted event class for the image, based on the pre-trained model.
    * Logic:
        1) Image Preprocessing (Optional):
            - Saves the image to a temporary file ("a.jpeg").
            - Loads the image using Pillow's `Image` class.
            - Applies a series of transformations to resize the image to the model's input requirements.
            - Converts the image to a PyTorch tensor and moves it to the appropriate device (CPU/GPU).
        2) Model Evaluation:
            - Sets the model to evaluation mode (disabling dropout, etc.).
            - Disables gradient calculation for efficiency.
            - Makes a prediction on the image tensor with the model.
        3) Returns the predicted event class name from the list of class names.
    * Example Call:
        predicted_event = classify_event(image, image_index)
    
"""


def classify_event(image, img_index):
    global model
    class_names = [
        "combat",
        "destroyed_buildings",
        "fire",
        "human_aid_rehabilitation",
        "military_vehicles",
    ]
    cv2.imwrite("a.jpeg", image)
    test_image = Image.open("a.jpeg")

    manual_transforms2 = transforms.Compose(
        [
            transforms.Resize((80, 80), antialias=True),
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
        ]
    )
    test_image = manual_transforms2(test_image)
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(test_image.unsqueeze(0).to(device))
    target_image_pred = output
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    target_image_pred_label = get_max(target_image_pred_label.item(), img_index)
    return class_names[target_image_pred_label]


"""
  
    * Function Name: classification
    * Input:
        * event_list: A list of image with 5 events.
    * Output:
        * list: A list of detected events found in the input images.
    * Logic:
        1rocesses each sub-image from the `event_list` one by one.
        2) Calls the `classify_event` function (assumed to exist) on each sub-image.
        3) Returns the `detected_list` containing all predicted events from the input sub-images.
    * Example Call:
        detected_events = classification(event_list)
        print("Detected events:", detected_events)  # Output: ["combat", "fire", ...]
    
"""


def classification(event_list):
    detected_list = []
    for img_index in range(0, 5):
        img = event_list[img_index]
        detected_event = classify_event(img, img_index)
        if detected_event == combat:
            detected_list.append("combat")
        if detected_event == human_aid_rehabilitation:
            detected_list.append("human_aid_rehabilitation")
        if detected_event == military_vehicles:
            detected_list.append("military_vehicles")
        if detected_event == fire:
            detected_list.append("fire")
        if detected_event == destroyed_buildings:
            detected_list.append("destroyed_buildings")
    return detected_list


"""

    * Function Name: detect_ArUco_details
    * Input: image (numpy.ndarray) - input image containing ArUco markers
    * Output: ArUco_details_dict (dict)
              ArUco_corners (dict) - dictionaries containing details of ArUco markers
    * Logic: 
        1) This function takes CV2 image as a parameter.
        2) The arucos are detected and there is perspective transformation of
        the CV2 image.
        3) Event_identification and classification functions are called inorder to 
        run the model and make predictions on the images.
        4) The images are also passed through filter which checks if the image is 
        present in the box or is it blank.
        5) After getting the respective predictions such as Fire, ... or Blank, the
        respective bounding boxes are drawn and the respective classified class is 
        printed above the bounding box.
        6) The CV2 image is displayed on another window.
    * Example Call: ArUco_details, ArUco_corners = detect_ArUco_details(image)
    
"""


def detect_ArUco_details(image):
    global detected, count, final_pts, global_points, details_aruco, init_flag_static, win_counter
    ArUco_details_dict = {}
    ArUco_corners = {}
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(image)
    desired_ids = [5, 7, 6, 4]
    marker_corners = []
    marker_id_order = []
    if ids is not None:
        for i in range(len(ids)):
            if ids[i][0] in desired_ids:
                marker_corners.append(corners[i][0])
                marker_id = int(ids[i][0])
                marker_id_order.append(marker_id)
                corners_list = corners[i][0]

                center_x = int(np.mean(corners_list[:, 0]))
                center_y = int(np.mean(corners_list[:, 1]))

                dx = corners_list[1][0] - corners_list[0][0]
                dy = corners_list[1][1] - corners_list[0][1]
                marker_orientation = int((180.0 / math.pi) * math.atan2(dy, dx))
                ArUco_details_dict[marker_id] = [
                    [center_x, center_y],
                    marker_orientation,
                ]
                ArUco_corners[marker_id] = corners_list.tolist()

        if len(marker_corners) >= 4:
            marker_centers = np.mean(np.array(marker_corners), axis=1).astype(int)
            combined_lists = list(zip(marker_id_order, marker_centers))
            sorted_combined_lists = sorted(combined_lists, key=lambda x: x[0])
            if count < 15:
                sorted_list1, final_pts = zip(*sorted_combined_lists)
                count += 1

            paper = image
            pts1 = np.float32(final_pts)
            pts2 = np.float32([[1000, 0], [0, 0], [1000, 1000], [0, 1000]])
            if pts1 is not None:
                global_points = pts1.copy()
            M = cv2.getPerspectiveTransform(global_points, pts2)
            dst = cv2.warpPerspective(paper, M, (1000, 1000))

            ## INITIAL ARUCO MAPPING FOR 200 ITERATIONS
            # if init_flag_static == 1:
            #     for i in range(0, 200):
            #         temp = all_aruco(dst)
            #         print(len(temp))
            #         details_aruco = temp
            #     if 100 in details_aruco:
            #         del details_aruco[100]
            #     init_flag_static = 0

            event_list = event_identification(dst)
            filter(dst)
            detected = classification(event_list)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (0, 255, 0)
            font_thickness = 2

            if events["A"] == 1:
                text_position = (185, 862)
                cv2.putText(
                    dst,
                    detected[0],
                    text_position,
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )
                cv2.rectangle(dst, (190, 869), (286, 972), (0, 255, 0), 4)
            else:
                text_position = (185, 862)
                cv2.putText(
                    dst,
                    "Blank",
                    text_position,
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )
                cv2.rectangle(dst, (190, 869), (286, 972), (0, 255, 0), 4)

            if events["B"] == 1:
                text_position = (658, 664)
                cv2.putText(
                    dst,
                    detected[1],
                    text_position,
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )
                cv2.rectangle(dst, (663, 671), (763, 771), (0, 255, 0), 4)
            else:
                text_position = (658, 664)
                cv2.putText(
                    dst,
                    "Blank",
                    text_position,
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )
                cv2.rectangle(dst, (663, 671), (763, 771), (0, 255, 0), 4)
            if events["C"] == 1:
                text_position = (666, 456)
                cv2.putText(
                    dst,
                    detected[2],
                    text_position,
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )
                cv2.rectangle(dst, (671, 463), (771, 563), (0, 255, 0), 4)
            else:
                text_position = (666, 456)
                cv2.putText(
                    dst,
                    "Blank",
                    text_position,
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )
                cv2.rectangle(dst, (671, 463), (771, 563), (0, 255, 0), 4)
            if events["D"] == 1:
                text_position = (167, 451)
                cv2.putText(
                    dst,
                    detected[3],
                    text_position,
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )
                cv2.rectangle(dst, (172, 458), (272, 558), (0, 255, 0), 4)
            else:
                text_position = (167, 451)
                cv2.putText(
                    dst,
                    "Blank",
                    text_position,
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )

                cv2.rectangle(dst, (172, 458), (272, 558), (0, 255, 0), 4)
            if events["E"] == 1:
                text_position = (188, 114)
                cv2.putText(
                    dst,
                    detected[4],
                    text_position,
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )
                cv2.rectangle(dst, (193, 121), (293, 223), (0, 255, 0), 4)
            else:
                text_position = (188, 114)
                cv2.putText(
                    dst,
                    "Blank",
                    text_position,
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )
                cv2.rectangle(dst, (193, 121), (293, 223), (0, 255, 0), 4)
            if win_counter >= 25:
                cv2.imshow("Arena Feed", dst)
                cv2.setWindowProperty(
                    "Arena Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                )
                key = cv2.waitKey(1) & 0xFF
            win_counter = win_counter + 1
        else:
            paper = image
            pts2 = np.float32([[1000, 0], [0, 0], [1000, 1000], [0, 1000]])
            global_points = global_points.astype(np.float32)
            M = cv2.getPerspectiveTransform(global_points, pts2)
            dst = cv2.warpPerspective(paper, M, (1000, 1000))
            event_list = event_identification(dst)
            filter(dst)
            detected = classification(event_list)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (0, 255, 0)
            font_thickness = 2

            if events["A"] == 1:
                text_position = (185, 862)
                cv2.putText(
                    dst,
                    detected[0],
                    text_position,
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )
                cv2.rectangle(dst, (190, 869), (286, 972), (0, 255, 0), 4)
            else:
                text_position = (185, 862)
                cv2.putText(
                    dst,
                    "Blank",
                    text_position,
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )
                cv2.rectangle(dst, (190, 869), (286, 972), (0, 255, 0), 4)

            if events["B"] == 1:
                text_position = (658, 664)
                cv2.putText(
                    dst,
                    detected[1],
                    text_position,
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )
                cv2.rectangle(dst, (663, 671), (763, 771), (0, 255, 0), 4)
            else:
                text_position = (658, 664)
                cv2.putText(
                    dst,
                    "Blank",
                    text_position,
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )
                cv2.rectangle(dst, (663, 671), (763, 771), (0, 255, 0), 4)
            if events["C"] == 1:
                text_position = (666, 456)
                cv2.putText(
                    dst,
                    detected[2],
                    text_position,
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )
                cv2.rectangle(dst, (671, 463), (771, 563), (0, 255, 0), 4)
            else:
                text_position = (666, 456)
                cv2.putText(
                    dst,
                    "Blank",
                    text_position,
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )
                cv2.rectangle(dst, (671, 463), (771, 563), (0, 255, 0), 4)
            if events["D"] == 1:
                text_position = (167, 451)
                cv2.putText(
                    dst,
                    detected[3],
                    text_position,
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )
                cv2.rectangle(dst, (172, 458), (272, 558), (0, 255, 0), 4)
            else:
                text_position = (167, 451)
                cv2.putText(
                    dst,
                    "Blank",
                    text_position,
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )

                cv2.rectangle(dst, (172, 458), (272, 558), (0, 255, 0), 4)
            if events["E"] == 1:
                text_position = (188, 114)
                cv2.putText(
                    dst,
                    detected[4],
                    text_position,
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )
                cv2.rectangle(dst, (193, 121), (293, 223), (0, 255, 0), 4)
            else:
                text_position = (188, 114)
                cv2.putText(
                    dst,
                    "Blank",
                    text_position,
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )
                cv2.rectangle(dst, (193, 121), (293, 223), (0, 255, 0), 4)
            if win_counter >= 100:
                cv2.imshow("Arena Feed", dst)
                cv2.setWindowProperty(
                    "Arena Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                )
                key = cv2.waitKey(1) & 0xFF
            win_counter = win_counter + 1
    return ArUco_details_dict, ArUco_corners


"""
    * Function Name: mark_ArUco_image
    * Input:
        * `image`: An OpenCV image object representing the input image.
        * `ArUco_details_dict`: A dictionary containing detected ArUco marker IDs as keys 
        and a list [center, rotation_vector] as values, where:
            - `center`: (x, y) coordinates of the marker center.
            - `rotation_vector`: Rotation vector describing the marker orientation.
        * `ArUco_corners`: A list of corner coordinates for each detected ArUco marker.
    * Output:
        * `image`: The input image with circles drawn around marker centers, IDs displayed
        next to the centers, and potentially other visual markings (not implemented here).
    * Logic:
        1. Iterates through each detected ArUco marker ID (`ids`) and its details in the 
        `ArUco_details_dict`.
        2. Extracts the marker center coordinates (`center`) from the details.
        3. Draws a blue circle around the marker center on the image.
        4. Calculates the center of the top-left and top-right corners of the marker.
        5. Calculates the distance between the marker center and the top-left/top-right center.
        6. Uses this distance as an offset to position the marker ID text next to the center.
        7. Displays the marker ID on the image in blue with white outline.
    * Example Call: 
        image = cv2.imread("image.jpg")
        marked_image = mark_ArUco_image(image, ArUco_details_dict, ArUco_corners)
        cv2.imshow("Marked Image", marked_image)
        cv2.waitKey(0)

"""


def mark_ArUco_image(image, ArUco_details_dict, ArUco_corners):
    for ids, details in ArUco_details_dict.items():
        center = details[0]
        cv2.circle(image, center, 5, (0, 0, 255), -1)
        corner = ArUco_corners[int(ids)]
        tl_tr_center_x = int((corner[0][0] + corner[1][0]) / 2)
        tl_tr_center_y = int((corner[0][1] + corner[1][1]) / 2)
        display_offset = int(
            math.sqrt(
                (tl_tr_center_x - center[0]) ** 2 + (tl_tr_center_y - center[1]) ** 2
            )
        )
        cv2.putText(
            image,
            str(ids),
            (center[0] + int(display_offset / 2), center[1]),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 0, 255),
            2,
        )
    return image


"""
    * Function Name: task_4a_return
    * Input: None (uses global variables `detected` and `events`)
    * Output:
        * Dictionary `identified_labels`: Containing letters ("A", "B", ...) as 
        keys and corresponding labels from the `detected` list as values, but 
        only includes labels where event was detected in the filter function.
        (events["A"] == 1, etc.).
    * Logic:
        1. Initializes an empty dictionary `identified_labels`.
        2. Iterates through each letter ("A", "B", ..., "E").
        3. Checks if the event label (key) in the `events` dictionary is 1 (presence of event).
        4. If the value is 1, gets the label for that letter from the `detected` list and adds 
        it as the value
        in the `identified_labels` dictionary with the letter as the key.
        5. Otherwise, adds an empty string to `identified_labels` for that letter.
        6. Returns the `identified_labels` dictionary with the event letters and 
        their labels as keys and values respectively .
    * Example Call:
        identified_labels = task_4a_return()
        print(identified_labels)  # Output: {'A': 'fire', 'B': '', 'C': 'destroyed_buildings', ...}
"""


def task_4a_return():
    identified_labels = {}
    global detected
    identified_labels = {
        "A": str(detected[0] if events["A"] == 1 else ""),
        "B": str(detected[1] if events["B"] == 1 else ""),
        "C": str(detected[2] if events["C"] == 1 else ""),
        "D": str(detected[3] if events["D"] == 1 else ""),
        "E": str(detected[4] if events["E"] == 1 else ""),
    }
    return identified_labels


"""
    * Function Name: filter_list
    * Input:
        * `identified_labels`: A dictionary containing letters ("A", "B", ...) as keys and corresponding labels
        from the `detected` list as values.
    * Output:
        * List `final_list`: Containing only the letters from `identified_labels` where the corresponding label
        matches specific keywords ("fire", "destroyed_buildings", etc.).
    * Logic:
        1. Initializes an empty list `final_list`.
        2. Iterates through each letter in `identified_labels`.
        3. Checks if the corresponding label matches a specific keyword ("fire", "destroyed_buildings", etc.).
        4. If there's a match, adds the letter to `final_list`.
        5. Repeats for all keywords and letters in `identified_labels`.
        6. Returns the final list of letters with labels matching the specified keywords.
    * Example Call:
        labels = {"A": "fire", "B": "military_vehicles", "C": "buildings"}
        filtered_list = filter_list(labels)
        print(filtered_list)  # Output: ['A', 'B']
"""


def filter_list(identified_labels):
    for i in identified_labels.keys():
        if identified_labels[i] == "fire":
            final_list.append(i)

    for i in identified_labels.keys():
        if identified_labels[i] == "destroyed_buildings":
            final_list.append(i)

    for i in identified_labels.keys():
        if identified_labels[i] == "human_aid_rehabilitation":
            final_list.append(i)

    for i in identified_labels.keys():
        if identified_labels[i] == "military_vehicles":
            final_list.append(i)

    for i in identified_labels.keys():
        if identified_labels[i] == "combat":
            final_list.append(i)

    return final_list


###############	Main Function	#################

#################### MODEL ####################

if __name__ == "__main__":

    """
    Logic:
    1)
        - Load a pre-trained model (`model5B.pth`).
        - Define parameters like marker type (`aruco`) and video path (`vid.mp4`).
        - Open the video capture and check for errors.
        - Set video resolution.
        - Initialize variables for time tracking and duration.
    2)
        - Loop for a specified duration.
        - Read each frame from the video.
        - Detect Aruco markers in the frame (`detect_ArUco_details`).
        - Mark the identified markers on the image (`mark_ArUco_image`).
        - Get identified labels from marked areas (`task_4a_return`).
        - Release video capture and close windows.
    3)
        - Filter and order the identified labels (`filter_list`).
        - Create a dictionary with non-empty labels.
        - Copy the dictionary for further processing.
        - Map identified labels to human-readable names.
    4.
        - Print the final predicted events as JSON and the priority-ordered list.
    """

    flag_path = 1
    model_path = r"model5B.pth"
    model = torch.load(model_path, map_location="cpu")
    model.eval()
    marker = "aruco"
    cap = cv2.VideoCapture("vid.mp4")
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    set_resolution(cap, 1440, 900)
    start_time = time.time()
    duration = 20

    while time.time() - start_time < duration:
        ret, img = cap.read()
        if not ret:
            break
        ArUco_details_dict, ArUco_corners = detect_ArUco_details(img)
        img = mark_ArUco_image(img, ArUco_details_dict, ArUco_corners)
        identified_labels = task_4a_return()
    cap.release()
    cv2.destroyAllWindows()
    priority_order_list = filter_list(identified_labels)
    final_dict = {}
    for i in identified_labels:
        if identified_labels[i] != "":
            final_dict[i] = identified_labels[i]
    event_x = final_dict.copy()
    for key, value in event_x.items():
        if value == "fire":
            event_x[key] = "Fire"
        if value == "destroyed_buildings":
            event_x[key] = "Destroyed Buildings"
        if value == "human_aid_rehabilitation":
            event_x[key] = "Humanitarian Aid and rehabilitation"
        if value == "military_vehicles":
            event_x[key] = "Military Vehicles"
        if value == "combat":
            event_x[key] = "Combat"

    print("-" * 64)
    print("FINAL PREDICTED EVENTS : ", json.dumps(event_x))
    print("FINAL LIST : ", priority_order_list)
    print("-" * 64)

    #################### PATH GENERATOR ####################

    path_gen = path_gen()
    data = path_gen.gendirect(path_gen.path_find(priority_order_list))
    nodes = data[1]
    direct = data[0]
    nodes_str = ""
    for i in nodes:
        nodes_str = nodes_str + str(i) + " "
    nodes_str = nodes_str + "z"
    direct_str = ""
    for i in direct:
        direct_str = direct_str + str(i)
    direct_str = direct_str + "z"

    # print(direct_str)
    # print(nodes_str)

    #################### ARUCO AND BOT COMS ####################

    lat_lon = read_csv()
    marker = "aruco"
    cap = cv2.VideoCapture("vid.mp4")
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    set_resolution(cap, 1280, 720)
    # set_resolution(cap, 1595, 999)
    # set_resolution(cap, 1440, 900)
    ip = "192.168.0.108"
    # ip = "10.42.0.1"
    flag_u = 0
    flag_c = 0
    u_turn = "no"
    start = 0
    end = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((ip, 8002))
        s.listen()
        conn, addr = s.accept()
        node_count = 0
        with conn:
            inputs = [conn]
            print(f"Connected by {addr}")
            while True:
                ret, img = cap.read()
                if not ret:
                    # print("Error: Failed to capture frame.")
                    break

                if flag_path:
                    conn.sendall(str.encode(str(direct_str)))
                    conn.sendall(str.encode(str(nodes_str)))
                    flag_path = 0

                ArUco_details_dict, ArUco_corners = aruco_details(img, lat_lon)
                img = mark_ArUco_image(img, ArUco_details_dict, ArUco_corners)
                if prev_is_sent != is_sent:
                    for key, value in is_sent.items():
                        if prev_is_sent[key] != value:
                            node = key
                            # print("KEY  :", key)
                            if priority_order_list[node_count] == node:
                                string = "eventz"
                                node_count = node_count + 1
                            else:

                                string = "nodez"
                                reset_counter = 0
                            # print(string)
                            conn.sendall(str.encode(str(string)))
                            # print("Sent")
                # print("COUNTER : " ,reset_counter)
                if reset_counter >= 0 and reset_counter < 100:
                    reset_counter += 1
                if reset_counter >= 100:
                    reset_counter = -1
                    prev_is_sent = default_is_sent.copy()
                    is_sent = default_is_sent.copy()

                prev_is_sent = is_sent.copy()

                readable, _, _ = select.select(inputs, [], [], 0)
                if conn in readable:
                    try:
                        data = conn.recv(1024)
                        print(data)
                        byte_data = data
                        string_data = byte_data.decode("utf-8")
                        print(string_data)
                        data = string_data
                        if data == "U TURN":
                            u_turn = "yes"
                            flag_u = 0
                    except socket.error as e:
                        print(e)

                if u_turn == "yes" and flag_u == 0:
                    flag_c = 1
                    u_dict_initial = all_aruco(img)
                    if 100 in u_dict_initial:
                        bot_aruco_initial = u_dict_initial[100]
                        bot_orient_initial = bot_aruco_initial[1]
                        # print(bot_orient_initial)
                        if bot_orient_initial - 150 < 0:
                            start = 360 + bot_orient_initial - 150
                            end = start - 35
                        else:
                            start = bot_orient_initial - 150
                            end = bot_orient_initial - 190
                        flag_u = 1
                    # print("start ",start, "  end ", end)
                if flag_c:
                    u_dict_cur = all_aruco(img)
                    if 100 in u_dict_cur:
                        bot_aruco_cur = u_dict_cur[100]
                        bot_orient_cur = bot_aruco_cur[1]
                        current = bot_orient_cur
                        # print(current)
                        if current < start and current > end:
                            u_turn = "no"
                            u_flag = 0
                            flag_c = 0
                            # print("Uturn done")
                            conn.sendall(str.encode(str("stopz")))

    cap.release()
    cv2.destroyAllWindows()
