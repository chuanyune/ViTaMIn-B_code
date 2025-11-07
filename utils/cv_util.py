from typing import Dict, Tuple

import copy
import numpy as np
import math
import cv2
import scipy.interpolate as si

# =================== intrinsics ===================

def parse_fisheye_intrinsics(json_data: dict) -> Dict[str, np.ndarray]:
    """
    Reads camera intrinsics from OpenCameraImuCalibration to opencv format.
    Example:
    {
        "final_reproj_error": 0.17053819312281043,
        "fps": 60.0,
        "image_height": 1080,
        "image_width": 1920,
        "intrinsic_type": "FISHEYE",
        "intrinsics": {
            "aspect_ratio": 1.0026582765352035,
            "focal_length": 420.56809123853304,
            "principal_pt_x": 959.857586309181,
            "principal_pt_y": 542.8155851051391,
            "radial_distortion_1": -0.011968137016185161,
            "radial_distortion_2": -0.03929790706019372,
            "radial_distortion_3": 0.018577224235396064,
            "radial_distortion_4": -0.005075629959840777,
            "skew": 0.0
        },
        "nr_calib_images": 129,
        "stabelized": false
    }
    """
    assert json_data['intrinsic_type'] == 'FISHEYE'
    intr_data = json_data['intrinsics']
    
    # img size
    h = json_data['image_height']
    w = json_data['image_width']

    # pinhole parameters
    f = intr_data['focal_length']
    px = intr_data['principal_pt_x']
    py = intr_data['principal_pt_y']
    
    # Kannala-Brandt non-linear parameters for distortion
    kb8 = [
        intr_data['radial_distortion_1'],
        intr_data['radial_distortion_2'],
        intr_data['radial_distortion_3'],
        intr_data['radial_distortion_4']
    ]

    opencv_intr_dict = {
        'DIM': np.array([w, h], dtype=np.int64),
        'K': np.array([
            [f, 0, px],
            [0, f, py],
            [0, 0, 1]
        ], dtype=np.float64),
        'D': np.array([kb8]).T
    }
    print(opencv_intr_dict)
    return opencv_intr_dict


def convert_fisheye_intrinsics_resolution(
        opencv_intr_dict: Dict[str, np.ndarray], 
        target_resolution: Tuple[int, int]
        ) -> Dict[str, np.ndarray]:
    """
    Convert fisheye intrinsics parameter to a different resolution,
    assuming that images are not cropped in the vertical dimension,
    and only symmetrically cropped/padded in horizontal dimension.
    """
    iw, ih = opencv_intr_dict['DIM']
    # print(f"iw: {iw}, ih: {ih}")
    iK = opencv_intr_dict['K']
    ifx = iK[0,0]
    ify = iK[1,1]
    ipx = iK[0,2]
    ipy = iK[1,2]

    ow, oh = target_resolution
    ofx = ifx / ih * oh
    ofy = ify / ih * oh
    opx = (ipx - (iw / 2)) / ih * oh + (ow / 2)
    opy = ipy / ih * oh
    oK = np.array([
        [ofx, 0, opx],
        [0, ofy, opy],
        [0, 0, 1]
    ], dtype=np.float64)

    out_intr_dict = copy.deepcopy(opencv_intr_dict)
    out_intr_dict['DIM'] = np.array([ow, oh], dtype=np.int64)
    out_intr_dict['K'] = oK
    return out_intr_dict


class FisheyeRectConverter:
    def __init__(self, K, D, DIM, out_size, out_fov):
        out_size = np.array(out_size)
        # vertical fov
        out_f = (out_size[1] / 2) / np.tan(out_fov/180*np.pi/2)
        out_K = np.array([
            [out_f, 0, out_size[0]/2],
            [0, out_f, out_size[1]/2],
            [0, 0, 1]
        ], dtype=np.float32)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), out_K, out_size, cv2.CV_16SC2)

        self.map1 = map1
        self.map2 = map2
    
    def forward(self, img):
        rect_img = cv2.remap(img, 
            self.map1, self.map2,
            interpolation=cv2.INTER_AREA, 
            borderMode=cv2.BORDER_CONSTANT)
        return rect_img

# ================= ArUcO tag =====================
def parse_aruco_config(aruco_config_dict: dict):
    """
    example:
    aruco_dict:
        predefined: DICT_4X4_50
    marker_size_map: # all unit in meters
        default: 0.15
        12: 0.2
    """
    aruco_dict = get_aruco_dict(**aruco_config_dict['aruco_dict'])

    n_markers = len(aruco_dict.bytesList)
    marker_size_map = aruco_config_dict['marker_size_map']
    default_size = marker_size_map.get('default', None)
    
    out_marker_size_map = dict()
    for marker_id in range(n_markers):
        size = default_size
        if marker_id in marker_size_map:
            size = marker_size_map[marker_id]
        out_marker_size_map[marker_id] = size
    
    result = {
        'aruco_dict': aruco_dict,
        'marker_size_map': out_marker_size_map
    }
    return result


def get_aruco_dict(predefined:str
                   ) -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, predefined))

def detect_localize_aruco_tags(
        img: np.ndarray, 
        aruco_dict: cv2.aruco.Dictionary, 
        marker_size_map: Dict[int, float], 
        fisheye_intr_dict: Dict[str, np.ndarray], 
        refine_subpix: bool=True):
    K = fisheye_intr_dict['K']
    D = fisheye_intr_dict['D']
    param = cv2.aruco.DetectorParameters()
    param.adaptiveThreshConstant = 8 # Lower this value if marker contrast is low
    if refine_subpix:
        param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    
    # Use new OpenCV 4.7+ ArUco API
    detector = cv2.aruco.ArucoDetector(aruco_dict, param)
    corners, ids, rejectedImgPoints = detector.detectMarkers(img)
    
    if ids is None or len(corners) == 0:
        return dict()

    tag_dict = dict()
    for this_id, this_corners in zip(ids, corners):
        this_id = int(this_id[0])
        if this_id not in marker_size_map:
            continue
        
        marker_size_m = marker_size_map[this_id]
        undistorted = cv2.fisheye.undistortPoints(this_corners, K, D, P=K)
        
        # Create 3D object points for ArUco marker (standard order: TL, TR, BR, BL)
        object_points = np.array([
            [-marker_size_m/2, marker_size_m/2, 0],   # Top-left
            [marker_size_m/2, marker_size_m/2, 0],    # Top-right
            [marker_size_m/2, -marker_size_m/2, 0],   # Bottom-right
            [-marker_size_m/2, -marker_size_m/2, 0]   # Bottom-left
        ], dtype=np.float32)
        
        # Use solvePnP instead of estimatePoseSingleMarkers
        success, rvec, tvec = cv2.solvePnP(
            object_points, undistorted.reshape(-1, 2), K, np.zeros((1,5)))
        
        if success:
            tag_dict[this_id] = {
                'rvec': rvec.squeeze(),
                'tvec': tvec.squeeze(),
                'corners': this_corners.squeeze()
            }
    return tag_dict

def get_gripper_width(tag_dict, left_id, right_id, nominal_z): 
    left_x = None
    if left_id in tag_dict:
        tvec = tag_dict[left_id]['tvec']
        left_x = tvec[0]

    right_x = None
    if right_id in tag_dict:
        tvec = tag_dict[right_id]['tvec']
        right_x = tvec[0]

    width = None
    if (left_x is not None) and (right_x is not None):
        width = np.abs(right_x - left_x)
    elif left_x is not None:
        width = abs(left_x) * 2
    elif right_x is not None:
        width = abs(right_x) * 2
    return width




def draw_slam_mask(img): # only for baseline in vitamin b
    points = [(239, 619),
                (302, 574),
                (333, 591),
                (428, 539),
                (540, 493),
                (699, 490),
                (853, 548),
                (957, 604),
                (1038, 537),
                (1080, 457),
                (1096, 324),
                (1279, 508),
                (1279, 799),
                (0, 799)]
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(img, [points], (0, 0, 0), lineType=cv2.LINE_AA)
    return img

def draw_fisheye_mask(img, radius=None, center=None, fill_color=(255, 255, 255)):
    h, w = img.shape[:2]
    
    # Default center is image center
    if center is None:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = center
    
    # Default radius is a reasonable size (adjustable based on specific fisheye lens)
    if radius is None:
        radius = min(w, h) // 2 - 50  # Leave some margin to avoid cutting effective area
    
    # Create circular mask: 255 inside circle, 0 outside
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    mask3 = cv2.merge([mask, mask, mask])  # Convert to 3 channels
    
    # Create fill background
    fill_bg = np.full_like(img, fill_color, dtype=img.dtype)
    
    # Keep original image inside circle, use fill color outside
    result = np.where(mask3 == 255, img, fill_bg)
    
    return result

def inpaint_tag(img, corners, tag_scale, n_samples=16):
    # scale corners with respect to geometric center
    center = np.mean(corners, axis=0)
    scaled_corners = tag_scale * (corners - center) + center
    
    # sample pixels on the boundary to obtain median color
    sample_points = si.interp1d(
        [0,1,2,3,4], list(scaled_corners) + [scaled_corners[0]], 
        axis=0)(np.linspace(0,4,n_samples)).astype(np.int32)
    sample_colors = img[
        np.clip(sample_points[:,1], 0, img.shape[0]-1), 
        np.clip(sample_points[:,0], 0, img.shape[1]-1)
    ]
    median_color = np.median(sample_colors, axis=0).astype(img.dtype)
    
    # draw tag with median color
    img = cv2.fillPoly(
        img, scaled_corners[None,...].astype(np.int32), 
        color=median_color.tolist())
    return img

# =========== other utils ====================
def get_fisheye_image_transform(in_res, out_res, crop_ratio:float = 1.0, bgr_to_rgb: bool=False):
    iw, ih = in_res
    ow, oh = out_res
    ch = round(ih * crop_ratio)
    cw = round(ih * crop_ratio / oh * ow)
    interp_method = cv2.INTER_AREA

    w_slice_start = (iw - cw) // 2
    w_slice = slice(w_slice_start, w_slice_start + cw)
    h_slice_start = (ih - ch) // 2
    h_slice = slice(h_slice_start, h_slice_start + ch)
    c_slice = slice(None)
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)

    def transform(img: np.ndarray):
        assert img.shape == ((ih,iw,3))
        # crop
        img = img[h_slice, w_slice, c_slice]
        # resize
        img = cv2.resize(img, out_res, interpolation=interp_method)
        return img
    
    return transform

# TODO: Directly copied, but definitely not reasonable, needs modification.
def get_tactile_image_transform(in_res, out_res, crop_ratio:float = 1.0, bgr_to_rgb: bool=False):
    iw, ih = in_res
    ow, oh = out_res
    ch = round(ih * crop_ratio)
    cw = round(ih * crop_ratio / oh * ow)
    interp_method = cv2.INTER_AREA

    w_slice_start = (iw - cw) // 2
    w_slice = slice(w_slice_start, w_slice_start + cw)
    h_slice_start = (ih - ch) // 2
    h_slice = slice(h_slice_start, h_slice_start + ch)
    c_slice = slice(None)
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)

    def transform(img: np.ndarray):
        # Flexible shape check - allow for different input resolutions
        if img.shape != (ih, iw, 3):
            print(f"Warning: Expected image shape ({ih}, {iw}, 3), got {img.shape}. Adjusting...")
            # Simply resize without cropping if dimensions don't match
            img = cv2.resize(img, out_res, interpolation=interp_method)
            return img
        
        # crop
        img = img[h_slice, w_slice, c_slice]
        # resize
        img = cv2.resize(img, out_res, interpolation=interp_method)
        return img
    
    return transform