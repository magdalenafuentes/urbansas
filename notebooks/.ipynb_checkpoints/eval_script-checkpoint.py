import json
import os
import glob
import librosa
import numpy as np
import networkx as nx
from tqdm import tqdm
import h5py
# import moviepy.editor as mpy
import skimage.transform
from skimage import img_as_float
from sklearn.metrics import ndcg_score
# from moviepy.editor import VideoFileClip
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

#from eval_script import eval, vis

SR = 48000
MAX_LEN_ID = 11
IMAGE_SHAPE = (224, 224)

# Video helper functions
def video_to_size(cm, shape=IMAGE_SHAPE):
    return skimage.transform.resize(cm, shape, order=0, preserve_range=True, anti_aliasing=False)

def video2audio_fname(fname, ext=None):
    # convert: tau2021aus/video/street_traffic-barcelona-161-4901.mp4
    #      to: tau2021aus/audio/street_traffic-barcelona-161-4901.wav
    fdir, _, fbase = fname.rsplit('/', 2)
    fstem = fbase.split('.')[0]
    fname = os.path.join(fdir, 'audio', fstem)
    if not ext:
        fname += '.*'
        fs = glob.glob(fname)
        assert fs, f'No files matching {fname}'
        return fs[0]
    return fname + ext

def load_sample(vid_fname, aud_fname, hop_size=0.1, fps=None, sr=SR, **kw):
    win_size = sr
    y, _ = librosa.load(aud_fname, sr=sr)
    aud = librosa.util.frame(
        librosa.util.pad_center(y, int((len(y) + win_size-1))), 
        frame_length=sr, hop_length=int(sr * hop_size)).T[:,None]
    
    clip = VideoFileClip(vid_fname)
    
    vid = np.array([
        video_to_size(clip.get_frame(i), **kw) 
        for i in np.linspace(0, len(y)/sr, len(aud))])
    vid = vid / 255 * 2 - 1
    return vid, aud


# def predict_cm_video_grid(cmodel, *video_fnames, **kw):  # load, predict, overlay correspondence
#     vids = []
#     for i, f in enumerate(video_fnames):
#         print(f'{i+1}/{len(video_fnames)}', f)
#         vid, aud = load_sample(f, video2audio_fname(f))
#         print('inputs:', vid.shape, vid.min(), vid.max(), aud.shape, aud.min(), aud.max())
#         corr = cmodel.predict([ vid, aud ])
#         print('correspondence:', corr.shape, corr.min(), corr.max())
#         vids.append(vis.make_video_overlay_frames(vid, corr, **kw))

#     return vis.frames_to_clip(vis.tile_overlay_frames(vids))

def load_av_data_from_files(video_fnames, audio_fnames, **kw):
    for vf, af in zip(video_fnames, audio_fnames):
        yield load_sample(vf, af, **kw)
        
def videos_to_h5(video_fnames, audio_fnames, **kw):
    n = len(video_fnames)
    with h5py.File(os.path.join(SHARE_DIR), 'a') as f:
        audio_files = f.require_dataset('audio_file', (n,))
        for vf, af in zip(video_fnames, audio_fnames):
            v, a = load_sample(vf, af, prepare=False, **kw)


def create_mask(coords, existing_mask=None, frame_size=(720, 1280)):
    '''
    Generate a mask given [x1,y1,x2,y2] coordinates.

    coords: 
        [x1,y1,x2,y2] coordinates of top-left and bottom right corner of box.
    existing_mask:
        if we have an existing mask we will update with the new coordinates we
        can pass this here, otherwise a mask of zeroes will be initialized.
    frame_size: 
        size of overall frame. a new mask created will be of this shape.

    Returns
    -------
    mask: 
        binary np.ndarray mask
    '''
    if existing_mask is not None:
        if existing_mask.shape != frame_size:
            raise ValueError(f"Existing mask is of shape {existing_mask.shape}, but target frame size is {frame_size} These must match.")
        mask = existing_mask
    else:
        mask = np.zeros(frame_size)

    if not np.any(coords) or not coords:
        return np.zeros(frame_size)
    
    curr_x = (coords[0], coords[2])
    curr_y = (coords[1], coords[3])
    mask[curr_y[0]:curr_y[1]+1, curr_x[0]:curr_x[1]+1] = 1

    return mask      


# IOU Scores     
def iou_score(pred_mask, gt_bbox_coords=None, gt_box_mask=None, th=0.5, frame_size=(720, 1280), target_size=None):
    '''
    Computes Intersection over Union (IoU) for a given mask and bounding boxes.
    Assumes that bboxes are according to original size, and box is in format [x1, y1, x2, y2] 
    where (x1,y1) is the upper left corner and (x2,y2) the lower right corner. This is for 
    one *class* at a time currently.

    pred_mask: 
        mask thats the size of the entire image (likelihood), predictions for one frame
    gt_bbox_coords:  
        list of coordinates from ground truth [x1, y1, x2, y2] 
    gt_box_mask: 
        ground truth box mask (if passed directly instead of creating)
    th: 
        threshold tau that determines how to binarize the predicted max (which has likelihood vals)
    frame_size: 
        size of the input frames (should be the same for pred/gt)
    target_size: 
        desired output frame size of both masks

    Returns
    -------
    (iou, gt_box_mask, pred_mask): 
        iou: IoU score computed (float)
        gt_box_mask: reshaped ground truth box mask (2D np.ndarray, shape=target_size)
        pred_mask: reshaped prediction mask (2D np.ndarray, shape=target_size)
        
    '''
    if gt_bbox_coords is None and gt_box_mask is None:
        raise ValueError("Need to pass either box or box mask")

    if len(pred_mask.shape)>2:
        pred_mask = pred_mask[...,0]
        
    # Convert the likelihood thresholds to a binary mask
    pred_mask = (pred_mask>th)**1 
    
    if target_size is None:
        target_size = frame_size

    if gt_box_mask is None:
        print('creating box mask')
        gt_box_mask = np.zeros(frame_size)

        # Make the overall mask
        for _box in gt_bbox_coords:
            if _box is None or sum(_box) == 0:
                continue
            # +1 is to take the "borders" into account
            gt_box_mask = create_mask(coords=_box, existing_mask=gt_box_mask, frame_size=frame_size)
        
        gt_box_mask = video_to_size(gt_box_mask, target_size)

    # We can adjust resolution if necessary
    pred_mask = video_to_size(pred_mask, target_size)

    # Intersection, multiply and sum the 1/0s
    overlap =  np.sum(pred_mask * gt_box_mask) 

    # Union: elementwise subtraction
    union = np.sum((pred_mask - gt_box_mask>0) + gt_box_mask)
    iou = overlap / union if union else 1 # This is where 1 is assigned to empty frames
    # Empty -> doesn't have GT and didn't predict anything
    return (iou, gt_box_mask, pred_mask)

def iou_frame(annot, pred, frame_id, **kw):
    '''
    Computes the IoU score for a single frame.

    annot: 
        video annotations for one video
    pred: 
        predictions for one video
    frame_id: 
        single frame ID, this allows us to filter down to evaluate only
        one frame at a time 

    Returns
    -------
    See return for `iou_score`. Returns this for the given `frame_id`.
    
    '''
    _annot = annot[annot.frame_id==frame_id]
    gth = []
    for _, a in _annot.iterrows():
        # ground truth 
        # this converts GT to x1,y1 x2,y2
        # the predictions
        gth.append(np.array([a.x, a.y, a.x + a.w, a.y + a.h]).astype(int))
    return iou_score(pred, gth, **kw) # each row in annotations -> one BB (in that frame)


def iou_frame_1D(annot, pred, frame_id, **kw):
    '''
    Computes the IoU score for a single frame, but for height=1 boxes. We use this
    as we evaluate the vertical regions of frames.

    annot: 
        video annotations for one video
    pred: 
        predictions for one video
    frame_id: 
        single frame ID, this allows us to filter down to evaluate only
        one frame at a time 

    Returns
    -------
    See return for `iou_score`. Returns this for the given `frame_id`.
    '''
    _annot = annot[annot.frame_id==frame_id]
    gth = []
    for _, a in _annot.iterrows():
        gth.append(np.array([a.x, 0, a.x + a.w, 1]).astype(int))
    return iou_score(pred, gth, **kw)


def iou_video(annot, corrs, **kw):
    '''
    Compute an array of frame by frame IOU scores for a video.
    
    annot:
        video annotations for one video
    corr:
    
    Returns
    -------
    See return for `iou_frame` and `iou_score`. Returns this for each frame in a video.
    '''
#     print(len(annot.frame_id.unique()),len(corrs))
    return [iou_frame(annot, pred, frame_id, **kw) for frame_id, pred in enumerate(corrs, 1)]


def eval_video(vfname, annotations, corrs, **kw):
    '''
    Compute an array of frame-by-frame IOU scores given a video filename.
    
    vfname:
        video filename
    annotations:
        video annotations for the given video filename.
    corrs:  

    Returns
    -------
    See return for `iou_video`. Gets final evaluation for a video. 
    
    '''

    if vfname not in annotations.filename.unique():
        raise ValueError(f'{vfname} is not a valid video name in the dataset')
    vannot = annotations[annotations.filename==vfname]
#     print([a for a in vannot.iterrows()])
    return iou_video(vannot, corrs, **kw)


## GIOU Code
def get_enclosing_mask(mask1, mask2):
    '''
    Generate coordinates [x1,y1,x2,y2] of the smallest enclosing convex
    rectangle that encloses the given two masks. 
    
    mask1:
        binary np.ndarray (1D or 2D+) representing a mask
    mask2:
        binary np.ndarray (1D or 2D+) representing a mask

    Returns
    -------
    coords: [x1,y1,x2,y2] list of coordinates of smallest enclosing mask.
    '''
    # Where these are 2D arrays of the two masks
    mask1_min_xy = min(list(zip(*np.where(mask1 == 1))))
    mask1_max_xy = max(list(zip(*np.where(mask1 == 1))))

    mask2_min_xy = min(list(zip(*np.where(mask2 == 1))))
    mask2_max_xy = max(list(zip(*np.where(mask2 == 1))))

    min_x = min(mask1_min_xy[1], mask2_min_xy[1])
    min_y = min(mask1_min_xy[0], mask2_min_xy[0])

    max_x = max(mask1_max_xy[1], mask2_max_xy[1])
    max_y = max(mask1_max_xy[0], mask2_max_xy[0])

    coords = [min_x, min_y, max_x, max_y]
    return coords

def giou_score(pred_mask, gt_mask, th=0.5, frame_size=(720, 1280), target_size=None):
    '''
    Computes Generalized Intersection over Union (GIoU) score for a given predicted
    mask and ground truth mask. **NOTE** that this metric is currently only designed to
    work when there is only one bounding box in a frame.

    The score itself is IoU - (C(AUB)/C)

    pred_mask: 
        prediction mask for one frame, (not binary - it will be in likelihood)
    gt_mask_coords:  
        coordinates of ground truth bbox [x1, y1, x2, y2] 
    th: 
        threshold tau that determines how to binarize the predicted max (which has likelihood vals)
    frame_size: 
        size of the input frames (should be the same for pred/gt)
   
    target_size: 
        desired output frame size of both masks

    Returns
    -------
    (giou, gt_mask, pred_mask, c_mask): 
        iou: IoU score computed (float)
        gt_mask: reshaped ground truth box mask (2D np.ndarray, shape=target_size)
        pred_mask: reshaped prediction mask (2D np.ndarray, shape=target_size)
        c_mask: reshaped mask of smallest rectangle enclosing gt_mask and pred_mask.
        
    '''
    # Convert coordinates to mask right away
#     gt_mask = create_mask(gt_mask_coords, frame_size=frame_size)
    # If both the prediction and the ground truth are empty (all zeros)
    if not np.any(pred_mask) and not np.any(gt_mask):
        return 1, gt_mask, pred_mask, create_mask([0,0,0,0], frame_size=frame_size)

    # One is empty the other is not
    if (not np.any(pred_mask) and np.any(gt_mask)) or (not np.any(gt_mask) and np.any(pred_mask)):
        return 0, gt_mask, pred_mask, create_mask([0,0,0,0], frame_size=frame_size)

    # Convert the likelihood thresholds to a binary mask
    pred_mask = (pred_mask>th)**1 
            
    # gt_mask = video_to_size(gt_mask, target_size)
    # pred_mask = video_to_size(pred_mask, target_size)

    overlap =  np.sum(pred_mask * gt_mask) 
    union = np.sum((pred_mask - gt_mask>0) + gt_mask)
    
    # Get enclosing mask
    get_c = get_enclosing_mask(gt_mask, pred_mask) # This gets (x,y) top left, width, height
    c_mask = create_mask(coords=get_c,
                         frame_size=frame_size)
    c_diff = np.sum(c_mask - ((pred_mask - gt_mask>0) + gt_mask))
    res = np.abs(c_diff) / np.abs(np.sum(c_mask))
    
    iou = overlap / union
    giou = iou - res
    
#     print("Norm term: ", res)
#     print("IoU: ", iou)
#     print("GIoU: ", giou)

    return giou, gt_mask, pred_mask, c_mask


def giou_frame(annot, pred, frame_id, **kw):
    '''
    Computes the GIoU score for a single frame.
    **Note** that this is only designed for single bounding boxes per frames.

    annot: 
        video annotations for one video
    pred: 
        predictions for one video
    frame_id: 
        single frame ID, this allows us to filter down to evaluate only
        one frame at a time 

    Returns
    -------
    See return for `giou_score`. Returns this for the given `frame_id`.
    
    '''
    _annot = annot[annot.frame_id==frame_id].iloc[0]
    gth = np.array([_annot.x, _annot.y, _annot.x + _annot.w, _annot.y + _annot.h]).astype(int)
    return giou_score(pred, gth, **kw) 


def giou_frame_1D(annot, pred, frame_id, **kw):
    '''
    Computes the GIoU score for a single frame, but for height=1 boxes. We use this
    as we evaluate the vertical regions of frames.
     **Note** that this is only designed for single bounding boxes per frames.

    annot: 
        video annotations for one video
    pred: 
        predictions for one video
    frame_id: 
        single frame ID, this allows us to filter down to evaluate only
        one frame at a time 

    Returns
    -------
    See return for `iou_score`. Returns this for the given `frame_id`.
    '''
    _annot = annot[annot.frame_id==frame_id].iloc[0]
    gth = np.array([_annot.x, 0, _annot.x + _annot.w, 1]).astype(int)
    return giou_score(pred, gth, **kw) 


def giou_video(annot, corrs, **kw):
    '''
    Compute an array of frame by frame GIOU scores for a video.
    
    annot:
        video annotations for one video
    corr:
    
    Returns
    -------
    See return for `iou_frame` and `iou_score`. Returns this for each frame in a video.
    '''
#     print(len(annot.frame_id.unique()),len(corrs))
    return [giou_frame(annot, pred, frame_id, **kw) for frame_id, pred in enumerate(corrs, 1)]


def giou_eval_video(vfname, annotations, corrs, **kw):
    '''
    Compute an array of frame-by-frame GIOU scores given a video filename.
    
    vfname:
        video filename
    annotations:
        video annotations for the given video filename.
    corrs:  

    Returns
    -------
    See return for `giou_video`. Gets final evaluation for a video. 
    
    '''

    if vfname not in annotations.filename.unique():
        raise ValueError(f'{vfname} is not a valid video name in the dataset')
    vannot = annotations[annotations.filename==vfname]
#     print([a for a in vannot.iterrows()])
    return giou_video(vannot, corrs, **kw)

def get_x1y1x2y2(mask):
    
    '''
    Given a binary mask, calculate [x1,y1,x2,y2] coordinate form of 
    this mask in a format convenient for plotting.
    
    mask:
        binary np.ndarray (1D or 2D+) representing a mask

    Returns
    -------
    (min_x, min_y): bottom left corner coordinates
    width: x-dimension width of box
    height: y-dimension height of box

    '''

    xy = list(zip(*np.where(mask == 1)))
    # If the mask is in 2d array form
    min_x = min([a[0] for a in xy])
    min_y = min([a[1] for a in xy])
    max_x = max([a[0] for a in xy])
    max_y = max([a[1] for a in xy])

    width = max_x-min_x
    height = max_y-min_y
    return (min_x, min_y), width, height

def plot_rects(gt_mask, pred_mask, enclosing_mask=None, frame_size=(10,10)):
    
    gt_format = get_x1y1x2y2(gt_mask)
    pred_format = get_x1y1x2y2(pred_mask)
    
    fig, ax = plt.subplots()

    ax.set_xlim([0, frame_size[1]])
    ax.set_ylim([0, frame_size[0]])
    #create simple line plot
    ax.plot()

    #add rectangle to plot
    ax.add_patch(Rectangle(gt_format[0][::-1], gt_format[2], gt_format[1],
                           ec="gray", fc="blue", zorder=10, alpha=0.5))

    ax.add_patch(Rectangle(pred_format[0][::-1], pred_format[2], pred_format[1],
                           ec="gray", fc="red", zorder=5, alpha=0.5))
    
    if enclosing_mask is not None:
        c_format = get_x1y1x2y2(enclosing_mask)
        ax.add_patch(Rectangle(c_format[0][::-1], c_format[2], c_format[1],
                           ec="red", fc="yellow", zorder=1, alpha=0.3, linewidth=2))

    #display plot
    plt.gca().invert_yaxis()
    plt.show()

# FIX
# def plot_rects_1d(gt_mask, pred_mask, enclosing_mask=None, frame_size=(10,10)):
    
#     gt_format = get_x1y1x2y2(gt_mask)
#     pred_format = get_x1y1x2y2(pred_mask)
    
#     fig, ax = plt.subplots()

#     ax.set_xlim([0, frame_size[1]])
#     ax.set_ylim([0, frame_size[0]])
#     #create simple line plot
#     ax.plot()
    
#     #add rectangle to plot
#     ax.add_patch(Rectangle(gt_format[0][::-1], gt_format[2], 1,
#                            ec="gray", fc="blue", zorder=10, alpha=0.5))

#     ax.add_patch(Rectangle(pred_format[0][::-1], pred_format[2], 1,
#                            ec="gray", fc="red", zorder=5, alpha=0.5))
    
#     if enclosing_mask is not None:
#         c_format = get_x1y1x2y2(enclosing_mask)
#         ax.add_patch(Rectangle(c_format[0][::-1], c_format[2], 1,
#                            ec="red", fc="yellow", zorder=1, alpha=0.3, linewidth=2))

#     #display plot
#     plt.gca().invert_yaxis()
#     plt.show()

# Keep this but need to figure out imagesequenceclip function
# def mask_to_clip(masked_frames, fps=2):
#     return ImageSequenceClip(masked_frames, fps=fps)
    

# # Copied from https://github.com/mgeier/python-audio/blob/master/audio-files/utility.py
# def pcm2float(sig, dtype='float64'):
#     """Convert PCM signal to floating point with a range from -1 to 1.
#     Use dtype='float32' for single precision.
#     Parameters
#     ----------
#     sig : array_like
#         Input array, must have integral type.
#     dtype : data type, optional
#         Desired (floating point) data type.
#     Returns
#     -------
#     numpy.ndarray
#         Normalized floating point data.
#     See Also
#     --------
#     float2pcm, dtype
#     """
#     sig = np.asarray(sig)
#     if sig.dtype.kind not in 'iu':
#         raise TypeError("'sig' must be an array of integers")
#     dtype = np.dtype(dtype)
#     if dtype.kind != 'f':
#         raise TypeError("'dtype' must be a floating point type")

#     i = np.iinfo(sig.dtype)
#     abs_max = 2 ** (i.bits - 1)
#     offset = i.min + abs_max
#     return (sig.astype(dtype) - offset) / abs_max