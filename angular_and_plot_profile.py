# %%
import os
import csv
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bioio import BioImage
from scipy.signal import find_peaks, peak_widths
import tifffile as tif
from tqdm import tqdm
#%%
# Parameters

GUV_CH = 0
DETECTION_SUFFIX = "_detected_vesicles.csv"

PATH = r"D:\Data\EVOLF"
IMAGE_FORMAT = '.czi'
num_angles = 360
length_excess = 1.5
dr = 1

SIZE_VIEW = length_excess

INNER_MARGIN = 1.2
OUTER_MARGIN = 2.0

PARAMETERS_PROFILES = np.array((num_angles, length_excess, dr))

PLOT_RESULTS = False

pixels_to_remove = 2
size_central_area = 1/4  # the central area radius (unit of measure: each vesicles' radius) considered during localization quantification
#%%
# Image loading and saving
def open_czi(path):
    img_bio = BioImage(path)
    img = img_bio.get_image_data()

    img, dims = squeeze_singleton_dims(img, img_bio.dims.order)

    return img

def open_tif(path):
    img = tif.imread(path)
    return img

def open_image(path):
    if path[-4:] == '.tif':
        return open_tif(path)
    if path[-4:] == '.czi':
        return open_czi(path)

def squeeze_singleton_dims(img, dims):
    keep_dims = []

    for size, dim in zip(img.shape, dims):
        if size != 1:
            keep_dims.append(dim)

    img_squeezed = np.squeeze(
        img,
        axis=tuple(axis for axis, size in enumerate(img.shape) if size == 1)
    )

    dims_squeezed = "".join(keep_dims)

    return img_squeezed, dims_squeezed

def get_output_folder(image_path, base_path):
    """
    Create and return the output folder corresponding to one image.

    Example
    -------
    base_path:
        D:\\Data\\EVOLF

    image_path:
        D:\\Data\\EVOLF\\a\\b\\file.czi

    returns:
        D:\\Data\\EVOLF\\output\\a\\b
    """

    relative_path = os.path.relpath(image_path, base_path)
    relative_folder = os.path.dirname(relative_path)

    output_folder = os.path.join(base_path, "output", relative_folder)
    os.makedirs(output_folder, exist_ok=True)

    return output_folder
# =============================================================================
# Analysis functions
# =============================================================================

def linear_profiles(channels_data, ves_coordinates, image_dim, parameters_profiles):
    """
    Calculation of the multiple linear profiles for a single vesicle.

    Input
    -----
    channels_data : np.ndarray
        The channels of the image to be analyzed.
        channels_data[0,:,:] always corresponds to the Membrane channel

    ves_coordinates : np.ndarrya
        1-dimensional array with the coordinates of the single vesicle we want 
        to focus on.  
        ves_coordinates[0]: vesicle id 
        ves_coordinates[1]: xc (in px) 
        ves_coordinates[2]: yc (in px) 
        ves_coordinates[3]: radius (in px)
        
    image_dim : np.ndarray
        The original image's dimensions. 
        image_dim[0] corresponds to the x direction.
        image_dim[1] corresponds to the y direction.
        
    parameters_profiles : np.ndarray
        The 3 important parameters for the linear profile calculation:
        parameters_profiles[0] : number of linear profiles to consider for a 
                                 single vesicle
        parameters_profiles[1] : length of the linear profiles 
                                 (unit of measure: vesicle radius)
        parameters_profiles[2] : the step (in px) for sampling along the vesicle
                                 radius
        
    Returns
    -------
    intensity_profiles : np.ndarray
        The intensity linear profiles calculated.
        Note: Data related to different channels are stored along the 3rd 
              dimension of the array. Element [:,:,0] is always based on the 
              membrane channel, element [:,:,1] is based on the single protein
              present. If both proteins are present, element [:,:,1] corresponds
              to the septin channel and element [:,:,2] corresponds to the actin
              channel.
        
    along_radius : np.ndarray
        The points along the linear profiles, starting from the center of the
        vesicle. 
    
    theta : np.ndarray
        The angles considered for the linear profiles.
        
    death_mark : bool
        If True, the vesicle's requested intensity profile is extending outside
        of the image borders and the vesicle is marked to be disregarded. 
        If False, the vesicle's position doesn't raise an issue and the analysis
        can be continued.

    """
    num_channels = channels_data.shape[0]

    xc = ves_coordinates[1]
    yc = ves_coordinates[2]
    radius = ves_coordinates[3]

    num_angles = int(parameters_profiles[0])
    length_excess = parameters_profiles[1]
    dr = parameters_profiles[2]

    profile_radius_limit = length_excess * radius

    theta = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    along_radius = np.arange(0, int(profile_radius_limit), dr)
    profile_radius = len(along_radius)

    intensity_profiles = np.zeros((profile_radius, num_angles, num_channels))

    death_mark = False

    if (
        xc + profile_radius_limit >= image_dim[0]
        or xc - profile_radius_limit < 0
        or yc + profile_radius_limit >= image_dim[1]
        or yc - profile_radius_limit < 0
    ):
        death_mark = True
        return intensity_profiles, along_radius, theta, death_mark

    line_x = (
        np.full((profile_radius, num_angles), int(xc))
        + np.rint(
            np.matmul(
                np.transpose(np.asmatrix(along_radius)),
                np.asmatrix(np.cos(theta))
            )
        )
    )

    line_y = (
        np.full((profile_radius, num_angles), int(yc))
        + np.rint(
            np.matmul(
                np.transpose(np.asmatrix(along_radius)),
                np.asmatrix(np.sin(theta))
            )
        )
    )

    for i in range(profile_radius):
        for j in range(num_angles):
            for k in range(num_channels):
                channel = channels_data[k, :, :]
                intensity_profiles[i, j, k] = channel[
                    int(line_y[i, j]),
                    int(line_x[i, j])
                ]

    return intensity_profiles, along_radius, theta, death_mark


def edit_output(path_to_output, exp_info, ves_coordinates, background, localization, comment):
    """
    Adds all relevant information for a single vesicle in a row of the previously
    created output file.

    Parameters
    ----------
    path_to_output : str
        The path to the folder to output folder.
    
    exp_info : list
        Contains information of the file's name in order:
        exp_info[0]: the experiment date
        exp_info[1]: the experiment name
        exp_info[2]: the name of the image to be analyzed
        exp_info[3]: full representative name by combining the previous information
    
    ves_coordinates : np.ndarrya
        1-dimensional array with the coordinates of the single vesicle we want 
        to focus on.  
        ves_coordinates[0]: vesicle id 
        ves_coordinates[1]: xc (in px) 
        ves_coordinates[2]: yc (in px) 
        ves_coordinates[3]: radius (in px)
        
     background : np.ndarray
         The background noise estimated for the different channels
         Data related to different 
         channels are stored along the 3rd dimension of the array:
         background[:,0] always corresponds to the background of the membrane channel.
         If only one protein is present:
         background[:,1] corresponds to the background of that protein's channel.
         If two proteins are present:
         background[:,1] corresponds to the background of the septin channel.    
         background[:,2] corresponds to the background of the actin channel.
     
     localization : np.ndarray
         The quantified localization of the protein(s) at the membrane.
         If only one protein is present, then localization has a single element
         corresponding to that protein. 
         If both proteins are present, localization[0] corresponds to the septin
         channel and localization[1] corresponds to the actin channel. 
     
    comment : list
         A list of strings with relevant comments explaining (possible) issues 
         related to the specific vesicle. 

    """
    list_exp_info = list(exp_info[:-1])
    list_ves_coordinates = list(ves_coordinates)
    list_background = list(background[0, :])
    list_localization = list(localization)

    vesicle_row = (
        list_exp_info
        + list_ves_coordinates
        + list_background
        + list_localization
        + comment
    )

    output_file_path = os.path.join(
        path_to_output,
        exp_info[3] + "-Output.csv"
    )

    with open(output_file_path, "a", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(vesicle_row)


def background_correction(num_channels, intensity_profiles, background):
    """ 
    Background correction of the intesity linear profiles. 
    
    Input
    -----
    num_channels : int
        The number of channels present in the image analyzed.
        
    intensity_profiles : np.ndarray
        The intensity linear profiles calculated.
        Note: Data related to different channels are stored along the 3rd 
              dimension of the array. Element [:,:,0] is always based on the 
              membrane channel, element [:,:,1] is based on the single protein
              present. If both proteins are present, element [:,:,1] corresponds
              to the septin channel and element [:,:,2] corresponds to the actin
              channel.
              
    background : np.ndarray
        The background noise estimated for the different channels
        Data related to different 
        channels are stored along the 3rd dimension of the array:
        background[:,0] always corresponds to the background of the membrane channel.
        If only one protein is present:
        background[:,1] corresponds to the background of that protein's channel.
        If two proteins are present:
        background[:,1] corresponds to the background of the septin channel.    
        background[:,2] corresponds to the background of the actin channel.
    
    Returns
    -------
    intensity_profiles_corrected : np.ndarray
        The intensity linear profiles after the background noise correction.
        Note: Data is stored as in the intensity_profiles array. 
              
    """
    intensity_profiles_corrected = np.zeros_like(intensity_profiles)

    for i in range(num_channels):
        intensity_profiles_corrected[:, :, i] = np.clip(
            intensity_profiles[:, :, i] - background[0, i],
            0,
            None
        )

    return intensity_profiles_corrected


def background_noise(
    channels_data,
    ves_coordinates,
    inner_margin=1.2,
    outer_margin=2.0
):
    """
    Estimate local background from an annulus around the vesicle.

    Uses pixels between inner_margin * radius and outer_margin * radius.
    """

    num_channels = channels_data.shape[0]

    xc = ves_coordinates[1]
    yc = ves_coordinates[2]
    radius = ves_coordinates[3]

    y, x = np.indices(channels_data.shape[1:])

    distance = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    background_mask = (
        (distance >= inner_margin * radius)
        & (distance <= outer_margin * radius)
    )

    if not np.any(background_mask):
        raise ValueError(
            f"No background pixels found for vesicle {ves_coordinates[0]}"
        )

    background = np.zeros((1, num_channels))

    for ch in range(num_channels):
        background[0, ch] = np.mean(channels_data[ch, :, :][background_mask])

    return background


def zoom_in_vesicle(channel, size_side, image_dim, ves_coordinates):
    """
    Creates a square croped view of a chosen vesicle. In case the vesicle is 
    close to the border the produced image is recentered so that the original
    image borders are not crossed. 
    
    Input
    -----
    channel : np.ndarray
        The image data corresponding to the channel we want to crop.
        
    size_side : float
        The length (unit of measure: each vesicles' radius) of the side of the 
        square zoomed-in image we want to create. 
        
    image_dim : np.ndarray
        The original image's dimensions. 
        image_dim[0] corresponds to the x direction.
        image_dim[1] corresponds to the y direction.
        
    ves_coordinates : np.ndarrya
        1-dimensional array with the coordinates of the single vesicle we want 
        to focus on.  
        ves_coordinates[0]: vesicle id 
        ves_coordinates[1]: xc (in px) 
        ves_coordinates[2]: yc (in px) 
        ves_coordinates[3]: radius (in px)
        
    Returns
    -------
    vesicle_box_channel : np.ndarray
        The 2-D array with the data of the cropped image.

    """
    xc = ves_coordinates[1]
    yc = ves_coordinates[2]
    radius = ves_coordinates[3]

    vesicle_box_side = int(size_side * radius)

    move_right = 0
    move_left = 0
    move_up = 0
    move_down = 0

    if xc - vesicle_box_side < 0:
        move_right = vesicle_box_side - xc

    if xc + vesicle_box_side > image_dim[0]:
        move_left = vesicle_box_side - (image_dim[0] - xc)

    if yc - vesicle_box_side < 0:
        move_down = vesicle_box_side - yc

    if yc + vesicle_box_side > image_dim[1]:
        move_up = vesicle_box_side - (image_dim[1] - yc)

    vesicle_box_channel = channel[
        int(yc - vesicle_box_side + move_down - move_up):
        int(yc + vesicle_box_side + move_down - move_up),

        int(xc - vesicle_box_side + move_right - move_left):
        int(xc + vesicle_box_side - move_left + move_right)
    ]

    return vesicle_box_channel


def radial_profile(num_channels, intensity_profiles):
    """
    Calculation of a collective radial profile for a given vesicle from multiple 
    equally-distances linear profiles along the vesicles radius. 
    
    Input
    -----
    num_channels : int
        The number of channels present in the image analyzed. 
        
    intensity_profiles : np.ndarray
        The intensity linear profiles calculated. Backgrounds corrections should
        take place beforehand. 
        Note: Data related to different channels are stored along the 3rd 
              dimension of the array. Element [:,:,0] is always based on the 
              membrane channel, element [:,:,1] is based on the single protein
              present. If both proteins are present, element [:,:,1] corresponds
              to the septin channel and element [:,:,2] corresponds to the actin
              channel.
    
    Returns
    -------
    radial_profiles : np.ndarray
        The collective radial intensity profile of the vesicle under study. 
    
    """
    radial_profiles = np.zeros_like(intensity_profiles[:, 0, :])

    for i in range(num_channels):
        radial_profiles[:, i] = np.average(intensity_profiles[:, :, i], axis=1)

    return radial_profiles

def membrane_detection(radial_profile_memb, radius):
    """
    Detection of the membrane inner and outer borders.
    Convention used: width at half maximum
    
    Input
    -----
    radial_profile_memb : np.ndarray
        The 1D array of the membrane radial intensity profile. 
    
    radius: float
        The radius of the vesicle detected by DisGUVery. 
        
    Returns
    -------
    index_border_in : int
        The index of the element corresponding to the inner border of the membrane.
        
    index_border_out : int
        The index of the element corresponding to the outer border of the membrane.
    
    comment : list
        List of strings that correspond to comments on issues encountered.
        
    death_mark : bool
        If True, no peak was detected and the vesicle is marked to be disregarded. 
        If False, peak/s was/were detected and the analysis can continued.
    """
    comment = []
    death_mark = False
    peaks, _ = find_peaks(radial_profile_memb, height=10, distance=5, prominence=1)
    
    if not np.any(peaks) :
        index_border_in  = 0
        index_border_out = 0 
        comment          = ["no_memb_peak"]
        death_mark       = True
        return index_border_in, index_border_out, comment, death_mark
        
    width_half_max = peak_widths(radial_profile_memb, peaks, rel_height=0.5)
    
    chosen_peak = -1
    
    if len(peaks) > 1:
        if peaks[chosen_peak] > radius and radial_profile_memb[peaks[chosen_peak]] < radial_profile_memb[peaks[chosen_peak-1]]:
            chosen_peak = chosen_peak - 1
            
            if len(peaks) > 2:
                if peaks[chosen_peak] > radius and radial_profile_memb[chosen_peak] < radial_profile_memb[chosen_peak - 1]:
                    chosen_peak = chosen_peak - 1
                    
        elif len(peaks)>2:
            if peaks[chosen_peak] > radius and radial_profile_memb[peaks[chosen_peak]] < radial_profile_memb[peaks[chosen_peak-2]]:
                chosen_peak = chosen_peak - 2
        
        if len(peaks[:chosen_peak]) != 0:
            comment = comment + ["confetti"] 
    
    # if radial_profile_memb[peaks[chosen_peak]] < max(radial_profile_memb[:peaks[chosen_peak]]):
    #     comment = comment + ["confetti"] 
   
    index_border_in = int(np.rint(width_half_max[2][chosen_peak]))
    index_border_out = int(np.rint(width_half_max[3][chosen_peak]))
    
    if index_border_out - index_border_in > radius/4:
        comment = comment + ["wide_peak"]
    
    if np.average(radial_profile_memb[:index_border_in]) >= 0.3 *radial_profile_memb[peaks[chosen_peak]]:
        comment = comment + ["lipids_inside"]
    
    if np.average(radial_profile_memb[index_border_out:]) >= 0.4 *radial_profile_memb[peaks[chosen_peak]]:
        comment = comment + ["clusters_outside"]
    
    return index_border_in, index_border_out, comment, death_mark


def angular_profile(num_channels, intensity_profiles, index_border_in, index_border_out, pixels_to_remove):
    """
    Calculation of the angular profile along the membrane contour for a given 
    vesicle.
    Current convention: Averaging over the detected membrane area for each angle
                        considered during the calcualtion of the linear profiles. 
                        
    Input
    -----
    num_channels : int
        The number of channels present in the image analyzed.

    intensity_profiles : np.ndarray
        The intensity linear profiles calculated. Backgrounds corrections should
        take place beforehand. 
        Note: Data related to different channels are stored along the 3rd 
              dimension of the array. Element [:,:,0] is always based on the 
              membrane channel, element [:,:,1] is based on the single protein
              present. If both proteins are present, element [:,:,1] corresponds
              to the septin channel and element [:,:,2] corresponds to the actin
              channel.
              
    index_border_in : int
        The index of the element corresponding to the inner border of the membrane.
        
    index_border_out : int
        The index of the element corresponding to the outer border of the membrane.
    
    pixels_to_remove : int
        The number of pixels at the center to be removed to avoid the inherent 
        pixel multicounting.
        
    Returns
    -------
    angular_profiles : np.ndarray
        The angular profile along the membrane contour for a given vesicle.
        The profiles corresponding to the different channels are stored along 
        the 3rd dimension. 
        angular_profiles[:,0] corresponds to the membrane channel.
        If only one protein is present:
        angular_profiles[:,1] corresponds to that protein channel. 
        If both proteins are peresent:
        angular_profiles[:,1] corresponds to the septin channel. 
        angular_profiles[:,2] corresponds to the actin channel.
    
    """
    # intensity_profiles = intensity_profiles[:, pixels_to_remove:, :]
    angular_profiles = np.ones_like(intensity_profiles[0,:,:])
    
    for i in range(num_channels):
        angular_profiles[:,i] = np.average(intensity_profiles[pixels_to_remove+index_border_in:pixels_to_remove+index_border_out,:,i], axis=0)
    
    # print("angular_profiles", np.average(angular_profiles[:,1]))
    return angular_profiles


def localization(
    num_channels,
    angular_profiles,
    radial_profiles,
    radius,
    size_central_area,
    pixels_to_remove,
    guv_channel=0
):
    """
    Quantification of the protein localization on the membrane.

    Convention used: Difference between the median signal within the detected
                     membrane width and the average signal at a central area of
                     the vesicle, normalized by the median signal within the
                     detected membrane width. So, (membrane - centre) / membrane

    The localization is calculated for all channels except guv_channel.

    Recommended values for classification --> intensity on the 
    membrane is at least 20% higher than in the centre --> 0.2/1.2 = 1/6 ~=0.1667:
        L < 1/6  : No localization
        1/6 < L <= 1   : Localization

    Input
    -----
    num_channels : int
        The number of channels present in the image analyzed.

    angular_profiles : np.ndarray
        The collective angular intensity profile along the detected membrane of
        the vesicle under study.

    radial_profiles : np.ndarray
        The collective radial intensity profile of the vesicle under study.

    index_border_in : int
        The index of the element corresponding to the inner border of the membrane.

    index_border_out : int
        The index of the element corresponding to the outer border of the membrane.

    radius : float
        The radius of the vesicle under study.

    size_central_area : float
        The central area radius, in units of vesicle radius, to be considered.

    pixels_to_remove : int
        The number of pixels at the center removed to avoid inherent pixel
        multicounting.

    guv_channel : int
        The channel corresponding to the GUV/membrane signal. This channel is
        excluded from protein localization quantification.

    Returns
    -------
    localization_index : np.ndarray
        The quantified localization for all non-GUV channels.

    localization_channels : list
        The original image channel indices corresponding to localization_index.

    comment : list
        List of strings corresponding to issues encountered.
    """

    comment = []

    protein_channels = [ch for ch in range(num_channels) if ch != guv_channel]

    index_centre = int(size_central_area * (radius - pixels_to_remove))

    localization_index = np.zeros(len(protein_channels))

    for protein_i, ch in enumerate(protein_channels):

        mean_angular_signal = np.mean(angular_profiles[:, ch])

        if mean_angular_signal > 0:
            rsd = np.std(angular_profiles[:, ch]) / mean_angular_signal
        else:
            rsd = np.nan
            comment = comment + [f"zero_mean_ch{ch}"]

        if not np.isnan(rsd) and rsd > 0.8:
            comment = comment + [f"high_rsd_ch{ch}"]

        median = np.median(angular_profiles[:, ch])
        centre = np.average(radial_profiles[0:index_centre, ch])

        if median > 0:
            localization_index[protein_i] = (median - centre) / median

        else:
            localization_index[protein_i] = 0
            comment = comment + [f"zero_median_ch{ch}"]

    return localization_index, protein_channels, comment

def plot_profile_and_zoom(
    channels_data,
    ves_coordinates,
    image_dim,
    radial_profiles,
    along_radius,
    angular_profiles=None,
    size_view=1.5,
    channels=(0, 1, 2),
    save_path=None,
    show=True
):
    """
    Plot cropped channel images stacked vertically, plus radial and angular profiles.

    No overlay.
    No circle.
    No distance normalization.
    """

    num_channels_to_show = len(channels)

    if angular_profiles is None:
        num_columns = 2
    else:
        num_columns = 3

    fig, axes = plt.subplots(
        num_channels_to_show,
        num_columns,
        figsize=(4 * num_columns, 3 * num_channels_to_show)
    )

    if num_channels_to_show == 1:
        axes = np.array([axes])

    for row, ch in enumerate(channels):
        crop = zoom_in_vesicle(
            channels_data[ch, :, :],
            size_view,
            image_dim,
            ves_coordinates
        )

        axes[row, 0].imshow(crop, cmap="gray")
        axes[row, 0].set_title(f"Channel {ch} crop")
        axes[row, 0].axis("off")

        axes[row, 1].plot(along_radius, radial_profiles[:, ch])
        axes[row, 1].set_title(f"Channel {ch} radial profile")
        axes[row, 1].set_xlabel("Distance from center (px)")
        axes[row, 1].set_ylabel("Intensity")

        if angular_profiles is not None:
            axes[row, 2].plot(angular_profiles[:, ch])
            axes[row, 2].set_title(f"Channel {ch} angular profile")
            axes[row, 2].set_xlabel("Angle index")
            axes[row, 2].set_ylabel("Intensity")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

def save_separate_profile_plots(
    radial_profiles,
    along_radius,
    angular_profiles,
    vesicle_id,
    channels,
    output_folder
):
    """
    Save separate radial and angular profile plots for each channel.

    Files are named:
        radial_ves{vesicle_id}_ch{ch}.png
        angular_ves{vesicle_id}_ch{ch}.png
    """

    for ch in channels:

        # Radial profile
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(along_radius, radial_profiles[:, ch])
        ax.set_title(f"Vesicle {vesicle_id} - channel {ch} radial profile")
        ax.set_xlabel("Distance from center (px)")
        ax.set_ylabel("Intensity")
        plt.tight_layout()

        radial_path = os.path.join(
            output_folder,
            f"radial_ves{vesicle_id}_ch{ch}.png"
        )

        fig.savefig(radial_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Angular profile
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(angular_profiles[:, ch])
        ax.set_title(f"Vesicle {vesicle_id} - channel {ch} angular profile")
        ax.set_xlabel("Angle index")
        ax.set_ylabel("Intensity")
        plt.tight_layout()

        angular_path = os.path.join(
            output_folder,
            f"angular_ves{vesicle_id}_ch{ch}.png"
        )

        fig.savefig(angular_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


#%%
# Run analysis
images = glob(os.path.join(PATH, "**", "*"+IMAGE_FORMAT), recursive=True)
all_results = []
for image_i, image_path in tqdm(enumerate(images), total=len(images),):
    print(image_path)
    comment = []
    # Open data for each image
    # get unique part of the path in case we want to save the results in an output folder. 

    output_folder = get_output_folder(image_path, PATH)

    img = open_image(image_path) #open image

    #changes .czi to the suffix you had set and if the CSv exists it opens it
    csv_path = image_path.replace(IMAGE_FORMAT, DETECTION_SUFFIX)

    if not os.path.exists(csv_path):
        print(f"Skipping image. Detection CSV not found: {csv_path}")
        continue

    locs = pd.read_csv(csv_path, index_col=False).to_numpy()

    num_vesicles = len(locs[:, 0])
    num_channels = img.shape[0]

    image_dim = np.array((img.shape[2], img.shape[1]))  # X, Y
   
    # Vesicle-level analysis
    for ves_i in range(num_vesicles):
        ves_coordinates = locs[ves_i, :]
        vesicle_id = ves_coordinates[0]

        # 1. Calculate linear profiles
        intensity_profiles, along_radius, theta, death_mark = linear_profiles(img, ves_coordinates, image_dim, PARAMETERS_PROFILES)

        # 2. Skip vesicles too close to the image border
        if death_mark:
            comment = ['margins']

            result_row = {
                "image_path": image_path,
                "vesicle_id": vesicle_id,
                "xc": ves_coordinates[1],
                "yc": ves_coordinates[2],
                "radius": ves_coordinates[3],
                "death_mark": death_mark,
                "comment": ";".join(comment),
            }

            all_results.append(result_row)

            print("Skipped GUV: profiles extend outside image margins.")
            continue

        # 3. Estimate local background using annulus around vesicle
        background = background_noise(
            img,
            ves_coordinates,
            inner_margin=INNER_MARGIN,
            outer_margin=OUTER_MARGIN
        )

        # 4. Background correction
        intensity_profiles_corrected = background_correction(
            num_channels,
            intensity_profiles,
            background
        )

        # 5. Calculate and correct radial profiles
        radial_profiles = radial_profile(
            num_channels,
            intensity_profiles_corrected
        )
        
        radial_profiles  = radial_profiles[pixels_to_remove:,:]
        along_radius     = along_radius[pixels_to_remove:]

        index_border_in, index_border_out, comment_peak, death_mark_peak = membrane_detection(radial_profiles[:,GUV_CH], locs[ves_i,3])

        ## Registration of the triggered conditions for automatic vesicle rejection:
        if comment_peak:
            comment = comment + comment_peak
        ## Output for profiles where no peak was detected is set to 0:
        if death_mark_peak:
            background = np.zeros((1, num_channels))
            localization_values = np.zeros(num_channels - 1)
            comment = comment_peak

            result_row = {
                "image_path": image_path,
                "vesicle_id": vesicle_id,
                "xc": ves_coordinates[1],
                "yc": ves_coordinates[2],
                "radius": ves_coordinates[3],
                "death_mark": True,
                "comment": ";".join(comment),
            }


            all_results.append(result_row)

            print("Skipped GUV: no membrane peak detected.")
            continue

        # 6. Calculation of the angular profile of the vesicle under study:
        angular_profiles = angular_profile(num_channels, intensity_profiles_corrected, index_border_in, index_border_out, pixels_to_remove)
        
        # 7. Quantification of protein localization on the membrane:
        localization_index, localization_channels, comment_loc = localization(
                num_channels,
                angular_profiles,
                radial_profiles,
                locs[ves_i, 3],
                size_central_area,
                pixels_to_remove,
                guv_channel=GUV_CH
            )
        if comment_loc:
            comment = comment + comment_loc
        # 8. Plot result for visual inspection
        if PLOT_RESULTS:
            channels_to_plot = tuple(range(num_channels))

            full_plot_path = os.path.join(
                output_folder,
                f"full_ves{vesicle_id}.png"
            )

            plot_profile_and_zoom(
                channels_data=img,
                ves_coordinates=ves_coordinates,
                image_dim=image_dim,
                radial_profiles=radial_profiles,
                along_radius=along_radius,
                angular_profiles=angular_profiles,
                size_view=SIZE_VIEW,
                channels=channels_to_plot,
                save_path=full_plot_path,
                show=PLOT_RESULTS
            )

            save_separate_profile_plots(
                radial_profiles=radial_profiles,
                along_radius=along_radius,
                angular_profiles=angular_profiles,
                vesicle_id=vesicle_id,
                channels=channels_to_plot,
                output_folder=output_folder
            )

        # 8. Store summary result
        result_row = {
            "image_path": image_path,
            "vesicle_id": vesicle_id,
            "xc": ves_coordinates[1],
            "yc": ves_coordinates[2],
            "radius": ves_coordinates[3],
            "death_mark": death_mark,
            "death_mark_peak": death_mark_peak,
            "index_border_in": index_border_in,
            "index_border_out": index_border_out,
            "comment": ";".join(comment),
        }

        for ch in range(num_channels):
            result_row[f"background_ch{ch}"] = background[0, ch]

        for loc_value, ch in zip(localization_index, localization_channels):
            result_row[f"localization_ch{ch}"] = loc_value

        all_results.append(result_row)


print(f"\nDone. Results saved to:")

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(PATH, 'results.csv'), index=False)