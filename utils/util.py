import cv2
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def otsus_thresholding(image):
    """
    Apply Otsu's thresholding to the input image.
    Parameters:
    - image: Input image as a 2D numpy array.
    Returns:
    - Binary mask as a 2D numpy array.
    """
    # Calculate histogram [np.arange(257) ranges from 0 to 256]
    histogram, _ = np.histogram(image, bins=np.arange(257), density=True)

    cumulative_sum = np.cumsum(histogram)
    cumulative_mean = np.cumsum(histogram * np.arange(256))

    global_intensity_mean = cumulative_mean[-1]
    max_variance = 0
    optimal_threshold = 0

    # Iterate through all possible thresholds
    for t in range(256):
        prob_background, prob_foreground = cumulative_sum[t], 1 - cumulative_sum[t]
        mean_background, mean_foreground = (
            cumulative_mean[t] / cumulative_sum[t],
            (
                (global_intensity_mean - cumulative_mean[t]) / (1 - cumulative_sum[t])
                if (1 - cumulative_sum[t]) > 0
                else 0
            ),
        )
        variance_between = (
            prob_background * prob_foreground * (mean_background - mean_foreground) ** 2
        )

        if variance_between > max_variance:
            max_variance = variance_between
            optimal_threshold = t

    # Create binary mask
    binary_mask = np.where(image > optimal_threshold, 1, 0).astype(np.uint8)

    return binary_mask, optimal_threshold

def interpolate(edge_magnitude, x, y, angle):
    """
    Interpolate the gradient magnitude of a pixel located at (x, y) in the direction specified by angle.
    """
    m, n = edge_magnitude.shape
    # Calculate the back and front coordinates based on angle
    x_back, y_back, x_front, y_front = None, None, None, None
    if angle == 0:
        x_back, y_back = x, max(y - 1, 0)
        x_front, y_front = x, min(y + 1, n - 1)
    elif angle == 90:
        x_back, y_back = max(x - 1, 0), y
        x_front, y_front = min(x + 1, m - 1), y
    elif angle == 135:
        x_back, y_back = max(x - 1, 0), min(y + 1, n - 1)
        x_front, y_front = min(x + 1, m - 1), max(y - 1, 0)
    elif angle == 45:
        x_back, y_back = max(x - 1, 0), max(y - 1, 0)
        x_front, y_front = min(x + 1, m - 1), min(y + 1, n - 1)

    # Interpolate the magnitude
    magnitude_back = edge_magnitude[x_back, y_back]
    magnitude_front = edge_magnitude[x_front, y_front]

    return magnitude_back, magnitude_front


def nms(edge_magnitude, edge_orientation):
    """
    Perform non-maximum suppression on the edge magnitude image.
    Parameters:
        edge_magnitude: The edge magnitude image
        edge_orientation: The edge orientation image
    Returns:
        thin_edges: The edge magnitude image after non-maximum suppression
    """
    # Initialize the output image with zeros (suppress all)
    thin_edges = np.zeros(edge_magnitude.shape, dtype=np.float32)

    # Convert orientation from radians to degrees for easier comparison
    angle = edge_orientation * 180.0 / np.pi
    angle[angle < 0] += 180

    # Map angles to the nearest 0, 45, 90, or 135 degrees for simplification
    angle = np.round(angle / 45) * 45
    # 180 maps to 0 degrees
    angle[angle == 182] = 0

    for i in range(1, edge_magnitude.shape[0] - 1):
        for j in range(1, edge_magnitude.shape[1] - 1):
            current_angle = angle[i, j] % 180  # Ensure angle is within 0-180
            # Find the angles of the two relevant neighbors
            q, r = interpolate(edge_magnitude, i, j, current_angle)

            # Suppress pixels that are not local maxima
            if edge_magnitude[i, j] >= q and edge_magnitude[i, j] >= r:
                thin_edges[i, j] = edge_magnitude[i, j]
            else:
                thin_edges[i, j] = 0

    return thin_edges


def myEdgeFilter(img0, sigma):
    """
    Apply the edge filter to an image.
    Parameters:
        img0: The input image
        sigma: The standard deviation of the Gaussian filter
    Returns:
        edge_magnitude: The edge magnitude image edge_orientation: The edge orientation image
    """
    hsize = int(2 * ceil(3 * sigma) + 1)

    smooth_img = cv2.GaussianBlur(img0, (hsize, hsize), sigma)

    grad_x = cv2.Sobel(smooth_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smooth_img, cv2.CV_64F, 0, 1, ksize=3)

    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    edge_orientation = np.arctan2(grad_y, grad_x)

    return edge_magnitude, edge_orientation

def hough_circle_detect(nms_edge_magnitude, edge_orientation, img_shape, radius_range,radius_step=1):
    """
    Detect circles in an image using the Hough transform.
    Parameters:
        nms_edge_magnitude: The non-maximum suppressed edge magnitude image
        edge_orientation: The edge orientation image
        img_shape: The shape of the original image
        radius_range: The range of radii to search for
        radius_step: The step size for the radius search
    Returns:
        circles: A tuple of arrays (x_indices, y_indices, radius_indices)
        accumulator: The Hough accumulator array of shape (img_shape[0], img_shape[1], len(radius_range))
    """
    # Initialize the accumulator (3D, for the x, y center positions and the radius)
    accumulator = np.zeros((img_shape[0], img_shape[1], len(radius_range)))
    radii = np.arange(radius_range[0], radius_range[1] + 1, radius_step)

    # Iterate over the edge pixels
    for y in range(img_shape[0]):
        for x in range(img_shape[1]):
            if nms_edge_magnitude[y, x] != 0:
                # For each radius
                for r_idx, r in enumerate(radii):
                    # Calculate possible centers based on the gradient direction
                    for sign in (-1, 1):
                        a = int(y - sign * r * np.sin(edge_orientation[y, x]))
                        b = int(x - sign * r * np.cos(edge_orientation[y, x]))
                        # Check if the center is within bounds
                        if a >= 0 and a < img_shape[0] and b >= 0 and b < img_shape[1]:
                            accumulator[a, b, r_idx] += 1

    # Threshold to find centers with enough votes
    circles = np.where(accumulator > np.max(accumulator) * 0.40)

    return circles, accumulator

def display_voting_lines(thresholded_nms, edge_orientation, img, radius=15):
    """
    Display the voting lines for a given radius based on edge pixels and their orientations.

    Parameters:
    - thresholded_nms: The thesholded nms magnitude response.
    - edge_orientation: The edge orientation image (in radians).
    - img: The original image for background display.
    - radius: The radius for which to visualize the voting process.
    """
    _, ax = plt.subplots()
    if img.ndim == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)

    # Threshold edge magnitude to identify edge pixels (simple binary thresholding for demonstration)
    #edge_pixels = np.where(thresholded_nms > np.percentile(thresholded_nms, 90))  # Adjust threshold as needed
    edge_pixels = np.where(thresholded_nms)

    for y, x in zip(*edge_pixels):
        orientation = edge_orientation[y, x]

        # Calculate the end points of the voting line in both directions
        dx1 = radius * np.cos(orientation)
        dy1 = radius * np.sin(orientation)
        dx2 = -dx1  # Opposite direction
        dy2 = -dy1

        # Draw arrows to represent voting lines
        ax.arrow(x, y, dx1, dy1, color='yellow', head_width=2, head_length=3, alpha=0.5)
        ax.arrow(x, y, dx2, dy2, color='yellow', head_width=2, head_length=3, alpha=0.5)

    plt.show()

def display_circles(circles, img, radius_range):
    _, ax = plt.subplots()
    ax.imshow(img)

    # circles is a tuple of arrays (x_indices, y_indices, radius_indices)
    # We need to iterate through them together
    for x, y, r_idx in zip(*circles):
        r = radius_range[r_idx]  # Map radius index back to radius value
        circ = Circle((y, x), r, edgecolor=(1, 0, 0), facecolor='none')  # Create circle patch
        ax.add_patch(circ)  # Add the circle patch to the axes

    plt.show()
