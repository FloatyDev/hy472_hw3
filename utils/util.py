import cv2
import numpy as np
from math import ceil
from matplotlib import pyplot as plt


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
        edge_magnitude: The edge magnitude image
        edge_orientation: The edge orientation image
    """
    # Determine size of the Gaussian filter
    hsize = int(2 * ceil(3 * sigma) + 1)

    # Apply Gaussian blur to smooth the image
    smooth_img = cv2.GaussianBlur(img0, (hsize, hsize), sigma)

    # Use Sobel operators to find x and y gradients
    grad_x = cv2.Sobel(smooth_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smooth_img, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the edge magnitude
    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Compute the edge orientation (in radians)
    edge_orientation = np.arctan2(grad_y, grad_x)

    return edge_magnitude, edge_orientation


##def find_circles(image, radius, region=5):
#    (m, n) = image.shape
#    min_radius, max_radius = radius
#    acc_array = np.zeros(
#        (max_radius - min_radius + 1, m + 2 * max_radius, n + 2 * max_radius)
#    )
#    edge_locs = np.argwhere(image[:, :])
#    theta = np.arange(0, 360, 1) * np.pi / 180
#
#    for r in range(min_radius, max_radius + 1):
#        circle = np.zeros((2 * (r + 1), 2 * (r + 1)))
#        center_x, center_y = r + 1, r + 1
#
#        for t in theta:
#            x = np.round(r * np.cos(t)).astype(int)
#            y = np.round(r * np.sin(t)).astype(int)
#            circle[center_x + x, center_y + y] = 1
#
#        for x, y in edge_locs:
#            x_start = max(0, x - center_x + max_radius)
#            x_end = min(m + 2 * max_radius, x + center_x + max_radius + 1)
#            y_start = max(0, y - center_y + max_radius)
#            y_end = min(n + 2 * max_radius, y + center_y + max_radius + 1)
#            acc_array[r - min_radius, x_start:x_end, y_start:y_end] += circle[
#                max(0, center_x - x) : min(2 * (r + 1), center_x - x + x_end - x_start),
#                max(0, center_y - y) : min(2 * (r + 1), center_y - y + y_end - y_start),
#            ]
#
#    output_array = np.zeros((max_radius - min_radius + 1, m, n))
#    for r, x, y in np.argwhere(acc_array):
#        temp = acc_array[
#            max(0, r - region) : min(max_radius - min_radius + 1, r + region + 1),
#            max(0, x - region) : min(m + 2 * max_radius, x + region + 1),
#            max(0, y - region) : min(n + 2 * max_radius, y + region + 1),
#        ]
#        p, a, b = np.unravel_index(np.argmax(temp), temp.shape)
#        output_array[r, x + a - region - max_radius, y + b - region - max_radius] = 1
#
#    return output_array
##def find_circles(image, radius, region=5, threshold=None):
#    (m, n) = image.shape
#    min_radius, max_radius = radius
#    acc_array = np.zeros(
#        (max_radius - min_radius + 1, m + 2 * max_radius, n + 2 * max_radius)
#    )
#    edge_locs = np.argwhere(image[:, :])
#    theta = np.arange(0, 360, 1) * np.pi / 180
#
#    for r in range(min_radius, max_radius + 1):
#        circle = np.zeros((2 * (r + 1), 2 * (r + 1)))
#        center_x, center_y = r + 1, r + 1
#
#        for t in theta:
#            x = np.round(r * np.cos(t)).astype(int)
#            y = np.round(r * np.sin(t)).astype(int)
#            circle[center_x + x, center_y + y] = 1
#
#        for x, y in edge_locs:
#            x_start = max(0, x - center_x + max_radius)
#            x_end = min(m + 2 * max_radius, x + center_x + max_radius)
#            y_start = max(0, y - center_y + max_radius)
#            y_end = min(n + 2 * max_radius, y + center_y + max_radius)
#            acc_array[r - min_radius, x_start:x_end, y_start:y_end] += circle
#
#    if threshold is not None:
#        acc_array[acc_array < threshold] = 0
#
#    output_array = np.zeros((max_radius - min_radius + 1, m, n))
#    for r in range(max_radius - min_radius + 1):
#        for x in range(m):
#            for y in range(n):
#                if acc_array[r, x + max_radius, y + max_radius] > 0:
#                    temp = acc_array[
#                        max(0, r - region) : min(
#                            max_radius - min_radius + 1, r + region + 1
#                        ),
#                        max(0, x + max_radius - region) : min(
#                            m + 2 * max_radius, x + max_radius + region + 1
#                        ),
#                        max(0, y + max_radius - region) : min(
#                            n + 2 * max_radius, y + max_radius + region + 1
#                        ),
#                    ]
#                    _, a, b = np.unravel_index(np.argmax(temp), temp.shape)
#                    output_array[r, x, y] = 1 if a == region and b == region else 0
#
#    return output_array
def detectCircles(img, threshold, region, radius=None):
    (M, N) = img.shape
    if radius == None:
        R_max = np.max((M, N))
        R_min = 3
    else:
        [R_max, R_min] = radius

    R = R_max - R_min
    # Initializing accumulator array.
    # Accumulator array is a 3 dimensional array with the dimensions representing
    # the radius, X coordinate and Y coordinate resectively.
    # Also appending a padding of 2 times R_max to overcome the problems of overflow
    A = np.zeros((R_max, M + 2 * R_max, N + 2 * R_max))
    B = np.zeros((R_max, M + 2 * R_max, N + 2 * R_max))

    # Precomputing all angles to increase the speed of the algorithm
    theta = np.arange(0, 360) * np.pi / 180
    edges = np.argwhere(img[:, :])  # Extracting all edge coordinates
    for val in range(R):
        r = R_min + val
        # Creating a Circle Blueprint
        bprint = np.zeros((2 * (r + 1), 2 * (r + 1)))
        (m, n) = (r + 1, r + 1)  # Finding out the center of the blueprint
        for angle in theta:
            x = int(np.round(r * np.cos(angle)))
            y = int(np.round(r * np.sin(angle)))
            bprint[m + x, n + y] = 1
        constant = np.argwhere(bprint).shape[0]
        for x, y in edges:  # For each edge coordinates
            # Centering the blueprint circle over the edges
            # and updating the accumulator array
            X = [x - m + R_max, x + m + R_max]  # Computing the extreme X values
            Y = [y - n + R_max, y + n + R_max]  # Computing the extreme Y values
            A[r, X[0] : X[1], Y[0] : Y[1]] += bprint
        A[r][A[r] < threshold * constant / r] = 0

    for r, x, y in np.argwhere(A):
        temp = A[
            r - region : r + region, x - region : x + region, y - region : y + region
        ]
        try:
            p, a, b = np.unravel_index(np.argmax(temp), temp.shape)
        except:
            continue
        B[r + (p - region), x + (a - region), y + (b - region)] = 1

    return B[:, R_max:-R_max, R_max:-R_max]


def display_voting_space(acc_array, radius):
    min_radius, max_radius = radius
    num_radii = max_radius - min_radius + 1

    fig, axes = plt.subplots(1, num_radii, figsize=(20, 5))

    for r in range(num_radii):
        axes[r].imshow(acc_array[r], cmap="gray")
        axes[r].set_title(f"Radius {r + min_radius}")
        axes[r].axis("off")

    plt.tight_layout()


# def display_detected_circles(image, output_array, radius):
#   min_radius, max_radius = radius
#   num_radii = max_radius - min_radius + 1
#
#   fig, ax = plt.subplots(figsize=(10, 10))
#   ax.imshow(image, cmap='gray')
#
#   for r in range(num_radii):
#       for x in range(image.shape[0]):
#           for y in range(image.shape[1]):
#               if output_array[r, x, y] == 1:
#                   circle = plt.Circle((y, x), r + min_radius, color='red', fill=False)
#                   ax.add_patch(circle)
#
#   ax.set_title('Detected Circles')
#   ax.axis('off')
#   plt.tight_layout()
#   plt.show()
def display_circles(acc_array, img):
    fig = plt.figure()
    plt.imshow(img)
    circleCoordinates = np.argwhere(acc_array)  # Extracting the circle information
    circle = []
    for r, x, y in circleCoordinates:
        circle.append(plt.Circle((y, x), r, color=(1, 0, 0), fill=False))
        fig.add_subplot(111).add_artist(circle[-1])
    plt.show()
