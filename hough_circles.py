from utils.util import (
    myEdgeFilter,
    nms,
    hough_circle_detect,
    display_circles,
    otsus_thresholding,
    display_voting_lines
)
import cv2
import matplotlib.pyplot as plt


def main():
    radii = range(20, 110)
    img = cv2.imread("./data/HoughCircles.jpg", cv2.IMREAD_COLOR)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply the edge filter form previous assignment
    edge_mag, edge_or = myEdgeFilter(gray_image, 3)
    # Perform non-maximum suppression
    nms_edge_mag = nms(edge_mag, edge_or)
    # Threshold the edge magnitude image
    thresholded_nms,_ = otsus_thresholding(nms_edge_mag)

    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 3, 1).add_artist(plt.imshow(img))
    plt.title("Original Image")
    fig.add_subplot(1, 3, 2).add_artist(plt.imshow(edge_mag, cmap="gray"))
    plt.title("Edge Image")
    fig.add_subplot(1, 3, 3).add_artist(plt.imshow(thresholded_nms, cmap="gray"))
    plt.title("NMS Image")
    plt.show()
    circles,acc = hough_circle_detect(thresholded_nms, edge_or, img.shape, radii,radius_step=3)
    print(f"detected {len(circles[0])} circles")
    display_voting_lines(thresholded_nms, edge_or, img, radius=15)
    display_circles(circles, img, radii)



if __name__ == "__main__":
    main()
