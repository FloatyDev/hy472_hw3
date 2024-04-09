from utils.util import (
    myEdgeFilter,
    nms,
    detectCircles,
    display_circles,
)
import cv2
import matplotlib.pyplot as plt


def main():
    radii = [100, 30]
    img = cv2.imread("./data/HoughCircles.jpg", cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply the edge filter form previous assignment
    edge_mag, edge_or = myEdgeFilter(gray_image, 3)
    # Perform non-maximum suppression
    nms_edge_mag = nms(edge_mag, edge_or)
    # Threshold the edge magnitude image
    # nms_edge_mag[nms_edge_mag < 0.23 * max(nms_edge_mag.flatten())] = 0
    # nms_edge_mag[nms_edge_mag >= 0.23 * max(nms_edge_mag.flatten())] = 1

    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 3, 1).add_artist(plt.imshow(img))
    plt.title("Original Image")
    fig.add_subplot(1, 3, 2).add_artist(plt.imshow(edge_mag, cmap="gray"))
    plt.title("Edge Image")
    fig.add_subplot(1, 3, 3).add_artist(plt.imshow(nms_edge_mag, cmap="gray"))
    plt.title("NMS Image")
    plt.show()
    # output_stack = find_circles(nms_edge_mag, radii)
    output_stack = detectCircles(nms_edge_mag, 8.1, 15, radii)
    display_circles(output_stack, img)


if __name__ == "__main__":
    main()
