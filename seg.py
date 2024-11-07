import numpy as np

def rescale_to_image(ground_truth_positions, armstrong_range, image_size):
    x_scale = image_size[1] / armstrong_range[0]
    y_scale = image_size[2] / armstrong_range[1]
    z_scale = image_size[0] / armstrong_range[2]

    rescaled_positions = []
    for point in ground_truth_positions:
        rescaled_x = int(point[0] * x_scale)
        rescaled_y = int(point[1] * y_scale)
        rescaled_z = int(point[2] * z_scale)
        rescaled_positions.append([rescaled_x, rescaled_y, rescaled_z])

    return np.array(rescaled_positions)

def create_segmentation_mask(rescaled_positions, image_size):
    mask = np.zeros(image_size)

    for point in rescaled_positions:
        x, y, z = point
        mask[z, x, y] = 1  

    return mask

def inverse_rescale_from_mask(segmentation_mask, armstrong_range=(1500, 6000, 6000), image_size=(184, 630, 630)):
    z, x, y = np.where(segmentation_mask == 1)

    x_scale_inv = armstrong_range[1] / image_size[1]
    y_scale_inv = armstrong_range[2] / image_size[2]
    z_scale_inv = armstrong_range[0] / image_size[0]

    original_coordinates = []
    for i in range(len(x)):
        original_x = x[i] * x_scale_inv
        original_y = y[i] * y_scale_inv
        original_z = z[i] * z_scale_inv
        original_coordinates.append([original_x, original_y, original_z])

    original_coordinates_sorted = sorted(original_coordinates, key=lambda coord: (coord[0], coord[1], coord[2]))

    return np.array(original_coordinates_sorted)

rescaled_positions = rescale_to_image(positions, armstrong_range, (184, 630, 630))
segmentation_mask = create_segmentation_mask(rescaled_positions, (184, 630, 630))
unscaled_points = inverse_rescale_from_mask(segmentation_mask)
