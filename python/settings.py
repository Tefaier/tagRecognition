import numpy as np

generated_info_folder = 'generatedInfo'
calibration_images_folder = 'calibration'
analyse_images_folder = 'analyse'
general_info_filename = 'profileInfo'
image_info_filename = 'imageInfo'
detection_info_filename = 'detectionInfo'
plots_folder = 'plots'

tag_images_folder = 'tagImages'

image_width = 1920
image_height = 1080

test_camera_matrix=np.array(
        [[1000.0, 0.0, image_width / 2.0],
         [0.0, 1000.0, image_height / 2.0],
         [0.0, 0.0, 1.0]])
test_distortion_coefficients=np.array([0.0, 0.0, 0.0, 0.0, 0.0])
