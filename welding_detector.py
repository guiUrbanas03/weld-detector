import cv2 as cv
import os
from Sample import Sample

BACKGROUND_SUBTRACTION = cv.createBackgroundSubtractorMOG2()
BRUTEFORCE_MATCHER = cv.BFMatcher_create()
ORB = cv.ORB_create(nfeatures=1000)


CLASSES_PATH = './images/classes'
SAMPLES_PATH = './images/samples'


def apply_image_filters(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_image = cv.GaussianBlur(gray_image, (7, 7), 0)
    return gray_image


def apply_background_subtraction(image):
    image_with_adaptive_threshold = cv.adaptiveThreshold(
        image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 10)

    foreground_mask = BACKGROUND_SUBTRACTION.apply(image_with_adaptive_threshold)
    return foreground_mask


def apply_orb(image):
    key_points, descriptors = ORB.detectAndCompute(image, None)
    return key_points, descriptors


def apply_bruteforce_matcher(base_descriptor, target_descriptors):
    matches = BRUTEFORCE_MATCHER.knnMatch(base_descriptor, target_descriptors, k=2)
    return matches


def get_good_matches(matches, threshold):
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append([m])

    return good_matches


def test_matches(classes):
    pass


def read_images(images):
    readed_images = []
    for image in images:
        cv_image = cv.imread(image)
        readed_images.append(cv_image)

    return readed_images


def get_files_path(directory):
    file_names = os.listdir(directory)
    paths = []

    for file_name in file_names:
        paths.append(relative_path(directory, file_name))

    return paths


def relative_path(directory, file_name):
    return os.path.relpath(os.path.join(directory, file_name))


def list_map(iterable, function):
    return list(map(lambda item: function(item), iterable))


def main():
    classes = {
        'good': [],
        'fusion': [],
        'perforation': [],
        'cold_joint': [],
    }

    class_images = read_images(get_files_path(CLASSES_PATH))
    sample_images = read_images(get_files_path(SAMPLES_PATH))

    gray_class_images = list_map(class_images, apply_image_filters)
    gray_sample_images = list_map(sample_images, apply_image_filters)

    background_subtracted_class_images = list_map(gray_class_images, apply_background_subtraction)
    background_subracted_sample_images = list_map(gray_sample_images, apply_background_subtraction)

    orb_class_images = list_map(background_subtracted_class_images, apply_orb)
    orb_sample_images = list_map(background_subracted_sample_images, apply_orb)

    # cv.imshow('gray', background_subtracted_class_images[0])
    # cv.imshow('gray 2', background_subracted_sample_images[0])


if __name__ == '__main__':
    main()
    cv.waitKey(0)

