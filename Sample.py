class Sample:
    def __init__(self, identifier, expected, image):
        self.__identifier = identifier
        self.__expected = expected
        self.__image = image
        self.__keypoints = []
        self.__descriptors = []
        self.__predicted = ''

    def __repr__(self):
        return 'repr'

    def get_attributes(self):
        return {
            'id': self.__identifier,
            'expected': self.__expected,
            'predicted': self.__predicted
        }

    def set_keypoints(self, keypoints):
        self.__keypoints = keypoints

    def set_descriptors(self, descriptors):
        self.__descriptors = descriptors


