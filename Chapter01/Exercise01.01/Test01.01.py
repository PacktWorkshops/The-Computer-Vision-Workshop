# Import the required packages for this script:
import import_ipynb
import cv2
import unittest


class Test(unittest.TestCase):
    def setUp(self):
        import Exercise01_01
        self.exercises = Exercise01_01
        print(self.exercises)

        self.path_img = 'Images/logo.png'
        self.img = cv2.imread(self.path_img)
        (height, width, channels) = self.img.shape
        print("Height: '{}', width: '{}', channels: '{}'".format(height, width, channels))

    def test_file_path(self):
        self.assertEqual(self.exercises.path_img, self.path_img)


if __name__ == '__main__':
    # unittest.main()
    unittest.main(argv=[''], verbosity=2, exit=False)
