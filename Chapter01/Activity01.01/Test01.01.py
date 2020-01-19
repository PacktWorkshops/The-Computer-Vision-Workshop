# Import the required packages:
import import_ipynb
import cv2
import unittest


class Test(unittest.TestCase):
    def setUp(self):
        import Activity01_01
        self.exercises = Activity01_01
        print(self.exercises)

        # self.path_img = 'C:/Users/Alberto/Desktop/opencv_workshop/01-chap/Exercise01.01/Images/logo.png'
        self.path_img = 'Images/lena.jpg'
        self.img = cv2.imread(self.path_img)
        (height, width, channels) = self.img.shape
        print("Height: '{}', width: '{}', channels: '{}'".format(height, width, channels))

        # Make a copy of the source image:
        self.img_masked_face = self.img.copy()

        # Origin: (x = 240, y = 230), width = 110, and height = 140
        # Calculate the coordinates and mask the image:
        y_start = 230
        y_end = y_start + 140
        x_start = 240
        x_end = 240 + 110
        self.img_masked_face[y_start:y_end, x_start:x_end] = (0, 0, 255)

    def test_file_path(self):
        self.assertEqual(self.exercises.path_img, self.path_img)
        self.assertEqual(self.exercises.img_masked_face.shape, self.img_masked_face.shape)


if __name__ == '__main__':
    # unittest.main()
    unittest.main(argv=[''], verbosity=2, exit=False)
