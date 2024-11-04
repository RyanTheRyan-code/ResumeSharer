from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np

class AugmentedRealityApp(App):
    def build(self):
        # Create an Image widget for displaying the output
        self.img = Image()
        # Start video capture from the webcam
        self.capture = cv2.VideoCapture(0)

        # Load target image and feature detection with ORB
        self.imgTarget = cv2.imread('PXL_20241104_031352252.MP~2.jpg')
        self.resume = cv2.imread('Resume - Ryan Makela 2024-1.png')
        self.orb = cv2.ORB_create(nfeatures=7500)
        self.kp1, self.des1 = self.orb.detectAndCompute(self.imgTarget, None)

        # Schedule the update method
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)  # 30 fps
        return self.img

    def update_frame(self, dt):
        ret, imgWebcam = self.capture.read()
        if ret:
            imgAug = imgWebcam.copy()
            kp2, des2 = self.orb.detectAndCompute(imgWebcam, None)

            # Check if descriptors are not None before continuing
            if des2 is None or self.des1 is None:
                return

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(self.des1, des2, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]

            if len(good) > 20:
                srcPts = np.float32([self.kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5.0)
                hT, wT, _ = self.imgTarget.shape
                pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)

                imgWarp = cv2.warpPerspective(self.resume, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))
                maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
                cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
                maskInv = cv2.bitwise_not(maskNew)
                imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
                imgAug = cv2.bitwise_or(imgWarp, imgAug)

            # Convert the augmented image to a Kivy texture and display it
            buf1 = cv2.flip(imgAug, 0)
            buf = buf1.tobytes()
            image_texture = Texture.create(size=(imgAug.shape[1], imgAug.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img.texture = image_texture

if __name__ == '__main__':
    AugmentedRealityApp().run()
