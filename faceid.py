# Import Kivy dependencies

# App Layout
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import Kivy UX components
from kivy.uix.image import Image as KivyImage
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy objects
from kivy.clock import Clock  # helps us get continuous real time feed from webcam in kivy app
from kivy.graphics.texture import Texture  # convert image from opencv to texture
from kivy.logger import Logger  # shows logs and app  metrics

# Import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
from PIL import Image
import pymysql
import drivers

# Build App and Layout
class CamApp(App):





    def build (self):
        # Main layout components
        #self.web_cam = Image(size_hint=(1, .8))
        self.web_cam = KivyImage(source ='download.jpg')
        #self.setimage = KivyImage(source ='download.jpg')

        self.verify_label = Label(text="Please Face The Camera And Select Verify, Then Scan Your ID", size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1, .1))
        self.button = Button(text="Verify", on_press = self.verify,  size_hint=(1, .1))

        #Add items to layout

        layout = BoxLayout(orientation = 'vertical',size=(600,600), pos_hint={'y': .1} )
        layout.add_widget(self.web_cam)
        #layout.add_widget(self.setimage)

        layout.add_widget(self.button)
        layout.add_widget(self.verify_label)

        layout.add_widget(self.verification_label)

        # Load tensorflow keras model
        self.model = tf.keras.models.load_model('siamesemodel3.h5', custom_objects={'L1Dist': L1Dist})

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)  # 33 times per 1 second

        display = drivers.Lcd()
        display.lcd_display_string('Face The Camera', 1)
        display.lcd_display_string('Select Verify', 2)

        return layout

    #Run continously  to  get   webcam   feed

    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[180:180 + 250, 180:180 + 250, :]

        # Flip horizontal and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        # passing buf through blit_buffer method
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Preprocess..Loads image from file folder and converts to 100x 100 pixels
    def preprocess(self, file_path):

        # load image file path
        byte_img = tf.io.read_file(file_path)
        # load image and decode jpeg
        img = tf.io.decode_jpeg(byte_img)
        # resize image to 100x100 pixels from 3 channels (RGB)
        img = tf.image.resize(img, (100, 100))
        # divide by 255 to scale be between 0 and 1
        img = img / 255.0
        # returns image
        return img

    # verification function to verify person
    def verify(self, *args):

        # read the card using the rfid card
        reader = SimpleMFRC522()
        display = drivers.Lcd()
        display.lcd_display_string('Hello Please', 1)
        display.lcd_display_string('Scan Your ID', 2)
        try:
            id, text = reader.read()
            print(id)
            display.lcd_clear()

            # Load the driver and set it to "display"
            # If you use something from the driver library use the "display." prefix first

            try:
                sql_con = pymysql.connect(host='localhost', user='rfidreader', passwd='password', db='rfidcardsdb')
                sqlcursor = sql_con.cursor()

                # first thing is to check if the card exist
                cardnumber = '{}'.format(id)

                sql_request = 'SELECT user_id FROM cardtbl WHERE serial_no = "' + cardnumber + '"'

                count = sqlcursor.execute(sql_request)
                if count > 0:
                    print("already in database")
                    T = sqlcursor.fetchone()
                    for i in T:
                        print(i)

                        display.lcd_display_string('Welcome '+i, 1)
                        display.lcd_display_string('Please Wait ', 2)


                    #def auth(self):



                    #

                        #Start Here
                        # specify thresholds....setting verification threshold to high helps ensure verification accuracy
                        #Adding more images could also help verification accuracy


                        detection_threshold = 0.85
                        verification_threshold = 0.8

                        # Capture input image from our webcam
                        SAVE_PATH = os.path.join('application_data', 'input_image','input_image.jpg')
                        ret, frame = self.capture.read()
                        frame = frame[180:180+250, 180:180+250,:]
                        cv2.imwrite(SAVE_PATH,frame)


                        # Build results array
                        results = []
                        # looping through and listing every image in verification image folder
                        for image in os.listdir(os.path.join('application_data', 'verification_images')):
                            # Grabbing input image from webcam, using prepocessing function created earlier
                            # to scale and resize imge and saving image in input_image folder
                            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))


                            # Looping through validation images and passing through preprocess function,
                            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))

                            # Make Predictions
                            # using model.predict method created earlier
                            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
                            # creating a big array of rseults
                            results.append(result)

                        #Evaluation
                        # Detection threshold is metric above which a prediction is considered positive
                        detection = np.sum(np.array(results) > detection_threshold)

                        # verification Threshold: Proportion of positive predictions / total positive samples( 50 selected images)
                        verification = detection / len(
                            os.listdir(os.path.join('application_data', 'verification_images')))  # dividing by 50 images
                        verified = verification > verification_threshold

                        # Set Verification text
                        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

                        #Log out details
                        Logger.info(results)
                        print("Detection ")
                        Logger.info(detection)
                        print("Verification ")
                        Logger.info(verification)
                        print("verified? ")
                        Logger.info(verified)




                        # calculating % of results above % threshold
                        print("above 30% ")
                        Logger.info(np.sum(np.array(results) > 0.3))

                        print("above 50% ")
                        Logger.info(np.sum(np.array(results) > 0.5))

                        print("above 80% ")
                        Logger.info(np.sum(np.array(results) > 0.8))


                        #return results, verified
                        #End Here


                        if verified == True:
                            display.lcd_clear()
                            display.lcd_display_string('You Are Verified!', 1)
                            display.lcd_display_string('Access Granted', 2)

                        else:
                            display.lcd_display_string('Unknown User!', 1)  # Write line of text to first line of display
                            display.lcd_display_string('Access Denied', 2)  # Write line of text to first line of display
                            self.verification_label.text = 'Unverified User'

                        return results, verified



                else:
                    display.lcd_display_string('Unknown user!', 1)  # Write line of text to first line of display
                    display.lcd_display_string('Access denied', 2)  # Write line of text to first line of display
                    print("Not in database")
                    self.verification_label.text = 'Unknown User ID'



            except KeyboardInterrupt:
                    # If there is a KeyboardInterrupt (when you press ctrl+c), exit the program and cleanup
                    print("Cleaning up!")
                    display.lcd_clear()


        finally:
            GPIO.cleanup()


if __name__ == '__main__':

    CamApp().run()