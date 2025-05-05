import kivy
from kivy.app import App
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
import numpy as np
import threading
import time

kivy.require('2.3.1')  # Ensure the version of Kivy is at least 2.3.1


class RedTextureApp(App):
    def build(self):
        # Create a BoxLayout to hold the image
        layout = BoxLayout(orientation='vertical')

        # Create an Image widget
        self.video_label = Image(allow_stretch=True, keep_ratio=True, size_hint=(1, 1))

        # Add the Image widget to the layout
        layout.add_widget(self.video_label)

        # Start the texture update thread
        self.texture_thread = threading.Thread(target=self.update_texture, daemon=True)
        self.texture_thread.start()

        return layout

    def update_texture(self):
        while True:
            time.sleep(1)  # Simulate a delay to update the texture every 1 second

            # Create a red image (480x640) in RGB format
            red_img = np.ones((480, 640, 3), dtype=np.uint8) * np.array([255, 0, 0], dtype=np.uint8)  # Pure red

            # Convert the image to a texture
            red_buf = red_img.tobytes()
            red_texture = Texture.create(size=(640, 480), colorfmt='rgb')
            red_texture.blit_buffer(red_buf, colorfmt='rgb', bufferfmt='ubyte')

            # Safely update the texture in the main thread using Clock.schedule_once
            Clock.schedule_once(self.update_gui_texture, 0, red_texture)

    def update_gui_texture(self, dt, texture):
        # This function will be executed on the main thread
        if self.video_label:
            self.video_label.texture = texture


if __name__ == '__main__':
    RedTextureApp().run()
