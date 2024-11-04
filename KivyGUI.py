from kivy.app import App
from kivy.uix.image import Image

class ResumeApp(App):
    def build(self):
        img = Image(source='example.jpg')
        return img

if __name__ == '__main__':
    ResumeApp().run()
