from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.core.window import Window
from kivy.clock import Clock  # To schedule an action (like hiding the spinner)
from image_screen import ImageScreen


class TextBehindImageApp(App):
    def build(self):
        Window.clearcolor = (1, 1, 1, 1)  # Setting the window color to white
        self.layout = FloatLayout()

        # Adding the loader before showing the option screen
        self.add_loader()

        return self.layout

    def add_loader(self):
        """Show a loading spinner before showing the content"""
        # Create a spinner widget that displays loading text
        self.spinner = Spinner(
            text='Loading...',
            size_hint=(None, None),
            size=(200, 100),
            pos_hint={'center_x': 0.5, 'center_y': 0.5},  # Center the spinner
            color=(0, 0, 0, 1),
        )
        
        # Add the spinner to the layout to show the loader
        self.layout.add_widget(self.spinner)
        
        # Use Clock to call load_options function after 2 seconds
        Clock.schedule_once(self.load_options, 2)

    def load_options(self, dt):
        """Remove the loader and load the option screen"""
        # Remove the spinner from the layout
        self.layout.clear_widgets()

        # Proceed to show the actual options screen after the loader disappears
        self.option_screen()

    def option_screen(self):
        """Create and add the content for the option screen"""
        # Title Label
        label = Label(text="Neko", size_hint=(None, None),
                      size=(500, 80), pos_hint={'center_x': 0.5, 'center_y': 0.8}, font_size=50, color=(255, 165, 0, 1))
        self.layout.add_widget(label)

        # Subtitle Label
        label = Label(text="Discover a new way to edit photos", size_hint=(None, None),
                  size=(300, 50), pos_hint={'center_x': 0.5, 'center_y': 0.7}, font_size=18, color=(0, 0, 0, 1))
        self.layout.add_widget(label)

        # Button for Text Behind Image
        btn1 = Button(text="Text Behind Image", size_hint=(None, None), size=(200, 100),
                      pos_hint={'center_x': 0.3, 'center_y': 0.5}, background_color=(0, 0.5, 1, 1))
        btn1.bind(on_press=self.go_to_image_screen)
        self.layout.add_widget(btn1)

        # Button for Carousel Frame
        btn2 = Button(text="Carousel Frame", size_hint=(None, None), size=(200, 100),
                      pos_hint={'center_x': 0.7, 'center_y': 0.5}, background_color=(0, 0.5, 1, 1))
        btn2.bind(on_press=self.go_to_corousel_screen)
        self.layout.add_widget(btn2)

        # Developer Info at the bottom
        dev_info = Label(text="Developed by Your AI Lab | 2025", size_hint=(None, None),
                         size=(300, 50), pos_hint={'center_x': 0.5, 'center_y': 0.05}, font_size=12, color=(0, 0, 0, 1))
        self.layout.add_widget(dev_info)

    def go_to_image_screen(self, instance):
        """Switch to image screen when the button is pressed"""
        image_screen = ImageScreen()
        self.layout.clear_widgets()
        self.layout.add_widget(image_screen)

    def go_to_corousel_screen(self, instance):
        """Switch to carousel screen when the button is pressed"""
        image_screen = ImageScreen()
        self.layout.clear_widgets()
        self.layout.add_widget(image_screen)


if __name__ == "__main__":
    TextBehindImageApp().run()
