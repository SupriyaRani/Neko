from kivy.uix.floatlayout import FloatLayout
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.image import Image as KivyImage
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.slider import Slider
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.progressbar import ProgressBar
from kivy.clock import Clock
from kivy.uix.label import Label
import os
from kivy.uix.popup import Popup
from kivy.app import App
from kivy.lang import Builder


# Kivy String for Custom Styling
KV = """
<FileChooserIconView>:
    Label:
        color: 0, 0, 0, 1  # Black text color

<BoxLayout>:
    FileChooserIconView:
        size_hint: (0.9, 0.7)
        pos_hint: {'center_x': 0.5, 'center_y': 0.6}
        filters: ['*.png', '*.jpg', '*.jpeg']  # Display only PNG, JPG, and JPEG
"""

# Update ImageScreen to add ProcessingScreen
class ImageScreen(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.screen_manager = ScreenManager()
        self.upload_screen = UploadScreen(name="upload")
        self.processing_screen = ProcessingScreen(name="processing")
        self.edit_screen = EditScreen(name="edit")

        # Add screens
        self.screen_manager.add_widget(self.upload_screen)
        self.screen_manager.add_widget(self.processing_screen)
        self.screen_manager.add_widget(self.edit_screen)

        # Display the screen manager
        self.add_widget(self.screen_manager)

class UploadScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()

        # File chooser for selecting an image
        self.file_chooser = FileChooserIconView(filters=['*.png', '*.jpg', '*.jpeg'], 
                                                size_hint=(0.9, 0.7), pos_hint={'center_x': 0.5, 'center_y': 0.6},
                                                )
        layout.add_widget(self.file_chooser)

        # Upload button
        upload_button = Button(text="Upload Image", size_hint=(None, None), size=(200, 50),
                               pos_hint={'center_x': 0.5, 'center_y': 0.2})
        upload_button.bind(on_press=self.upload_image)
        layout.add_widget(upload_button)

        self.add_widget(layout)

    def upload_image(self, instance):
        if self.file_chooser.selection:
            global img_path
            img_path = self.file_chooser.selection[0]

            # Switch to ProcessingScreen
            self.manager.transition.direction = 'left'
            self.manager.current = "processing"

            # Start processing
            processing_screen = self.manager.get_screen("processing")
            processing_screen.start_processing(self.go_to_edit_screen)

    def go_to_edit_screen(self):
        self.manager.transition.direction = 'left'
        self.manager.current = "edit"
        self.manager.get_screen("edit").load_image(img_path)

class ProcessingScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()

        # Add a label to show the message
        label = Label(text="Uploading and Processing Image...",
                      font_size=24,
                      size_hint=(0.8, 0.1),
                      pos_hint={'center_x': 0.5, 'center_y': 0.6},
                      color=(0, 0, 0, 1))
        layout.add_widget(label)

        # Add a progress bar
        self.progress_bar = ProgressBar(max=100, size_hint=(0.8, 0.1),
                                        pos_hint={'center_x': 0.5, 'center_y': 0.4})
        layout.add_widget(self.progress_bar)

        self.add_widget(layout)

    def start_processing(self, next_screen_callback):
        # Start simulating progress
        self.progress = 0
        self.next_screen_callback = next_screen_callback
        Clock.schedule_interval(self.update_progress, 0.1)

    def update_progress(self, dt):
        self.progress += 10
        self.progress_bar.value = self.progress

        if self.progress >= 100:
            Clock.unschedule(self.update_progress)
            if self.next_screen_callback:
                self.next_screen_callback()

class EditScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_display = None
        self.text_layer = TextLayer(default_text="Text Behind Image", position=(100, 500))
        self.object_mask = None
        self.img_cv2 = None
        self.history = []  # History list to store changes

        self.layout = FloatLayout()
        self.add_widget(self.layout)

        # Add menu at the bottom
        self.menu = BoxLayout(orientation='vertical', size_hint=(1, 0.2), pos_hint={'center_x': 0.5, 'y': 0})
        self.add_menu_options()
        self.layout.add_widget(self.menu)

    def save_current_state(self):
        """
        Save the current state of the image, text, and any other parameters.
        """
        state = {
            "img_cv2": self.img_cv2.copy() if self.img_cv2 is not None else None,
            "text_layer": TextLayer(
                default_text=self.text_layer.default_text,
                position=self.text_layer.position,
                font_size=self.text_layer.font_size,
                color=self.text_layer.color,
                font_width=self.text_layer.font_width,
            )
        }
        self.history.append(state)

    def undo_last_change(self, instance):
        """
        Undo the last change by reverting to the previous state in the history.
        """
        if self.history:
            last_state = self.history.pop()  # Remove and get the last saved state
            self.img_cv2 = last_state["img_cv2"]
            self.text_layer = last_state["text_layer"]
            self.update_image_display()
        else:
            print("No changes to undo.")

    def add_menu_options(self):
        # First row of the menu
        first_row = BoxLayout(orientation='horizontal', size_hint=(1, 0.5))
        
        # Text Input for live updates
        self.text_input = TextInput(
            text="Text Behind Image",
            size_hint=(0.4, 1),
            multiline=False
        )
        self.text_input.bind(text=self.update_text)
        first_row.add_widget(self.text_input)

        # Font Size Button
        size_button = Button(text="Size", size_hint=(0.2, 1))
        size_button.bind(on_press=self.increase_font_size)
        first_row.add_widget(size_button)

        # Color Button
        self.color_button = Button(text="Color", size_hint=(0.2, 1))
        self.color_button.bind(on_press=self.change_text_color)
        first_row.add_widget(self.color_button)

        # Brightness Button
        # brightness_button = Button(text="Brightness", size_hint=(0.2, 1))
        # brightness_button.bind(on_press=self.apply_brightness)
        # first_row.add_widget(brightness_button)
        # Replace the Brightness Button with Beautify Button
        beautify_button = Button(text="Beautify", size_hint=(0.2, 1))
        beautify_button.bind(on_press=self.apply_beautify)
        first_row.add_widget(beautify_button)

        # Sharpness Button
        sharpness_button = Button(text="Sharpness", size_hint=(0.2, 1))
        sharpness_button.bind(on_press=self.apply_sharpness)
        first_row.add_widget(sharpness_button)

        # Save Image Button
        save_button = Button(text="Save", size_hint=(0.2, 1))
        save_button.bind(on_press=self.save_image)
        first_row.add_widget(save_button)

        # Second row of the menu
        second_row = BoxLayout(orientation='horizontal', size_hint=(1, 0.5))
        
        # Font Width Slider
        self.font_width_slider = Slider(min=1, max=25, value=8, size_hint=(0.4, 1))
        self.font_width_slider.bind(value=self.update_font_width)
        second_row.add_widget(self.font_width_slider)

        # Back Button
        back_button = Button(text="Back", size_hint=(0.2, 1))
        back_button.bind(on_press=self.undo_last_change)
        second_row.add_widget(back_button)

        # Add rows to the menu
        self.menu.add_widget(first_row)
        self.menu.add_widget(second_row)

    def change_text_color(self, instance):
        if self.text_layer:
            # Toggle between white, black, red, yellow, orange, and blue
            colors = [
                (255, 255, 255),  # White
                (0, 0, 0),        # Black
                (255, 0, 0),      # Red
                (255, 255, 0),    # Yellow
                (255, 165, 0),    # Orange
                (0, 0, 255)       # Blue
            ]
            
            # Check current color in RGB
            current_color_index = next((index for index, color in enumerate(colors) if color == self.text_layer.color), -1)
            
            if current_color_index == -1:
                current_color_index = 0  # Default to the first color if current is not found
            
            # Toggle to the next color
            next_color_index = (current_color_index + 1) % len(colors)
            new_color = colors[next_color_index]
            
            # Update color and image display
            self.text_layer.update_color(new_color)
            self.update_image_display()

    # def apply_brightness(self, instance):
    #     if self.img_cv2 is not None:
    #         self.save_current_state()  # Save current state before making changes
    #         self.img_cv2 = self.adjust_brightness(self.img_cv2, 1.2)  # Adjust brightness factor as needed
    #         self.update_image_display()
    def apply_beautify(self, instance):
        if self.img_cv2 is not None and self.object_mask is not None:
            self.save_current_state()  # Save current state before making changes
            self.img_cv2 = self.beautify_segment(self.img_cv2, self.object_mask)
            self.update_image_display()

    # def beautify_segment(self, image, mask):
    #     # Apply balanced corrections to the masked segment
    #     segment = cv2.bitwise_and(image, image, mask=mask)

    #     # Example corrections (adjust as needed)
    #     segment = cv2.convertScaleAbs(segment, alpha=1.2, beta=20)  # Increase brightness and contrast
    #     segment = cv2.GaussianBlur(segment, (5, 5), 0)  # Apply Gaussian blur for smoothing

    #     # Combine the corrected segment back with the original image
    #     inverted_mask = cv2.bitwise_not(mask)
    #     background = cv2.bitwise_and(image, image, mask=inverted_mask)
    #     beautified_image = cv2.add(background, segment)

    #     return beautified_image


    # Function to beautify the masked segment of the image

    def beautify_segment(self, image, mask):
        if image is None or mask is None:
            return image  # Return the original image if no mask or image is provided

        # Ensure the mask is resized to match the image dimensions
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Convert mask to a single channel if needed
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Convert mask to 8-bit unsigned integer type
        mask = mask.astype(np.uint8)

        # Debug prints
        print("Image shape:", image.shape)
        print("Mask shape (before resize):", mask.shape)

        # Extract the masked area
        masked_area = cv2.bitwise_and(image, image, mask=mask)
        
        # Apply bilateral filtering for edge-preserving smoothing
        smoothed = cv2.bilateralFilter(masked_area, 5, 15, 15)
        
        # Slightly enhance contrast and brightness of the masked area (subtle beautification)
        enhanced = cv2.convertScaleAbs(smoothed, alpha=1.1, beta=10)
        
        # Create a blurred mask for smooth edges
        blurred_mask = cv2.GaussianBlur(mask, (5, 5), 1)
        blurred_mask = blurred_mask.astype(np.float32) / 255.0

        # Ensure blurred_mask has the same number of channels as enhanced
        if len(blurred_mask.shape) == 2:
            blurred_mask = cv2.cvtColor(blurred_mask, cv2.COLOR_GRAY2BGR)

        # Debug prints
        print("Mask shape (after resize):", blurred_mask.shape)
        
        # Blend the enhanced region with the original image using the blurred mask
        masked_correction = cv2.multiply(enhanced.astype(np.float32), blurred_mask)
        original_contribution = cv2.multiply(image.astype(np.float32), 1 - blurred_mask)
        blended_masked_area = cv2.add(masked_correction, original_contribution).astype(np.uint8)
        
        # Merge the processed masked region back to the original image
        mask_inverse = cv2.bitwise_not(mask)
        untouched_area = cv2.bitwise_and(image, image, mask=mask_inverse)
        beautified_image = cv2.add(untouched_area, blended_masked_area)
        
        # Optionally enhance the entire image slightly
        beautified_image = cv2.convertScaleAbs(beautified_image, alpha=1.05, beta=5)
        
        return beautified_image

    def apply_sharpness(self, instance):
        if self.img_cv2 is not None:
            self.save_current_state()  # Save current state before making changes
            self.img_cv2 = self.sharpen_image(self.img_cv2)
            self.update_image_display()

    def go_back(self, instance):
        self.manager.transition.direction = 'right'
        self.manager.current = "upload"

    def adjust_brightness(self, image, factor):
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)

    def sharpen_image(self, image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening kernel
        return cv2.filter2D(image, -1, kernel)

    def load_image(self, img_path):
        # Clear widgets (except the menu)
        self.layout.clear_widgets()
        self.layout.add_widget(self.menu)

        # Load image
        self.img_cv2, self.object_mask = self.image_segmentation(img_path)

        # Display image in background
        self.image_display = KivyImage(source="temp_image.png", size_hint=(1, 0.8), pos_hint={'center_x': 0.5, 'y': 0.2})
        self.layout.add_widget(self.image_display)
        self.adjust_text_position_based_on_mask(self.object_mask)

    def image_segmentation(self, img_path):
        # Use YOLO model to perform segmentation
        model = YOLO("yolov8m-seg.pt")
        results = model.predict(img_path)
        result = results[0]
        masks = result.masks
        class_labels = result.names # Get class labels # Define the classes to keep (persons, vehicles, and animals) 
        classes_to_keep = ["person", "car", "truck", "bus", "motorbike", "bicycle", "dog", "cat", "horse", "sheep", "cow"]

        # Convert image
        img = Image.open(img_path)
        img_cv2 = np.array(img)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

        # Create object mask
        object_mask = np.zeros((img_cv2.shape[0], img_cv2.shape[1]), dtype=np.uint8)

        # Iterate through the results and apply mask only for the selected classes
        for mask_idx in range(len(masks)):
            class_idx = int(result.boxes.cls[mask_idx].item())  # Get class index for current mask
            class_name = class_labels[class_idx]  # Get the class name

            # Only process masks for person and animal classes
            if class_name in classes_to_keep:
                mask = masks[mask_idx].data[0].numpy()
                mask = (mask * 255).astype(np.uint8)  # Convert to binary mask (0 or 255)

                # Resize the mask to match the size of the original image
                mask_resized = cv2.resize(mask, (img_cv2.shape[1], img_cv2.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Add this mask to the overall object mask
                object_mask = cv2.bitwise_or(object_mask, mask_resized)

        # Save temporary masked image for display
        cv2.imwrite("temp_image.png", img_cv2)
        return img_cv2, object_mask 

    # def apply_text_on_image(self, img_cv2, text_layer, object_mask):
    #     text_canvas = np.zeros_like(img_cv2, dtype=np.uint8)
    #     cv2.putText(
    #         text_canvas,
    #         text_layer.default_text,
    #         (int(text_layer.position[0]), int(text_layer.position[1])),  # Ensure position is a tuple of integers
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         text_layer.font_size,
    #         text_layer.color,
    #         thickness=text_layer.font_width,  # Use font width from text layer
    #     )

    #     inverted_mask = cv2.bitwise_not(object_mask)
    #     text_visible = cv2.addWeighted(img_cv2, 1, text_canvas, 1, 0)
    #     masked_image = cv2.bitwise_and(img_cv2, img_cv2, mask=object_mask)
    #     background = cv2.bitwise_and(text_visible, text_visible, mask=inverted_mask)
    #     final_image = cv2.add(masked_image, background)

    #     # Convert to PIL and save temporarily
    #     final_image_pil = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    #     final_image_pil.save("temp_image.png")
        # return final_image_pil
    # Update the apply_text_on_image function to place text over predicted segments

    def apply_text_on_image(self, img_cv2, text_layer, object_mask):
        text_canvas = np.zeros_like(img_cv2, dtype=np.uint8)
        cv2.putText(
            text_canvas,
            text_layer.default_text,
            (int(text_layer.position[0]), int(text_layer.position[1])),  # Ensure position is a tuple of integers
            cv2.FONT_HERSHEY_SIMPLEX,
            text_layer.font_size,
            text_layer.color,
            thickness=text_layer.font_width,  # Use font width from text layer
        )

        # Now overlap the text on top of the object mask with 10% offset
        inverted_mask = cv2.bitwise_not(object_mask)
        text_visible = cv2.addWeighted(img_cv2, 1, text_canvas, 1, 0)
        masked_image = cv2.bitwise_and(img_cv2, img_cv2, mask=object_mask)
        background = cv2.bitwise_and(text_visible, text_visible, mask=inverted_mask)
        final_image = cv2.add(masked_image, background)

        # Convert to PIL and save temporarily
        final_image_pil = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        final_image_pil.save("temp_image.png")

        return final_image_pil

    # Adjust position based on segment overlap by 10% (this method applies while setting the text)
    def adjust_text_position_based_on_mask(self, object_mask):
        # Calculate the position of the top of the predicted segment
        x, y, w, h = cv2.boundingRect(object_mask)

        # Adjust Y position: 10% offset from the top of the bounding box
        adjusted_y_position = y + h * 0.1  # Overlay the text 10% from the top

        # Now update the text layer's position with adjusted coordinates
        self.text_layer.position = (x, adjusted_y_position)

        # Refresh the display
        self.update_image_display()


    def update_text(self, instance, value):
        if self.text_layer:
            self.save_current_state()  # Save current state before making changes
            self.text_layer.default_text = value
            self.update_image_display()

    # def save_image(self, instance):
    #     final_image_pil = self.apply_text_on_image(self.img_cv2, self.text_layer, self.object_mask)
    #     final_image_pil.save("final_image.png")
    #     print("Image saved as final_image.png")
    def save_image(self, instance):
        # Get the image name with prefix 'neko_' and replace spaces with underscores
        img_name = f"neko_{os.path.basename(img_path)}"
        img_name = img_name.replace(" ", "_")  # Replace spaces in the name if any

        # Check if the file already exists
        if os.path.exists(img_name):
            self.ask_to_replace_image(img_name)
        else:
            self.save_and_notify(img_name)

    def ask_to_replace_image(self, img_name):
        # Create a vertical BoxLayout for the popup content
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)
        
        # Add label
        label = Label(text="Do you want to replace the image?")
        content.add_widget(label)

        # Create a horizontal BoxLayout for the buttons
        button_layout = BoxLayout(orientation='horizontal', spacing=10, size_hint_y=0.3)

        # Yes button
        yes_button = Button(text="Yes", size_hint=(0.5, 1))
        yes_button.bind(on_press=lambda instance: self.handle_yes(img_name))
        button_layout.add_widget(yes_button)

        # No button
        no_button = Button(text="No", size_hint=(0.5, 1))
        no_button.bind(on_press=lambda instance: self.close_popup())
        button_layout.add_widget(no_button)

        # Add button layout to content
        content.add_widget(button_layout)

        # Create the Popup
        self.popup = Popup(
            title="This image already exists.", 
            content=content, 
            size_hint=(None, None), 
            size=(400, 300)
        )
        self.popup.open()


    def handle_yes(self, img_name):
        # Save the image and close the popup
        self.save_and_notify(img_name)
        self.close_popup()

    def close_popup(self):
        # Dismiss the popup
        if self.popup:
            self.popup.dismiss()

    def save_and_notify(self, img_name):
        # Apply text on image and save with the generated name
        final_image_pil = self.apply_text_on_image(self.img_cv2, self.text_layer, self.object_mask)
        final_image_pil.save(img_name)

        # Show the self-disappearing "Image Saved" message
        self.show_disappearing_message(f"Image saved!")

    def show_disappearing_message(self, message):
        # Create label for the message
        message_label = Label(
            text=message,
            font_size=18,
            size_hint=(None, None),
            size=(300, 50),
            pos_hint={'center_x': 0.5, 'center_y': 0.8},
            color=(0, 0, 0, 0)  # Black text color
        )
        self.layout.add_widget(message_label)

        # Set the label to disappear after 3 seconds
        Clock.schedule_once(lambda dt: self.remove_message(message_label), 3)

    def remove_message(self, message_label):
        # Remove the message label
        self.layout.remove_widget(message_label)

    def increase_font_size(self, instance):
        if self.text_layer:
            self.save_current_state()  # Save current state before making changes
            self.text_layer.update_font_size(1)
            self.update_image_display()

    # def change_text_color(self, instance):
    #     if self.text_layer:
    #         self.save_current_state()  # Save current state before making changes
    #         # Toggle between red and white
    #         new_color = (255, 0, 0) if self.text_layer.color == (255, 255, 255) else (255, 255, 255)
    #         self.text_layer.update_color(new_color)
    #         self.update_image_display()

    def update_font_width(self, instance, value):
        if self.text_layer:
            self.save_current_state()  # Save current state before making changes
            self.text_layer.update_font_width(int(value))
            self.update_image_display()

    def update_image_display(self):
        final_image_pil = self.apply_text_on_image(self.img_cv2, self.text_layer, self.object_mask)
        self.image_display.source = "temp_image.png"
        self.image_display.reload()

    def on_touch_move(self, touch):
        if self.text_layer and self.image_display.collide_point(*touch.pos):
            self.save_current_state()  # Save current state before making changes
            # Convert touch coordinates to image coordinates
            img_x = (touch.x - self.image_display.pos[0]) * (self.img_cv2.shape[1] / self.image_display.size[0])
            img_y = (self.image_display.size[1] - (touch.y - self.image_display.pos[1])) * (self.img_cv2.shape[0] / self.image_display.size[1])
            
            # Adjust x-coordinate scaling
            img_x = min(max(img_x, 0), self.img_cv2.shape[1])
            img_y = min(max(img_y, 0), self.img_cv2.shape[0])
            
            # Debug prints
            print(f"Touch coordinates: ({touch.x}, {touch.y})")
            print(f"Image coordinates: ({img_x}, {img_y})")
            
            self.text_layer.position = (img_x, img_y)
            self.update_image_display()


class TextLayer:
    def __init__(self, default_text="Text Behind Image", position=(100, 500), font_size=2, color=(255, 255, 255), font_width=2):
        self.default_text = default_text
        self.position = position
        self.font_size = font_size
        self.color = color
        self.font_width = font_width

    def update_font_size(self, increment):
        self.font_size = max(1, self.font_size + increment)

    def update_color(self, color):
        self.color = color

    def update_font_width(self, width):
        self.font_width = width


