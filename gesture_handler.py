# gesture_handler.py
import cv2
import os
import random

# Function to get the path of a random gesture corresponding to each letter
def get_random_gesture_path(letter):
    gesture_folder = f'Data_Collected/{letter}'  # Each letter has its own folder

    # Check if the folder exists
    if os.path.exists(gesture_folder):
        # Get all images in the folder
        gesture_images = [f for f in os.listdir(gesture_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if gesture_images:
            # Randomly select an image
            selected_image = random.choice(gesture_images)
            gesture_path = os.path.join(gesture_folder, selected_image)
            return gesture_path
    return None

# Function to read the stored word
def read_stored_word(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()

# Main function to handle gesture translation and return gesture image paths
def handle_gesture_translation(file_path):
    constructed_word = read_stored_word(file_path)
    gesture_paths = []

    print(f"Reading constructed word: {constructed_word}")

    if constructed_word in os.listdir("Data_Collected") and len(constructed_word) > 2:
        gesture_path = get_random_gesture_path(constructed_word)
        if gesture_path:
            gesture_paths.append(gesture_path)
    else:
        for letter in constructed_word:
            gesture_path = get_random_gesture_path(letter)
            if gesture_path:
                gesture_paths.append(gesture_path)

    return gesture_paths
