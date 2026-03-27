from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
import io

app = Flask(__name__)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Function to decode Braille image (same as your Python script)
def parseDecodePattern(image):
    kernel = np.ones((7, 7), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=4)

    kernel[:, :] = 0
    kernel[:, 3] = 1
    image = cv2.erode(image, kernel, iterations=4)

    return image

def parseBraille(image):
    kernel = np.zeros((13, 13), np.uint8)
    kernel[:, 6] = 1
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=4)

    kernel[:, 6] = 0
    kernel[6, 2:10] = 1
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=4)

    return image

def getNeighbours(labels, pixel):
    return [
        labels[pixel[0], pixel[1] - 1], 
        labels[pixel[0] - 1, pixel[1] - 1], 
        labels[pixel[0] - 1, pixel[1]], 
        labels[pixel[0] - 1, pixel[1] + 1]
    ]

def cca(image, v):
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    labels = np.zeros(image.shape, dtype=np.int32)
    label = 0
    equivalence_table = defaultdict(set)

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if image[i, j] in v:
                neighbours = getNeighbours(labels, [i, j])
                non_zero_neighbours = [x for x in neighbours if x != 0]
                if not non_zero_neighbours:
                    label += 1
                    labels[i, j] = label
                else:
                    min_label = min(non_zero_neighbours)
                    labels[i, j] = min_label
                    for lbl in non_zero_neighbours:
                        equivalence_table[min_label].add(lbl)
                        equivalence_table[lbl].add(min_label)

    for lbl, linked_labels in equivalence_table.items():
        for l in linked_labels:
            equivalence_table[l] = equivalence_table[l].union(linked_labels)

    resolved_labels = {lbl: min(equivalence_table[lbl]) for lbl in equivalence_table}

    for i in range(1, labels.shape[0] - 1):
        for j in range(1, labels.shape[1] - 1):
            if labels[i, j] in resolved_labels:
                labels[i, j] = resolved_labels[labels[i, j]]

    object_count = len(set(resolved_labels.values()))
    return object_count, labels[1:-1, 1:-1]

def cropImage(labels, image, space_flag=False, gap_threshold=15):
    label_numbers = np.unique(labels)
    label_numbers = label_numbers[label_numbers != 0]
    cropped_images = []
    previous_right = None

    for label in label_numbers:
        rows, cols = np.where(labels == label)
        top, bottom = rows.min(), rows.max() 
        left, right = cols.min(), cols.max()

        if space_flag and previous_right is not None:
            gap = left - previous_right
            if gap > gap_threshold:
                cropped_images.append(" ")

        cropped = image[top:bottom + 1, left:right + 1]
        cropped_images.append(cropped)
        previous_right = right

    return cropped_images

def matchImages(key, braille):
    characters = []
    for image1 in braille:
        if isinstance(image1, str) and image1 == " ":
            characters.append(" ")
            continue

        differences = []
        for image2 in key:
            resized1 = cv2.resize(image1, (50, 50), interpolation=cv2.INTER_AREA)
            resized2 = cv2.resize(image2, (50, 50), interpolation=cv2.INTER_AREA)
            differences.append(mse(resized1, resized2))

        match_index = np.argmin(differences)
        characters.append(chr(match_index + 97))

    return "".join(characters)

def mse(imageA, imageB):
    err = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)
    return err

@app.route('/decode', methods=['POST'])
def decode_braille():
    if 'image' not in request.files:
        return jsonify({"text": "No image uploaded."})

    file = request.files['image']
    img = Image.open(file.stream)
    braille_image = np.array(img.convert('L'))

    decode_pattern = cv2.imread(r"C:\Users\marve\Desktop\first\first\key.jpg", 0)  # Load key image
    connected_decode_pattern = parseDecodePattern(decode_pattern)
    _, labels_decode_pattern = cca(connected_decode_pattern, [0])
    decode_key = cropImage(labels_decode_pattern, decode_pattern)

    connected_braille = parseBraille(braille_image)
    _, labels_braille = cca(connected_braille, [0])
    braille_images = cropImage(labels_braille, braille_image, space_flag=True)
    english_text = matchImages(decode_key, braille_images)

    return jsonify({"text": english_text})

if __name__ == '__main__':
    app.run(debug=True)
