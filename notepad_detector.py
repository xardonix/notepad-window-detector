import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import json 
import argparse

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary

def perform_ocr_on_roi(roi):
    #Perform OCR on the given region of interest
    roi_preprocessed = preprocess_image(roi)
    text = pytesseract.image_to_string(roi_preprocessed, lang='eng+rus', config='--psm 6')
    return text

def draw_text_pillow(image, text, position, font_path='arial.ttf'):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, 20)
    draw.text(position, text, font=font, fill=(0, 0, 255))
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def save_to_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

def detect_and_draw_window(image_path, output_path):
    img = cv2.imread(image_path)
    template = cv2.imread(r'images\templates\template.png', cv2.IMREAD_GRAYSCALE)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tH, tW = template.shape[:2]

    best_score = -1
    best_loc = None
    best_scale = 1

    # Perform scalable template matching for the icon
    for scale in np.linspace(0.5, 1.5, 20)[::-1]:
        resized = cv2.resize(template, (int(tW * scale), int(tH * scale)))
        if resized.shape[0] > gray_img.shape[0] or resized.shape[1] > gray_img.shape[1]:
            continue

        res = cv2.matchTemplate(gray_img, resized, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val > best_score:
            best_score = max_val
            best_loc = max_loc
            best_scale = scale

    if best_loc is not None:
        # Draw the detected icon on the image
        top_left = best_loc
        w_icon, h_icon = int(tW * best_scale), int(tH * best_scale)
        icon_positions = [(top_left[0], top_left[1], w_icon, h_icon)]
        center_x = top_left[0] + w_icon // 2
        center_y = top_left[1] + h_icon // 2
        cv2.circle(img, (center_x, center_y), 9, (0, 0, 255), 2)
    else:
        print("No icons found with the given threshold.")
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([0, 0, 200])
    upper_gray = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_gray, upper_gray)
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    best_score = 0
    tolerance = 5

    # Find the best contour that have icon on top left corner
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        for (icon_x, icon_y, icon_w, icon_h) in icon_positions:
            if (x - tolerance <= icon_x <= x + w + tolerance) and (y - tolerance <= icon_y <= y + h + tolerance):
                score = res[icon_y, icon_x]
                if score > best_score:
                    best_score = score
                    best_contour = (x, y, w, h)

    if best_contour is not None:
        x, y, w, h = best_contour
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3) #draw find contour
        
        window_tl = [x, y]
        window_br = [x + w, y + h]

        #define ocr roi coordinates
        ocr_roi_x_start = top_left[0] + w_icon + 2
        ocr_roi_y_start = top_left[1]
        ocr_roi_x_end = int(ocr_roi_x_start + w/2)
        ocr_roi_y_end = ocr_roi_y_start + h_icon

        ocr_roi = img[ocr_roi_y_start:ocr_roi_y_end, ocr_roi_x_start:ocr_roi_x_end]

        text = perform_ocr_on_roi(ocr_roi)
        img = draw_text_pillow(img, text, (ocr_roi_x_start + 5, ocr_roi_y_start))

        #define button roi coordinates
        btn_roi_x_start = x + w - 200
        btn_roi = img[y:y + 35, btn_roi_x_start:x + w]
        gray_roi = cv2.cvtColor(btn_roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_roi, 250, 255, cv2.THRESH_BINARY)
        button_contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        button_centers = []
        for button_contour in button_contours:
            bx, by, bw, bh = cv2.boundingRect(button_contour)
            cx, cy = bx + bw // 2, by + bh // 2
            button_centers.append((cx, cy))

        y_dict = {}
        for cx, cy in button_centers:
            if cy not in y_dict:
                y_dict[cy] = []
            y_dict[cy].append(cx)

        found = False
        button_positions = {'btn_minimize': None, 'btn_maximize': None, 'btn_close': None}
        
        #finding 3 coords that have same y and distinct x
        for cy, x_coords in y_dict.items():
            unique_x_coords = sorted(set(x_coords)) #drop duplicates
            if len(unique_x_coords) >= 3:
                button_positions['btn_minimize'] = (unique_x_coords[0] + btn_roi_x_start, cy + y)
                button_positions['btn_maximize'] = (unique_x_coords[1] + btn_roi_x_start, cy + y)
                button_positions['btn_close'] = (unique_x_coords[2] + btn_roi_x_start, cy + y)
                found = True
                for btn_name, (bx, by) in button_positions.items():
                    cv2.circle(img, (bx, by), 9, (0, 0, 255), 2)

        if not found:
            print("No suitable button centers found with the same y-coordinate and distinct x-coordinates.")

    
        caption = text.strip()

        json_data = {
            "window_tl": window_tl,
            "window_br": window_br,
            "caption": caption,
            "btn_minimize": button_positions['btn_minimize'],
            "btn_maximize": button_positions['btn_maximize'],
            "btn_close": button_positions['btn_close']
        }

        # save to JSON
        save_to_json(json_data, 'output_data.json')

    else:
        print("No suitable contour found.")

    cv2.imwrite(output_path, img)

def main():
    parser = argparse.ArgumentParser(description='Process images for icon detection and OCR.')
    parser.add_argument('--image_path', type=str, help='Path to the input image file.')
    parser.add_argument('--output_path', type=str, help='Path to save the output image file.', default='result.png')
    args = parser.parse_args()

    detect_and_draw_window(args.image_path, args.output_path)

if __name__ == '__main__':
    main()
