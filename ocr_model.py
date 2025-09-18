import cv2, pytesseract

pytesseract.pytesseract.tesseract_cmd = r"D:\Downloads\tesseract.exe"

def extract_text_with_boxes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config="--psm 6")
    results = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if text and len(text) > 1:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            results.append((text, (x, y, w, h)))
    return results
