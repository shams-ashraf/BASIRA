from deepface import DeepFace

def get_emotion(face_image):
    try:
        result = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
        dominant = result[0]['dominant_emotion']
        return dominant.capitalize()
    except:
        return "UNKNOWN"
