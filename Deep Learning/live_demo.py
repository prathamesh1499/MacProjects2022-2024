import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
import tensorflow.keras as keras


model = load_model('model.h5', compile=False)

# Define the attributes
attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 
              'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 
              'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 
              'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
              'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 
              'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 
              'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 
              'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 
              'Wearing_Necklace', 'Wearing_Necktie', 'Young']

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    if not ret:
        break
        
    # Convert the frame to grayscale
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the frame
    face_cascade = cv2.CascadeClassifier('C:/Users/pmals/Desktop/MEST Automation and Smart Systems/SEP 740 - Deep Learning/SEP 740 Deep Learning Final Project/haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(rgb, 1.3, 5)
    
    for (x, y, w, h) in faces:
       
        face = frame[y:y+h, x:x+w]
        
        # Preprocess the face image
        face = cv2.resize(face, (228, 228))
        face = np.expand_dims(face, axis=0)
        face = face / 255.0
        
        # Make predictions
        predictions = model.predict(face)[0]
        percentages = [round(p*100, 2) for p in predictions]
        
        # Here we Draw the bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        label_y = y + 220
        for i, attribute in enumerate(attributes):
            label = attribute + ': ' + str(percentages[i]) + '%'
            cv2.putText(frame, label, (0, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            label_y -= 15
        

    cv2.imshow('Facial Attribute Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
