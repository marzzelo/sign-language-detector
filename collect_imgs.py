import os
import handDetector as hd
import cv2


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 5
dataset_size = 100


hd = hd.HandDetector(max_num_hands=1)

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        
        cv2.putText(frame, 'Ready? Press "k" [esc EXIT]', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 128, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)
        if key == 27:
            done = True
            break
        elif key == ord('k'):
            break

    if done:
        print('Data collection abborted.')
        break
    
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        
        # Display sampled points and connections
        frame = hd.findHands(frame, connections=True, points=True)
        
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

    print('Data collection for class {} finished.'.format(j))
    
cap.release()
cv2.destroyAllWindows()
