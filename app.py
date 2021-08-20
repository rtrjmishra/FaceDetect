from flask import Flask, render_template, Response,request
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)


def generate_frames():
    while True:

        ## read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/index2')
def index2():
    return render_template('detections.html')
@app.route('/detect')
def detect():
    return Response(detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

def detection():
    while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
                eye = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
                smile = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face.detectMultiScale(gray, 1.3, 7)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = frame[y:y + h, x:x + w]
                    eyes = eye.detectMultiScale(roi_gray, 2.5, 10)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    smiles = smile.detectMultiScale(roi_gray, 7, 10)
                    for (sx, sy, sw, sh) in smiles:
                        cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0,0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
if __name__ == "__main__":
    app.run(debug=True)