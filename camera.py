import cv2
import torch
import torch.nn as nn
import numpy as np

# =====================
# DEVICE
# =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# =====================
# MODEL
# =====================
class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            nn.Linear(64*5*5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )


    def forward(self, x):

        x = self.conv(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


# =====================
# LOAD MODEL
# =====================
model = CNN().to(device)

model.load_state_dict(
    torch.load("mnist_cnn.pth", map_location=device)
)

model.eval()

print("Model loaded!")

# =====================
# CAMERA
# =====================
cap = cv2.VideoCapture(0)

# ROI position
x1, y1 = 100, 100
x2, y2 = 300, 300

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # flip để giống gương
    frame = cv2.flip(frame, 1)

    # vẽ khung ROI
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # crop ROI
    roi = frame[y1:y2, x1:x2]

    # =====================
    # PREPROCESS
    # =====================

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # blur giảm noise
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # threshold → giống MNIST
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # resize 28x28
    img = cv2.resize(thresh, (28,28))

    # normalize
    img = img.astype(np.float32) / 255.0

    # convert tensor
    tensor = torch.from_numpy(img)\
                  .unsqueeze(0)\
                  .unsqueeze(0)\
                  .to(device)

    # =====================
    # PREDICT
    # =====================
    with torch.no_grad():

        output = model(tensor)

        pred = torch.argmax(output, dim=1).item()

        confidence = torch.softmax(output, dim=1).max().item()

    # =====================
    # SHOW RESULT
    # =====================

    cv2.putText(frame,
                f"Digit: {pred} ({confidence:.2f})",
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("Camera", frame)

    cv2.imshow("Processed (model input)", img)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()