import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import os
os.environ["OPENCV_VIDEOIO_MS MF_ENABLE_HW_TRANSFORMS"] = "0"

# Initialize webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Mirror effect

    # Find hands
    hands, img = detector.findHands(img)

    if hands:
        for hand in hands:
            handType = hand["type"]
            lmList = hand["lmList"]
            bbox = hand["bbox"]

            # Display hand type
            cv2.putText(img, handType, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Check if two hands are detected
        if len(hands) == 2:
            hand1 = hands[0]
            hand2 = hands[1]

            lmList1 = hand1["lmList"]
            lmList2 = hand2["lmList"]

            # Get index fingertips (landmark 8)
            x1, y1 = lmList1[8][0], lmList1[8][1]
            x2, y2 = lmList2[8][0], lmList2[8][1]

            # Draw circles on index fingertips
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)

            # Draw line between fingertips
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # Calculate and show distance
            distance = int(math.hypot(x2 - x1, y2 - y1))
            cv2.putText(img, f"Distance: {distance}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Optional: Show "Touching" message if close
            if distance < 40:
                cv2.putText(img, "Touching!", (200, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # Show result
    cv2.imshow("Hand Tracking", img)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
