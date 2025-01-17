# LookAhead
Inspired by [WebGazer.js](https://webgazer.cs.brown.edu/), in this project, I explored accessible technology and its real-world applications.
LookAhead allows users to predict gaze positions with iris coordinates entirely in Python.

This project consists of **two** core parts:
1. **Blendshape-Based Gaze Prediction**: A basic approach that used predefined facial landmarks for gaze estimation.
2. **On-the-Go Model Training**: A system that continuously learns and adapts to user-specific gaze behavior through live interactions (mouse clicks).

The simpler version uses blendshapes built into Mediapipe to detect the direction in which the user is looking, and executes a left-click when the user puckers their lips.

The more advanced (and frankly, cooler) version lets users interact with a set of dots on the screen, capturing their mouse position and iris data with each click. With every click, two regression models (one for x-coor and the other for y-coor) are trained incrementally with this new data. The system then predicts gaze positions based on real-time iris data and visualises them as a blue circle on the screen.

---
![lookahead-train](https://github.com/user-attachments/assets/efca9c22-bee0-4e65-954b-e81d35afb1d2)
*Training the model by clicking dots on the screen.*

![lookahead-test](https://github.com/user-attachments/assets/5a748303-7407-452f-bd6d-c14eadefc1a2)
*Predicted iris position vs actual mouse cursor location*

---
Technologies used:
- Python
- Mediapipe Iris
- OpenCV
- Pygame
- Scikit-learn
