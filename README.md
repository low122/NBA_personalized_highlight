# NBA Personalized Highlight Generator

This project builds an ai-powered floating widget that provides play-by-play analysis with an AI generated audio or text digests of the latest highlights and commentary.  The system processes NBA game footage to identify players, track their movements, and assign them to teams based on their jersey colors.  The output is an annotated video showing player and ball tracking.


## Features

* **Object Detection and Tracking:** Uses a pre-trained YOLO model to detect players, balls, and rims in NBA game videos.  The ByteTrack algorithm tracks objects across frames.
* **Team Assignment:** Assigns players to teams by analyzing their jersey colors using KMeans clustering and color histograms. Includes temporal smoothing for accuracy.
* **Ball Possession Assignment:** Determines which player possesses the ball based on proximity.
* **Video Annotation:**  Generates an output video with bounding boxes, team colors, and distance annotations for players and the ball.
* **Data Persistence:** Uses pickle files to store and load tracking data for faster processing.



## Usage

1.  **Prepare Input Video:** Place your NBA game video in the `input_videos` directory.
2.  **Run the script:** Execute `main.py`.  This will process the video and save the annotated output to the `output_videos` directory.
3.  **(Future Development):**  Integrate with a system for generating personalized highlight digests.


## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    ```
2.  **(Optional) Download pre-trained model:** Download the `best.pt` model and place it in the `models` directory.


## Technologies Used

* **Python:** The primary programming language.
* **Ultralytics YOLO:**  A powerful object detection model for identifying players, balls, and rims.
* **ByteTrack:**  An object tracking algorithm to maintain object identities across video frames.
* **OpenCV (cv2):**  Used for image processing tasks, video reading/writing, and drawing annotations.
* **Scikit-learn (sklearn):** Used for KMeans clustering in team color assignment.
* **Pandas:** Used for data manipulation and interpolation of ball positions.
* **Pickle:** Used for saving and loading tracking data.


## Configuration

The main configuration is through the paths to input and output videos and the pre-trained model in `main.py`:

```python
video_frames, fps = read_video('input_videos/Screen Recording 2025-02-08 at 00.10.37.mov')
# ... other code ...
save_video(output_video_frames, 'output_videos/team_assigner4.mp4', fps=fps)
```

You can modify these paths to point to your own videos and desired output locations.  The model path is also configurable within the `Tracker` class.


## Dependencies

The dependencies are listed in the `requirements.txt` file.  Install them using `pip install -r requirements.txt`.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request.


## Testing

Currently, no formal testing framework is implemented.  Testing would involve comparing the accuracy of object detection and tracking, as well as the robustness of team and ball assignment.


## License

MIT License

Copyright (c) 2025 ChickenTrapped

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



*README.md was made with [Etchr](https://etchr.dev)*
