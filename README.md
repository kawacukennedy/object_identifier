# Object Identifier App

A real-time computer vision application that uses your webcam to detect objects, matches them against a predefined list, and announces the object names using text-to-speech with a male voice.

## 🚀 Features

- **Real-time Object Detection**: Uses your webcam to continuously detect objects in real-time
- **Pre-trained Model**: Leverages Faster R-CNN with MobileNetV3 backbone for accurate detection
- **Voice Announcements**: Automatically announces detected object names using text-to-speech
- **Customizable Object List**: Easy to modify which objects to track and announce
- **Visual Feedback**: Displays bounding boxes and confidence scores on the video feed
- **Multi-threaded Processing**: Separate threads for video capture and object detection for smooth performance

## 📋 Requirements

- **Python**: 3.8 or higher
- **Webcam**: Any USB webcam or built-in camera
- **Operating System**: Windows, macOS, or Linux
- **Memory**: At least 4GB RAM recommended
- **Storage**: ~500MB for model downloads

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kawacukennedy/object_identifier.git
   cd object_identifier
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional**: Install additional TTS voices for better voice quality:
   - **Windows**: Voices are typically pre-installed
   - **macOS**: Go to System Preferences > Accessibility > Speech > System Voice
   - **Linux**: Install `espeak` or `festival` for additional voices

## 🎯 Usage

1. **Run the application**:
   ```bash
   python object_identifier/app.py
   ```

2. **Controls**:
   - Press `q` to quit the application
   - The application will automatically start detecting objects

3. **What you'll see**:
   - Live webcam feed with bounding boxes around detected objects
   - Object names and confidence scores displayed on screen
   - Voice announcements when new objects are detected

## ⚙️ Configuration

### Customizing Detected Objects

Edit the `KNOWN_OBJECTS` list in `object_identifier/app.py` to change which objects trigger voice announcements:

```python
KNOWN_OBJECTS = [
    'person', 'bottle', 'cup', 'cell phone', 'laptop', 
    'book', 'chair', 'dog', 'cat', 'car'
]
```

### Available Object Categories

The application uses the COCO dataset labels. Some common objects include:
- **People**: `person`
- **Animals**: `dog`, `cat`, `bird`, `horse`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`
- **Vehicles**: `car`, `truck`, `bus`, `motorcycle`, `bicycle`, `airplane`, `boat`
- **Electronics**: `laptop`, `cell phone`, `tv`, `remote`, `keyboard`, `mouse`
- **Furniture**: `chair`, `couch`, `bed`, `dining table`
- **Food**: `apple`, `banana`, `orange`, `pizza`, `hot dog`, `cake`
- **Kitchen items**: `cup`, `bottle`, `fork`, `knife`, `spoon`, `bowl`
- **And many more...**

### Adjusting Detection Sensitivity

Modify the confidence threshold in the `detection_worker` function:

```python
if score > 0.7:  # Change this value (0.0 to 1.0)
```

- **Higher values** (e.g., 0.8-0.9): Fewer false positives, may miss some objects
- **Lower values** (e.g., 0.5-0.6): More detections, may include false positives

## 🔧 How It Works

1. **Video Capture**: Continuously captures frames from your webcam
2. **Object Detection**: Uses a pre-trained Faster R-CNN model to detect objects in each frame
3. **Object Matching**: Compares detected objects against the `KNOWN_OBJECTS` list
4. **Voice Announcement**: When a new known object is detected, announces its name using text-to-speech
5. **Visual Display**: Draws bounding boxes and labels on the video feed

### Technical Details

- **Model**: Faster R-CNN with MobileNetV3-Large FPN backbone
- **Dataset**: Pre-trained on COCO dataset (80 object categories)
- **Processing**: Multi-threaded architecture for smooth performance
- **TTS Engine**: Uses `pyttsx3` for cross-platform text-to-speech

## 🐛 Troubleshooting

### Common Issues

1. **Camera not working**:
   - Ensure your webcam is connected and not in use by another application
   - Try changing the camera index in `cv2.VideoCapture(0)` to `1` or `2`

2. **No voice announcements**:
   - Check if your system has TTS voices installed
   - On macOS, ensure System Voice is set in Accessibility settings
   - On Linux, install `espeak`: `sudo apt-get install espeak`

3. **Poor performance**:
   - Close other applications using the camera
   - Reduce the detection frequency by increasing the sleep time in `detection_worker`
   - Lower the input resolution by modifying the resize dimensions

4. **Model download issues**:
   - Ensure you have a stable internet connection for the first run
   - The model will be cached locally after the first download

### Performance Optimization

- **Lower resolution**: Modify the resize dimensions in `detection_worker`
- **Higher confidence threshold**: Reduce false positives
- **Longer sleep intervals**: Reduce CPU usage

## 📦 Dependencies

- **opencv-python**: Computer vision and video processing
- **torch**: PyTorch deep learning framework
- **torchvision**: Pre-trained models and transforms
- **pyttsx3**: Text-to-speech functionality

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [COCO Dataset](https://cocodataset.org/) for the object detection model
- [OpenCV](https://opencv.org/) for computer vision capabilities

## 📞 Support

If you encounter any issues or have questions, please:
1. Check the troubleshooting section above
2. Search existing issues on GitHub
3. Create a new issue with detailed information about your problem

---

**Made with by [kawacukennedy](https://github.com/kawacukennedy)** 
