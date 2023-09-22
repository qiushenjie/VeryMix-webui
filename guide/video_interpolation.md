## Video Frame Interpolation Tool Based on AI Models
**Interpolate video frames using AI frame interpolation models**

### Usage
1. **Basic Settings**
    - **Select Model**: Choose a model from the dropdown menu.
    - **Interpolation Factor**: Specify the frame interpolation factor, supporting 2x, 4x, and 6x frame interpolation.
2. **Advanced Settings**
    - **Inference Frame Scale**: Frame scale used for inference. Smaller scales may result in faster inference speed (in theory), but the quality may be lower.
    - **Inference Flow Scale**: Scale of the optical flow generated during inference. Smaller scales may result in faster inference speed but lower quality.
    - **Scene Cut Threshold**: Frames with a similarity below this threshold to their adjacent frames will be considered transition frames and will not undergo frame interpolation.
    - **Bad Frame Threshold**: Frames with a similarity below this threshold to their interpolated counterparts will be considered bad frames, and their interpolation results will not be retained.
    - **TTA (Test Time Augmentation)**: Test-time data augmentation that enhances prediction stability and accuracy by integrating multiple prediction results, but it may reduce model inference speed.
    - **Codec**: Supports x264 and x265 encoders.
    - **CRF (Constant Rate Factor)**: Encoding quality. Lower values produce larger video files with better quality.
    - **Output Video Extension**: Output video container format.
3. **Input Video**:
    - **Upload Video**: Drag and drop a video file or select a local video file.
4. **Output Video**:
    - **Interpolated Video Preview**: A preview of the interpolated video, with a duration of up to 60 seconds, for reviewing the results.
    - **Interpolated Video Download Link**: Link to download the interpolated video.
5. **Run**: Click the "Run" button after configuring the parameters to start the video frame interpolation process.
6. **Cancel**: Click "Stop" to pause the process at any time.
7. **Logs**: View operation and error information for the current session through the logs.
### Important Notes
- N/A