# Video Batch Processing

## Scope
The python application "batch-videos.py" offer the feature of batch processing N amount of video for Upscaling, Frame Rate and Face restoracion.
This python application is a refactor of the original "app.py" where already located all the code logic for a single operation.

## Requierements
* The UI must be done in Gradio
* Section 1: user upload all videos
* Section 2: user select the operation to be applyied to each video in Section 1. Valid operations will be Upscale, Frame Rate and Face Restoration.
* Section 3: a informative section to show the detail operations of each video in the process
* Section 4: a list of video processed that the user can be click and download. Each video processed must be appear as soon it finished.
* Buttons: There must be two important buttons, "Start Process" that will trigger the batch processing and "Download All" that will allow the user to download all videos compressed in a single TAR file.

# Logic of the Buttons
When clic on "Start Process" must validate that at least exist one video to process. The label of the button must change to "Stop Process" to allow the user to stop the batch processing at once.
The "Download All" button will only be enabled only when all the videos had been processed.

# Important Notes
The code logic is already implemented in "app.py" python application for a single and manual video process. The new application must use the code already in place. "app.py" call "face_restorer.py", "seedvr2_upscaler" and "frame_interpolator.py". Keep the same logic for the new app.
Validations must be take in place to keep the process working fine. 
The output of each video must keep the original file name plus "_slp" at the end of the name
The information section must show dynimically and very clear each detailed opertions that is been applied to each video.
Detailed technical documentation of the Video Batch Processing.