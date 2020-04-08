import os

import cv2

try:
    if not os.path.exists(
        "../Data_Memoir"
    ):
        os.mkdir(
            "../Data_Memoir"
        )
    # Check for the folder 'Data_Memoir'.
    # It not present, create it.
except OSError:
    print(
        "Could not make the Data_Memoir directory."
    )


def process(vid_type, series, threshold):

    # Function to convert videos to frames.
    '''

    # vid_type: Type of video: Real or Animated. For now we have only these options. But the code is ready for future additions.
    # series: Name of the series the video to be converted is from.
    # threshold: Number of frame to make for a particular series.

    '''

    videos = os.listdir("../Memoir_Videos/" + vid_type +
                        "/" + series + "/Video/"
                        # To take into consideration that there might be more than 1 video for a series.
                        )

    currentframe = 0

    for vid in videos:

        # Looping over the videos of a particular series

        cam = cv2.VideoCapture(
            "../Memoir_Videos/" + vid_type + "/" + series + "/Video/" + vid
            # Reading a video using open-cv
        )

        while True:

            # Loop tp iterate over frames of a video

            if currentframe == threshold:
                # Checking for threshold
                return

            ret, frame = cam.read()

            # Reading a frame
            # ret: Boolean value. True if a frame is read, else False.
            # frame: Read frame.

            if ret:

                try:
                    if not os.path.exists(
                        "../Data_Memoir/" + vid_type
                    ):
                        os.mkdir(
                            "../Data_Memoir/" + vid_type)
                # Check for the folder 'Video_Type' (Real or Animated) in Data_Memoir folder.
                # If not present, create it.
                except OSError:
                    print(
                        "OSError: Could not create "
                        + vid_type
                        + " directory in Data_Memoir."
                    )

                try:
                    if not os.path.exists(
                        "../Data_Memoir/" + vid_type + "/" + series
                    ):
                        os.mkdir(
                            "../Data_Memoir/" + vid_type + "/" + series
                        )
                # Check for the folder 'Series_Name' in 'Video_Type' folder.
                # It not present, create it.
                except OSError:
                    print(
                        "OSError: Could not create "
                        + series
                        + " directory in Data_Memoir/"
                        + vid_type
                        + "."
                    )

                try:
                    if not os.path.exists(
                        "../Data_Memoir/" + vid_type + "/" + series + "/Frames"
                    ):
                        os.makedirs(
                            "../Data_Memoir/" + vid_type + "/" + series + "/Frames"
                        )
                # Check for the folder 'Frames' in 'Series_Name' folder.
                # It not present, create it.
                except OSError:
                    print(
                        "OSError: Could not create Frames directory in Data_Memoir/"
                        + vid_type
                        + "/"
                        + series
                        + "."
                    )

                name_of_frame = (
                    "../Data_Memoir/"
                    + vid_type
                    + "/"
                    + series
                    + "/Frames/"
                    + series
                    + "_frame_"
                    + str(currentframe).zfill(5)
                    + ".jpg"
                    # The name of the frame.
                )
                print("Creating..." + name_of_frame)

                cv2.imwrite(name_of_frame, frame)  # Saving the frame.

                currentframe += 1
            else:
                break

        cam.release()
        cv2.destroyAllWindows()


which_data = input(
    "Enter the type of data to generate frame for? [default: All] \n"
    # Asking for the type of video to make frames of. For now only Real and Animated videos are considered.
)
threshold = int(
    input(
        'Enter the number of frames to make for a particular series. \n'
        # Asking for threshold: Number of frames to make for a series.
    ))

types_of_videos = []

if which_data == "All":
    # Make frame for all the videos.
    types_of_videos = sorted(os.listdir("../Memoir_Videos"))
else:
    # Make frame for only the type entered by the user.
    types_of_videos.append(which_data)

for v_t in types_of_videos:
    # v_t: Looping over the types of videos to make frame of (Real or/and Animated).
    for sr in sorted(os.listdir("../Memoir_Videos/" + v_t)):
        # sr: Looping over series of each type of videos.
        try:
            os.listdir("../Memoir_Videos/" + v_t + "/" + sr + "/Video")
        # Checking if the path is broken or not. If broken, the loop will skip that path.
        except OSError:
            continue
        process(v_t, sr, threshold)
