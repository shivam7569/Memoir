import json
import os

import cv2


def vid_to_frames(vid_type, series, threshold):

    try:
        if not os.path.exists(paths['frame_dir']):
            os.mkdir(paths['frame_dir'])
        # Check for the folder 'Data_Memoir'.
        # If not present, create it.
    except OSError:
        print("Could not make the Data_Memoir directory.")

    videos = os.listdir(
        paths['video_dir']
        + str(vid_type)
        + "/"
        + series
        + "/Video/"
        # To take into consideration that
        # there might be more than 1 video for a series.
    )

    currentframe = 0

    for vid in videos:

        # Looping over the videos of a particular series

        cam = cv2.VideoCapture(
            paths['video_dir']
            + vid_type
            + "/"
            + series
            + "/Video/"
            + vid
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
                    if not os.path.exists(paths['frame_dir'] + vid_type):
                        os.mkdir(paths['frame_dir'] + vid_type)
                # Check for the folder 'Video_Type' in Data_Memoir.
                # If not present, create it.
                except OSError:
                    print(
                        "OSError: Could not create "
                        + vid_type
                        + " directory in Data_Memoir."
                    )

                try:
                    if not os.path.exists(paths['frame_dir'] + vid_type + "/" + series):
                        os.mkdir(paths['frame_dir'] + vid_type + "/" + series)
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
                        paths['frame_dir'] + vid_type + "/" + series + "/Frames"
                    ):
                        os.makedirs(
                            paths['frame_dir'] + vid_type + "/" + series + "/Frames"
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
                    paths['frame_dir']
                    + vid_type
                    + "/"
                    + series
                    + "/Frames/"
                    + series
                    + "_frame_"
                    + str(currentframe).zfill(len(str(threshold)))
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


def process_call(v_type, threshold):

    # Function to start the vid_to_frames.

    which_data = v_type

    types_of_videos = []

    if which_data == "All" or which_data == os.listdir(paths['video_dir']):
        # Make frames for all the videos.
        types_of_videos = sorted(os.listdir(paths['video_dir']))
    elif isinstance(which_data, list):
        for w_d in which_data:
            types_of_videos.append(w_d)
    elif isinstance(which_data, str):
        types_of_videos.append(which_data)

    for v_t in types_of_videos:
        # v_t: Looping over the types of videos to make frame of (entered by the user).
        for sr in sorted(os.listdir(paths['video_dir'] + v_t)):
            # sr: Looping over series of each type of videos.
            try:
                os.listdir(paths['video_dir'] + v_t + "/" + sr + "/Video")
            # Checking if the path is broken or not. If broken, the loop will skip that path.
            except OSError:
                continue
            vid_to_frames(v_t, sr, threshold)


def convert_vid_to_frames(vid_type="All", series='All', threshold=50000):
    """
    This function converts the videos into frames. This is useful in making dataset for training.

    Args:
        vid_type (str): Type of the video to make frames from, for now we only use 'Real' and 'Animated'. Use list for multiple entries. Default: 'All'
        series (str): Name(s) of a series from the entered vid_type to make frames from. Use list for multiple entries. Default: 'All'
        threshold (int): Number of frames to make. Make sure the videos are long enough for desired threshold. Default: 50000

    Returns:
        Nothing. It creates a directory 'Data_Memoir' inside the repository containing the generated frames.

    Raises:
        OSError: If videos to generate frames from are not found.
        OSError: When `series` does not belong to `vid_type`.
    """

    def check_videos(v_type_, ser_):
        try:
            if os.path.exists(paths['video_dir']):
                print(
                    "\nVideos found for {0} in {1}. Starting the process...\n".format(
                        ser_, v_type_
                    )
                )
                return
        except:
            raise OSError("Videos not found.")

    file_path = os.path.dirname(os.path.realpath(__file__)) 
    os.chdir(file_path)

    with open('../../../paths.json', 'r') as file:
        global paths
        paths = json.load(file)

    if series == "All":

        # If series is 'All', then it will generate frame from all the series for the given vid_type.

        if os.path.exists(paths['video_dir']):

            # Checking for videos.

            print("\nVideos Found. Starting the process.")
            process_call(v_type=vid_type, threshold=threshold)
        else:
            raise OSError("Videos not found.")
            exit(1)

    else:

        if isinstance(vid_type, list):
            # If the vid_type entered is a list, then further error handling...
            if len(vid_type) > 1:
                # If the number of vid_types entered is more than 1, then it's an invalid entry.
                raise OSError("A series cannot belong to more than one video types.")
                exit(1)
            if len(vid_type) == 1:
                # If only one entry for vid_type, then take it out.
                vid_type = vid_type[0]
        if isinstance(vid_type, str):
            if vid_type == "All":
                raise OSError("A series cannot belong to all the video types.")
                exit(1)

        if isinstance(series, str):

            if series not in os.listdir(paths['video_dir'] + vid_type + "/"):
                raise OSError("Series does not belong to entered video type.\n")
                exit(1)
            check_videos(v_type_=vid_type, ser_=series)
            vid_to_frames(vid_type=vid_type, series=series, threshold=threshold)

        if isinstance(series, list):
            for srs in series:

                if srs not in os.listdir(paths['video_dir'] + vid_type + "/"):
                    raise OSError(
                        "Series {0} does not belong to entered video type: {1}.\n".format(
                            srs, vid_type
                        )
                    )
                    exit(1)
                else:
                    check_videos(v_type_=vid_type, ser_=srs)
                    vid_to_frames(vid_type=vid_type, series=srs, threshold=threshold)
