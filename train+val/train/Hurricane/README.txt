Team No. - H2 (city-hurricane)
Team Members
    ANIKET UPADHYAY - 2017HD031005P
    Mehul Gera - 2017HD120579P
    Shivam Goyal - 2018HS030167P
    TARESH BANSAL - 2020B2A71945P

In this dataset, we have included a variety of images capturing after effects of the damage caused by hurricanes, for e.g. images containing hurricane damage in city in dusk dim light. This would help us to make the model more robust to be able to detect disaster even in poor lighting conditions. We purposefully numbered images taken from the same video together, so that incase we need to do something with similar images, we can do that simply by considering the name given to the images. Incase the requirement of the deep learning task is to randomize the order in which images are fed into the model, then that can easily be done in a data-preprocessing step.

Process of Extracting Frames from the YouTube video - 
    Firstly we downloaded the YouTube video using yt-dlp command line tool (terminal command - yt-dlp -f best -a file_with_links.txt). Then we preprocessed the video using Adobe Premiere Pro to crop out names of News Channels, and other watermarks, or text displayed on the video, to resize the video, and to merge all the videos that we have downloaded into one single video. Then we used two Python scripts. N_Frame.py was used to extract every 20th frame from the input video, and script.py to resize already preprocessed square images to the desired shape of 256x256, and also to rename the files according to the requirement. We have pasted the code for both these scripts in this ReadMe itself at the end.


We referred to a collection of YouTube videos that contained drone images. The list of the disasters searched for -
    Hurricane Idalia
    Hurricane Ian
    Hurricane Michael
    Hurricane Maria
    Hurricane Dominica

The following is the list of YouTube videos that we used to extract frames - 

    1. Drone video of Treasure Island after Hurricane Idalia
    https://www.youtube.com/watch?v=TlKOAhQmEh0&ab_channel=ABCActionNews

    2. Hurricane Idalia: Drone footage reveals devastating Florida flooding
    https://www.youtube.com/watch?v=Hwe7f3hEa-U&ab_channel=TheIndependent

    3. VIDEO: Drone captures Idalia impact on Treasure Island
    https://www.youtube.com/watch?v=Y3vf4o49CCs&ab_channel=ABCActionNews

    4. St James City Pine Island, FL Hurricane Ian drone damage covering most of City in 4k
    https://www.youtube.com/watch?v=lh-JMwD_Ccs

    5. Hurricane IAN's unbelievable never before seen destruction.
    https://www.youtube.com/watch?v=lhG9lkdVh3I

    6. Hurricane Ian- Ft. Myers Beach Florida- Tsunami like power of storm surge from drone - 4k
    https://www.youtube.com/watch?v=OHRuCKbKPek

    7. Fort Myers Beach Hurricane Ian Aerial Drone Footage
    https://www.youtube.com/watch?v=XnZmjdB8raM

    8. Coast Guard Drone Post Hurricane Ian
    https://www.youtube.com/watch?v=ouqYU9wUhaE

    9. Port Charlotte heavily damaged after Hurricane Ian
    https://www.youtube.com/watch?v=0KAKWU232vU

    10. DRONE VIDEO: Hurricane Michael devastates Panama City, Florida
        https://www.youtube.com/watch?v=13cngHwY1eo&ab_channel=CBS17

    11. Before & after: Devastating drone footage of hurricane hit Dominica | UNICEF
        https://www.youtube.com/watch?v=AlzpXQY6JOM&ab_channel=UNICEF

    12. Drone video of Treasure Island after Hurricane Idalia
        https://www.youtube.com/watch?v=TlKOAhQmEh0&ab_channel=ABCActionNews

    13. Drone video captures Tampa flooding from Hurricane Idalia
        https://www.youtube.com/watch?v=g4T-MusVmG8&ab_channel=NBCNews

    14. Drone footage captures Hurricane Ian's destruction in Fort Myers Beach | USA TODAY
        https://www.youtube.com/watch?v=ZFu8qwwU4V8&ab_channel=USATODAY

    15. DRONE VIDEO: Hurricane Michael devastates Panama City, Florida
        https://www.youtube.com/watch?v=13cngHwY1eo&ab_channel=CBS17

    16. Extreme 4K Video of Category 5 Hurricane Michael
        https://www.youtube.com/watch?v=wSXvcveNSTQ

    17. Hurricane Idalia slams Florida's Big Bend
        https://www.youtube.com/watch?v=4tiayyaypRM





N_Frame.py code
    import cv2

    def extract_frames(video_path, n):
        video = cv2.VideoCapture(video_path)
        frame_number = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            if frame_number % n == 0:
                cv2.imwrite('H2_{}.jpg'.format(frame_number), frame)
            frame_number += 1
        video.release()
        cv2.destroyAllWindows()

    # Usage
    extract_frames('/Users/tareshbansal/Desktop/ACADS/DL/Project/Dataset/5_frame/DL_2_25fps.mp4', n=20)


    ## N means the nth frame you want to extract

script.py code
    from PIL import Image
    import os

    def crop_and_resize(input_folder, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        files = os.listdir(input_folder)

        files = [file_name for file_name in files if file_name.startswith("H2_") and file_name.endswith(".jpg")]

        files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

        for idx, file_name in enumerate(files, start=1):
            input_path = os.path.join(input_folder, file_name)

            original_image = Image.open(input_path)

            cropped_image = original_image.crop((187, 0, 667, 480)) 

            resized_image = cropped_image.resize((256, 256), Image.Resampling.LANCZOS)

            new_name = f"H2_{idx}.jpg"

            output_path = os.path.join(output_folder, new_name)

            resized_image.save(output_path, quality=95)

    if __name__ == "__main__":
        input_folder = "./Final"
        output_folder = "./FinalImageDataset_TeamH2"

        crop_and_resize(input_folder, output_folder)
