import cv2
import os
import pandas as pd
from src.helper.constant import get_images_path, get_marked_images_path, get_videos_path, get_spots_vel_csv

class GenerateVideo:
    def __init__(self, seq, num_frames, sgap=1, fps=1, show_tid=True, show_next=False, show_link=False):
        self.seq = seq
        self.images_path = get_images_path(seq)
        self.spots_velocity = get_spots_vel_csv(seq)
        self.marked_images_path = get_marked_images_path(seq)
        self.videos_path = get_videos_path(seq)
        self.sparse_gap = sgap
        self.fps = fps
        self.frames = num_frames // sgap
        self.show_tid = show_tid
        self.show_next = show_next
        self.show_link = show_link

    def get_pc_image_name(self, num):
        """Generate the file path for the original image based on frame number."""
        return os.path.join(self.images_path, f"pc_{num:04}.tif")

    def get_pc_marked_image_name(self, num):
        """Generate the file path for the marked image based on frame number."""
        return os.path.join(self.marked_images_path, f"pc_{num:04}.tif")

    def pc_make_video(self):
        """Create a video from the marked images."""
        video_filename = os.path.join(self.videos_path, f"{self.seq}_sgap{self.sparse_gap}_{self.fps}fps.mp4")
        cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_filename, cv2_fourcc, self.fps, (1224, 904))

        for num in range(1, self.frames + 1):
            marked_img_path = self.get_pc_marked_image_name((num - 1) * self.sparse_gap)
            image_name = os.path.basename(marked_img_path)
            print("Marking image:", image_name)
            marked_img = cv2.imread(marked_img_path)
            video.write(marked_img)

        video.release()
        print("Video saved at:", video_filename)
        return 0

    def pc_mark_spots(self):
        """Mark spots on images based on the velocity data and save marked images."""
        spots_df = pd.read_csv(self.spots_velocity)
        colors = {
            'green': (0, 255, 0),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (0, 0, 255),
            'blue': (255, 0, 0)
        }

        for num in range(1, self.frames + 1):
            img_path = self.get_pc_image_name((num - 1) * self.sparse_gap)
            marked_img_path = self.get_pc_marked_image_name((num - 1) * self.sparse_gap)
            image_name = os.path.basename(marked_img_path)
            print("Marking image:", image_name)
            img = cv2.imread(img_path)

            # Filter spots DataFrame for the current frame
            frame_spots = spots_df[spots_df['FRAME'] == (num - 1)]

            for _, row in frame_spots.iterrows():
                pos_x, pos_y = int(row['pos_x']), int(row['pos_y'])
                dx, dy = int(row[f'dt1_n0_dx']), int(row[f'dt1_n0_dy'])
                track_id = row['track_id_unique']

                # Draw spot
                cv2.circle(img, (pos_x, pos_y), 2, colors['blue'], 2)

                # Display track ID if required
                if self.show_tid:
                    cv2.putText(
                        img, f'TID: {int(track_id)}', 
                        (pos_x - 30, pos_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['white'], 2
                    )

                # Highlight the next spot if required
                if self.show_next:
                    cv2.circle(img, (pos_x + dx, pos_y + dy), 2, colors['red'], 2)

                # Draw link between spots if required
                if self.show_link:
                    cv2.arrowedLine(img, (pos_x, pos_y), (pos_x + dx, pos_y + dy), colors['black'], 1)

            # Save the marked image
            cv2.imwrite(marked_img_path, img)

        cv2.destroyAllWindows()
        print("Marked images saved in:", self.marked_images_path)
        return 0
