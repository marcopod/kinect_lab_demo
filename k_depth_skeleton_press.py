import cv2
import json
import pykinect_azure as pykinect
from datetime import datetime

filename = None  # Filename for saving data
recording = False


def save_skeleton_data(body_frame, filename):
    try:
        with open(filename, "r") as file:
            skeleton_data = json.load(file)
    except (IOError, ValueError):
        skeleton_data = []

    current_timestamp = datetime.now().isoformat()

    num_bodies = body_frame.get_num_bodies()
    for i in range(num_bodies):
        body_id = body_frame.get_body_id(i)
        skeleton = body_frame.get_body_skeleton(i)
        joints = []
        for joint in skeleton.joints:
            joints.append(
                {
                    "position": {
                        "x": joint.position.xyz.x,
                        "y": joint.position.xyz.y,
                        "z": joint.position.xyz.z,
                    },
                    "orientation": {
                        "x": joint.orientation.wxyz.x,
                        "y": joint.orientation.wxyz.y,
                        "z": joint.orientation.wxyz.z,
                        "w": joint.orientation.wxyz.w,
                    },
                    "confidence_level": joint.confidence_level,
                }
            )
        skeleton_data.append(
            {
                "timestamp": current_timestamp,
                "body_id": body_id,
                "joints": joints,
            }
        )

    with open(filename, "w") as file:
        json.dump(skeleton_data, file, indent=4)


if __name__ == "__main__":
    pykinect.initialize_libraries(track_body=True)
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    device = pykinect.start_device(config=device_config)
    bodyTracker = pykinect.start_body_tracker()

    cv2.namedWindow("Depth image with skeleton", cv2.WINDOW_NORMAL)

    while True:
        capture = device.update()
        body_frame = bodyTracker.update()
        ret_depth, depth_color_image = capture.get_colored_depth_image()
        ret_color, body_image_color = body_frame.get_segmentation_image()

        if not ret_depth or not ret_color:
            continue

        combined_image = cv2.addWeighted(
            depth_color_image, 0.6, body_image_color, 0.4, 0
        )
        combined_image = body_frame.draw_bodies(combined_image)
        cv2.imshow("Depth image with skeleton", combined_image)

        key = cv2.waitKey(1)
        if key == ord("s"):
            recording = not recording
            if recording:
                filename = f"./skeleton_data_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"
            else:
                filename = None  # Reset the filename after stopping
        elif key == ord("q"):
            break

        if recording and filename:
            save_skeleton_data(body_frame, filename)

    cv2.destroyAllWindows()
    device.stop()
