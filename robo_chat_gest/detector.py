#!/usr/bin/python3

import cv2
import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from .gesture_recognizer import HandGestRecognizer
from .utils import draw_bones, draw_joints


class Detector(Node):
    def __init__(self):
        # Initiate the Node class's constructor and give it a name
        super().__init__('detector')

        self.declare_parameter("config_path", "")

        # Create the subscriber. This subscriber will receive an Image
        # from the video_frames topic. The queue size is 10 messages.
        self.subscription = self.create_subscription(
            Image,
            'image',
            self.listener_callback,
            10)

        # Create the publisher. This publisher will publish an Image
        # to the video_frames topic. The queue size is 10 messages.
        self.img_publisher_ = self.create_publisher(Image, '/robo_chat_gest/result', 10)
        self.msg_publisher_ = self.create_publisher(Int32, '/robo_chat_gest/detected_id', 10)

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

        self.obj_classes = {
            0: "call",
            1: "dislike",
            2: "fist",
            3: "four",
            4: "like",
            5: "mute",
            6: "ok",
            7: "one",
            8: "palm",
            9: "peace",
            10: "peace_inverted",
            11: "rock",
            12: "stop",
            13: "stop_inverted",
            14: "three",
            15: "three2",
            16: "two_up",
            17: "two_up_inverted",
            18: "no_gesture",
        }

        config_path = self.get_parameter("config_path").value
        with open(config_path, "r") as stream:
            try:
                data_cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                assert False, exc
        self.gest_recog = HandGestRecognizer(**data_cfg)

        self.int32_msg = Int32()

    def listener_callback(self, data):
        # Convert ROS Image message to OpenCV image
        frame = self.br.imgmsg_to_cv2(data)

        token, landmarks, box = self.gest_recog.inference(frame)

        if token != -1:
            frame = draw_bones(frame, landmarks)
            frame = draw_joints(frame, landmarks)
            frame = cv2.rectangle(
                frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            frame = cv2.putText(
                frame,
                "Prediction: {}".format(self.obj_classes[token]),
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            print("Hand gestures detected: [{}]".format(self.obj_classes[token]))
        else:
            print("No gestures detected.")

        self.img_publisher_.publish(self.br.cv2_to_imgmsg(frame))
        self.int32_msg.data = int(token)
        self.msg_publisher_.publish(self.int32_msg)


def main(args=None):

    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    image_subscriber = Detector()

    # Spin the node so the callback function is called.
    rclpy.spin(image_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    image_subscriber.destroy_node()

    # Shutdown the ROS client library for Python
    rclpy.shutdown()


if __name__ == "__main__":
    main()
