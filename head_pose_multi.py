from head_pose_geometry import HeadPoseGeometry
from head_pose_model import HeadPoseModel
from head_pose_tracker import HeadPoseTracker


class MultiHeadPoseEstimator:
    """
    Head pose estimation using all 3 implemented methods.
    """
    def __init__(self):
        self.__geom = HeadPoseGeometry()
        self.__track = HeadPoseTracker()
        self.__model = HeadPoseModel()
        self.landmarks = []

    def get_name(self):
        return "multi"

    def pose_for_landmarks(self, image, landmarks):
        geom_res = self.__geom.pose_for_landmarks(image, landmarks)
        track_res = self.__track.pose_for_landmarks(image, landmarks)
        model_res = self.__model.pose_for_landmarks(image, landmarks)
        self.landmarks = self.__geom.landmarks
        return self.__format_group(geom_res[0], track_res[0], model_res[0]), \
               self.__format_group(geom_res[1], track_res[1], model_res[1]), \
               self.__format_group(geom_res[2], track_res[2], model_res[2])

    def __format_group(self, arg1, arg2, arg3):
        return 'geom: {}, track: {}, model: {}'.format(arg1, arg2, arg3)
