from head_pose_geometry import HeadPoseGeometry
from head_pose_model import HeadPoseModel
from head_pose_tracker import HeadPoseTracker


class MultiHeadPoseEstimator:
    """
    Head pose estimation using all 3 implemented methods.
    """
    def __init__(self, detector, predictor):
        self.__geom = HeadPoseGeometry(detector, predictor)
        self.__track = HeadPoseTracker(detector, predictor)
        self.__model = HeadPoseModel(detector, predictor)
        self.landmarks = []

    def get_name(self):
        return "multi"

    def pose_for_image(self, image):
        geom_res = self.__geom.pose_for_image(image)
        track_res = self.__track.pose_for_image(image)
        model_res = self.__model.pose_for_image(image)
        self.landmarks = self.__geom.landmarks
        return geom_res[0], self.__format_group(geom_res[1], track_res[1], model_res[1]), \
               self.__format_group(geom_res[2], track_res[2], model_res[2]), \
               self.__format_group(geom_res[3], track_res[3], model_res[3])

    def __format_group(self, arg1, arg2, arg3):
        return 'geom: {}, track: {}, model: {}'.format(arg1, arg2, arg3)
