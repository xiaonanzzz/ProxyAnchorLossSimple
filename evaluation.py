from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.preprocessing import LabelEncoder



class ClusteringEvaluation():
    def __init__(self, ytrue, ycluster):
        self.true_label_encoder = LabelEncoder()
        self.cluster_label_encoder = LabelEncoder()

        ytrue_int = self.true_label_encoder.fit_transform(ytrue)
        ycluster_int = self.cluster_label_encoder.fit_transform(ycluster)

        self.homogeneity_score, self.completeness_score, self.v_measure_score = \
            homogeneity_completeness_v_measure(ytrue_int, ycluster_int)
