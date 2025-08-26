from mlxtend.feature_selection import SequentialFeatureSelector as sfs
import os
from sklearn.model_selection import  StratifiedKFold, RepeatedStratifiedKFold
from sklearn.feature_selection import mutual_info_classif
import json
import warnings

class featureSelectionHandler: 
    def __init__(self, pathConfig, trainConfig):
        self.pathConfig = pathConfig
        self.trainConfig = trainConfig
    
    def readParamsFromConfig(self):
        self.nFeatures = int(self.trainConfig['FeatureSelection']['k_features'])
        self.floating = self.trainConfig.getboolean('FeatureSelection','floating')
        self.scoring = self.trainConfig['FeatureSelection']['scoring']
        self.nJobs = int(self.trainConfig['FeatureSelection']['n_jobs'])

    def forwardSelection(self, X, y, mlModel, cv, imputationSetup, ID):
        self.ID = ID

        self.readParamsFromConfig()

        X = imputationSetup.imputationCV(cv, X, y)

        sfs1 = sfs(mlModel.mlModel,
        k_features = self.nFeatures,
        forward = True,
        floating = self.floating,
        verbose = 2,
        scoring = self.scoring,
        cv = cv,
        n_jobs = self.nJobs
        )
        warnings.filterwarnings("ignore",message=" Missing values detected.",category=UserWarning)
        sfs1 = sfs1.fit(X, y)
        mlModel.features = list(sfs1.k_feature_names_)
        self.saveMetricDict(sfs1)

        return mlModel.features

    def saveMetricDict(self, sfsObject):
        metricDict = sfsObject.get_metric_dict()
        for i in range(len(metricDict)):
            metricDict[i+1]['cv_scores'] = metricDict[i+1]['cv_scores'].tolist()
        filePath = os.path.join(self.pathConfig['Paths']['featureSelectionResultsPath'],f'{self.ID}_FeatureSelection_{self.scoring}.json')
        with open(filePath, 'w') as file:
            json.dump(metricDict, file)

