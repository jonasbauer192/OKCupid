import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# --- GLOBAL VARIABLES ---
path = "D:\\OKCupid\\"
file = "profiles.csv"

# -- CLASS DEFINITIONS ---
class DataframeML():
    def __init__(self):
        self.df = pd.read_csv(path + file)
        self.dropColumns()
        self.dropRows()
        self.cleanDataFrame()

    def dropColumns(self):
        lst1 = ["essay" + str(i) for i in range(10)]
        lst2 = ["diet", "education", "ethnicity", "job", "last_online",
                "location", "offspring", "pets", "religion", "speaks"]
        self.df = self.df.drop(columns = lst1+lst2)

    def dropRows(self):
        self.df = self.df.dropna()

    def cleanDataFrame(self):
        self.transformType()
        self.modifyStatus()
        self.modifySmoking()
        self.modifyOrientation()
        self.modifySex()
        self.modifyDrinks()
        self.modifyDrugs()
        self.modifyBodyType()
        self.modifySign()

    def transformType(self):
        self.df = self.df.astype({"age": "float", "income": "float"})

    def modifyStatus(self):
        # remove the status "unknown"
        self.df["status"] = self.df["status"].replace(["unknown"], np.nan)
        self.df = self.df.dropna(subset=["status"])

        # status = 1 if "seeing someone" / "married"; status = 0 if "single" / "available"
        lambdaFunction = lambda col: int(0) if col == "available" or col == "single" else int(1)
        self.df["status"] = self.df["status"].apply(lambdaFunction)

    def modifySmoking(self):
        lambdaFunction = lambda col: int(0) if col == "no" or col == np.nan else int(1)
        self.df["smokes"] = self.df["smokes"].apply(lambdaFunction)

    def modifyOrientation(self):
        lambdaFunction = lambda col: int(1) if col == "straight" else int(0)
        self.df["orientation"] = self.df["orientation"].apply(lambdaFunction)

    def modifySex(self):
        lambdaFunction = lambda col: int(1) if col == "f" else int(0)
        self.df["sex"] = self.df["sex"].apply(lambdaFunction)

    def modifyDrinks(self):
        self.df = self.df.fillna(value={"drinks": "socially"})
        self.df["drinks"] = pd.Categorical(self.df["drinks"], ["not at all", "rarely", "socially", "often",
                                                               "very often", "desperately"], ordered=True)
        self.df["drinks"] = self.df["drinks"].cat.codes

    def modifyDrugs(self):
        self.df = self.df.fillna(value={"drugs": "never"})
        self.df["drugs"] = pd.Categorical(self.df["drugs"], ["never", "sometimes", "often"], ordered=True)
        self.df["drugs"] = self.df["drugs"].cat.codes

    def modifyBodyType(self):

        categories = {"lowest category":        ["skinny", "rather not say", "used up", "overweight"],
                      "lower middle category":  ["curvy", "a little extra", "full figured"],
                      "higher middle category": ["thin", "average"],
                      "highest category":       ["fit", "athletic", "jacked"]
                      }
        for category in categories.keys():
            lambdaFunction = lambda col: category if col in categories[category] else col
            self.df["body_type"] = self.df["body_type"].apply(lambdaFunction)

        # defining the order of different categories
        self.df["body_type"] = pd.Categorical( self.df["body_type"], ["lowest category", "lower middle category",
                                                "higher middle category", "highest category"], ordered=True)
        self.df["body_type"] = self.df["body_type"].cat.codes

    def modifySign(self):
        lambdaFunction = lambda col: word_tokenize(str(col))
        self.df["sign"] = self.df["sign"].apply(lambdaFunction)
        lambdaFunction = lambda col: col[0]
        self.df["sign"] = self.df["sign"].apply(lambdaFunction)

class ML():
    def __init__(self, df):
        self.sign = df[["sign"]]
        self.features = df.drop(columns=["sign"])
        self.splitData()
        self.kNearestNeighbours()
        self.supportVectorMachine()
        self.randomForest()

    def splitData(self):
        self.trainFeatures, self.testFeatures, self.trainSign, self.testSign = train_test_split(
            self.features, self.sign, train_size=0.8, test_size=0.2, random_state=1)

    def kNearestNeighbours(self):
        accuracies = []
        # find best k
        for k in range(1, 3):
            classifier = KNeighborsClassifier(n_neighbors=k)
            confusionMatrix, accuracy = self.classifierEvaluation(classifier)
            accuracies.append(accuracy)
            if accuracy >= max(accuracies):
                bestConfusionMatrix = confusionMatrix
        self.plotConfusionMatrix(bestConfusionMatrix, "KNearestNeighbours")

    def supportVectorMachine(self):
        classifier = SVC(gamma=0.5, C=0.75)
        confusionMatrix = self.classifierEvaluation(classifier)[0]
        self.plotConfusionMatrix(confusionMatrix, "SupportVectorMachine")

    def randomForest(self):
        classifier = RandomForestClassifier(n_estimators=2, random_state = 0)
        confusionMatrix = self.classifierEvaluation(classifier)[0]
        self.plotConfusionMatrix(confusionMatrix, "RandomForest")

    def classifierEvaluation(self, classifier):
        classifier.fit(self.trainFeatures, self.trainSign)
        accuracy = classifier.score(self.testFeatures, self.testSign)
        predictions = classifier.predict(self.testFeatures)
        confusionMatrix = confusion_matrix(self.testSign, predictions)
        return confusionMatrix, accuracy

    def plotConfusionMatrix(self, confusionMatrix, figname):
        plt.clf()
        ax = sns.heatmap(confusionMatrix, annot=True, cmap='Blues')
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted sign")
        ax.set_ylabel("Actual sign")
        labels=list(self.sign["sign"].unique())
        labels.sort()
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        plt.savefig(figname + ".png")

if __name__ == '__main__':
    dfML = DataframeML()
    ml = ML(dfML.df)

