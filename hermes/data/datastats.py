from hermes.data.dataloader import DataLoader
import pandas as pd

class DataStats:
    @staticmethod
    def classifierStats(envName:str):
        dl = DataLoader(envName=envName)
        classes = dl.getClasses()

        data = {"classe":[],"nb":[]}
        for c in classes:
            nb = len(dl.getData(c)["classifier"])
            data["classe"].append(c)
            data["nb"].append(nb)

        mx = max(data["nb"])
        mn = min(data["nb"])
        variation = (mx-mn)/mx

        return  pd.DataFrame(data, columns=["classe","nb"]), variation
    
    @staticmethod
    def NerStats(envName:str):
        #Repartition des classes
        dl = DataLoader(envName=envName)
        classes = dl.getClasses()

        data = {"classe":[],"nb":[]}
        for c in classes:
            nb = len(dl.getData(c)["ner"])
            data["classe"].append(c)
            data["nb"].append(nb)

        mx = max(data["nb"])
        mn = min(data["nb"])
        variation = (mx-mn)/mx

        labels = {}
        nbEntites = {}
        for c in classes:
            d = dl.getData(c)["ner"]
            for l in d:
                for e in l[1]["entities"]:
                    if e[2] not in labels:
                        labels[e[2]] = 1
                    else:
                        labels[e[2]] = labels[e[2]]+1
                if f"e{len(l[1]["entities"])}" in nbEntites:
                    nbEntites[f"e{len(l[1]["entities"])}"] = nbEntites[f"e{len(l[1]["entities"])}"]+1
                else:
                    nbEntites[f"e{len(l[1]["entities"])}"] = 1

        dataLabels = {"label":[], "nb":[]}
        for l in labels:
            dataLabels["label"].append(l)
            dataLabels["nb"].append(labels[l])

        dataNbLabels = {"croise": [], "nb":[]}
        for e in nbEntites:
            dataNbLabels["croise"].append(e)
            dataNbLabels["nb"].append(nbEntites[e])

        return  pd.DataFrame(data, columns=["classe","nb"]), variation, pd.DataFrame(dataLabels, columns=["label","nb"]), pd.DataFrame(dataNbLabels, columns=["croise","nb"])
        
