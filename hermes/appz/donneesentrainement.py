from hermes.data.dataloader import DataLoader
from flask import render_template
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Backend non-interactif pour serveur
import base64
import io

class DonneesEntrainement:
    def __init__(self, envName:str):
        self.envName = envName
        self.dl = DataLoader(envName=envName)
        self.classes = self.dl.getClasses()
        
    def render(self):
        graph1, var1, nbClasses1, count1 = self.__graphClassificationClasses()
        graph2, var2, nbClasses2, count2 = self.__graphNerClasses()
        graph3, graph4 = self.__graphNerLabels()
        return render_template('donneesentrainement.html', graph1=graph1, var1=var1, nbClasses1=nbClasses1, count1=count1, graph2=graph2, var2=var2, nbClasses2=nbClasses2, count2=count2, graph3=graph3, graph4=graph4)



    def __graphClassificationClasses(self):
        data = {"classe":[],"nb":[]}
        total = 0
        for c in self.classes:
            nb = len(self.dl.getData(c)["classifier"])
            data["classe"].append(c)
            data["nb"].append(nb)
            total = total+nb

        mx = max(data["nb"])
        mn = min(data["nb"])
        variation = (mx-mn)/mx

        # Créer le graphique répartition classes classification
        plt.figure(figsize=(10, 6))
        plt.bar(data["classe"], data["nb"])
        plt.xlabel('Commande')
        plt.ylabel('Nombre')
        plt.title("Répartition des données d'exemple par commande")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Sauvegarder le graphique en mémoire
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        
        # Encoder en base64 pour l'HTML
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()  # Libérer la mémoire

        return image_base64, variation, len(self.classes), total
    
    def __graphNerClasses(self):
        data = {"classe":[],"nb":[]}
        total = 0
        for c in self.classes:
            nb = len(self.dl.getData(c)["ner"])
            data["classe"].append(c)
            data["nb"].append(nb)
            total = total+nb

        mx = max(data["nb"])
        mn = min(data["nb"])
        variation = (mx-mn)/mx

        plt.figure(figsize=(10, 6))
        plt.bar(data["classe"], data["nb"])
        plt.xlabel('Commande')
        plt.ylabel('Nombre')
        plt.title("Répartition des données d'exemple par commande")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Sauvegarder le graphique en mémoire
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        
        # Encoder en base64 pour l'HTML
        image_base64Ner = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()  # Libérer la mémoire

        return image_base64Ner, variation, len(self.classes), total
    
    def __graphNerLabels(self):
        labels = {}
        nbEntites = {}
        for c in self.classes:
            d = self.dl.getData(c)["ner"]
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

        #Créer le graphique répartition labels NER
        plt.figure(figsize=(10, 6))
        plt.bar(dataLabels["label"], dataLabels["nb"])
        plt.xlabel('Label')
        plt.ylabel('Nombre')
        plt.title("Répartition des données d'exemple par label")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Sauvegarder le graphique en mémoire
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        
        # Encoder en base64 pour l'HTML
        image_base64Labels = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()  # Libérer la mémoire

        # Créer le graphique répartition nb labels NER
        plt.figure(figsize=(10, 6))
        plt.bar(dataNbLabels["croise"], dataNbLabels["nb"])
        plt.xlabel('Nombre de labels')
        plt.ylabel('Nombre')
        plt.title("Répartition des données d'exemple par nombre de labels associés")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Sauvegarder le graphique en mémoire
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        
        # Encoder en base64 pour l'HTML
        image_base64NbLabels = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()  # Libérer la mémoire

        return image_base64Labels, image_base64NbLabels