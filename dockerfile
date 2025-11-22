#Image de python
FROM python:3.12.3
  
#Dossier de travail
WORKDIR /

#Copie des dépendances
COPY requirements.txt .

#Commandes à executer
RUN apt-get update && apt-get clean

#Ici intéressant, l'option --no-cache-dir permet de ne pas avoir de cache restant
#Option v pour la verbosité (on voit ce qui se passe)
RUN pip3 install --no-cache-dir -r requirements.txt -v

#Copie des fichiers restants
COPY . .

#Definition du port
EXPOSE 8080

#Commande pour lancer l'application
CMD ["python3", "run.py"]

