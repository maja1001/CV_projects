#!/bin/bash

# Überprüfen, ob "models.zip" bereits existiert
if [ -f "models.zip" ]; then
    echo "models.zip existiert bereits. Überspringe den Download."
else
    echo "Lade models.zip herunter..."
    wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
fi

# Überprüfen, ob der Ordner "models" bereits existiert
if [ -d "models" ]; then
    echo "Der Ordner 'models' existiert bereits. Überspringe das Entpacken."
else
    echo "Entpacke models.zip..."
    unzip models.zip
fi


# wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
# unzip models.zip
