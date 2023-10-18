# Plan Trenowania


1. Na samym Początku odpal pierwszy trening
```bash
python train_model.py \
--epochs 1 \
--n_batch 5000
```
To tylko raz odpal, powinno trwać ~40 min

2. Następnie wygeneruj plik midi - pamiętaj, żeby nadawać w nazwie im kolejne numery
```bash

```
3. Przekształć go w plik mp3
```bash

```

4. Przenieś wygenerowane obrazki z pliku images do images/run_1 
5. Kontynuuj trenowanie
```bash
python train_model.py \
--epochs 1 \
--n_batch 5000 \
--load_from_path 'gan_save_2023_10_18_16_43'
```
zmień datę i godzinę w pliku

6. Powtarzaj kroki 2-5 za każdym razem:
- odpal sieć ładując kolejne modele, jeżeli 5000 się policzy szybko to odpal na noc 8 razy tyle. 
- zapisując obrazki do kolejnych folderów run_1, run_2, itd
- tworząc i zapisując plik midi i mp3 z kolejnymi nazwami name=1, name=2 itd.
- PAMIĘTAJ, żeby zapisywać wszystkie ustawienia ile epok kiedy puściłaś, który to numer odpalenia itd.

# Installing Miniconda
Miniconda is a free, open-source package management system and environment management system. It allows you to create and manage virtual environments for different projects and install packages in those environments.

## Mac OS
If you're using a Mac, you can use Homebrew to install Miniconda. Follow the steps below:

1. Open the Terminal app on your Mac.
2. Install Homebrew by running the following command:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Once Homebrew is installed, run the following command to install Miniconda:

   ```bash
   brew install --cask miniconda
   ```
   This will install the latest version of Miniconda.

## Windows and Linux (using Miniconda)
If you're using Windows or Linux, you can download and install Miniconda directly from the Miniconda website. Follow the steps below:

1. Go to the [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html) and download the appropriate installer for your system.

2. Run the installer and follow the instructions on the screen to install Miniconda.
   
   Note: On Linux, you may need to make the installer executable by running the following command:
   ```bash
   chmod +x <installer-filename>
   ```
   Replace `<installer-filename>` with the name of the installer file you downloaded.

## Create Conda Environment
Once Conda is installed, you can create a new environment using the `requirements.yaml` file. This file specifies the packages required for the project. Follow the steps below to create a Conda environment:

1. Open a terminal or command prompt.
2. Navigate to the directory where `requirements.yml` is located.
3. Run the following command to create the environment:
   ```bash
   conda env create --file requirements.yml
   ```
   This will create a new environment with the specified packages installed.

# Download the MAESTRO Data and Generate Training Dataset
To download the MAESTRO dataset, you can use the `download_data.py` Python script. Follow the steps below to download the dataset:

1. Open a terminal or command prompt.

2. Activate the Conda environment created in the previous step:

    ```bash
    conda activate <environment-name>
    ```
    Replace `<environment-name>` with the name of the environment you created.

3. Navigate to the directory where download_data.py is located.

4. Run the following command to download the MAESTRO data:

    ```bash
    python download_data.py
    ```
    This will download the MAESTRO dataset from the web.
5. Run the following command to generate the training dataset: 
    
   ```bash
   python generate_dataset.py
   ```
