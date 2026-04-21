
---
title: World Happiness Analytics Engine
emoji: 🌍
colorFrom: pink
colorTo: blue
sdk: gradio
app_file: app.py
python_version: 3.12
---

# World Happiness Analytics Engine

This is an interactive Gradio application that visualizes the World Happiness Report data from 2015 to 2019. It allows users to explore happiness scores globally, analyze correlations between happiness and various factors, and observe historical trends.

## Features

*   **Globe View:** Interactive choropleth map showing happiness scores by country for a selected year.
*   **Correlations:** Scatter plots to analyze the relationship between happiness scores and different socio-economic factors, with dynamic correlation coefficients.
*   **Trajectories:** Time-series line charts to track happiness trends for the world, continents, regions, and individual countries.

## Setup and Run Locally

1.  **Clone the repository:**
    ```bash
    git clone <your-huggingface-space-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a `data` folder and place CSVs inside:**
    Ensure you have the `2015.csv`, `2016.csv`, `2017.csv`, `2018.csv`, and `2019.csv` files inside a subfolder named `data` in the root directory of the application.

    Your directory structure should look like this:
    ```
    .
    ├── app.py
    ├── requirements.txt
    ├── README.md
    └── data/
        ├── 2015.csv
        ├── 2016.csv
        ├── 2017.csv
        ├── 2018.csv
        └── 2019.csv
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python app.py
    ```

    The application will launch in your browser, or provide a local URL.

## Data Source

The data used in this application is derived from the World Happiness Report datasets for the years 2015-2019.

## Deployment on Hugging Face Spaces

This application is designed to be easily deployable on Hugging Face Spaces. Simply create a new Space, upload all files (`app.py`, `requirements.txt`, `README.md`, and the `data` folder with CSVs), and select `Gradio` as the SDK.

