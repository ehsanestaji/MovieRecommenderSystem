# Node2Vec Movie Recommender System

## Overview
The Node2Vec Movie Recommender System is an innovative application leveraging the Node2Vec algorithm to provide movie recommendations based on user preferences. This system utilizes a graph-based approach to learn low-dimensional representations of movies, enabling intuitive and accurate recommendations.

## Features
- **Movie Recommendations:** Users can select a movie from the dropdown menu to receive recommendations for similar movies.
- **Custom Model Training:** The application allows users to train the Node2Vec model with custom parameters, ensuring flexibility and adaptability to diverse datasets.
- **Interactive Graph Visualization:** View an interactive graph showing the network of recommended movies in relation to the selected movie.
- **Node2Vec Model Insights:** The application displays the current Node2Vec model parameters used for generating recommendations.

## Installation

To run this application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/node2vec-movie-recommender.git
   ```
2. Navigate to the project directory:
   ```bash
   cd node2vec-movie-recommender
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run moviematcher.py
   ```

## Usage

Upon launching the app, select a movie title from the dropdown menu in the sidebar. Click 'Recommend' to view similar movies and an interactive graph illustrating their connections.

For custom model training, navigate to the 'Custom Node2Vec Training' tab and adjust the parameters as desired. Click 'Train and Save Model' to retrain the model with your specified parameters.

## Technologies Used
- Python
- Streamlit
- Node2Vec
- NetworkX
- Pyvis

## Contributions

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

The Node2Vec method used in this application is based on the paper "node2vec: Scalable Feature Learning for Networks" by Aditya Grover and Jure Leskovec. Access the paper [here](https://arxiv.org/abs/1607.00653).
