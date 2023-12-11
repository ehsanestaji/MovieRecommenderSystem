import networkx as nx
from collections import defaultdict
from node2vec import Node2Vec
import streamlit as st
import pandas as pd
import pickle
import os
import streamlit.components.v1 as components
from pathlib import Path
import time
import json
from pyvis.network import Network

def bootstrap():
    bootstrap_css = """
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
    """
    st.markdown(bootstrap_css, unsafe_allow_html=True)

bootstrap()

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def load_data():
    ratings = pd.read_csv('data/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'unix_timestamp'])
    movies = pd.read_csv('data/u.item', sep='|', usecols=range(2), names=['movie_id', 'title'], encoding='latin-1')
    return ratings, movies

def filter_movies_by_model_vocabulary(movies, model):
    model_vocab = set(int(key) for key in model.wv.key_to_index)
    return movies[movies['movie_id'].isin(model_vocab)]

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

def create_movie_subgraph(model, movies, selected_movie_title, depth=2):
    G = nx.Graph()
    try:
        def add_nodes_and_edges(G, current_movie_id, current_depth):
            if current_depth > depth:
                return
            if current_movie_id not in model.wv.key_to_index:
                return
            current_movie_title = movies[movies.movie_id == int(current_movie_id)].title.values[0]
            if current_movie_title not in G:
                G.add_node(current_movie_title)
            similar_movies = model.wv.most_similar(current_movie_id, topn=5)
            for id, _ in similar_movies:
                similar_movie_title = movies[movies.movie_id == int(id)].title.values[0]
                G.add_node(similar_movie_title)
                G.add_edge(current_movie_title, similar_movie_title)
                add_nodes_and_edges(G, id, current_depth + 1)
        selected_movie_id = str(movies[movies.title == selected_movie_title].movie_id.values[0])
        add_nodes_and_edges(G, selected_movie_id, 0)
    except Exception as e:
        print(f"Error creating subgraph: {e}")
    return G

def filter_ratings(ratings):
    return ratings[ratings.rating >= 4]

def create_pairs(ratings):
    pairs = defaultdict(int)
    for group in ratings.groupby("user_id"):
        user_movies = list(group[1]["movie_id"])
        for i in range(len(user_movies)):
            for j in range(i+1, len(user_movies)):
                pairs[(user_movies[i], user_movies[j])] += 1
    return pairs

def build_graph(pairs):
    G = nx.Graph()
    for pair, score in pairs.items():
        if score >= 20:
            G.add_edge(pair[0], pair[1], weight=score)
    return G

def train_node2vec(G, dimensions, walk_length, num_walks, p, q):
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p, q=q, workers=4)
    return node2vec.fit(window=10, min_count=1, batch_words=4)

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def visualize_movie_subgraph(G, selected_movie_title):
    net = Network(height='600px', bgcolor='#FFFFFF', font_color='black')
    default_color = "#99CCFF"
    selected_movie_color = "#FF6666"
    for node in G.nodes():
        if node == selected_movie_title:
            G.nodes[node]['color'] = selected_movie_color
            G.nodes[node]['size'] = 20
        else:
            G.nodes[node]['color'] = default_color
    net.from_nx(G)
    net.repulsion(node_distance=600, central_gravity=0.05, spring_length=150, spring_strength=0.08, damping=0.85)
    path = Path('html_files')
    path.mkdir(exist_ok=True)
    file_path = path / 'movie_subgraph.html'
    net.save_graph(str(file_path))
    return file_path

def save_model_params(params, filename):
    with open(filename, 'w') as file:
        json.dump(params, file)

def load_model_params(filename):
    with open(filename, 'r') as file:
        return json.load(file)


def main():
    st.title("Movie Recommender System")
    st.subheader("Node2Vec (Random Walk Based Embedding)")


    # Credits
    st.markdown("---")
    st.markdown("Developed by Ehsan Estaji © 2023")

    # Initialize session state for tab control
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Movie Recommendations"


    model = load_model('node2vec_movie_recommender.pkl')

    # Load data
    ratings, movies = load_data()
    filtered_movies = filter_movies_by_model_vocabulary(movies, model)

    # Initialize the model variable
    model = None

    # Check if a trained model already exists
    model_exists = os.path.exists("node2vec_movie_recommender.pkl")
    if model_exists:
        model = load_model('node2vec_movie_recommender.pkl')

    # Tab layout
    tab1, tab2 = st.tabs(["Movie Recommendations", "Custom Node2Vec Training"])

    with tab1:
        if st.session_state.active_tab == "Movie Recommendations":
            # Existing code for movie recommendations
            # Initialize session state for the first-time graph load
            if 'graph_loaded' not in st.session_state:
                st.session_state.graph_loaded = False

            # Sidebar for movie selection and recommendations
            with st.sidebar:
                movie_titles = filtered_movies['title'].tolist()
                selected_movie = st.selectbox("Choose a movie title:", movie_titles, key="unique_select_movie")


                if st.button("Recommend"):
                    try:
                        movie_id = str(movies[movies.title == selected_movie].movie_id.values[0])
                        similar_movies = model.wv.most_similar(movie_id, topn=5)

                        expander = st.expander("Recommended Movies:", expanded=True)

                        for id, score in similar_movies:
                            with expander:
                                placeholder = st.empty()
                                movie_title = movies[movies.movie_id == int(id)].title.values[0]
                                time.sleep(0.2)  # Delay for the 'animation' effect
                                placeholder.markdown(f"- **{movie_title}** (Score: {score:.2f})")

                        # Create and display the graph
                        subgraph = create_movie_subgraph(model, movies, selected_movie)
                        file_path = visualize_movie_subgraph(subgraph, selected_movie)
                        HtmlFile = open(file_path, 'r', encoding='utf-8')
                        st.session_state.graph_html = HtmlFile.read()
                        st.session_state.graph_loaded = True
                    except IndexError:
                        st.error("Movie not found in dataset.")

            # Main area for the graph
            if st.session_state.graph_loaded:
                components.html(st.session_state.graph_html, height=600)


    #with tab2:
    with tab2:
            st.markdown("## Custom Node2Vec Model Training")

            dimensions = st.number_input('Dimensions', min_value=10, max_value=300, value=64, step=10)
            walk_length = st.number_input('Walk Length', min_value=10, max_value=100, value=20, step=5)
            num_walks = st.number_input('Number of Walks', min_value=10, max_value=500, value=200, step=10)
            p = st.number_input('Return Hyperparameter p', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            q = st.number_input('In-out Hyperparameter q', min_value=0.0, max_value=10.0, value=1.0, step=0.1)

            if st.button('Train and Save Model'):
                with st.spinner('Training Node2Vec model...'):
                    filtered_ratings = filter_ratings(ratings)
                    pairs = create_pairs(filtered_ratings)
                    G = build_graph(pairs)
                    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p,
                                        q=q, workers=4)
                    model = node2vec.fit(window=10, min_count=1, batch_words=4)
                    # Inside the 'Train and Save Model' button click event
                    params = {'dimensions': dimensions, 'walk_length': walk_length, 'num_walks': num_walks, 'p': p,
                              'q': q}
                    save_model_params(params, 'node2vec_params.json')

                    save_model(model, 'node2vec_movie_recommender.pkl')

                st.success("Node2Vec model trained and saved successfully!")
            # Short description of the method used

    st.header("Current Model Parameters")
    params = load_model_params('node2vec_params.json')

    col1, col2 = st.columns(2)
    with col1:
        st.text("Dimensions:")
        st.text("Walk Length:")
        st.text("Number of Walks:")
        st.text("Return:")
        st.text("In-Out:")

    with col2:
        st.text(f"{params.get('dimensions', 'N/A')}")
        st.text(f"{params.get('walk_length', 'N/A')}")
        st.text(f"{params.get('num_walks', 'N/A')}")
        st.text(f"{params.get('p', 'N/A'):.2f}")
        st.text(f"{params.get('q', 'N/A'):.2f}")

    st.markdown("---")

    st.markdown("""
    ## Node2Vec Method Citation

    The Node2Vec embedding method, used in this application, is based on the work by Aditya Grover and Jure Leskovec. The method is detailed in their paper titled "node2vec: Scalable Feature Learning for Networks." The paper was published in 2016 and is a significant contribution to the field of graph-based learning algorithms.

    You can access the paper here: [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653) (Aditya Grover and Jure Leskovec, 2016).
    """)


if __name__ == '__main__':
    main()