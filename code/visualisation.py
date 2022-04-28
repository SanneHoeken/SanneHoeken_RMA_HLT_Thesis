import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
import json

def main(embeddings_path1, embeddings_path2, contexts_path1, contexts_path2, targets_path, label_names, output_dir):

    # Load usage representations
    embeddings1 = np.load(embeddings_path1)
    embeddings2 = np.load(embeddings_path2)

    # Get contexts
    with open(contexts_path1, "r") as infile:
        texts1 = json.load(infile)
    with open(contexts_path2, "r") as infile:
        texts2 = json.load(infile)

    # Load targets
    with open(targets_path, "r") as infile:
        targets = [line.strip('\n') for line in infile.readlines()]

    # Generate plots for every target
    for target in targets:

        # Usage representations of target
        target_embeddings1 = embeddings1[target]
        target_embeddings2 = embeddings2[target]
        target_embeddings = np.concatenate((target_embeddings1, target_embeddings2), axis=0)

        # Contexts of target
        text = texts1[target] + texts2[target]

        # Initialize labels
        labels1 = [label_names[0] for i in range(len(target_embeddings1))]
        labels2 = [label_names[1] for i in range(len(target_embeddings2))]
        labels = labels1 + labels2

        # Create tsne-plot
        tsne = TSNE(verbose=1, random_state=123)
        embedded_space = tsne.fit_transform(target_embeddings)

        df = pd.DataFrame()
        df["comp-1"] = embedded_space[:,0]
        df["comp-2"] = embedded_space[:,1]
        df["text"] = text
        df["community"] = labels

        fig = px.scatter(df, x="comp-1", y="comp-2", color="community", labels=None,
                        hover_data={'community': True, 'text': True, 'comp-1':False, 'comp-2': False}, 
                        title=f"T-SNE projection of word representations of '{target}'")
        
        fig.update_layout(yaxis_title=None, xaxis_title=None)
        fig.write_html(f'{output_dir}/tsne_plot_{target}.html')
        fig.write_image(f'{output_dir}/tsne_plot_{target}.png')
    
    
if __name__ == '__main__':
    
    embeddings_path1 = '../output/embeddings/bertje-ft-all_embeddings_FD1.npz'
    embeddings_path2 = '../output/embeddings/bertje-ft-all_embeddings_PS.npz'
    contexts_path1 = '../output/contexts/contexts_FD1.json'
    contexts_path2 = '../output/contexts/contexts_PS.json'
    targets_path = '../data/targets.txt'
    labels =['Forum_Democratie1', 'Poldersocialisme']
    output_dir = '../output/visualisations/FD1-ft-all_PS-ft-all'

    main(embeddings_path1, embeddings_path2, contexts_path1, contexts_path2, targets_path, labels, output_dir)