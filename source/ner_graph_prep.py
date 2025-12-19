import pandas as pd
import networkx as nx
import community.community_louvain as community  # pip install python-louvain
from pyvis.network import Network
import numpy as np
from collections import defaultdict, Counter
import pkgutil
from jinja2 import Template

# ---------------------------------------------------
# 1. Decode BIO NER tags into entity spans
# ---------------------------------------------------

def extract_entities(tokens, labels):
    """
    Robustly extracts entities from BIO-tagged tokens.
    Handles type mismatches (e.g. B-PER followed by I-LOC) correctly.
    """
    entities = []
    current_tokens = []
    current_type = None

    for tok, tag in zip(tokens, labels):
        if tag.startswith("B-"):
            # 1. New entity starts: Save previous if exists
            if current_tokens:
                entities.append((" ".join(current_tokens), current_type))
            
            # 2. Reset for new entity
            current_tokens = [tok]
            current_type = tag[2:]
            
        elif tag.startswith("I-"):
            # 3. Inside entity: Check if type matches
            tag_type = tag[2:]
            
            # Only append if we are in an entity AND types match
            if current_tokens and current_type == tag_type:
                current_tokens.append(tok)
            else:
                # Mismatch or Orphan I-tag: Treat as new B-tag (or O-tag depending on preference)
                # Here we treat it as starting a new entity of that type to be safe
                if current_tokens:
                    entities.append((" ".join(current_tokens), current_type))
                current_tokens = [tok]
                current_type = tag_type
                
        else:
            # 4. Outside (O): Save previous and reset
            if current_tokens:
                entities.append((" ".join(current_tokens), current_type))
            current_tokens = []
            current_type = None

    # Flush remaining
    if current_tokens:
        entities.append((" ".join(current_tokens), current_type))

    return entities

# ---------------------------------------------------
# 2. Extract DISH + LOCATION + CUISINE entities
# ---------------------------------------------------

# Adjust these to match your real labels:
# DISH_TAGS = {"food", "dish", "meal"}
LOC_TAGS = {"loc", "geo", "car", "gpe"}        
CUISINE_TAGS = {"grp"}  # grp might include ethnicities

def classify_entity(ent_type, location_included):
    et = ent_type.lower()
    # if et in DISH_TAGS:
    #     return "DISH"
    if location_included and et in LOC_TAGS:
        return "LOC"
    elif et in CUISINE_TAGS:
        return "CUISINE"
    else:
        return None  # ignore others
    
# -----------------------------------------------------------
# 3. SPARSIFY GRAPH (From Code 1)
# -----------------------------------------------------------
# This keeps only the top-K strongest connections per node to reduce clutter
def sparsify_graph(graph, k=5):
    new_edges = set()
    for n in graph.nodes():
        neighbors = graph[n].items()
        # Sort neighbors by edge weight
        sorted_neighbors = sorted(neighbors, key=lambda x: x[1]['weight'], reverse=True)
        # Keep top k
        for nbr, data in sorted_neighbors[:k]:
            # Store as tuple sorted alphabetically to avoid duplicates (u,v) vs (v,u)
            edge = tuple(sorted((n, nbr)))
            new_edges.add((edge[0], edge[1], data['weight']))
            
    G_clean = nx.Graph()
    G_clean.add_nodes_from(graph.nodes(data=True)) 
    
    for u, v, w in new_edges:
        G_clean.add_edge(u, v, weight=w)
        
    # Remove isolated nodes that lost all edges
    G_clean.remove_nodes_from(list(nx.isolates(G_clean)))
    return G_clean

    
# ---------------------------------------------------
# 4. Community Detection (Louvain)
# ---------------------------------------------------

def community_detection(G):
    try:
        partition = community.best_partition(G, weight="weight")
        return partition
    except ImportError:
        print("Install python-louvain: pip install python-louvain")
        partition = {node: 0 for node in G.nodes()}  # fallback
        return partition
    
# ---------------------------------------------------
# 5. Infer Names and Create DataFrame
# ---------------------------------------------------
    
def infer_group_name(nodes, graph, location_included):
    """
    Infers a name for a cluster based on the importance (weighted degree)
    of its constituent nodes.
    Priority: CUISINE > LOC > DISH
    """
    # dictionaries to hold total weight for each entity in the cluster
    cuisine_scores = defaultdict(int)
    dish_scores = defaultdict(int)
    if location_included:
        loc_scores = defaultdict(int)
    
    for node in nodes:
        # Get the type we assigned in Step 3
        if not graph.has_node(node): 
            continue
            
        attrs = graph.nodes[node]
        ent_type = attrs.get('type')
        
        # Calculate importance: The weighted degree of the node within the whole graph
        # (or you could restrict this to degree within the subgraph)
        score = graph.degree(node, weight='weight')
        
        if ent_type == "CUISINE":
            cuisine_scores[node] += score
        elif location_included: 
            if ent_type == "LOC":
                loc_scores[node] += score
        elif ent_type == "DISH":
            dish_scores[node] += score
            
    # Heuristic 1: Dominant Cuisine (Highest weighted cuisine node)
    if cuisine_scores:
        best_cuisine = max(cuisine_scores, key=cuisine_scores.get)
        return f"{best_cuisine.title()} Cuisine"
    
    # Heuristic 2: Dominant Location (Highest weighted location node)
    if location_included: 
        if loc_scores:
            best_loc = max(loc_scores, key=loc_scores.get)
            return f"{best_loc.title()} Location"
    
    # Heuristic 3: Top Dishes (Top 2 highest weighted dishes)
    if dish_scores:
        # Sort dishes by score descending
        sorted_dishes = sorted(dish_scores, key=dish_scores.get, reverse=True)
        top_two = [d.title() for d in sorted_dishes[:2]]
        return f"{' & '.join(top_two)} Cluster"
        
    return "General Cluster"
