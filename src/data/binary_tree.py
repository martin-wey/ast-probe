from .code2ast import get_id, get_root_ast
import networkx as nx
import numpy as np

SEPARATOR = '<sep>'


def ast2binary(G):
    # fussion non-terminals with on non-terminal child
    def ast2binary_aux(current_node_G, G, new_G, parent_in_new_G):
        out_edges = list(G.out_edges(current_node_G))
        if len(out_edges) == 2:
            for _, m in out_edges:
                id_m_new = get_id(new_G)
                new_G.add_node(id_m_new, **G.nodes[m])
                new_G.add_edge(parent_in_new_G, id_m_new)
                ast2binary_aux(m, G, new_G, id_m_new)
        elif len(out_edges) == 1:
            m = out_edges[0][1]
            if not G.nodes[m]['is_terminal']:
                new_G.nodes[parent_in_new_G]['type'] = new_G.nodes[parent_in_new_G]['type'] + SEPARATOR + G.nodes[m][
                    'type']
                ast2binary_aux(m, G, new_G, parent_in_new_G)
            else:
                # todo: check this, unary things
                new_G.nodes[parent_in_new_G]['is_terminal'] = True
                new_G.nodes[parent_in_new_G]['unary'] = new_G.nodes[parent_in_new_G]['type']
        elif len(out_edges) > 2:
            out_nodes = [m for _, m in out_edges]
            out_nodes.sort(key=lambda m: G.nodes[m]['start'])
            id_m_new = get_id(new_G)
            new_G.add_node(id_m_new, **G.nodes[out_nodes[0]])
            new_G.add_edge(parent_in_new_G, id_m_new)
            ast2binary_aux(out_nodes[0], G, new_G, id_m_new)
            new_empty_id = get_id(new_G)
            new_G.add_node(new_empty_id, type='<empty>')
            new_G.add_edge(parent_in_new_G, new_empty_id)
            for j, out_node in enumerate(out_nodes[1:]):
                if len(list(new_G.out_edges(new_empty_id))) == 1 and len(out_nodes[1:]) - j > 1:
                    new_empty_id_new = get_id(new_G)
                    new_G.add_node(new_empty_id_new, type='<empty>')
                    new_G.add_edge(new_empty_id, new_empty_id_new)
                    new_empty_id = new_empty_id_new
                id_m_new = get_id(new_G)
                new_G.add_node(id_m_new, **G.nodes[out_node])
                new_G.add_edge(new_empty_id, id_m_new)
                ast2binary_aux(out_node, G, new_G, id_m_new)

    new_G = nx.DiGraph()
    root_G = get_root_ast(G)
    new_G.add_node(0, **G.nodes[root_G])
    ast2binary_aux(root_G, G, new_G, 0)
    return new_G


def get_leaves(tree, node):
    if tree.out_degree(node) == 0:
        return [node]
    else:
        result = []
        for _, m in tree.out_edges(node):
            result += get_leaves(tree, m)
        return result


def get_most_left(tree, nodes):
    nodes_sort = sorted(nodes, key=lambda n: tree.nodes[n]['start'])
    return tree.nodes[nodes_sort[0]]['start']


def get_left_right_child(tree, node):
    child_1 = list(tree.out_edges(node))[0][1]
    child_2 = list(tree.out_edges(node))[1][1]
    child_1_leaves = get_leaves(tree, child_1)
    child_2_leaves = get_leaves(tree, child_2)
    child_1_left_most = get_most_left(tree, child_1_leaves)
    child_2_left_most = get_most_left(tree, child_2_leaves)
    if child_1_left_most < child_2_left_most:
        return child_1, child_2
    else:
        return child_2, child_1


def tree_to_distance(tree, node):
    if tree.out_degree(node) == 0:
        d = []
        c = []
        h = 0
        if 'unary' in tree.nodes[node]:
            u = [tree.nodes[node]['unary']]
        else:
            u = ['<empty>']
    else:
        left_child, right_child = get_left_right_child(tree, node)
        d_l, c_l, h_l, u_l = tree_to_distance(tree, left_child)
        d_r, c_r, h_r, u_r = tree_to_distance(tree, right_child)
        h = max(h_r, h_l) + 1
        d = d_l + [h] + d_r
        c = c_l + [tree.nodes[node]['type']] + c_r
        u = u_l + u_r
    return d, c, h, u


def distance_to_tree(d, c, u, tokens):
    def distance_to_tree_aux(G, d, c, u, father, tokens, start_token):
        if d == []:
            new_id = get_id(G)
            G.add_node(new_id, type=tokens[0], start=start_token, unary=u[0])
            G.add_edge(father, new_id)
        else:
            i = np.argmax(d)
            new_id = get_id(G)
            G.add_node(new_id, type=c[i])
            if father != None:
                G.add_edge(father, new_id)
            distance_to_tree_aux(G, d[0:i], c[0:i], u[0:i + 1], new_id, tokens[0:i + 1], start_token)
            distance_to_tree_aux(G, d[i + 1:], c[i + 1:], u[i + 1:], new_id, tokens[i + 1:], start_token + i + 1)

    G = nx.DiGraph()
    distance_to_tree_aux(G, d, c, u, None, tokens, 0)
    return G


def remove_empty_nodes(G):
    g = G.copy()
    while len([n for n in g if g.nodes[n]['type'] == '<empty>']) != 0:
        g0 = g.copy()
        n = [n for n in g if g.nodes[n]['type'] == '<empty>'][0]
        if g.in_degree(n) == 0:
            g0.nodes[n]['type'] = 'bad_root'
            g = g0
            continue
        edges_in = list(g.in_edges(n))
        edges_out = list(g.out_edges(n))
        if len(edges_in) != 0:
            u, _ = edges_in[0]
            for _, v in edges_out:
                g0.add_edge(u, v)
        g0.remove_node(n)
        g = g0
        # print(len([n for n in g if not has_terminals(g, n)]))
    return g


def extend_complex_nodes(G):
    g = G.copy()
    while len([n for n in g if SEPARATOR in g.nodes[n]['type'] and g.out_edges(n) != 0]) != 0:
        g0 = g.copy()
        n = [n for n in g if SEPARATOR in g.nodes[n]['type'] and g.out_edges(n) != 0][0]
        edges_in = list(g.in_edges(n))
        edges_out = list(g.out_edges(n))
        labels = g.nodes[n]['type'].split(SEPARATOR)
        new_nodes = []
        for l in labels:
            new_id = get_id(g0)
            g0.add_node(new_id, type=l)
            if len(new_nodes) != 0:
                g0.add_edge(new_nodes[-1], new_id)
            new_nodes.append(new_id)
        for u, _ in edges_in:
            g0.add_edge(u, new_nodes[0])
        for _, v in edges_out:
            g0.add_edge(new_nodes[-1], v)
        g0.remove_node(n)
        g = g0
        # print(len([n for n in g if not has_terminals(g, n)]))
    return g


def add_unary(G):
    g = G.copy()
    for n in [n for n in g if g.out_degree(n) == 0]:
        if g.nodes[n]['unary'] != '<empty>':
            new_id = get_id(g)
            g.add_node(new_id, type=g.nodes[n]['unary'])
            g.add_edge(new_id, n)
            u, _ = list(g.in_edges(n))[0]
            g.add_edge(u, new_id)
            g.remove_edge(u, n)
    return g


def get_multiset_ast(G, filter_non_terminal=None):
    result = []
    for n in G:
        if G.out_degree(n) > 0:
            if filter_non_terminal is None:
                leaves = get_leaves(G, n)
                leaves = sorted(leaves, key=lambda n: G.nodes[n]['start'])
                result.append(G.nodes[n]['type'] + '-' + '-'.join([str(G.nodes[l]['start']) for l in leaves]))
            else:
                if G.nodes[n]['type'] == filter_non_terminal:
                    leaves = get_leaves(G, n)
                    leaves = sorted(leaves, key=lambda n: G.nodes[n]['start'])
                    result.append(G.nodes[n]['type'] + '-' + '-'.join([str(G.nodes[l]['start']) for l in leaves]))
    return result


def get_precision_recall_f1(G_true, G_pred, filter_non_terminal=None):
    m_true = get_multiset_ast(G_true, filter_non_terminal)
    m_pred = get_multiset_ast(G_pred, None)
    prec = float(len([n for n in m_pred if n in m_true])) / float(len(m_pred))
    rec = float(len([n for n in m_pred if n in m_true])) / float(len(m_true))
    if prec + rec == 0:
        return 0, 0, 0
    f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def get_recall_non_terminal(G_true, G_pred):
    non_terminals = set([])
    for n in G_true:
        if G_true.out_degree(n) > 0:
            non_terminals.add(G_true.nodes[n]['type'])
    dic_results = {}
    m_pred = get_multiset_ast(G_pred, None)
    for n in non_terminals:
        m_true = get_multiset_ast(G_true, n)
        rec = float(len([n for n in m_pred if n in m_true])) / float(len(m_true))
        dic_results[n] = rec
    return dic_results
