import networkx as nx

from .utils import remove_comments_and_docstrings_python, \
    remove_comments_and_docstrings_java_js, \
    remove_comments_php


# aux function, get a new id in the graph
def get_id(G):
    if len(G) == 0:
        return 0
    return max(list(G)) + 1


# aux function used to get the graph associated to the ast
def get_graph_from_tree(node, G, id_father):
    # traverse children
    for child in node.children:
        is_terminal_child = (len(child.children) == 0)
        id_child = get_id(G)
        G.add_node(id_child, type=child.type,
                   is_terminal=is_terminal_child,
                   start=child.start_byte,
                   end=child.end_byte)
        G.add_edge(id_father, id_child)
        get_graph_from_tree(child, G, id_child)


# get token given the code, the start byte and the end byte
def get_token(code, start, end):
    return bytes(code, "utf8")[start:end].decode("utf-8")


def solve_string_problems(G):
    strings = [n for n in G if (G.nodes[n]['type'] == 'string' or ('string_literal' in G.nodes[n]['type']))
               and not G.nodes[n]['is_terminal']]
    for n in strings:
        if n not in G:
            continue
        for v in nx.single_source_shortest_path(G, n).keys():
            if v != n:
                G.remove_node(v)
        G.nodes[n]['is_terminal'] = True


# preprocess code, obtain the ast and returns a network graph.
# it returns the graph of the ast and the preprocessed code
# directed graph
def code2ast(code, parser, lang='python'):
    if lang == 'python':
        # preprocess
        code = remove_comments_and_docstrings_python(code)
        tree = parser.parse(bytes(code, "utf8"))

        G = nx.DiGraph()
        # add root
        G.add_node(0, type=tree.root_node.type,
                   is_terminal=False,
                   start=tree.root_node.start_byte,
                   end=tree.root_node.end_byte)
        get_graph_from_tree(tree.root_node, G, 0)
    elif lang == 'javascript' or lang == 'go':
        code = remove_comments_and_docstrings_java_js(code)
        tree = parser.parse(bytes(code, "utf8"))

        G = nx.DiGraph()
        # add root
        G.add_node(0, type=tree.root_node.type,
                   is_terminal=False,
                   start=tree.root_node.start_byte,
                   end=tree.root_node.end_byte)
        get_graph_from_tree(tree.root_node, G, 0)
    elif lang == 'php':
        if not code.startswith('<?php'):
            code = '<?php\n' + code
        if not code.endswith('?>'):
            code = code + '\n?>'
        code = remove_comments_php(code)
        tree = parser.parse(bytes(code, "utf8"))

        G = nx.DiGraph()
        # add root
        G.add_node(0, type=tree.root_node.type,
                   is_terminal=False,
                   start=tree.root_node.start_byte,
                   end=tree.root_node.end_byte)
        get_graph_from_tree(tree.root_node, G, 0)
    elif lang == 'java':
        code = 'public class Main {\n' + code + '\n}'
        code = remove_comments_and_docstrings_java_js(code)
        tree = parser.parse(bytes(code, "utf8"))

        G = nx.DiGraph()
        # add root
        G.add_node(0, type=tree.root_node.type,
                   is_terminal=False,
                   start=tree.root_node.start_byte,
                   end=tree.root_node.end_byte)
        get_graph_from_tree(tree.root_node, G, 0)
    elif lang == 'ruby':
        tree = parser.parse(bytes(code, "utf8"))
        G = nx.DiGraph()
        # add root
        G.add_node(0, type=tree.root_node.type,
                   is_terminal=False,
                   start=tree.root_node.start_byte,
                   end=tree.root_node.end_byte)
        get_graph_from_tree(tree.root_node, G, 0)

        #remove comments and parse again, not the best way to do that
        code = remove_comments_ast(G, code)
        tree = parser.parse(bytes(code, "utf8"))
        G = nx.DiGraph()
        # add root
        G.add_node(0, type=tree.root_node.type,
                   is_terminal=False,
                   start=tree.root_node.start_byte,
                   end=tree.root_node.end_byte)
        get_graph_from_tree(tree.root_node, G, 0)
    solve_string_problems(G)
    return G, code


def remove_comments_ast(G, code):
    start_end_comments = [(G.nodes[n]['start'], G.nodes[n]['end']) for n in G if G.nodes[n]['type'] == 'comment']
    code_bytes = bytes(code, "utf8")
    new_bytes = []
    for j, b in enumerate(code_bytes):
        to_add = True
        for s, e in start_end_comments:
            if e > j >= s:
                to_add = False
        if to_add:
            new_bytes.append(code_bytes[j:j+1])
    new_bytes = b''.join(new_bytes)
    return new_bytes.decode("utf-8")


def get_tokens_ast(T, code):
    return [get_token(code, T.nodes[t]['start'], T.nodes[t]['end']) for t in
            sorted([n for n in T if T.nodes[n]['is_terminal']],
                   key=lambda n: T.nodes[n]['start'])]


def get_root_ast(G):
    for n in G:
        if G.in_degree(n) == 0:
            return n


def has_error(G):
    for n in G:
        if G.nodes[n]['type'] == 'ERROR':
            return True
    return False
