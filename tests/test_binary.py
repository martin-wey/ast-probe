import unittest

import matplotlib.pyplot as plt
import networkx as nx
import torch
from tree_sitter import Language, Parser

from src.data.binary_tree import ast2binary, tree_to_distance, \
    distance_to_tree, remove_empty_nodes, extend_complex_nodes, \
    get_multiset_ast, get_precision_recall_f1, add_unary
from src.data.code2ast import code2ast, get_tokens_ast, get_token

code = """'''Compute the maximum'''
def max(a,b):
    #compare a and b
    if a > b:
        return
    return b
"""

PY_LANGUAGE = Language('grammars/languages.so', 'python')
parser = Parser()
parser.set_language(PY_LANGUAGE)


def rankloss(input, target, mask, exp=False):
    diff = input[:, :, None] - input[:, None, :]
    target_diff = ((target[:, :, None] - target[:, None, :]) > 0).float()
    mask = mask[:, :, None] * mask[:, None, :] * target_diff
    if exp:
        loss = torch.exp(torch.relu(target_diff - diff)) - 1
    else:
        loss = torch.relu(target_diff - diff)
    loss = (loss * mask).sum() / (mask.sum() + 1e-9)
    return loss


class TestBinary(unittest.TestCase):

    def test_visualization(self):

        def hierarchy_pos(G, root, levels=None, width=1., height=1.):
            '''If there is a cycle that is reachable from root, then this will see infinite recursion.
               G: the graph
               root: the root node
               levels: a dictionary
                       key: level number (starting from 0)
                       value: number of nodes in this level
               width: horizontal space allocated for drawing
               height: vertical space allocated for drawing'''
            TOTAL = "total"
            CURRENT = "current"

            def make_levels(levels, node=root, currentLevel=0, parent=None):
                """Compute the number of nodes for each level
                """
                if not currentLevel in levels:
                    levels[currentLevel] = {TOTAL: 0, CURRENT: 0}
                levels[currentLevel][TOTAL] += 1
                neighbors = G.neighbors(node)
                for neighbor in neighbors:
                    if not neighbor == parent:
                        levels = make_levels(levels, neighbor, currentLevel + 1, node)
                return levels

            def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
                dx = 1 / levels[currentLevel][TOTAL]
                left = dx / 2
                pos[node] = ((left + dx * levels[currentLevel][CURRENT]) * width, vert_loc)
                levels[currentLevel][CURRENT] += 1
                neighbors = G.neighbors(node)
                for neighbor in neighbors:
                    if not neighbor == parent:
                        pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc - vert_gap)
                return pos

            if levels is None:
                levels = make_levels({})
            else:
                levels = {l: {TOTAL: levels[l], CURRENT: 0} for l in levels}
            vert_gap = height / (max([l for l in levels]) + 1)
            return make_pos({})

        G, pre_code = code2ast(code, parser)
        pos = hierarchy_pos(G, 0)
        plt.figure(figsize=(12, 12))
        nx.draw(G, pos=pos,
                with_labels=True, labels=nx.get_node_attributes(G, 'type'))
        plt.show()

        # for n in G:
        #    G.nodes[n]['label'] = G.nodes[n]['type']
        # T = nx.drawing.nx_pydot.to_pydot(G)
        # T.write_png('example.png')

    def test_torch_loss(self):
        seqlen_minus_one = 3
        length_batch = torch.tensor([4, 4])
        norm = length_batch.float() * (length_batch.float() - 1) / 2
        d_real = torch.tensor([[3, 2, 1],
                               [1, 5, 6]])
        d_pred = torch.tensor([[33, 32, 31],
                               [34, 54, 25]])
        d_pred_masked = d_pred.unsqueeze(2).expand(-1, -1, seqlen_minus_one)
        d_real_masked = d_real.unsqueeze(2).expand(-1, -1, seqlen_minus_one)
        d_pred_masked_transposed = d_pred_masked.transpose(1, 2)
        d_real_masked_transposed = d_real_masked.transpose(1, 2)
        d_hat = d_pred_masked - d_pred_masked_transposed  # b x seq-1 x seq-1
        print(d_hat)
        d_no_hat = d_real_masked - d_real_masked_transposed
        print(d_no_hat)
        tri = torch.triu(torch.relu(1 - torch.sign(d_no_hat) * d_hat), diagonal=1)
        print(tri, norm)
        loss_d = torch.sum(tri.view(2, -1), dim=1) / norm
        print(loss_d)
        loss_d = torch.sum(loss_d) / 2
        print(loss_d)

        lens = torch.tensor([3, 3])
        max_len = 3
        mask = torch.arange(max_len)[None, :] < lens[:, None]
        print(rankloss(d_pred, d_real, mask, exp=False))

    def test_random(self):
        d = [12.749075889587402, 12.086353302001953, 3.5092380046844482, 2.8773586750030518, 1.8203082084655762,
             1.105534553527832, 16.498552322387695, 14.007007598876953, 10.105134963989258, 8.575981140136719,
             9.284015655517578, 10.76449203491211, 9.185318946838379, 6.908447265625, 2.8699421882629395,
             1.5084803104400635, 3.9279606342315674, 1.9249317646026611, 1.1106932163238525, 5.392538070678711,
             1.335496187210083]
        c = ['module<sep>function_definition', '<empty>', 'parameters', '<empty>', '<empty>', '<empty>', '<empty>',
             '<empty>',
             'block<sep>for_statement', '<empty>', '<empty>', '<empty>', '<empty>', 'block<sep>if_statement',
             'comparison_operator',
             '<empty>', '<empty>', '<empty>', 'block<sep>return_statement', '<empty>', 'return_statement']
        u = ['<empty>', '<empty>', '<empty>', '<empty>', '<empty>', '<empty>', '<empty>', '<empty>', '<empty>',
             '<empty>',
             '<empty>', '<empty>', '<empty>', '<empty>', '<empty>', '<empty>', '<empty>', '<empty>', '<empty>',
             '<empty>',
             '<empty>', '<empty>']

        binary_ast_recov = distance_to_tree(d, c, u, [str(i) for i in range(len(u))])

        nx.draw(nx.Graph(binary_ast_recov), labels=nx.get_node_attributes(binary_ast_recov, 'type'), with_labels=True)
        plt.show()

        binary_ast_recov_full = extend_complex_nodes(add_unary(remove_empty_nodes(binary_ast_recov)))

        nx.draw(nx.Graph(binary_ast_recov_full), labels=nx.get_node_attributes(binary_ast_recov_full, 'type'),
                with_labels=True)
        plt.show()

    def test_binary(self):

        def label_leaves(g, code):
            G = nx.DiGraph(g)
            for n in G:
                if G.nodes[n]['is_terminal']:
                    G.nodes[n]['type'] = get_token(code, G.nodes[n]['start'], G.nodes[n]['end'])
            return G

        G, pre_code = code2ast(code, parser)
        G_label_leaves = label_leaves(G, pre_code)
        binary_ast = ast2binary(G)

        nx.draw(nx.Graph(G_label_leaves), labels=nx.get_node_attributes(G_label_leaves, 'type'), with_labels=True)
        plt.show()

        nx.draw(nx.Graph(binary_ast), labels=nx.get_node_attributes(binary_ast, 'type'), with_labels=True)
        plt.show()

        self.assertTrue(nx.is_tree(binary_ast))

        print([binary_ast.out_degree(n) for n in binary_ast])
        d, c, _, u = tree_to_distance(binary_ast, 0)
        self.assertTrue(len(u), len(get_tokens_ast(G, pre_code)))
        print(u)
        binary_ast_recov = distance_to_tree(d, c, u, get_tokens_ast(G, pre_code))

        self.assertTrue(nx.is_tree(binary_ast_recov))
        self.assertEqual(len(binary_ast_recov), len(binary_ast))

        nx.draw(nx.Graph(binary_ast_recov), labels=nx.get_node_attributes(binary_ast_recov, 'type'), with_labels=True)
        plt.show()

        print(binary_ast_recov.nodes(data=True))
        binary_ast_recov_full = extend_complex_nodes(add_unary(remove_empty_nodes(binary_ast_recov)))

        def node_match_type(n1, n2):
            return n1['type'] == n2['type']

        self.assertTrue(nx.is_isomorphic(binary_ast_recov_full, G_label_leaves, node_match_type))

        nx.draw(nx.Graph(binary_ast_recov_full), labels=nx.get_node_attributes(binary_ast_recov_full, 'type'),
                with_labels=True)
        plt.show()

        print(get_precision_recall_f1(binary_ast_recov_full, binary_ast_recov_full))

        perturbed = nx.DiGraph(binary_ast_recov)
        for n in perturbed:
            if perturbed.nodes[n]['type'] == 'comparison_operator':
                perturbed.nodes[n]['type'] = 'binary_operator'
        perturbed = extend_complex_nodes(add_unary(remove_empty_nodes(perturbed)))
        print(get_precision_recall_f1(binary_ast_recov_full, perturbed))

        nx.draw(nx.Graph(perturbed), labels=nx.get_node_attributes(perturbed, 'type'),
                with_labels=True)
        plt.show()

        print(get_multiset_ast(binary_ast_recov_full))
        print(get_multiset_ast(G_label_leaves))
        # print(binary_ast.nodes(data=True))


if __name__ == '__main__':
    unittest.main()
