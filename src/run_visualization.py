from data.utils import match_tokenized_to_untokenized_roberta
from data.code2ast import code2ast, get_tokens_ast
from data.binary_tree import ast2binary, tree_to_distance, distance_to_tree, \
    extend_complex_nodes, add_unary, remove_empty_nodes, get_precision_recall_f1, \
    get_recall_non_terminal
import torch
from probe.utils import get_embeddings, align_function
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from data import PY_LANGUAGE, JS_LANGUAGE, GO_LANGUAGE
from probe import ParserProbe
import os
from transformers import AutoModel, AutoTokenizer, RobertaModel, T5EncoderModel
from run_probing import generate_baseline
from tree_sitter import Parser
import glob
import logging
import pickle

logger = logging.getLogger(__name__)


# todo: add the dictionaries for the classification
def run_visualization(args):
    code_samples = []
    if args.lang == 'python':
        for filename in glob.glob('code_samples/*.py'):
            with open(filename, 'r') as f:
                code_samples.append(f.read())
    elif args.lang == 'javascript':
        for filename in glob.glob('code_samples/*.js'):
            with open(filename, 'r') as f:
                code_samples.append(f.read())
    elif args.lang == 'go':
        for filename in glob.glob('code_samples/*.go'):
            with open(filename, 'r') as f:
                code_samples.append(f.read())

    # @todo: load lmodel and tokenizer from checkpoint
    # @todo: model_type in ProgramArguments
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    if args.model_type == 't5':
        lmodel = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, output_hidden_states=True)
        lmodel = lmodel.to(args.device)
    else:
        lmodel = AutoModel.from_pretrained(args.pretrained_model_name_or_path, output_hidden_states=True)
        if '-baseline' in args.run_name:
            lmodel = generate_baseline(lmodel)
        lmodel = lmodel.to(args.device)

    # select the parser
    parser = Parser()
    if args.lang == 'python':
        parser.set_language(PY_LANGUAGE)
    elif args.lang == 'javascript':
        parser.set_language(JS_LANGUAGE)
    elif args.lang == 'go':
        parser.set_language(GO_LANGUAGE)

    # load the labels
    labels_file_path = os.path.join(args.dataset_name_or_path, 'labels.pkl')
    with open(labels_file_path, 'rb') as f:
        data = pickle.load(f)
        labels_to_ids_c = data['labels_to_ids_c']
        ids_to_labels_c = data['ids_to_labels_c']
        labels_to_ids_u = data['labels_to_ids_u']
        ids_to_labels_u = data['ids_to_labels_u']

    final_probe_model = ParserProbe(
        probe_rank=args.rank,
        hidden_dim=args.hidden,
        number_labels_c=len(labels_to_ids_c),
        number_labels_u=len(labels_to_ids_u)).to(args.device)

    final_probe_model.load_state_dict(torch.load(os.path.join(args.model_checkpoint, f'pytorch_model.bin'),
                                                 map_location=torch.device(args.device)))

    __run_visualization_code_samples(lmodel, tokenizer, final_probe_model, code_samples, parser,
                                     ids_to_labels_c, ids_to_labels_u, args)
    __run_visualization_vectors(final_probe_model, ids_to_labels_c, ids_to_labels_u, args)


def __run_visualization_code_samples(lmodel, tokenizer, probe_model, code_samples,
                                     parser, ids_to_labels_c, ids_to_labels_u, args):
    lmodel.eval()
    probe_model.eval()

    for c, code in enumerate(code_samples):
        G, pre_code = code2ast(code, parser, args.lang)
        binary_ast = ast2binary(G)
        ds_current, cs_labels, _, us_labels = tree_to_distance(binary_ast, 0)
        tokens = get_tokens_ast(G, pre_code)

        # align tokens with subtokens
        to_convert, mapping = match_tokenized_to_untokenized_roberta(tokens, tokenizer)
        # generate inputs and masks
        inputs = torch.tensor([tokenizer.convert_tokens_to_ids([tokenizer.cls_token] +
                                                               to_convert +
                                                               [tokenizer.sep_token])]).to(args.device)
        mask = torch.tensor([[1] * inputs.shape[1]]).to(args.device)

        # get align tensor
        j = 0
        indices = []
        for t in range(len(mapping)):
            indices += [j] * len(mapping[t])
            j += 1
        indices += [j] * (inputs.shape[1] - 1 - len(indices))
        alig = torch.tensor([indices]).to(args.device)

        # get embeddings from the lmodel
        emb = get_embeddings(inputs, mask, lmodel, args.layer, args.model_type)
        emb = align_function(emb, alig)

        # generating distance matrix
        d_pred, scores_c, scores_u = probe_model(emb.to(args.device))
        scores_c = torch.argmax(scores_c, dim=2)
        scores_u = torch.argmax(scores_u, dim=2)
        len_tokens = len(tokens)

        d_pred_current = d_pred[0, 0:len_tokens - 1].tolist()
        score_c_current = scores_c[0, 0:len_tokens - 1].tolist()
        score_u_current = scores_u[0, 0:len_tokens].tolist()

        scores_c_labels = [ids_to_labels_c[s] for s in score_c_current]
        scores_u_labels = [ids_to_labels_u[s] for s in score_u_current]

        ground_truth_tree = distance_to_tree(ds_current, cs_labels, us_labels, tokens)
        ground_truth_tree = extend_complex_nodes(add_unary(remove_empty_nodes(ground_truth_tree)))

        pred_tree = distance_to_tree(d_pred_current, scores_c_labels, scores_u_labels, tokens)
        pred_tree = extend_complex_nodes(add_unary(remove_empty_nodes(pred_tree)))

        prec_score, recall_score, f1_score = get_precision_recall_f1(ground_truth_tree, pred_tree)
        # _, recall_block, _ = get_precision_recall_f1(ground_truth_tree, pred_tree, filter_non_terminal='block')

        logger.info(f'For code {c}, prec = {prec_score}, recall = {recall_score}, f1 = {f1_score}.')
        # logger.info(f'For code {c}, recall block = {recall_block}.')

        recall_score = get_recall_non_terminal(ground_truth_tree, pred_tree)
        for k, s in recall_score.items():
            logger.info(f'Non-terminal {k} | recall {s}')

        figure, axis = plt.subplots(2, figsize=(15, 15))
        nx.draw(nx.Graph(ground_truth_tree), labels=nx.get_node_attributes(ground_truth_tree, 'type'), with_labels=True,
                ax=axis[0])
        axis[0].set_title("True ast")
        nx.draw(nx.Graph(pred_tree), labels=nx.get_node_attributes(pred_tree, 'type'), with_labels=True,
                ax=axis[1])
        axis[1].set_title("Pred ast")
        plt.show()
        plt.savefig(f'fig_{c}_{args.lang}.png')

        labels_axis = [tokens[i] + '-' + tokens[i + 1] for i in range(0, len(tokens) - 1)]
        figure, axis = plt.subplots(2, figsize=(15, 15))
        axis[0].bar(labels_axis, ds_current)
        axis[0].set_title("True dist")
        for ix, label in enumerate(scores_c_labels):
            axis[0].annotate(label, (labels_axis[ix], ds_current[ix]))

        axis[1].bar(labels_axis, d_pred_current)
        axis[1].set_title("Pred dist")
        for ix, label in enumerate(cs_labels):
            axis[1].annotate(label, (labels_axis[ix], d_pred_current[ix]))
        plt.show()
        plt.savefig(f'fig_{c}_{args.lang}_syn_dis.png')


def __run_visualization_vectors(probe_model, ids_to_labels_c, ids_to_labels_u, args):
    vectors_c = probe_model.vectors_c.detach().cpu().numpy().T
    vectors_u = probe_model.vectors_u.detach().cpu().numpy().T

    v_c_2d = TSNE(n_components=2, learning_rate='auto',
                  init='random', random_state=args.seed).fit_transform(vectors_c)
    v_u_2d = TSNE(n_components=2, learning_rate='auto',
                  init='random', random_state=args.seed).fit_transform(vectors_u)

    figure, axis = plt.subplots(2, figsize=(15, 15))
    axis[0].scatter(v_c_2d[:, 0], v_c_2d[:, 1])
    axis[0].set_title("Vectors constituency")
    for ix, label in ids_to_labels_c.items():
        axis[0].annotate(label, (v_c_2d[ix, 0], v_c_2d[ix, 1]))

    axis[1].scatter(v_u_2d[:, 0], v_u_2d[:, 1])
    axis[1].set_title("Vectors unary")
    for ix, label in ids_to_labels_u.items():
        axis[1].annotate(label, (v_u_2d[ix, 0], v_u_2d[ix, 1]))

    plt.show()
    plt.savefig(f'vectors.png')
