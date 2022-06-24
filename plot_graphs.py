import glob
import argparse
import os
import pickle

import pandas as pd
from plotnine import ggplot, aes, geom_line, \
    scale_x_continuous, labs, scale_color_discrete, \
    theme, element_text


def main():
    parser = argparse.ArgumentParser(description='Script for generating the plots of the paper')
    parser.add_argument('--run_dir', default='./runs', help='Path of the run logs')
    args = parser.parse_args()

    data = {'model': [], 'lang': [], 'layer': [], 'rank': [],
            'precision': [], 'recall': [], 'f1': [], 'rq4': []}
    for file in glob.glob(args.run_dir + "/*/metrics.log"):
        parent = os.path.dirname(file).split('/')[-1]
        rq4 = False
        if '_rq4' not in parent:
            model, lang, layer, rank = parent.split('_')
        else:
            model, lang, layer, rank, _ = parent.split('_')
            rq4 = True
        with open(file, 'rb') as f:
            results = pickle.load(f)
        if model == 'codebert0':
            model = 'codebert-baseline'
        data['model'].append(model)
        data['lang'].append(lang)
        data['layer'].append(int(layer))
        data['rank'].append(int(rank))
        data['precision'].append(results['test_precision'])
        data['recall'].append(results['test_recall'])
        data['f1'].append(results['test_f1'])
        data['rq4'].append(rq4)

    df = pd.DataFrame(data)
    df_renamed = df.replace({'codebert': 'CodeBERT',
                             'codebert-baseline': 'CodeBERTrand',
                             'codeberta': 'CodeBERTa',
                             'codet5': 'CodeT5',
                             'graphcodebert': 'GraphCodeBERT',
                             'roberta': 'RoBERTa'})
    for lang in ['python', 'javascript', 'go']:
        myPlot = (
                ggplot(df_renamed[(df_renamed['lang'] == lang) & (df_renamed['rq4'] == False)])
                + aes(x="layer", y="f1", color='model')
                + geom_line()
                + scale_x_continuous(breaks=range(0, 13, 1))
                + labs(x="Layer", y="F1", color="Model")
                + scale_color_discrete(breaks=['GraphCodeBERT',
                                               'CodeBERT',
                                               'CodeT5',
                                               'RoBERTa',
                                               'CodeBERTa',
                                               'CodeBERTrand'])
                + theme(text=element_text(size=16))
        )
        myPlot.save(f"myplot_{lang}.pdf", dpi=600)

    for lang in ['python', 'javascript', 'go']:
        for model in ['codebert', 'graphcodebert', 'codet5', 'codeberta', 'roberta', 'codebert-baseline']:
            df_filtered = df[(df['lang'] == lang) & (df['model'] == model) & (df['rq4'] == False)]
            row = df.iloc[df_filtered['f1'].idxmax()]
            print(model, lang, row['layer'], row['precision'], row['recall'], row['f1'])
            if model == 'codebert-baseline':
                print(df[(df['lang'] == lang) & (df['model'] == model)])

    for lang in ['python', 'go', 'javascript']:
        myPlot = (
                ggplot(df_renamed[(df_renamed['lang'] == lang) & (df_renamed['rq4'] != False)])
                + aes(x="rank", y="f1", color='model')
                + geom_line()
                + scale_x_continuous(trans='log2')
                + labs(x="Rank", y="F1", color="Model")
                + scale_color_discrete(breaks=['GraphCodeBERT',
                                               'CodeBERT',
                                               'CodeT5',
                                               'RoBERTa',
                                               'CodeBERTa',
                                               'CodeBERTrand'])
                + theme(text=element_text(size=16))
        )
        myPlot.save(f"myplot_rank_{lang}.pdf", dpi=600)


if __name__ == '__main__':
    main()
