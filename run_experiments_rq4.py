import os


def main():
    models = ['microsoft/codebert-base', 'microsoft/graphcodebert-base',
              'Salesforce/codet5-base', 'huggingface/CodeBERTa-small-v1',
              'roberta-base']
    folders = ['codebert', 'graphcodebert', 'codet5', 'codeberta', 'roberta']
    model_types = ['roberta', 'roberta', 't5', 'roberta', 'roberta']

    for lang in ['python', 'javascript', 'go']:
        for model, folder, model_type in zip(models, folders, model_types):
            layers = [get_layer_model(lang, folder)]
            for layer in layers:
                for rank in [8, 16, 32, 64, 128, 256, 512]:
                    run_name = '_'.join([folder, lang, str(layer), str(rank), 'rq4'])
                    os.system(f"CUDA_VISIBLE_DEVICES=0 python src/main.py --do_train --run_name {run_name} "
                              f"--pretrained_model_name_or_path {model} "
                              f"--model_type {model_type} --lang {lang} "
                              f"--layer {layer} --rank {rank}")


def get_layer_model(lang, folder):
    if lang == 'python':
        if folder == 'codebert':
            return 5
        if folder == 'graphcodebert':
            return 4
        if folder == 'codet5':
            return 7
        if folder == 'codeberta':
            return 4
        if folder == 'roberta':
            return 5
    if lang == 'go':
        if folder == 'codebert':
            return 5
        if folder == 'graphcodebert':
            return 5
        if folder == 'codet5':
            return 8
        if folder == 'codeberta':
            return 4
        if folder == 'roberta':
            return 5
    if lang == 'javascript':
        if folder == 'codebert':
            return 5
        if folder == 'graphcodebert':
            return 4
        if folder == 'codet5':
            return 6
        if folder == 'codeberta':
            return 5
        if folder == 'roberta':
            return 8


if __name__ == '__main__':
    main()
