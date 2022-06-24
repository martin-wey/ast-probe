import os


def main():
    models = ['microsoft/codebert-base', 'microsoft/graphcodebert-base',
                                         'Salesforce/codet5-base', 'huggingface/CodeBERTa-small-v1',
              'roberta-base', 'microsoft/codebert-base', 'microsoft/codebert-base']
    folders = ['codebert', 'graphcodebert', 'codet5', 'codeberta', 'roberta',
               'codebert-baseline', 'codebert0']
    model_types = ['roberta', 'roberta', 't5', 'roberta', 'roberta', 'roberta', 'roberta']

    for lang in ['python', 'javascript', 'go']:
        for model, folder, model_type in zip(models, folders, model_types):
            if model == 'huggingface/CodeBERTa-small-v1':
                layers = list(range(1, 7))
            elif folder == 'codebert0':
                layers = [0]
            else:
                layers = list(range(1, 13))
            for layer in layers:
                run_name = '_'.join([folder, lang, str(layer), '128'])
                os.system(f"CUDA_VISIBLE_DEVICES=3 python src/main.py --do_train --run_name {run_name} "
                          f"--pretrained_model_name_or_path {model} "
                          f"--model_type {model_type} --lang {lang} "
                          f"--layer {layer} --rank 128")


if __name__ == '__main__':
    main()
