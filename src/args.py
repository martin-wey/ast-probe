from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProgramArguments:
    run_base_path: Optional[str] = field(
        default='./runs',
        metadata={'help': 'Base path where to save runs.'}
    )

    run_name: Optional[str] = field(
        default=None,
        metadata={'help': 'Name to identify the run and logging directory.'}
    )

    pretrained_model_name_or_path: Optional[str] = field(
        default='microsoft/codebert-base',
        metadata={'help': 'Path to the pretrained language model or its name on Huggingface hub.'}
    )

    model_type: Optional[str] = field(
        default='roberta',
        metadata={'help': 'Architecture of the transformer model. Currently just supported t5 and roberta.'}
    )

    model_checkpoint: Optional[str] = field(
        default=None,
        metadata={'help': 'Model checkpoint directory.'}
    )
    model_source_checkpoint: Optional[str] = field(
        default=None,
        metadata={'help': 'Model source checkpoint directory for direct transfer.'}
    )

    dataset_name_or_path: Optional[str] = field(
        default='./dataset',
        metadata={'help': 'Path to the folder that contains the dataset.'}
    )

    lang: Optional[str] = field(
        default='python',
        metadata={'help': 'Programming language used in the experiments.'}
    )

    lr: float = field(
        default=1e-3,
        metadata={'help': 'The initial learning rate for AdamW.'}
    )

    epochs: Optional[int] = field(
        default=20,
        metadata={'help': 'Number of training epochs.'}
    )

    batch_size: Optional[int] = field(
        default=32,
        metadata={'help': 'Train and validation batch size.'}
    )

    patience: Optional[int] = field(
        default=5,
        metadata={'help': 'Patience for early stopping.'}
    )

    layer: Optional[int] = field(
        default=5,
        metadata={'help': 'Layer used to get the embeddings.'}
    )

    rank: Optional[int] = field(
        default=128,
        metadata={'help': 'Maximum rank of the probe.'}
    )

    orthogonal_reg: float = field(
        default=5,
        metadata={'help': 'Orthogonal regularized term.'}
    )

    hidden: Optional[int] = field(
        default=768,
        metadata={'help': 'Dimension of the feature word vectors.'}
    )

    seed: Optional[int] = field(
        default=42,
        metadata={'help': 'Seed for experiments replication.'}
    )

    max_tokens: Optional[int] = field(
        default=100,
        metadata={'help': 'Max tokens considered.'}
    )

    download_csn: bool = field(default=False, metadata={'help': 'Download CodeSearchNet dataset.'})
    do_train: bool = field(default=False, metadata={'help': 'Run probe training.'})
    do_test: bool = field(default=False, metadata={'help': 'Run probe training.'})
    do_visualization: bool = field(default=False, metadata={'help': 'Run visualizations.'})
    do_train_direct_transfer: bool = field(default=False, metadata={'help': 'Run probe training direct transfer.'})
