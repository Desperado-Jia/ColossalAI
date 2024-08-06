import argparse
import tempfile

from src.components.data.dataset import prepare_hf_dataset
from src.components.data.packer import setup_packer, PackerType
from src.components.models import setup_llm, ChatLLMType


def str2bool(v: str) -> bool:
    if v.lower() in ["true", "yes", "on", "y", "1"]:
        return True
    elif v.lower() in ["false", "no", "off", "n", "0"]:
        return False
    else:
        raise ValueError(f"Invalid v[{v}] to be booleanized.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Define arguments for preparing tokenized huggingface dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_files", nargs="+", default=[],
        help="All paths of existed '.jsonl' files containing `RawSample`."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./dataset.tk",
        help="Output directory with suffix '.tk' containing `TokenizedSample`."
    )
    parser.add_argument(
        "--llm", type=str, default=ChatLLMType.LLAMA3.value,
        help="Type of `ChatLLM`, must be one of registered key in `ChatLLMType`."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, required=True,
        help="A path to a directory containing configuration files, vocabulary files and "
             "model weights used for chat LLM."
    )
    parser.add_argument(
        "--trust_remote_code", type=str2bool, default=True,
        help="Whether or not to allow for custom models defined on the Hub in their own modeling files. "
             "This option should only be set to `True` for repositories you trust and in which "
             "you have read the code, as it will execute code present on the Hub on your local machine."
    )
    parser.add_argument(
        "--packer", type=str, default=PackerType.INTEGRITY.value,
        help="Type of `Packer`, must be one of registered key in `PackerType`."
    )
    parser.add_argument(
        "--max_sequence_length", type=int, default=4096,
        help="Maximum length of each (packed) tokenized sequence."
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Number of CPU cores for processing dataset."
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    llm = setup_llm(
        llm_type=args.llm,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        trust_remote_code=args.trust_remote_code
    )
    llm.init_tokenizer()
    packer = setup_packer(packer_type=args.packer, max_length=args.max_sequence_length)

    with tempfile.TemporaryDirectory() as cache_dir:
        dataset = prepare_hf_dataset(
            filenames=args.input_files,
            llm=llm,
            packer=packer,
            shuffle=True,
            num_workers=args.num_workers,
            cache_dir=cache_dir
        )
        dataset.save_to_disk(
            dataset_path=args.output_dir,
            num_proc=args.num_workers
        )


if __name__ == '__main__':
    main(parse_args())
