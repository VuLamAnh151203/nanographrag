def parse_args():
    import yaml
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--model_name_or_path", default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument(
            "--train_type",
            type=str,
            default="train",
            help="chose from [train, distill]")

    parser.add_argument("--train_dataset", help='trainset')
    parser.add_argument("--train_dataset_vec", help='distillion trainset embedding')
    parser.add_argument("--val_dataset", help="validation file", default=None)
    parser.add_argument('--shuffle', action='store_true', help='if shuffle')

    parser.add_argument('--neg_nums', type=int, default=15)
    parser.add_argument('--query_max_len', type=int, default=128)
    parser.add_argument('--passage_max_len', type=int, default=512)
    parser.add_argument('--teatch_emebedding_dim', type=int)

    parser.add_argument('--output_dir', help='output dir')
    parser.add_argument('--save_on_epoch_end', type=int, default=1, help='if save_on_epoch_end')
    parser.add_argument('--num_max_checkpoints', type=int, default=5)


    parser.add_argument('--epochs', type=int, default=2, help='epoch nums')
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument("--warmup_proportion", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument('--mixed_precision', default='fp16', help='')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    parser.add_argument("--log_with", type=str, default='wandb', help='wandb,tensorboard')
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=None)

    parser.add_argument('--use_mrl', action='store_true', help='if use mrl loss')
    parser.add_argument('--mrl_dims', type=str, help='list of mrl dims', default='128, 256, 512, 768, 1024, 1280, 1536, 1792')

    args = parser.parse_args()

    print(args.config)
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    print(config)
    for key, value in config.items():
        setattr(args, key, value)

    return args

if __name__ == "__main__":
    parse_args()