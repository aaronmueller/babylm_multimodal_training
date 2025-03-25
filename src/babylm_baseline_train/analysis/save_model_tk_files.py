from babylm_baseline_train.analysis import use_lm_eval
from tqdm import tqdm
import ipdb
import os


def get_parser():
    parser = use_lm_eval.get_parser()
    parser.add_argument(
            '--save_folder',
            default='/nese/mit/group/evlab/u/chengxuz/babyLM_related/more_models', type=str,
            action='store')
    return parser


class ModelSaver(use_lm_eval.LMEvalRunner):
    def run(self):
        print(self.all_ckpts)
        for _ckpt in tqdm(self.all_ckpts):
            self.curr_ckpt = _ckpt
            ckpt_path = os.path.join(self.exp_folder, _ckpt)
            self.load_ckpt(ckpt_path)
            save_folder = os.path.join(
                    self.args.save_folder,
                    self.col_name,
                    self.exp_id + '_' + _ckpt[:-4])
            self.lm.model.save_pretrained(save_folder)
            self.lm.tokenizer.save_pretrained(save_folder)


def main():
    parser = get_parser()
    args = parser.parse_args()

    model_saver = ModelSaver(args)
    model_saver.run()


if __name__ == '__main__':
    main()
