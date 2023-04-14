import torch

from config import cfg
from model.detector import KeypointDetector
from data.build import make_data_loader, build_test_loader

from utils.init_fn import random_seed, build_optimizer, build_scheduler
from utils.logger import set_logger
from utils.save_load_fn import load_model
from engine.trainer_iter import do_train
from engine.inference import inference
from utils.visualize_infer import draw_dt


class NetTrainer:
    def __init__(self):
        self.setup()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model = KeypointDetector(cfg).to(self.device)

    @staticmethod
    def setup():
        cfg.merge_from_file(r"E:\fy_works\projects\2022_mono3d\AdCenter_mono\config\adc.yaml")
        # cfg.merge_from_list(args.opts)
        # cfg.merge_from_other_cfg()
        # cfg.freeze()
        print(f'USE NORM: {cfg.MODEL.BACKBONE.USE_NORMALIZATION}')
        print(f"USE RIGHT IMAGES: {cfg.DATASETS.USE_RIGHT_IMAGE}")
        # print(f'USE Dropout: {cfg.MODEL.SMOKE_HEAD.USE_DROPOUT}')
        random_seed(deterministic=False)

    def train_for_val(self, save_path=None, finetune_path=None):
        logger = set_logger(cfg.LOG_OUTPUT_DIR, logger_name='adc_3712')

        tr_loader = make_data_loader(cfg, split='train', is_train=True)
        va_loader = build_test_loader(cfg, split='val', is_train=False, is_visualize=True)
        # src = tgt
        src_loader = None  # make_data_loader(cfg, split='train', is_train=True, bsize=4)
        tgt_loader = None  # make_data_loader(cfg, split='val', is_train=True, bsize=4)

        iters_each_epoch = len(tr_loader.dataset) // cfg.SOLVER.BATCH_SIZE
        # use epoch rather than iterations for saving checkpoint and validation
        if cfg.SOLVER.EVAL_AND_SAVE_EPOCH:
            cfg.SOLVER.MAX_ITERATION = cfg.SOLVER.MAX_EPOCHS * iters_each_epoch
            cfg.SOLVER.SAVE_CHECKPOINT_INTERVAL = iters_each_epoch * cfg.SOLVER.SAVE_CHECKPOINT_EPOCH_INTERVAL
            cfg.SOLVER.EVAL_INTERVAL = iters_each_epoch * cfg.SOLVER.EVAL_EPOCH_INTERVAL
            cfg.SOLVER.STEPS = [iters_each_epoch * x for x in cfg.SOLVER.DECAY_EPOCH_STEPS]
            cfg.SOLVER.WARMUP_STEPS = cfg.SOLVER.WARMUP_EPOCH * iters_each_epoch
        # cfg.freeze()

        opt1 = build_optimizer(self.model.backbone, cfg)
        opt2 = build_optimizer([self.model.neck.feat_conv1, self.model.heads.predictor[0]], cfg)
        opt3 = build_optimizer([self.model.neck.feat_conv2, self.model.heads.predictor[1]], cfg)
        optimizer = [opt1, opt2, opt3]

        if finetune_path:
            self.model, optimizer, start_iter = load_model(finetune_path, self.model, optimizer,
                                                           base_lr=cfg.SOLVER.BASE_LR,
                                                           lr_decay=cfg.SOLVER.LR_DECAY,
                                                           lr_steps=cfg.SOLVER.STEPS)
            logger.info(f"***** Training with iter {start_iter}, LR STEPS: {cfg.SOLVER.STEPS} *****")
            # logger.debug(f"configs: \n {cfg}")
        else:
            start_iter = 0
            logger.debug(f"configs: \n {cfg}")

        sch1 = build_scheduler(opt1, optim_cfg=cfg.SOLVER, last_epoch=start_iter+1)
        sch2 = build_scheduler(opt2, optim_cfg=cfg.SOLVER, last_epoch=start_iter+1)
        sch3 = build_scheduler(opt3, optim_cfg=cfg.SOLVER, last_epoch=start_iter+1)
        scheduler = [sch1, sch2, sch3]

        do_train(cfg, self.model, save_path, start_iter,
                 tr_loader, va_loader, src_loader, tgt_loader,
                 optimizer, scheduler, logger
                 )
        exit('Training finished.')

    def train_for_test(self):
        pass

    def dataset_evaluation(self, model_path, results_folder, threshold=None):
        self.model = load_model(model_path, self.model)
        va_loader = build_test_loader(cfg, split='val', is_train=False, is_visualize=False)
        if threshold:
            cfg.TEST.DETECTIONS_THRESHOLD = threshold
            # print(f'Change detection threshold to {cfg.TEST.DETECTIONS_THRESHOLD}')
        print(f'Detection threshold: {cfg.TEST.DETECTIONS_THRESHOLD}')

        inference(
            self.model,
            va_loader,
            dataset_name='kitti',
            results_folder=results_folder,
            metrics=['R40', 'R11'],
        )

    def visualization(self, model_path, split='val'):
        self.model = load_model(model_path, self.model)
        if split == 'val':
            va_loader = build_test_loader(cfg, split='val', is_train=False, is_visualize=True)
        else:
            va_loader = build_test_loader(cfg, split='test', is_train=False, is_visualize=False)

        self.model = self.model.eval()
        for i in range(3):
            idx = torch.randint(va_loader.dataset.__len__(), size=(1,)).item()
            img, target, original_idx = va_loader.dataset.__getitem__(idx)
            with torch.no_grad():
                img = img.to(self.device)
                target_cuda = target.to(self.device)

                output, out1, out2 = self.model(img.unsqueeze(0), [target_cuda], single_det=True)
                # show_image_with_boxes(target.get_field('ori_img'), output, target, visualize_preds)
                draw_dt(out1, original_idx, target)
                draw_dt(out2, original_idx, target)
                draw_dt(output, original_idx, target)


if __name__ == "__main__":
    import os
    import yaml

    learner = NetTrainer()

    save_dir = r"E:\fy_works\save_model\adcenter_mono\train3712_3cls"
    # os.makedirs(save_dir, exist_ok=True)
    assert os.path.exists(save_dir)

    # finetune = None
    finetune = os.path.join(save_dir, 'weights(iter42224)')

    # result_dir = r'E:\fy_works\save_model\adcenter_mono\train3712_3cls\results_iter24128'
    # learner.dataset_evaluation(finetune, result_dir)

    # if input('Train? (y/n)\t').lower() == 'y':
    try:
        # learner.visualization(finetune, split='val')
        learner.train_for_val(save_dir, finetune)
        # exit()
    except RuntimeError:  # RuntimeError
        print('+++++++ Runtime Error +++++++ ')
        # for i in range(5):
        learner = NetTrainer()
        yaml_file = os.path.join(save_dir, r'car_moderate_best_records.yaml')
        with open(yaml_file, 'r', encoding='utf-8') as f:
            cur_iter = yaml.safe_load(f)['cur_iter']
        finetune = os.path.join(save_dir, fr'weights(iter{cur_iter})')
        learner.train_for_val(save_dir, finetune)
    exit()

    # detection:
    # result_dir = r'E:\fy_works\save_model\adcenter\ad_from0\results_ep120'
    result_dir = r'E:\fy_works\save_model\adcenter_mono\train3712_3cls\results_iter928'
    model_pt = r"E:\fy_works\save_model\adcenter_mono\train3712_3cls\weights(iter928)"
    learner.visualization(model_pt, split='val')
    learner.dataset_evaluation(model_pt, result_dir)
