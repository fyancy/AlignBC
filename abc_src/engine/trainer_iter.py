import os
from visdom import Visdom
from tqdm import tqdm
import yaml

import torch
# from torch.nn.utils import clip_grad_norm_

from utils.init_fn import get_lr
from utils.visdom_fn import vis_line
from engine.inference import inference
from utils.metric_logger import MetricLogger
from utils.save_load_fn import save_model
from utils.visualize_infer import show_image_with_boxes, draw_dt


def do_train(cfg, model, save_path, start_iter,
             tr_loader, va_loader, src_loader, tgt_loader,
             optimizer, scheduler, logger, mcd: bool = True):
    default_depth_method = cfg.MODEL.HEAD.OUTPUT_DEPTH

    best_mAP, best_result_str, best_iter, eval_iter = get_history_from_yaml(save_path)

    model.train()
    logger.info('Start training ...')
    vis = Visdom(env='AdCenter')

    batch_size = cfg.SOLVER.BATCH_SIZE  # 8, 16
    iters_each_epoch = len(tr_loader.dataset) // batch_size
    max_iter = cfg.SOLVER.MAX_ITERATION
    max_epoch = max_iter // iters_each_epoch
    device = cfg.MODEL.DEVICE
    mcd_start_ep = cfg.SOLVER.MCD_START_EP if mcd else max_epoch * 10  # 4

    tgt_loader = iter(tgt_loader) if tgt_loader is not None else None
    # src_loader = iter(src_loader)

    c_meters = c1_meters = c2_meters = dis_meters = None
    for batch, iteration in zip(tr_loader, range(max(0, start_iter), max_iter)):
        epoch = iteration // iters_each_epoch  # 0, 1, 2
        epi = (iteration + 1) % iters_each_epoch
        epoch_end = epi == 0
        epoch_start = epi == 1
        # is_epoch = True  # for test the codes.

        if epoch_start or c_meters is None:
            c_meters = MetricLogger()
            c1_meters = MetricLogger()
            c2_meters = MetricLogger()
            dis_meters = MetricLogger()
            pbar = tqdm(desc=f'Epoch {epoch + 1}/{max_epoch}', total=iters_each_epoch,
                        postfix=dict, mininterval=0.3)

        # [MCD]-Step A:
        images = batch["images"].to(device)
        targets = [target.to(device) for target in batch["targets"]]
        loss_dict, log_loss_dict, log_loss_dict1, log_loss_dict2 = \
            model(images, targets, dis_only=False)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer[0].step()
        optimizer[1].step()
        optimizer[2].step()
        optimizer[0].zero_grad()
        optimizer[1].zero_grad()
        optimizer[2].zero_grad()

        if (epoch + 1) > mcd_start_ep:  # mcd_start_ep=10
            # [MCD]-Step B:
            '''
            if (epi + 1) % 2 == 0:  # TODO: tgt_loader, meters, loss, ...
                s_batch, t_batch = src_loader.__next__(), tgt_loader.__next__()
                images = s_batch["images"].to(device)
                targets = [target.to(device) for target in s_batch["targets"]]
                loss_dict, _, _, _ = model(images, targets, dis_only=False)
                images = t_batch["images"].to(device)
                targets = [target.to(device) for target in t_batch["targets"]]
                loss_dict_d, _ = model(images, targets, dis_only=True)

                losses = sum(loss for loss in loss_dict.values()) - \
                         sum(loss for loss in loss_dict_d.values())
                losses.backward()
                optimizer[1].step()
                optimizer[2].step()
                optimizer[0].zero_grad()
                optimizer[1].zero_grad()
                optimizer[2].zero_grad()
            '''
            # [MCD]-Step C:
            if tgt_loader is not None:
                t_batch = tgt_loader.__next__()
                images = t_batch["images"].to(device)
                targets = [target.to(device) for target in t_batch["targets"]]
            loss_dict_d, log_loss_dict_d = model(images, targets, dis_only=True)
            losses = sum(loss for loss in loss_dict_d.values())
            losses.backward()
            # optimizer[0].step()
            optimizer[1].step()
            optimizer[2].step()
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            optimizer[2].zero_grad()

            log_losses_reduced_d = sum(loss for key, loss in log_loss_dict_d.items()
                                       if key.find('loss') >= 0)
            dis_meters.update(loss=log_losses_reduced_d, **log_loss_dict_d)

        # logging loss:
        log_losses_reduced = sum(loss for key, loss in log_loss_dict.items() if key.find('loss') >= 0)
        c_meters.update(loss=log_losses_reduced, **log_loss_dict)
        log_losses_reduced1 = sum(loss for key, loss in log_loss_dict1.items() if key.find('loss') >= 0)
        c1_meters.update(loss=log_losses_reduced1, **log_loss_dict1)
        log_losses_reduced2 = sum(loss for key, loss in log_loss_dict2.items() if key.find('loss') >= 0)
        c2_meters.update(loss=log_losses_reduced2, **log_loss_dict2)

        pbar.set_postfix(**{'epi_loss': c_meters.loss.value,
                            'mean_loss': c_meters.loss.global_avg,
                            'lr': get_lr(optimizer[0])
                            })
        pbar.update(1)
        for sch in scheduler:
            sch.step()

        if (iteration + 1) % 50 == 0:
            train_metric1 = {name: meter.avg for name, meter in c1_meters.meters.items()}
            train_metric2 = {name: meter.avg for name, meter in c2_meters.meters.items()}
            vis_line(vis, iteration, list(train_metric1.values()),
                     legend_list=list(train_metric1.keys()), win_name='c1_train_metric@50epi')
            vis_line(vis, iteration, list(train_metric2.values()),
                     legend_list=list(train_metric2.keys()), win_name='c2_train_metric@50epi')
            # depth_errors_dict = {key: c2_meters.meters[key].value
            #                      for key in c2_meters.meters.keys() if key.find('MAE') >= 0}
            # vis_line(vis, iteration, list(depth_errors_dict.values()),
            #          legend_list=list(depth_errors_dict.keys()), win_name='c2_depth_errors@40epi')

            # train_loss, iou_metric = {}, {}
            # for name, meter in c1_meters.meters.items():
            #     if name.find('loss') >= 0:
            #         train_loss.update({name: meter.avg})  # past 20 values
            #     elif name.find('IoU') >= 0:
            #         iou_metric.update({name: meter.avg})  # past 20 values
            # vis_line(vis, iteration, list(train_loss.values()),
            #          legend_list=list(train_loss.keys()), win_name='c1_train_loss@40epi')
            # vis_line(vis, iteration, list(iou_metric.values()),
            #          legend_list=list(iou_metric.keys()), win_name='c1_iou_metric@40epi')

            # if (epoch + 1) > mcd_start_ep:
            #     dis_loss_dict = {name: meter.avg for name, meter in dis_meters.meters.items()}
            #     vis_line(vis, iteration, list(dis_loss_dict.values()),
            #              legend_list=list(dis_loss_dict.keys()), win_name='d_train_loss@30epi')

        if epoch_end and iteration > 0:
            pbar.close()

            train_metric = {}
            for name, meter in c_meters.meters.items():
                if name.find('MAE') >= 0:
                    continue
                else:
                    train_metric.update({name: meter.global_avg})  # avg loss in an epoch
            vis_line(vis, epoch + 1, list(train_metric.values()),
                     legend_list=list(train_metric.keys()), win_name='train_metric@epoch')
            if (epoch + 1) > mcd_start_ep:
                dis_loss_dict = {name: meter.global_avg for name, meter in dis_meters.meters.items()}
                vis_line(vis, iteration, list(dis_loss_dict.values()),
                         legend_list=list(dis_loss_dict.keys()), win_name='d_loss@epoch')

            vis_line(vis, epoch + 1, [c1_meters.loss.global_avg, c2_meters.loss.global_avg],
                     legend_list=['c1_tot_loss', 'c2_tot_loss'], win_name='total_loss@epoch')

            all_metrics = {name: meter.global_avg for name, meter in c_meters.meters.items()}
            all_metrics.update({name: meter.global_avg for name, meter in dis_meters.meters.items()
                                if name != 'loss'})
            logger.debug(f'All metrics at epoch {epoch + 1}: \n {all_metrics}')

        # save and eval:
        if (iteration + 1) % cfg.SOLVER.SAVE_CHECKPOINT_INTERVAL == 0 or iteration == 0:
            path = do_save(model, optimizer, save_path, va_loader, iteration, logger, epoch=epoch)
            logger.info('[iter {}, ep {}] Save checkpoint at: {}'.format(iteration + 1, epoch + 1, path))

        if (iteration + 1) == max_iter:
            do_save(model, optimizer, save_path, va_loader, iteration, logger)

        if ((epoch + 1) <= 30 and (iteration + 1) % (5 * iters_each_epoch) == 0) or \
                ((epoch + 1) > 30 and (iteration + 1) % cfg.SOLVER.EVAL_INTERVAL == 0) or \
                ((epoch + 1) >= 60 and (iteration + 1) % (1 * iters_each_epoch) == 0):
            logger.info('iteration = {}, evaluate model on validation set with depth {}'.format(
                iteration + 1, default_depth_method))
            result_dict, result_str = do_eval(model, save_path, va_loader, iteration, logger)
            result_dict = result_dict[0]  # 'R40'
            result_str = result_str[0]  # 'R40'

            # record the best model according to the AP_3D, Car, Moderate, IoU=0.7
            important_key = '{}_3d_{:.2f}/moderate'.format('Car', 0.7)
            eval_mAP = float(result_dict[important_key])
            if eval_mAP >= best_mAP:
                best_mAP = eval_mAP  # save best mAP and corresponding iterations
                best_iter = iteration + 1
                best_result_str = result_str
                eval_iter += 1
                data = {'best_mAP': best_mAP, 'best_result_str': result_str,
                        'best_iter': best_iter, 'eval_iter': eval_iter}
                write_history_to_yaml(save_path, data)
                save_model(save_path, model, optimizer, best_iter - 1,
                           epoch_style=False, file_name=r'weights_car_moderate_best')

                logger.info(
                    'iteration = {}, best_mAP = {:.2f}, '
                    'updating best checkpoint for depth {} \n'.format(iteration + 1, eval_mAP,
                                                                      default_depth_method))

            model.train()

    logger.info(
        "Training finished! The best model is achieved at iteration = {}".format(best_iter)
    )
    logger.info('The best performance is as follows:')
    logger.info('\n' + best_result_str)


def do_eval(model, output_folder, va_loader, iteration, logger, device='cuda'):
    results_folder = os.path.join(output_folder, fr'results_iter{iteration + 1}')

    evaluate_metric, result_str = inference(
        model,
        va_loader,
        dataset_name='kitti',
        device=device,
        results_folder=results_folder,
        metrics=['R40', 'R11'],
    )

    return evaluate_metric, result_str


def do_save(model, optimizer, output_folder, va_loader, iteration, logger, device='cuda', epoch=4):
    os.makedirs(output_folder, exist_ok=True)
    new_path = save_model(output_folder, model, optimizer,
                          iteration, epoch_style=False, file_name=r'weights')
    if iteration == 0:
        logger.info('Test for Model Save operation at iteration 0: fine.')
    if (epoch + 1) % 4 > 0:
        return new_path

    model.eval()
    idx = torch.randint(va_loader.dataset.__len__(), size=(1,)).item()
    img, target, original_idx = va_loader.dataset.__getitem__(idx)
    with torch.no_grad():
        img = img.to(device)
        target_cuda = target.to(device)
        output, out1, out2 = model(img.unsqueeze(0), [target_cuda], single_det=True)
        # show_image_with_boxes(target.get_field('ori_img'), output, target, visualize_preds)
        draw_dt(out1, original_idx, target)
        draw_dt(out2, original_idx, target)
        draw_dt(output, original_idx, target,
                fig_save_path=os.path.join(output_folder, f'iter{iteration + 1}_'))

    model.train()  # NOTE: important

    return new_path


#########################
# tools
#########################


def get_history_from_yaml(his_dir):
    his_path = os.path.join(his_dir, r'car_moderate_best_records.yaml')
    if not os.path.exists(his_path):
        with open(his_path, 'w', encoding='utf-8') as f:
            data = {'best_mAP': 0, 'best_result_str': None,
                    'best_iter': 0, 'eval_iter': 0, 'cur_iter': 0}
            yaml.safe_dump(data, stream=f)
        return 0, None, 0, 0
    else:
        with open(his_path, 'r', encoding='utf-8') as f:
            doc = yaml.safe_load(f)
        return doc['best_mAP'], doc['best_result_str'], \
               doc['best_iter'], doc['eval_iter']


def write_history_to_yaml(file_dir, data):
    his_path = os.path.join(file_dir, r'car_moderate_best_records.yaml')

    with open(his_path, 'r', encoding='utf-8') as ff:
        doc = yaml.safe_load(ff)

    for key, value in data.items():
        doc[key] = value
        # doc.update({key: value})

    with open(his_path, 'w', encoding='utf-8') as ff:
        yaml.safe_dump(doc, stream=ff)
