import torch
from pretraining.segutils import DiceCELoss, diceloss, dice_score_fn, MyBCELoss, IoU, update_teacher, set_deterministic, seed_worker
import pretraining.classification_stuff.evaluate as ev
import os
from paths import pretraining_LOGS_PATH






def validation_seg(model, teacher, data_loader, writer, epoch, device, num_at_once, loss_fn):
    model.eval()
    teacher.eval()
    with torch.no_grad():
        loss, loss_ema = 0, 0
        up, down, up_ema, down_ema = 0, 0, 0, 0
        intersections, unions, intersections_ema, unions_ema = 0, 0, 0, 0
        for input, label in data_loader:
            input, label = input.to(device), label.to(device)
            for i in range(0, label.shape[0], num_at_once):
                step = min(i + num_at_once, label.shape[0])
                inp, lab = input[i:step], label[i:step]
                __1, __2, __3, pred = model(inp, mode='segmentation')
                loss += loss_fn(pred, lab) / max((label.shape[0] // num_at_once), 1) / len(data_loader)
                up_tmp, down_tmp = dice_score_fn(pred, lab)
                up += up_tmp.item()
                down += down_tmp.item()
                inter, uni = IoU(pred, lab, hard_label=True)
                intersections += inter.item()
                unions += uni.item()

                __1, __2, __3, pred_ema = teacher(inp, mode='segmentation')
                loss_ema += loss_fn(pred_ema, lab) / max((label.shape[0] // num_at_once), 1) / len(data_loader)
                up_tmp, down_tmp = dice_score_fn(pred_ema, lab)
                up_ema += up_tmp.item()
                down_ema += down_tmp.item()
                inter, uni = IoU(pred_ema, lab, hard_label=True)
                intersections_ema += inter.item()
                unions_ema += uni.item()
        iou= (intersections + 1) / (unions + 1)
        dice_score = (up + 1) / (down + 1)
        iou_ema = (intersections_ema + 1) / (unions_ema + 1)
        dice_score_ema = (up_ema + 1) / (down_ema + 1)
        print('--------------------')
        #print('val loss:', loss)
        #print('dice_score:', dice_score)
        #writer.add_scalar('val loss:', loss, epoch)
        #writer.add_scalar('dice_score:', dice_score, epoch)
        #writer.add_scalar('IoU:', iou, epoch)
        writer.add_scalar('val loss ema:', loss_ema.item(), epoch)
        writer.add_scalar('dice_score ema:', dice_score_ema, epoch)
        writer.add_scalar('IoU ema:', iou_ema, epoch)
        print('--------------------')
    model.train()
    teacher.train()
    return dice_score_ema



def supervised_step_seg(input, label, model, loss_fn, writer, iteration, config, num_at_once):
    input, label = input.to(config.DEVICE), label.to(config.DEVICE)
    loss_sum = 0
    for i in range(0, label.shape[0], num_at_once):
        step = min(i + num_at_once, label.shape[0])
        inp, lab = input[i:step], label[i:step]
        __1, __2, __3, pred = model(inp, mode='segmentation')
        loss = loss_fn(pred, lab) / max((label.shape[0] // num_at_once), 1)
        loss_sum += loss.item()
        loss.backward()
    writer.add_scalar('train loss', loss_sum, iteration)



def supervised_step_cls(input, label, model, loss_fn, writer, iteration, config, num_at_once):
    input, label = input.to(config.DEVICE), label.to(config.DEVICE)
    loss_sum = 0
    for i in range(0, label.shape[0], num_at_once):
        step = min(i + num_at_once, label.shape[0])
        inp, lab = input[i:step], label[i:step]
        pred, __1, __2, __3 = model(inp, mode='classification')
        loss = loss_fn(pred, lab, lab) / max((label.shape[0] // num_at_once), 1)
        loss_sum += loss.item()
        loss.backward()
    #print('train loss', loss_sum)
    writer.add_scalar('train loss', loss_sum, iteration)


def supervised_step_mul(input, label_cls, label_seg, model, loss_fn_seg, loss_fn_cls, writer, iteration, config, num_at_once):
    input, label_cls, label_seg = input.to(config.DEVICE), label_cls.to(config.DEVICE), label_seg.to(config.DEVICE)
    loss_sum, loss_sum_cls, loss_sum_seg = 0, 0, 0
    for i in range(0, label_cls.shape[0], num_at_once):
        step = min(i + num_at_once, label_cls.shape[0])
        inp, lab_cls, lab_seg = input[i:step], label_cls[i:step], label_seg[i:step]
        pred_cls, __1, __2, pred_seg = model(inp, mode='both')
        loss_cls = loss_fn_cls(pred_cls, lab_cls, lab_cls) / max((label_cls.shape[0] // num_at_once), 1)
        loss_seg = loss_fn_seg(pred_seg, lab_seg) / max((label_cls.shape[0] // num_at_once), 1)
        loss = loss_cls + loss_seg
        loss_sum += loss.item()
        loss_sum_cls += loss_cls.item()
        loss_sum_seg += loss_seg.item()
        loss.backward()
    #print('train loss', loss_sum)
    writer.add_scalar('train loss', loss_sum, iteration)
    writer.add_scalar('train loss cls', loss_sum_cls, iteration)
    writer.add_scalar('train loss seg', loss_sum_seg, iteration)




def validation_cls(data_loader, model, config, loss_fn, writer, epoch):
    val_results = evaluate(data_loader, model, loss_fn, config, num_at_once = 1, mode='classification')
    #writer.add_scalar('validation loss', val_results["loss"], iteration)
    writer.add_scalar('AUC EMA', val_results["auc_sev2"], epoch)
    writer.add_scalar('val loss ema:', val_results["loss"], epoch)
    return val_results["auc_sev2"], val_results["loss"]


#copied from module evaluate -> litte changes were needed
def evaluate(eval_loader, model, loss_fn, config, num_at_once=1, mode='stoic'):

    model.eval()

    all_preds = []
    all_sev_gt = []
    all_patients = []

    running_loss = 0.
    running_loss_sev0 = 0.
    running_loss_sev1 = 0.

    with torch.no_grad():
        for sample in eval_loader:
            input, seg_gt, sev_gt = sample

            #if type(sev_gt) != torch.Tensor: sev_gt = torch.tensor([sev_gt]).unsqueeze(0)
            #if len(input.shape) != 5: input.unsqueeze(0)
            all_sev_gt.append(sev_gt)

            loss_scaling = max((input.shape[0] // num_at_once), 1)

            preds = []
            for i in range(0, input.shape[0], num_at_once):
                step = min(i + num_at_once, input.shape[0])
                # select subset for gradient accumulation
                v_ten, s_gt, i_gt = input[i:step], sev_gt[i:step], torch.ones_like(sev_gt[i:step])
                v_ten, s_gt, i_gt = v_ten.to(config.DEVICE), s_gt.to(config.DEVICE), i_gt.to(config.DEVICE)
                # metadata
                a = None
                s = None
                # forward pass
                pred, __1, __2, __3 = model(v_ten, a, s, mode=mode)

                running_loss += loss_fn(pred, i_gt, s_gt).cpu().item() / loss_scaling

                # calculate extra loss for severe and non-severe cases
                l0, l1 = loss_fn.partial_loss(pred, i_gt, s_gt)
                running_loss_sev0 += l0.cpu().item() / loss_scaling
                running_loss_sev1 += l1.cpu().item() / loss_scaling

                preds.append(pred.cpu())

            all_preds.append(torch.cat(preds, dim=0))

    all_sev_gt = torch.cat(all_sev_gt)

    running_loss = running_loss / len(all_preds) # do this before torch.cat(all_preds)!
    running_loss_sev0 = running_loss_sev0 / len(all_preds)
    running_loss_sev1 = running_loss_sev1 / len(all_preds)
    all_preds = torch.cat(all_preds)

    pred_inf, pred_sev = loss_fn.finalize(all_preds)

    sev_roc = ev.rocauc_safe(all_sev_gt, pred_sev)

    submission_sev_roc = ev.rocauc_safe(all_sev_gt[:], pred_sev[:])

    model.train()

    return {
        "auc_sev": sev_roc,
        "auc_sev2": submission_sev_roc,
        "loss": running_loss,
        "loss_sev0": running_loss_sev0,
        "loss_sev1": running_loss_sev1,
        "all_data": {
            "inf_pred": pred_inf,
            "sev_gt": all_sev_gt,
            "sev_pred": pred_sev,
        },
    }



def saveSequential(model, epoch, config, identifier):
    #savepath = os.path.join(config.LOGS_PATH, 'checkpoints', identifier.split('/')[-1])
    savepath = os.path.join(config.LOGS_PATH, os.path.basename(identifier), 'checkpoints')
    os.makedirs(savepath, exist_ok=True)
    if epoch is not None:
        savepath = os.path.join(savepath, 'checkpoint_' + str(epoch) + '.tar')
    else:
        savepath = os.path.join(savepath, 'bestmodel_' + str(epoch) + '.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, savepath)

    print('saved model')



def loadSequential(model, savepath=os.path.join(pretraining_LOGS_PATH, 'saved_sup0', 'saved_sup0.pth')):
    checkpoint = torch.load(savepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    return model, epoch
