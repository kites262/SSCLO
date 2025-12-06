import os

import hydra
import numpy as np
import swanlab
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms

import utils.data_load_operate as data_load_operate
from utils.artifact import save_json_file, save_src_files
from utils.evaluation import Evaluator
from utils.HSICommonUtils import ImageStretching
from utils.logger import setup_logger
from utils.Loss import head_loss
from utils.seed import setup_seed
from utils.visual_predict import vis_a_image

# Train exception detection
# torch.autograd.set_detect_anomaly(True)


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def train(cfg: DictConfig) -> None:
    setup_logger()
    save_src_files(cfg.root_dir, cfg.exp.src, "src")
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(config_dict, dict)

    seed = cfg.exp.seed

    train_samples = cfg.exp.train_samples
    val_samples = cfg.exp.val_samples

    max_epochs = cfg.exp.epoch
    learning_rate = cfg.exp.lr

    split_image = cfg.exp.split_image

    exp_model_name = cfg.exp.model.name
    dataset_name = cfg.exp.dataset_name

    swanlab_mode = cfg.swanlab

    runtime = HydraConfig.get().runtime.output_dir.split("/")[-1]

    swanlab.init(
        project="MambaHSI",
        workspace="kites",
        experiment_name=f"{exp_model_name}_{runtime}",
        config=config_dict,
        mode=swanlab_mode,
    )

    # if data_set_name in ["HanChuan", "Houston"]:
    #     split_image = True
    # else:
    #     split_image = False

    transform = transforms.Compose(
        [
            # transforms.Resize((2048, 1024)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.Normalize(mean=[123.6750, 116.2800, 103.5300], std=[58.395, 57.120, 57.3750]),
        ]
    )

    dataset_path = cfg.data_dir

    torch.cuda.empty_cache()

    data, gt = data_load_operate.load_data(dataset_name, dataset_path)

    data_height, data_width, data_channels = data.shape

    gt_reshape = gt.reshape(-1)
    data_height, data_width, data_channels = data.shape
    img = ImageStretching(data)

    class_count = int(max(np.unique(gt)))

    flag_list = [1, 0]  # ratio or num
    ratio_list = [0.1, 0.01]  # [train_ratio,val_ratio]

    loss_func = torch.nn.CrossEntropyLoss(
        ignore_index=-1,
        label_smoothing=cfg.exp.optimizer.label_smoothing,
    )
    net = None

    evaluator = Evaluator(num_class=class_count)

    setup_seed(seed)

    train_data_index, val_data_index, test_data_index, all_data_index = (
        data_load_operate.sampling(
            ratio_list,
            [train_samples, val_samples],
            gt_reshape,
            class_count,
            flag_list[0],
        )
    )
    index = (train_data_index, val_data_index, test_data_index)
    train_label, val_label, test_label = data_load_operate.generate_image_iter(
        data, data_height, data_width, gt_reshape, index
    )

    net = instantiate(
        cfg.exp.model.instance,
        in_channels=data_channels,
        num_classes=class_count,
    )
    logger.debug(net)

    x = transform(np.array(img))
    x = x.unsqueeze(0).float().to(device)  # type: ignore

    train_label = train_label.to(device)
    test_label = test_label.to(device)
    val_label = val_label.to(device)

    # ############################################
    # val_label = test_label
    # ############################################

    net.to(device)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate)

    logger.debug(optimizer)

    best_OA = 0

    epoch = 0
    best_model_path = "best.pth"
    for epoch in range(max_epochs):
        y_train = train_label.unsqueeze(0)

        net.train()
        if split_image:
            x_part1 = x[:, :, : x.shape[2] // 2 + 5, :]
            y_part1 = y_train[:, : x.shape[2] // 2 + 5, :]
            x_part2 = x[:, :, x.shape[2] // 2 - 5 :, :]
            y_part2 = y_train[:, x.shape[2] // 2 - 5 :, :]
            y_pred_part1 = net(x_part1)

            ls1 = head_loss(loss_func, y_pred_part1, y_part1.long())
            optimizer.zero_grad()
            ls1.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            y_pred_part2 = net(x_part2)
            ls2 = head_loss(loss_func, y_pred_part2, y_part2.long())
            optimizer.zero_grad()
            ls2.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            loss = (ls1 + ls2).detach().cpu().numpy()
            logger.debug(f"epoch {epoch}")
            logger.debug(f"loss {loss}")
            swanlab.log({"train/loss": loss}, step=epoch)
        else:
            y_pred = net(x)
            ls = head_loss(loss_func, y_pred, y_train.long())
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
            loss = ls.detach().cpu().numpy()
            logger.debug(f"epoch {epoch}")
            logger.debug(f"loss {loss}")
            swanlab.log({"train/loss": loss}, step=epoch)

        torch.cuda.empty_cache()
        result, predict = evaluator.eval_and_log(net, x, val_label, epoch, stage="val")
        OA = result["OA"]
        # save weight
        if OA >= max(best_OA, 0.90):
            best_OA = OA
            save_json_file(result, "val_best_result.json")
            torch.save(net.state_dict(), best_model_path)
        if (epoch + 1) % 50 == 0:
            vis_a_image(
                gt,
                predict,
                f"visualization/val_predict_epoch{epoch + 1}.png",
                f"visualization/val_gt_epoch{epoch + 1}.png",
            )

        torch.cuda.empty_cache()

    logger.info("==== Stage Test ====")

    if not os.path.exists(best_model_path):
        logger.info("No best model found")
    else:
        net.load_state_dict(torch.load(best_model_path))
        net.to(device)
        result, predict = evaluator.eval_and_log(
            net, x, test_label, epoch, stage="test"
        )
        save_json_file(result, "test_result.json")

        vis_a_image(
            gt,
            predict,
            "test_predict.png",
            "test_gt.png",
        )

    del net


if __name__ == "__main__":
    train()
