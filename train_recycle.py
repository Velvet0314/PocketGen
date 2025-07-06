import shutil
import argparse
from functools import partial
import torch
torch.autograd.set_detect_anomaly(True)  # 启用自动梯度异常检测，用于调试
import esm  # Facebook的ESM蛋白质语言模型
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from torch_geometric.transforms import Compose
import numpy as np
from models.PD import Pocket_Design_new, sample_from_categorical, interpolation_init_new
from utils.datasets import *
from utils.misc import *
from utils.train import *
from utils.data import *
from utils.transforms import *
from torch.utils.data import DataLoader
import wandb

if __name__ == '__main__':
    # 1. 参数解析和配置加载
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train_model.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='./logs')
    args = parser.parse_args()

    # 加载配置文件
    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed) # 设置随机种子保证实验可重复性

    # 2. 日志和实验跟踪设置
    # Logging
    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)

    # 保存配置文件和模型代码，便于实验重现
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))

    # Wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="pocket generation",

        # track hyperparameters and run metadata
        config=config
    )

    # 3. 数据预处理和特征化
    # Transforms
    protein_featurizer = FeaturizeProteinAtom() # 蛋白质原子特征化
    ligand_featurizer = FeaturizeLigandAtom()   # 配体原子特征化
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
    ])

    # 4. ESM蛋白质语言模型设置
    # esm
    name = 'esm1b_t33_650M_UR50S'   # 使用ESM-1b模型
    pretrained_model, alphabet = esm.pretrained.load_model_and_alphabet_hub(name)
    batch_converter = alphabet.get_batch_converter()
    del pretrained_model    # 只需要alphabet，删除模型释放内存

    # 5. 数据集和数据加载器
    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(config=config.dataset, transform=transform, skip_process=True)
    train_set, val_set = subsets['train'], subsets['test']

    # 训练集使用无限迭代器，验证集使用普通加载器
    train_iterator = inf_iterator(DataLoader(train_set, batch_size=config.train.batch_size,
                                             shuffle=True, num_workers=config.train.num_workers,
                                             collate_fn=partial(collate_mols_block, batch_converter=batch_converter)))
    val_loader = DataLoader(val_set, batch_size=config.train.batch_size, shuffle=False,
                            num_workers=config.train.num_workers, collate_fn=partial(collate_mols_block, batch_converter=batch_converter))

    # 6. 模型初始化
    # Model
    logger.info('Building model...')
    model = Pocket_Design_new(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        device=args.device
    ).to(args.device)
    #ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    #model.load_state_dict(ckpt['model'])

    #model.apply(init_weight)
    total = sum([param.nelement() for param in model.parameters()])

    print("Number of parameter: %.2fM" % (total/1e6))

    # 7. 优化器和调度器
    # Optimizer and scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    loss_list = [0., 0., 0.]
    metric_list = [0., 0.]

    # 8. 训练函数定义
    def train(it, loss_list, metric_list):
        """
        训练函数 - 这是整个脚本的核心
        实现了循环训练策略：多次迭代优化同一个结构
        """

        model.train()
        batch = next(train_iterator)    # 获取下一个训练批次

        # 将数据移到GPU
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(args.device)

        # loss, loss_list, aar, rmsd = model(batch)
        # 准备训练数据
        residue_mask = batch['protein_edit_residue']    # 需要编辑的蛋白质残基掩码
        label_ligand = copy.deepcopy(batch['ligand_pos'])   # 真实配体位置
        atom_mask = model.residue_atom_mask[batch['amino_acid'][residue_mask]].bool()   # 原子掩码
        label_X = copy.deepcopy(batch['residue_pos'])   # 真实残基位置
        res_S = copy.deepcopy(batch['amino_acid_processed'])    # 处理后的氨基酸序列

        # 随机循环步数训练：随机选择1-3步循环
        total_steps = torch.randint(1, 4, (1,)).item() # random sample from 1,2,3
        
        # 初始化模型状态
        res_H, res_X, res_S, res_batch, pred_ligand, ligand_feat, ligand_mask, edit_residue_num, residue_mask = model.init(batch)
        
        # 循环优化过程
        for t in range(total_steps, -1, -1):    # 从total_steps倒数到 0
            if t == 0:
                # 最后一步：开启梯度计算进行训练
                model.train()
                res_H, res_X, ligand_pos, ligand_feat, pred_res_type = model(res_H, res_X, res_S, res_batch, pred_ligand, ligand_feat, ligand_mask, edit_residue_num, residue_mask)
            else:
                # 前面的步骤：无梯度推理，用于结构优化
                model.eval()
                with torch.no_grad():
                    res_H, res_X, ligand_pos, ligand_feat, pred_res_type = model(res_H, res_X, res_S, res_batch, pred_ligand, ligand_feat, ligand_mask, edit_residue_num, residue_mask)
        
        # 采样预测的氨基酸类型
        sampled_type, _ = sample_from_categorical(pred_res_type.detach())

        # 计算三个损失函数
        # 1. Huber损失：用于回归任务，对异常值不敏感
        huber_loss = model.huber_loss(res_X[residue_mask][atom_mask], label_X[residue_mask][atom_mask]) + model.huber_loss(ligand_pos[ligand_mask.bool()], label_ligand[ligand_mask.bool()])
        
        # 2. 预测损失：氨基酸类型分类损失
        pred_loss = model.pred_loss(pred_res_type, model.standard2alphabet[batch['amino_acid'][residue_mask] - 1])
        
        # 3. 结构损失：蛋白质结构约束损失（键长、键角等）
        struct_loss = 2 * model.proteinloss.structure_loss(res_X[residue_mask], label_X[residue_mask], batch['amino_acid'][residue_mask] - 1, batch['res_idx'][residue_mask], batch['amino_acid_batch'][residue_mask])
        
        # 总损失
        loss = huber_loss + pred_loss + struct_loss
        loss_list[0] += huber_loss
        loss_list[1] += pred_loss
        loss_list[2] += struct_loss

        # 计算评估指标
        # AAR (Amino Acid Recovery): 氨基酸恢复率
        aar = (model.standard2alphabet[batch['amino_acid'][residue_mask] - 1] == sampled_type).sum() / len(res_S[residue_mask])
        
        # RMSD (Root Mean Square Deviation): 均方根偏差
        rmsd = torch.sqrt((res_X[residue_mask][:, :4].reshape(-1, 3) - label_X[residue_mask][:, :4].reshape(-1, 3)).norm(dim=1).sum() / len(res_S[residue_mask]) / 4)
        metric_list[0] += aar
        metric_list[1] += rmsd

        # 反向传播
        loss.backward()

        # 累积梯度策略：每32步更新一次参数
        # freq = 32
        freq = 2
        if it % freq == 0:
            # 梯度裁剪防止梯度爆炸
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            # 计算平均损失和指标
            total_loss = (loss_list[0] + loss_list[1] + loss_list[2]).item()/freq

            # 日志记录
            logger.info('[Train] Iter %d | Loss %.6f | Loss(huber) %.6f | Loss(pred) %.6f | Loss(bond & andgle) %.6f | AAR %.6f | RMSD %.6f '
                        '|Orig_grad_norm %.6f' % (it, total_loss, loss_list[0].item()/freq, loss_list[1].item()/freq, loss_list[2]/freq, metric_list[0].item()/freq, metric_list[1].item()/freq, orig_grad_norm))
            
            # Wandb和TensorBoard记录
            wandb.log({"loss": total_loss, "Loss(huber)": loss_list[0].item()/freq, "Loss(pred)": loss_list[1].item()/freq, "aar": metric_list[0].item()/freq, "rmsd": metric_list[1].item()/freq})
            writer.add_scalar('train/loss', total_loss, it)
            writer.add_scalar('train/huber_loss', loss_list[0].item()/freq, it)
            writer.add_scalar('train/pred_loss', loss_list[1].item()/freq, it)
            writer.add_scalar('train/bondangle_loss', loss_list[2]/freq, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()

            # 重置累积器
            loss_list = [0., 0., 0.]
            metric_list = [0., 0.]
        return loss_list, metric_list

    # 9. 验证函数定义
    def validate(it):
        """
        验证函数：在验证集上评估模型性能
        使用固定的3步循环进行推理
        """

        sum_loss, sum_n, aar, rmsd = 0, 0, 0, 0
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                for key in batch:
                    # 数据移到GPU
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(args.device)

                # 准备验证数据
                residue_mask = batch['protein_edit_residue']
                label_ligand = copy.deepcopy(batch['ligand_pos'])
                atom_mask = model.residue_atom_mask[batch['amino_acid'][residue_mask]].bool()
                label_X = copy.deepcopy(batch['residue_pos'])
                res_H, res_X, res_S, res_batch, pred_ligand, ligand_feat, ligand_mask, edit_residue_num, residue_mask = model.init(batch)
                
                # 初始化并进行3步循环推理
                for _ in range(3):  # 固定3步循环
                    res_H, res_X, ligand_pos, ligand_feat, pred_res_type = model(res_H, res_X, res_S, res_batch, pred_ligand, ligand_feat, ligand_mask, edit_residue_num, residue_mask)
                
                # 计算验证损失和指标
                ligand_mask = batch['ligand_mask'].bool()
                sampled_type, _ = sample_from_categorical(pred_res_type.detach())
                loss = model.huber_loss(res_X[residue_mask][atom_mask], label_X[residue_mask][atom_mask]) + model.huber_loss(ligand_pos[ligand_mask], label_ligand[ligand_mask])
                loss += model.pred_loss(pred_res_type, model.standard2alphabet[batch['amino_acid'][residue_mask] - 1])
                loss += 2 * model.proteinloss.structure_loss(res_X[residue_mask], label_X[residue_mask], batch['amino_acid'][residue_mask] - 1, batch['res_idx'][residue_mask], batch['amino_acid_batch'][residue_mask])
                sum_loss += loss.item()
                sum_n += 1
                aar += (model.standard2alphabet[batch['amino_acid'][residue_mask] - 1] == sampled_type).sum() / len(res_S[residue_mask])
                rmsd += torch.sqrt((res_X[residue_mask][:, :4].reshape(-1, 3) - label_X[residue_mask][:, :4].reshape(-1, 3)).norm(dim=1).sum() / len(res_S[residue_mask]) / 4)
        
        # 计算平均指标
        avg_loss = sum_loss / sum_n
        aar = aar / sum_n
        rmsd = rmsd / sum_n

        # 学习率调度
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info('[Validate] Iter %05d | Loss %.6f' % (it, avg_loss,))
        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/aar', aar, it)
        writer.add_scalar('val/rmsd', rmsd, it)
        writer.flush()
        wandb.log(
            {"val_loss": avg_loss, "val_aar": aar, "val_rmsd": rmsd})
        return avg_loss

    # 10. 主训练循环
    try:
        for it in range(1, config.train.max_iters + 1):
            # 训练一步
            loss_list, metric_list = train(it, loss_list, metric_list)
            
            # 定期验证和保存模型
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                validate(it)
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')
        wandb.finish()
