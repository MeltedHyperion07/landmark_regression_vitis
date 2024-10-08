import os
import sys
import argparse
from pytorch_nndct.apis import torch_quantizer
import torch
import torchvision.transforms as transforms
import random
from my_hrnet768_vck import get_pose_net

from tqdm import tqdm

import _init_paths
import dataset
from config import cfg
from config import update_config
from core.function import validate
from core.loss import JointsMSELoss

#####################################################################################################################################################################################################
#device = torch.device("cuda")
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#####################################################################################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--cfg',
    help='experiment configure file name',
    required=True,
    type=str)
parser.add_argument('opts',
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER)
parser.add_argument('--modelDir',
    help='model directory',
    type=str,
    default='')
parser.add_argument('--logDir',
    help='log directory',
    type=str,
    default='')
parser.add_argument('--dataDir',
    help='data directory',
    type=str,
    default='')
parser.add_argument('--prevModelDir',
    help='prev Model directory',
    type=str,
    default='')

#################################################
# cfg args
#################################################
parser.add_argument('--data_dir',
    default="/path/to/imagenet/",
    help='Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation')
parser.add_argument('--model_dir',
    default="/path/to/trained_model/",
    help='Trained model file path. Download pretrained model from the following url and put it in model_dir specified path: https://download.pytorch.org/models/resnet18-5c106cde.pth'
)
parser.add_argument('--config_file',
    default=None,
    help='quantization configuration file')
parser.add_argument('--subset_len',
    default=None,
    type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
parser.add_argument('--batch_size',
    default=1,
    type=int,
    help='input data batch size to evaluate model')
parser.add_argument('--quant_mode', 
    default='calib', 
    choices=['float', 'calib', 'test'], 
    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
parser.add_argument('--fast_finetune', 
    dest='fast_finetune',
    action='store_true',
    help='fast finetune model before calibration')
parser.add_argument('--deploy', 
    dest='deploy',
    action='store_true',
    help='export xmodel for deployment')
parser.add_argument('--inspect', 
    dest='inspect',
    action='store_true',
    help='inspect model')
parser.add_argument('--target', 
    dest='target',
    nargs="?",
    const="",
    help='specify target device')

args, _ = parser.parse_known_args()

#####################################################################################################################################################################################################


def load_data(cfg, subset_len=None,
              sample_method='random',):


  # Data loading code
  normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
  dataset_ = dataset.PEdataset_test(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

  if subset_len:
    assert subset_len <= len(dataset_)
    if sample_method == 'random':
      dataset_ = torch.utils.data.Subset(
          dataset_, random.sample(range(0, len(dataset_)), subset_len))
    else:
      dataset_ = torch.utils.data.Subset(dataset_, list(range(subset_len)))
  
  data_loader = torch.utils.data.DataLoader(
        dataset_,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )
  return data_loader



def quantization(file_path=''): 

  data_dir = args.data_dir

  quant_mode = args.quant_mode
  deploy = args.deploy
  batch_size = args.batch_size
  subset_len = args.subset_len
  inspect = args.inspect
  config_file = args.config_file
  target = args.target
  if quant_mode != 'test' and deploy:
    deploy = False
    print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
  if deploy and (batch_size != 1 or subset_len != 1):
    print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
    batch_size = 1
    subset_len = 1

  model = get_pose_net(cfg, False).to(device)
  model = model.eval()
  model.load_state_dict(torch.load(file_path))

  input = torch.randn([batch_size, 3, 768, 768]).to(device)
  if quant_mode == 'float':
    quant_model = model
    if inspect:
      if not target:
          raise RuntimeError("A target should be specified for inspector.")
      import sys
      from pytorch_nndct.apis import Inspector
      # create inspector
      inspector = Inspector(target)  # by name
      # start to inspect
      inspector.inspect(quant_model, (input), device=device)
      sys.exit()
      
  else:
    ## new api
    ####################################################################################
    quantizer = torch_quantizer(
        quant_mode, model, (input), device=device, quant_config_file=config_file, target=target)

    quant_model = quantizer.quant_model
    quant_model = quant_model.to(device)
    #####################################################################################
  normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
  dataset_ = dataset.PEdataset_test(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
  criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
  # to get loss value after evaluation
  val_loader = load_data(cfg, subset_len=450,
              sample_method='random')
  validate(cfg, val_loader, dataset_, quant_model, criterion, '/workspace/GSAT_12R_FIN_TEST/PEdataset/my_hrnet768/cfg768/model_best.pth', None)

  # handle quantization result
  if quant_mode == 'calib':
    quantizer.export_quant_config()
  if deploy:
    print("here")
    quantizer.export_torch_script()
    print("here1")
    quantizer.export_onnx_model()
    print("here2")
    quantizer.export_xmodel(deploy_check=False)


#####################################################################################################################################################################################################
if __name__ == '__main__':

  update_config(cfg, args)
  
  model_name = 'model_best'
  file_path = os.path.join(args.model_dir, model_name + '.pth')

  feature_test = ' float model evaluation'
  if args.quant_mode != 'float':
    feature_test = ' quantization'
    # force to merge BN with CONV for better quantization accuracy
    args.optimize = 1
    feature_test += ' with optimization'
  else:
    feature_test = ' float model evaluation'
  title = model_name + feature_test

  print("-------- Start {} test ".format(model_name))

  # calibration or evaluation
  quantization(file_path=file_path)

  print("-------- End of {} test ".format(model_name))














