import argparse
import copy

import mon
from model.nets import my_model
from utils.common import *
from utils.loss_util import *

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


def predict(args: argparse.Namespace):
    # General config
    data         = args.data
    save_dir     = args.save_dir
    weights      = mon.Path(args.weights)
    device       = mon.set_device(args.device)
    imgsz        = args.imgsz
    resize       = args.resize
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    use_fullpath = args.use_fullpath
    
    # Device
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.GENERAL["GPU_ID"]

    # Seed
    random.seed(args.GENERAL["SEED"])
    np.random.seed(args.GENERAL["SEED"])
    torch.manual_seed(args.GENERAL["SEED"])
    torch.cuda.manual_seed_all(args.GENERAL["SEED"])
    if args.GENERAL["SEED"] == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark     = True
    
    # Model
    model = my_model(
        en_feature_num = args.MODEL["EN_FEATURE_NUM"],
        en_inter_num   = args.MODEL["EN_INTER_NUM"],
        de_feature_num = args.MODEL["DE_FEATURE_NUM"],
        de_inter_num   = args.MODEL["DE_INTER_NUM"],
        sam_number     = args.MODEL["SAM_NUMBER"],
    ).to(device)
    if weights.is_ckpt_file():
        model_state_dict = torch.load(weights)["state_dict"]
    elif weights.is_weights_file():
        model_state_dict = torch.load(weights, weights_only=True)
    else:
        raise ValueError(f"Invalid weights file: {weights}")
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = mon.compute_efficiency_score(
            model      = copy.deepcopy(model),
            image_size = imgsz,
            channels   = 3,
            runs       = 1000,
            use_cuda   = True,
            verbose    = False,
        )
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.17f}")
        
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = data,
        dst         = save_dir,
        to_tensor   = True,
        denormalize = True,
        verbose     = False,
    )
    
    # Predicting
    timer = mon.Timer()
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                # Input
                meta       = datapoint.get("meta")
                image_path = mon.Path(meta["path"])
                image      = datapoint["image"]
                image      = image.to(device)
                
                # Resize
                _, _, h0, w0 = image.size()
                # if h0 != 2000 or w0 != 2992:
                #     image = mon.resize(image, [2000, 2992])
                
                # Infer
                timer.tick()
                # Pad image such that the resolution is a multiple of 32
                b, c, h, w = image.size()
                w_pad      = (math.ceil(w / 32) * 32 - w) // 2
                h_pad      = (math.ceil(h / 32) * 32 - h) // 2
                w_odd_pad  = w_pad
                h_odd_pad  = h_pad
                if w % 2 == 1:
                    w_odd_pad += 1
                if h % 2 == 1:
                    h_odd_pad += 1
                image = img_pad(image, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)
                out_1, out_2, out_3 = model(image)
                if h_pad != 0:
                    out_1 = out_1[:, :, h_pad:-h_odd_pad, :]
                if w_pad != 0:
                    out_1 = out_1[:, :, :, w_pad:-w_odd_pad]
                timer.tock()
                
                enhanced = out_1.detach().cpu()
                # if h0 != 2000 or w0 != 2992:
                #     enhanced = mon.resize(enhanced, [h0, w0])
                
                # Save
                if save_image:
                    if use_fullpath:
                        rel_path    = image_path.relative_path(data_name)
                        output_path = save_dir / rel_path.parent / f"{image_path.stem}.png"
                    else:
                        output_path = save_dir / data_name / f"{image_path.stem}.png"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    torchvision.utils.save_image(enhanced, str(output_path))
                

# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()

# endregion
