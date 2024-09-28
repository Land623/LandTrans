import os
import argparse
import json
import numpy as np

from utils.Raster import validate_LUCC_files, compare_geotiff_properties, read_tif_info
from utils.DataEnvironment import DataEnvironment


def parse_args():
    parser = argparse.ArgumentParser(description="LandTrans")
    # model define
    parser.add_argument("--seq_len", type=int, default=6, help="")
    parser.add_argument("--label_len", type=int, default=6, help="")
    parser.add_argument("--pred_len", type=int, default=6, help="")
    parser.add_argument("--batch_size", type=int, default=16, help="")
    parser.add_argument("--spatial_patch_size", type=int, default=16, help="")
    parser.add_argument("--temporal_patch_size", type=int, default=2, help="")
    parser.add_argument(
        "--slide_interval",
        type=int,
        default=2,
        help="The interval at which time series data is sampled using a sliding window approach",
    )
    parser.add_argument("--embed_dim", type=int, default=32, help="")
    parser.add_argument("--num_heads", type=int, default=4, help="")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="")
    parser.add_argument("--dropout", type=float, default=0.1, help="")
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=128,
        help="A common starting point is 2-4 times the input embed_dim",
    )
    parser.add_argument("--epochs", type=int, default=10, help="")
    parser.add_argument("--patience", type=int, default=10, help="")

    parser.add_argument("--region_size", type=int, default=256, help="")

    # data loader
    parser.add_argument(
        "--lucc_data_path",
        type=str,
        default="E:/Coding/LUCC_DATA",
        help="path of the lucc data files",
    )
    parser.add_argument(
        "--mask_area",
        type=str,
        default="",
        help="if a location is located in the mask area, it will not be processed during the simulation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="E:/Coding/Result",
        help="the folder used to store the output data of the model",
    )
    return parser


def save_args_to_json(args, filename):
    with open(filename, "w") as f:
        json.dump(vars(args), f, indent=4)


def load_args_from_json(filename):
    args = parse_args()
    with open(filename, "r") as f:
        args_dict = json.load(f)
    args.set_defaults(**args_dict)
    args = args.parse_args()
    return args


def args_validation(args):
    result = {
        "status": "success",
        "message": "",
        "number_of_lucc_data_files": 0,
        "lucc_file_info": {},
        "land_use_class": [],
    }
    chk_lu_results = check_land_use_data(args)
    if chk_lu_results["status"] != "success":
        result["status"] = "error"
        result["message"] = chk_lu_results["message"]
        return result
    total_time_point = chk_lu_results["number_of_lucc_data_files"]
    all_lucc_file_info = chk_lu_results["lucc_file_info"]
    lucc_class_info = []
    for lucc_file_info in all_lucc_file_info.values():
        lucc_class_info.extend(lucc_file_info["unique_values"])
    lucc_class_info_set=set(lucc_class_info)
    if 0 in lucc_class_info_set:
        lucc_class_info_set=lucc_class_info_set.remove(0)
    lucc_class_info = list(lucc_class_info_set)
    result["land_use_class"] = lucc_class_info
    lucc_class_count = len(lucc_class_info)
    if args.epochs <= 0:
        result["status"] = "error"
        result["message"] = "Invalid configuration: Number of epochs must be positive."
        return result
    if args.batch_size <= 0:
        result["status"] = "error"
        result["message"] = "Invalid configuration: Batch size must be positive."
        return result
    
    if not (0 < args.learning_rate < 0.01):
        result["status"] = "error"
        result["message"] = (
            "Invalid configuration: Learning rate must be between 0 and 0.01."
        )
        return result
    if args.label_len < 2:
        result["status"] = "error"
        result["message"] = (
            "Invalid configuration: label_len must be greater than 1."
        )
        return result
    if args.seq_len < 2:
        result["status"] = "error"
        result["message"] = (
            "Invalid configuration: seq_len must be greater than 1."
        )
        return result
    if args.seq_len < args.label_len:
        result["status"] = "error"
        result["message"] = (
            "Invalid configuration: seq_len cannot be less than label_len."
        )
        return result
    if not (0 < args.dropout <= 0.3):
        result["status"] = "error"
        result["message"] = "Invalid configuration: dropout must be between 0 and 0.3."
        return result
    if not (0 < args.num_encoder_layers <= 32):
        result["status"] = "error"
        result["message"] = (
            "Invalid configuration: num_encoder_layers must be between 1 and 32."
        )
        return result
    if not (0 < args.num_decoder_layers <= 32):
        result["status"] = "error"
        result["message"] = (
            "Invalid configuration: num_decoder_layers must be between 1 and 32."
        )
        return result
    if not (0 < args.patience < 20):
        result["status"] = "error"
        result["message"] = "Invalid configuration: patience must be between 1 and 20."
        return result
    if not (0 < args.seq_len or args.seq_len < total_time_point):
        result["status"] = "error"
        result["message"] = (
            "Invalid configuration: seq_len must greater than 0, and smaller than the total periods of LUCC data."
        )
        return result
    if not (0 < args.label_len or args.label_len < total_time_point):
        result["status"] = "error"
        result["message"] = (
            "Invalid configuration: label_len must greater than 0, and smaller than the total periods of LUCC data."
        )
        return result
    if not (0 < args.label_len * 2 + args.seq_len <= total_time_point):
        result["status"] = "error"
        result["message"] = (
            "Invalid configuration: Invalid combination of label_length and sequence_length parameters. The following condition must be satisfied: 0 < 2 * label_length + sequence_length < total_time_points. Please adjust the values of label_length and sequence_length, or increase the number of time steps in your land use data."
        )
        return result
    if not (0 < args.pred_len):
        result["status"] = "error"
        result["message"] = "Invalid configuration: pred_len must be greater than 0."
        return result    
    if not ( 32<= args.region_size <=2048 ):
        result["status"] = "error"
        result["message"] = "Invalid configuration: region_size must be between 32 and 2048"
        return result
    
    if not (0 < args.spatial_patch_size <= 32):
        result["status"] = "error"
        result["message"] = "Invalid configuration: pred_len must be between 1 and 32"
        return result        
    if not (args.region_size % args.spatial_patch_size ==0 ):
        result["status"] = "error"
        result["message"] = "Invalid configuration:The provided region size must be a multiple of the spatial window size."
        return result   
    if args.model_type == 3: 
        if not (args.label_len % args.temporal_patch_size ==0 ):
            result["status"] = "error"
            result["message"] = "Invalid configuration:The provided label length must be a multiple of the temporal window size."
            return result
        if not (args.seq_len % args.temporal_patch_size ==0 ):
            result["status"] = "error"
            result["message"] = "Invalid configuration:The provided sequence length must be a multiple of the temporal window size."
            return result    
    
        if not (
            0 < args.temporal_patch_size 
            or args.temporal_patch_size <= args.seq_len
            or args.temporal_patch_size <= args.label_len
        ):
            result["status"] = "error"
            result["message"] = (
                "Invalid configuration: temporal_window_size must be smaller than seq_len and label_len."
            )
            return result
  
    if not (4 <= args.embed_dim <= 255 or args.embed_dim > lucc_class_count):
        result["status"] = "error"
        result["message"] = (
            "Invalid configuration: embed_dim dim_feedforward must be between 1 and 8 times of the number of land use classes."
        )
        return result    
    if not (args.embed_dim % args.num_heads ==0 ):
        result["status"] = "error"
        result["message"] = "Invalid configuration:The provided embed_dim must be a multiple of the num_heads."
        return result  
    if not (
        0 < args.dim_feedforward <= 10240000
        or args.dim_feedforward / args.embed_dim < 6
    ):
        result["status"] = "error"
        result["message"] = (
            "Invalid configuration: dim_feedforward must be between 2 and 4 times of embed_dim"
        )
        return result
    result["number_of_lucc_data_files"] = chk_lu_results["number_of_lucc_data_files"]
    result["lucc_file_info"] = chk_lu_results["lucc_file_info"]
    return result


def check_land_use_data(args):
    result = {
        "status": "success",
        "message": "",
        "validate": True,
        "number_of_lucc_data_files": 0,
        'lucc_file_info':{}    
    }
    lucc_data_path = args.lucc_data_path
    lucc_data_list = os.path.join(lucc_data_path, "data_list.txt")
    if not os.path.exists(lucc_data_list):
        result["status"] = "error"
        result["message"] = f"{lucc_data_list} does not exists."
        result["validate"] = False
        return result
    val_lucc_result = validate_LUCC_files(lucc_data_path, lucc_data_list)
    if val_lucc_result["status"] != "success":
        result["status"] = "error"
        result["message"] = val_lucc_result["message"]
        result["validate"] = False
        return result
    if val_lucc_result["number_of_files"] < 4:
        result["status"] = "error"
        result["message"] = (
            "Invalid configuration: This model requires at least 4 periods of LUCC data."
        )
        result["validate"] = False
        return result

    base_line_data_info = val_lucc_result["base_line_property"]
    if args.mask_area != "" and os.path.exists(args.mask_area):
        read_mask_result = read_tif_info(args.mask_area)
        if read_mask_result["status"] != "success":
            result["status"] = "error"
            result["message"] = read_mask_result["message"]
            return result
        else:
            mask_prop = read_mask_result["raster_properties"]
            cmp_mask_result = compare_geotiff_properties(base_line_data_info, mask_prop)
            if cmp_mask_result["status"] != "success":
                result["status"] = "error"
                result["message"] = cmp_mask_result["message"]
                return result
            if mask_prop["band_count"] > 1:
                result["status"] = "error"
                result["message"] = (
                    f"Invalid configuration: Restricted area data used for simulation must be a single band geo-tiff file."
                )
                return result
            if not np.issubdtype(mask_prop["dtype"], np.integer):
                result["status"] = "error"
                result["message"] = (
                    f"Invalid configuration: Restricted area data used for simulation must be of integer data type."
                )
                return result

    result["number_of_lucc_data_files"] = val_lucc_result["number_of_files"]
    result["lucc_file_info"] = val_lucc_result["tif_info_dict"]
    return result
