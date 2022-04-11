import torch
import os
import yaml
import json
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import argparse

print("\nModules import completed...")
print(" - Current Directory ------------ \n>>> {}".format(os.getcwd()))
FILE = Path(os.getcwd()).absolute() if Path(os.getcwd()).name=='yolov5' else str(Path(os.getcwd()).absolute() / 'yolov5')
print(" - Predict Directory ------------ \n>>> {}".format(FILE))

# Google Colab
print("-"*30+"\n ### Detect Google Colab Environment...")
isColab = True
'''
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ModuleNotFoundError as e:
    print("{},it's not in Colab.".format(e))
    isColab = False
'''
print(" - Is Google Colab -------------- {}".format(isColab)+'\n')

# CUDA Check
print("-"*30+"\n ### CUDA Check...")
print(" - Torch CUDA available --------- {}".format(torch.cuda.is_available()))
print(">>> "+'Using torch %s %s'%(
    torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'
)+'\n')

try:
    from models.experimental import attempt_load
    from utils.general import increment_path
    from utils.torch_utils import select_device
    from detect import run
except ImportError as e:
    print("'detect-modify-v2.py' is not in the directory 'yolov5' which git clone from 'ultralytics/yolov5'")

# Basic Function
def universal_cmd(cmd= ""):
    # try:
    #     try:
    #         get_ipython().magic("%"+cmd)
    #     except :
    #         get_ipython().system(cmd)
    # except:
    os.system(cmd)
    
# Environment Init
def InitRequirements(DO=False, wandbLogin=False):
    if DO:
        if input(
            "We found you seem to set 'ultralytics/yolov5' environment,\n"+
            "This will make a new folder '/yolov5' to save requirement program,\n"+
            "and install requirement environment in 'yolov5/requirements.txt'.\n"+
            "If you want to install, type 'install' to continue\n"+
            "type any key to cancel installation.\n"+
            "(This asking wants to make sure it's not a mistake.)\n>>> "
        ) == 'install':
            print("Going to installation...")
        else: 
            print("Cancel installation...")
            return None
    else:
        return None
    os.getcwd()
    
    print("-"*30+"\n ### Module Installation")
    universal_cmd('rm -r "yolov5"')
    universal_cmd('git clone https://github.com/ultralytics/yolov5')
    universal_cmd('pip install -U -r yolov5/requirements.txt --user')
    if wandbLogin:
        universal_cmd('pip install wandb')
        universal_cmd('wandb login')
    universal_cmd('cd ./yolov5')
    universal_cmd("pip install pytorch torchvision -U")

def InitUltralytics(tgtFolder):
    print("-"*30+"\n ### Ultralytics YOLOv5 Environment")
    tgtFolder = Path(tgtFolder)
    print(tgtFolder)
    if tgtFolder.exists():
        os.chdir(tgtFolder)
        if not (tgtFolder / 'detect.py').exists():
            raise FileNotFoundError("Couldn't find 'detect.py' in %s" % tgtFolder)
        if not (tgtFolder / 'train.py').exists():
            raise FileNotFoundError("Couldn't find 'train.py' in %s" % tgtFolder)
        if not (tgtFolder / 'models' / 'yolo.py').exists():
            raise FileNotFoundError(
                "Couldn't find the folder 'model' in %s,"\
                " the git clone may have some error" % tgtFolder)
        print("Checking Completed")
    else:
        InitRequirements(True)
    
    from models.experimental import attempt_load
    from utils.general import increment_path
    from utils.torch_utils import select_device
    from detect import run
    print("Import requiring function from ultralytics YOLOv5.")
    
    print("Change Directory...")
    os.chdir(tgtFolder)
    print(">>> {}".format(os.getcwd()))
    
def MatchFile(Matching, BeMatched, show_match_result=False):
    Unmatch = 0
    unmatchList = []
    Matching = Path(Matching)
    BeMatched = Path(BeMatched)
    for ImgName in glob.iglob(str(Matching / '*')):
        print("-"*20) if show_match_result else None
        ImgName = Path(ImgName)
        print(ImgName.name) if show_match_result else None
        if (BeMatched / ImgName.name).exists:
            print("Match success.") if show_match_result else None
        else:
            print("Match fail.") if show_match_result else None
            unmatchList.append(ImgName.name)
            Unmatch += 1
    print("#"*20) if show_match_result else None
    print("Processing End...") if show_match_result else None
    print("Unmatch File ------- {}".format(Unmatch))
    print("'{}' dosn't have the following files compared from '{}'.".format(BeMatched, Matching)) if unmatchList else None
    [ print(l) for l in unmatchList ]
    return unmatchList

def RawedCatchClass(weights):
    # copy from ultrralytics/yolov5/detect.py, the part works on model loading.
    # because there is no other way can get info of classes without extra file.
    # since '.pt' is a format of yolov5 result, it should be less updated,
    # in other word, it can avoid program no longer working due to frequently update from ultralytics.
    half = False
    device = select_device('cpu')
    
    w = weights[0] if isinstance(weights, list) else weights
    suffix = Path(w).suffix.lower()
    pt, onnx, tflite, pb, graph_def = (suffix == x for x in ['.pt', '.onnx', '.tflite', '.pb', ''])  # backend
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        return names
    else:
        print("we can only get class name from '.pt'.")
        return None


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--tfl-int8', action='store_true', help='INT8 quantized TFLite model')
    
    parser.add_argument('--cropname-full', action='store_true', help='use the name with more description')
    parser.add_argument('--yolov5-locate', default=str(
        str(os.getcwd()) + ('/' if Path(os.getcwd()).name=='yolov5' else 'yolov5/' ) 
    ), help='INT8 quantized TFLite model')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

def main(opt):
    all_vars = vars(opt)
    print(all_vars)
    '''
    {
        'weights': 'yolov5s.pt', 'source': 'data/images', 'imgsz': [640, 640],
        'conf_thres': 0.25, 'iou_thres': 0.45, 'max_det': 1000, 'device': '',
        'view_img': False, 'nosave': False, 'classes': None, 'agnostic_nms': False,
        'augment': False, 'update': False, 'project': 'runs/detect',
        'name': 'exp', 'exist_ok': False, 'line_thickness': 3,
        'hide_labels': False, 'hide_conf': False, 'half': False,
        'tfl_int8': False, 'cropname_full': False, 'yolov5_locate': '.'
    }
    '''
    ## Environment
    UltralyticsYOLOv5Folder = all_vars['yolov5_locate']
    WandbInstall = False

    ## Make Json File
    Project = all_vars['project']
    Increment = all_vars['exist_ok']

    os.chdir(Project)
    all_vars['name'] = increment_path(Path(str(os.getcwd())) / all_vars['name'], exist_ok= False).name
    os.chdir('../..')

    # print(" - Current Directory ------------ \n>>> {}".format(os.getcwd()))

    all_vars['name'] = increment_path(Path(all_vars['name']), exist_ok=Increment)
    OutputFolderName = all_vars['name']
    ImgSize = all_vars['imgsz']
    ConfOverLowExclude = all_vars['conf_thres']
    DetectSourceFolder = Path(all_vars['source'])
    WeightFilePath = Path(all_vars['weights'][0])
    names = RawedCatchClass(WeightFilePath)
    ImagesFolder = Path(UltralyticsYOLOv5Folder) / Project / OutputFolderName / ''
    LabelsFolder = ImagesFolder / 'labels'

    save_dir = ''
    AllLabel = {}
    ## Edge Cropped
    FullCropName = all_vars['cropname_full']
    CropFolder = ImagesFolder / 'crops'

    ## Run `'detect.py'`
    run_vars = all_vars.copy()
    run_vars.pop('cropname_full')
    run_vars.pop('yolov5_locate')
    run_vars['save_txt'] = True
    run_vars['save_conf'] = True
    run_vars['save_crop'] = False

    InitUltralytics(UltralyticsYOLOv5Folder)
    #########################
    # ReadyDetect
    print("-"*30+"\n ### Run `'detect.py'`")
    save_dir = increment_path(Path(Project) / OutputFolderName, exist_ok=Increment)
    OutputFolderName = save_dir.name
    print(" - Project(project) ------------- {}\n>>> {}".format(os.path.exists(Project), Project))
    print(" - Increment(exists_ok) --------- {}".format(Increment))
    print(" - OutputFolderName(name) ------- {}\n>>> {}".format(save_dir.name, (
        'Do increment since this name has been used.' if Increment else 'Replace existed folder.'
    )))
    print(" - ImgSize(size) ---------------- {}".format(ImgSize))
    print(" - ConfOverLowExclude(conf) ----- {}".format(ConfOverLowExclude))
    if os.path.exists(DetectSourceFolder):
        print(" - DetectSourceFolder(source) --- {}\n>>> {}".format(DetectSourceFolder.exists(), DetectSourceFolder))
    else:
        raise FileNotFoundError("Folder no found, Please Check the directory of the location of original images.")
    print(" - WeightFilePath(weight) ------- {}\n>>> {}".format(WeightFilePath.exists(), WeightFilePath))

    ## Run `detect.py`
    run(**run_vars)

    #########################
    ## ReadyWriteJson
    print("-"*30+"\n ### Write Json File")
    print(" - ImagesFolder ----------------- {}\n>>> {}".format(ImagesFolder.exists(), ImagesFolder))
    print(" - LabelsFolder ----------------- {}\n>>> {}".format(LabelsFolder.exists(), LabelsFolder))
    print(" - LabelsFolder ----------------- \n>>> {}".format(names))

    ## Make Json File
    j1 = 0
    null_class_img = []
    for img in glob.iglob(str(ImagesFolder)+'/*'):
        img = Path(img)
        if img.suffix in ['.jpg', '.png', '.jpeg']:
            imgName = Path(img.name)
            imgName.with_suffix('')
            labelTxt = ImagesFolder / 'labels' / (str(imgName.with_suffix(''))+'.txt')
        
            AllLabel[j1] = {
                "name": imgName.name,
                "label_file": labelTxt.name if labelTxt.exists() else None,
                "class": {}
            }
            print("-"*20)
            print("{}: {} -> ./label/{}".format(j1, imgName, labelTxt.name))
            try:
                with open(labelTxt, 'r') as content:
                    j2=0
                    for line in content.readlines():
                        line = line.replace('\n', '').split(' ')
                        subclass_num = int(line[0])
                        print(line)
                        AllLabel[j1]["class"][j2] = {
                            "file_name_full": '{}.cut.{}.{}.jpg'.format(
                                imgName.with_suffix('').name, str(j2).rjust(3, '0'), names[subclass_num]),
                            "file_name_serial_num": "{}-{}.jpg".format(
                                imgName.with_suffix('').name, str(j2).rjust(3, '0')),
                            "subclass_str": names[subclass_num],
                            "subclass_num": subclass_num,
                            "loc": [float(loc) for loc in line[1:5]],
                            "score": float(line[5])
                        }
                        j2 += 1
            except FileNotFoundError as e:
                AllLabel[j1]["class"] = {}
                null_class_img.append(labelTxt)
                print(e, "This image has no any target to be recongnized.")
                
            labelJson = ImagesFolder / 'labels' / (str(imgName.with_suffix(''))+'.json')
            with open(labelJson, 'a', encoding='utf-8') as content:
                json.dump(AllLabel[j1], content, indent=2, ensure_ascii=False)
            j1 += 1
    with open((ImagesFolder / 'AllLabel.json' ), 'w', encoding='utf-8') as content:
        json.dump(AllLabel, content, indent=2, ensure_ascii=False)
        
    with open((ImagesFolder / "LabelClass.json"), 'w', encoding='utf-8') as jfile:
        json.dump({ 'class': names }, jfile, ensure_ascii=False, indent=2)
    
    print("#"*20)
    print("Processing End...")
    print("#"*20)

    #########################
    ## ReadyCropImg
    print("-"*30+"\n ### Edge Location Get and Cut")
    print(" - Project(project) ------------- {}\n>>> {}".format(os.path.exists(Project), Project))
    print(" - OutputFolderName(name) ------- {}\n>>> {}".format(save_dir.name, (
        'Do increment since this name has been used.' if Increment else 'Replace existed folder.'
    )))
    print(" - DetectSourceFolder(source) --- \n>>> {}".format(DetectSourceFolder))
    for f in ['AllLabel.json', 'LabelClass.json']:
        if (ImagesFolder / f).exists():
            print("'{}' has found".format(f))
        else:
            raise FileNotFoundError("File no found, Please Check the directory of '{}', or whether it has been export correctly.".format(f))
    # Original File Completation Checking
    print("-"*30+"\n ### Original File Completation Checking")
    print("Matching:", str(DetectSourceFolder))
    print("BeMatched:", str(ImagesFolder))
    UnMatchedList = MatchFile(str(DetectSourceFolder), str(ImagesFolder))
    print("These files will be skipped when crop the image." if UnMatchedList else "No file lost, all images will be export.")
    print("Make folder 'crops'.")
    print("The folder has existed.") if CropFolder.exists() else os.mkdir(CropFolder)

    ## Edge Location Get and Cut      
    for k in AllLabel.keys():
        print("-"*20)
        imgName = AllLabel[k]['name']
        imgLabelPath = ImagesFolder / imgName
        imgOriginPath = DetectSourceFolder / imgName
        
        imgBeCroppedPath = imgOriginPath if imgOriginPath.exists() else imgLabelPath
        print(f"Using {'original' if imgOriginPath.exists() else 'label'} image as source.")
        imgBeCropped = cv2.imread(str(imgBeCroppedPath), cv2.IMREAD_UNCHANGED)
        print(imgName)
        
        for k2 in AllLabel[k]['class'].keys():
            cropinfo = AllLabel[k]['class'][k2]
            # print(cropinfo)
            [relX, relY, relW, relH] = cropinfo['loc']
            print("\n>>> {}: {}, {}".format(k2, cropinfo['subclass_str'], [relX, relY, relW, relH]))
            [absX, absY, halfAbsW, halfAbsH] = [
                relX*imgBeCropped.shape[1], relY*imgBeCropped.shape[0],
                relW*imgBeCropped.shape[1]/2, relH*imgBeCropped.shape[0]/2
            ]
            afterCrop = imgBeCropped[ int(absY-halfAbsH):int(absY+halfAbsH), int(absX-halfAbsW):int(absX+halfAbsW) ]
            plt.imshow(afterCrop)
            fileName = cropinfo["file_name_full"] if FullCropName else cropinfo["file_name_serial_num"]
            # fileName = Path(imgName).with_suffix('').name + '.cut.{}.{}'.format(classinfo[0], classinfo[1]) + Path(AllLabel['10']['name']).suffix
            print(">>> {}\n".format(fileName))
            print(str(CropFolder / fileName))
            cv2.imwrite(str(CropFolder / fileName), afterCrop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("#"*20)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)