import torch
from pathlib import Path
import cv2
import os
import matplotlib.pyplot as plt
import glob
import sys

# 定义面包价格字典（
bread_prices = {
    "ball": 8.50,      # 圆面包
    "bean": 6.00,      # 豆沙面包
    "black": 9.00,     # 黑麦面包
    "castera": 7.50,   # 卡斯提拉面包
    "cookie": 8.00,    # 曲奇
    "cream": 10.50,    # 奶油面包
    "foot": 7.00,      # 法棍
    "guabaegi": 9.50,  # 花式甜面包
    "heart": 5.50,     # 心形面包
    "soboro": 6.50     # 酥粒面包
}

# 添加本地 YOLOv5 路径到系统路径
yolov5_path = Path(r"d:/Lynn/25/exp_2/bread")  # 调整为您的 YOLOv5 目录路径
sys.path.append(str(yolov5_path))

# 从本地 YOLOv5 导入所需模块
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes, check_img_size

# 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "runs/train/exp20/weights/best.pt"  # 模型路径，可以是相对路径或绝对路径
model = DetectMultiBackend(model_path, device=device, fuse=True)
model.eval()

# 设置推理参数
img_size = 416  # 推理时的图像大小
conf_thres = 0.25  # 置信度阈值
iou_thres = 0.45  # NMS 的 IoU 阈值
# 修正 stride 的获取方式
if isinstance(model.stride, (list, torch.Tensor)):
    stride = int(max(model.stride))  # 如果是列表或张量，取最大值
else:
    stride = int(model.stride)  # 如果是单一整数，直接使用
model.warmup(imgsz=(1, 3, img_size, img_size))  # 模型预热

# 获取用户输入
input_path = input("请输入要识别的图片路径（可以是单张图片路径或文件夹路径）: ").strip()
path_obj = Path(input_path)

# 判断是文件还是文件夹
if path_obj.is_file():
    image_paths = [path_obj]
elif path_obj.is_dir():
    # 支持的图片格式
    exts = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = [p for p in path_obj.iterdir() if p.suffix.lower() in exts]
    if not image_paths:
        print("该文件夹下没有图片")
        exit()
else:
    print("路径无效")
    exit()

# 动态获取最新的检测输出目录
def get_latest_exp_dir(base_dir='runs/detect'):
    exp_dirs = glob.glob(os.path.join(base_dir, 'exp*'))
    if not exp_dirs:
        return os.path.join(base_dir, 'exp')
    exp_nums = [int(d.split('exp')[-1]) for d in exp_dirs if d.split('exp')[-1].isdigit()]
    latest_exp = max(exp_nums) if exp_nums else ''
    return os.path.join(base_dir, f'exp{latest_exp}' if latest_exp else 'exp')

# 处理每张图片
for img_path in image_paths:
    print(f"\n正在检测: {img_path}")
    
    # 使用 LoadImages 加载和预处理图片
    dataset = LoadImages(str(img_path), img_size=img_size, stride=stride)
    
    # 获取类别名称
    names = model.names
    
    # 遍历数据集（通常只有一张图片）
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0  # 归一化
        if len(im.shape) == 3:
            im = im[None]  # 扩展为 (1, C, H, W)
        
        # 模型推理
        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=1000)
        
        # 处理检测结果
        detections = []
        for i, det in enumerate(pred):  # 每个批次
            p, im0 = path, im0s.copy()
            
            # 将检测框缩放到原始图像大小
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            
            # 保存结果图片
            save_path = str(Path(get_latest_exp_dir()) / Path(p).name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 在原始图像上绘制检测结果
            for *xyxy, conf, cls in det:
                label = f'{names[int(cls)]} {conf:.2f}'
                cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(im0, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                detections.append({
                    'class': names[int(cls)],
                    'confidence': float(conf),
                    'bbox': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                })
            
            # 保存结果
            cv2.imwrite(save_path, im0)
            print(f"检测结果已保存至: {save_path}")

    # 打印检测信息（模拟 results.print()）
    print(f"检测到 {len(detections)} 个对象")
    
    # 读取结果图片
    result_img_path = Path(get_latest_exp_dir()) / img_path.name
    img = cv2.imread(str(result_img_path))
    if img is None:
        print(f"无法加载结果图片: {result_img_path}")
        continue
    
    # 转换为RGB用于Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 计算总价
    total_price = 0
    price_list = []
    
    # 打印检测结果和价格
    if detections:
        print("\n检测到的对象:")
        for i, det in enumerate(detections, 1):
            bread_class = det['class']
            # 检查类别是否在价格字典中
            if bread_class in bread_prices:
                price = bread_prices[bread_class]
                total_price += price
                price_list.append(price)
                print(f"{i}. 类别: {bread_class}, 价格: ¥{price:.2f}, 置信度: {det['confidence']:.2f}, 边界框: {det['bbox']}")
            else:
                print(f"{i}. 类别: {bread_class}, 价格: 未知, 置信度: {det['confidence']:.2f}, 边界框: {det['bbox']}")
        
        # 输出总价
        print(f"\n总价: ¥{total_price:.2f}")
    else:
        print("未检测到任何对象。")

    # 使用OpenCV显示结果
    cv2.imshow(f"检测结果: {img_path.name} | 总价: ¥{total_price:.2f}", img)
    cv2.waitKey(0)  # 等待按键，0表示无限等待
    cv2.destroyAllWindows()  # 关闭所有窗口