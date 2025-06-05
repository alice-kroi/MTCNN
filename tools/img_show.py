import matplotlib.pyplot as plt
from PIL import Image
import os

def visualize_annotation(img_path, x1, y1, x2, y2, cls=None, index=None, attributes=None):
    """
    可视化图像标注信息
    参数：
    img_path: 图片绝对路径
    x1,y1,x2,y2: 边界框坐标
    cls: 分类标签（可选）
    index: 样本索引（可选）
    attributes: 其他属性字典（可选）
    """
    # 读取图像
    img = Image.open(img_path)
    
    # 创建画布
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    # 绘制边界框
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                        linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    # 组合标注信息
    annotation = []
    if cls is not None:
        annotation.append(f"Class: {cls}")
    if index is not None:
        annotation.append(f"Index: {index}")
    if attributes:
        attr_str = '\n'.join([f"{k}: {v}" for k,v in attributes.items()])
        annotation.append(attr_str)
    
    # 添加文本标注
    if annotation:
        text = '\n'.join(annotation)
        ax.text(x1, y1-5, text, 
               color='white', 
               verticalalignment='bottom',
               bbox={'color': 'red', 'pad': 0})
    
    plt.axis('off')
    plt.show()
    plt.close()

# 示例用法
if __name__ == "__main__":
    # 示例参数
    img_path = "E:/test.jpg"
    x1, y1, x2, y2 = 100, 200, 300, 400
    visualize_annotation(img_path, x1, y1, x2, y2,
                       cls="face",
                       index=1,
                       attributes={"age": 25, "gender": "male"})