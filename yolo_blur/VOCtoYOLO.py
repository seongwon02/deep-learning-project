import os
import xml.etree.ElementTree as ET
import glob

def convert(size, box):
    # YOLO format uses normalized coordinates (0 to 1)
    dw = 1. / size[0]
    dh = 1. / size[1]
    
    # Center of the bounding box
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    
    # Width and height of the bounding box
    w = box[1] - box[0]
    h = box[3] - box[2]
    
    # Normalize
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_path, classes):
    try:
        in_file = open(xml_path, 'r', encoding='utf-8')
        tree = ET.parse(in_file)
        root = tree.getroot()
        
        size_elem = root.find('size')
        if size_elem is None:
            return
            
        w = int(size_elem.find('width').text)
        h = int(size_elem.find('height').text)

        txt_path = xml_path.replace('.xml', '.txt')
        out_file = open(txt_path, 'w', encoding='utf-8')

        for obj in root.iter('object'):
            difficult = obj.find('difficult')
            difficult_val = difficult.text if difficult is not None else '0'
            if int(difficult_val) == 1:
                continue
                
            cls = obj.find('name').text
            if cls not in classes:
                continue
                
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        
        in_file.close()
        out_file.close()
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")

if __name__ == '__main__':
    # CrowdHuman Face dataset has 'face' class
    classes = ['face']
    
    # Target the downloaded dataset's train folder
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset/crowdhuman-face/CrowdHuman Face Voc/train')
    
    if not os.path.exists(dataset_path):
        print(f"경로를 찾을 수 없습니다: {dataset_path}")
        exit(1)
        
    xml_files = glob.glob(os.path.join(dataset_path, '*.xml'))
    print(f"총 {len(xml_files)}개의 XML 파일을 찾았습니다. 변환을 시작합니다...")
    
    for xml_file in xml_files:
        convert_annotation(xml_file, classes)
        
    print("VOC(.xml)에서 YOLO(.txt) 형식으로의 변환이 완료되었습니다!")
