import cv2, json, random, os, PIL
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances

if __name__ == "__main__":
    imagesDir = '/datasets/Corn_syn_dataset/corn_dataset_v2/camera_main_camera/rect'
    ogAnnsDir = '/datasets/Corn_syn_dataset/corn_dataset_v2/camera_main_camera_annotations/bounding_box'

    datasetName = 'syn_coco_box'
    newAnnsDir = f'/datasets/Corn_syn_dataset/corn_dataset_v2/{datasetName}'

    newAnns = {}

    newAnns['info'] = {'description': f'Synthetic dataset ({datasetName})',
                       'url': 'None',
                       'version': 'None',
                       'year': '2022',
                       'contributor': 'None',
                       'date_created': 'None'}

    newAnns['licenses'] = [{'url': 'None', 'id': 1, 'name': 'None'}]

    newAnns['categories'] = [{'supercategory': 'plants', 'id': 1, 'name': 'Weeds'},
                             {'supercategory': 'plants', 'id': 2, 'name': 'Maize'},
                             {'supercategory': 'residue', 'id': 3, 'name': 'Bark'}]

    newAnns['images'] = []
    newAnns['annotations'] = []
    annsId = 0
    cat_id = 0
    for imageFile in os.listdir(imagesDir):
        imageId = int(imageFile.split('.')[0])

        width, height = PIL.Image.open(os.path.join(imagesDir, imageFile)).size
        newAnns['images'].append({'license': 1,
                                  'file_name': imageFile,
                                  'coco_url': 'None',
                                  'height': height,
                                  'width': width,
                                  'date_captured': 'None',
                                  'flickr_url': 'None',
                                  'id': imageId})

        with open(os.path.join(ogAnnsDir, '{:04}.txt'.format(imageId)), 'r') as f:
            for line in f.readlines():
                newBbox = [float(i) for i in line[:-1].split(' ')[1:]]
                newBbox[0] *= width
                newBbox[1] *= height
                newBbox[2] *= width
                newBbox[3] *= height
                newBbox[0] = newBbox[0] - newBbox[2] / 2
                newBbox[1] = newBbox[1] - newBbox[3] / 2
                newBbox = [round(i, 2) for i in newBbox]
                if int(float(line.split(' ')[0])) == 1:
                    cat_id = 2
                elif int(float(line.split(' ')[0])) == 2:
                    cat_id = 1
                else:
                    cat_id = 3
                newAnns['annotations'].append({'segmentation': [],
                                               'area': newBbox[2] * newBbox[3],  # Not sure about this
                                               'iscrowd': 0,
                                               'image_id': imageId,
                                               'bbox': newBbox,
                                               'category_id': cat_id,
                                               'id': annsId})
                annsId += 1
    os.makedirs(newAnnsDir, exist_ok=True)
    with open(os.path.join(newAnnsDir, f'{datasetName}NewAnnotations.json'), 'w+') as f:
        json.dump(newAnns, f)

    register_coco_instances("new_dataset", {}, os.path.join(newAnnsDir, f'{datasetName}NewAnnotations.json'), imagesDir)
    my_dataset_metadata = MetadataCatalog.get("new_dataset")
    dataset_dicts = DatasetCatalog.get("new_dataset")

    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_metadata, scale=2)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow('image', vis.get_image()[:, :, ::-1])
        cv2.waitKey()
        cv2.destroyAllWindows()