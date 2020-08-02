from utils import *
from darknet import Darknet
import cv2
from torchvision import transforms
from patch.nn_modules import PatchApplier

def demo(cfgfile, weightfile, videofile):
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    class_names = load_class_names(namesfile)
 
    use_cuda = 1
    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(videofile)
    if not cap.isOpened():
        print("Unable to open camera")
        exit(-1)

    patch = Image.open('final_patch_w_alpha2.png').convert('RGBA')
    patch_t = transforms.ToTensor()(patch)
    patch_applier = PatchApplier()

    attack = True

    while True:
        res, img = cap.read()
        if res:
            img_detect = sized = Image.fromarray(img).resize((m.width, m.height))
            if attack:
                sized_t = transforms.ToTensor()(sized)
                applied = patch_applier(sized_t, patch_t[:3], patch_t[-1])
                img_detect = transforms.ToPILImage()(applied)
            bboxes = do_detect(m, img_detect, 0.5, 0.4, use_cuda)

            print('------')
            draw_img = plot_boxes_cv2(np.asarray(img_detect), bboxes, None, class_names)
            cv2.imshow(cfgfile, draw_img)
            cv2.waitKey(1)
        else:
             # print("Unable to read image")
             # exit(-1)
            break


############################################
if __name__ == '__main__':
    if len(sys.argv) == 3 or True:
        # cfgfile = sys.argv[1]
        # weightfile = sys.argv[2]
        cfgfile = 'cfg\yolo_v2-608.cfg'
        weightfile = 'weights\yolo_v2-608.weights'
        videofile = 'videos/stop2.mp4'
        demo(cfgfile, weightfile, videofile)
        #demo('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights')
    else:
        print('Usage:')
        print('    python demo.py cfgfile weightfile')
        print('')
        print('    perform detection on camera')
