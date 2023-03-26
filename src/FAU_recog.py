import cv2
import time
import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from face.FAU_net.resnet_multi_view import ResNet_GCN_two_views

fusion_mode = 0
database = 0
use_web = 0

AU_num = 12
AU_idx = [0,1,2,3,4,5,6,7,8,9,10,11]
net = ResNet_GCN_two_views(AU_num=AU_num, AU_idx=AU_idx, output=2,
                           fusion_mode=fusion_mode, database=database)

model_path = './face/EmotioNet_model.pth.tar'
temp = torch.load(model_path)
net.load_state_dict(temp['net'])
net.cuda()

transform_test = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5355, 0.4249, 0.3801), (0.2832, 0.2578, 0.2548)),
])

def facial_action_unit_recog(img):
    img = transform_test(img)
    img = Variable(img).cuda()
    img = img.view(1, img.size(0), img.size(1), img.size(2))

    start_time = time.time()
    AU_view1, AU_view2, AU_fusion = net(img)
    print(time.time() - start_time)

    AU_view1 = torch.sigmoid(AU_view1)
    AU_view2 = torch.sigmoid(AU_view2)

    AU = (AU_view1 + AU_view2)/2
    AU = AU[0].cpu().detach().numpy()
    return AU

if __name__ == '__main__':
    #img = Image.open('./test_img.jpg').convert('RGB')
    img = cv2.imread('./face/test_img.jpg')
    img = Image.fromarray(img)
    start_time = time.time()
    facial_action_unit_recog(img)
    print(time.time() - start_time)