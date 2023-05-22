from headers import *
from config import *



class dataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, num_classes):
        if isinstance(annotations_file, str):
            annotations_file = pd.read_csv(annotations_file)
        self.images_csv = annotations_file.reset_index(drop=True) #pd.read_csv(annotations_file).reset_index(drop=True)
        self.img_dir = img_dir
        self.num_classes = num_classes
        self.transforms_ = tv.transforms.Compose([
            tv.transforms.Resize(40),
            # tv.transforms.CenterCrop(40),
            tv.transforms.ConvertImageDtype(torch.float),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    # def transforms_(self, image):
    #     # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    #     # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    #     normed= cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #     # skeleton = cv2.ximgproc.thinning(thresh, None, 1)
    #     image = cv2.resize(image, (100, 100))
    #     return image

    def __len__(self):
        return len(self.images_csv)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images_csv.iloc[index, 0])
        image = tv.io.image.read_image(img_path)
        # image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image_ = self.transforms_(image) #.to(device_)
        # image = torch.from_numpy(image_)
        label = self.images_csv.iloc[index, 1]
        # label = torch.nn.functional.one_hot(torch.tensor(label).to(torch.int64), num_classes=self.num_classes)
        return image.unsqueeze(0), label
    
    
def dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)