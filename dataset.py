from headers import *
from config import *
from preprocess.feature_generator import get_histogram_pixels



class dataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, num_classes, lstm=False):
        if isinstance(annotations_file, str):
            annotations_file = pd.read_csv(annotations_file)
        self.images_csv = annotations_file.reset_index(drop=True) #pd.read_csv(annotations_file).reset_index(drop=True)
        self.img_dir = img_dir
        self.num_classes = num_classes
        self.lstm = lstm
        self.transforms_ = tv.transforms.Compose([
            tv.transforms.Resize(40),
           
            tv.transforms.ConvertImageDtype(torch.float),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tv.transforms.Grayscale(num_output_channels=1) #if lstm else  tv.transforms.CenterCrop(40)
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
        label = torch.nn.functional.one_hot(torch.tensor(label).to(torch.int64), num_classes=self.num_classes)
        if self.lstm:
            image_linear_arr = torch.nn.Flatten()(image)
            return image_linear_arr.unsqueeze(0), label   
        image_linear_arr = torch.nn.Flatten()(image_)
        
        return image_linear_arr, label #image_.unsqueeze(0), label


class HistDataset(dataset):
    def __init__(self, annotations_file, img_dir, num_classes, lstm=False):
        super().__init__(annotations_file, img_dir, num_classes, lstm)
        self.transforms_ = get_histogram_pixels

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images_csv[index, 0])
        image = cv2.imread(img_path)
        image_ = self.transforms_(image)
        label = self.images_csv.iloc[index, 1]
        label = torch.nn.functional.one_hot(torch.tensor(label).to(torch.int64), num_classes=44)
        return torch.from_numpy(np.array(image_)), label

    
# defining dataset
class PreGeneratedDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        label = self.dataframe.iloc[index, 1].astype(dtype=int)
        features = self.dataframe.iloc[index, 2:].to_numpy()
        label = torch.nn.functional.one_hot(torch.tensor(label).to(torch.int64), num_classes=44)
        return torch.from_numpy(features.astype("float")), label
    

def dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)