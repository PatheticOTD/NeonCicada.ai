
class Config:
    
    yolo_model_path = "src/models/yolo_model.pt"
    
    img_path_test = 'data/validate/Hugh Jackman/Hugh Jackman17293.jpg'

    encodings_facenet = 'data/encodings(facenet).pkl'
    train_labels_facenet = 'data/train_labels(facenet).pkl'
    train_imgs_facenet = 'data/train_imgs.pkl'
    
    
    encodings_path_arc = 'data/encodings(arcface).pkl'
    labels_path_arc = 'data/train_labels(arcface).pkl'
    train_imgs_arc = 'data/train_imgs(arcface).pkl'
    