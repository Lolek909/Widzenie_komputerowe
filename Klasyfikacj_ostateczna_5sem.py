import random

from torch.nn import CrossEntropyLoss
import cv2
import torch
from torch import nn
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Subset
from  torchvision import transforms, datasets
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
from ultralytics import YOLO
from norfair import Detection, Tracker, draw_tracked_boxes

target_size = (80, 80)

learning_rate = 0.0001
num_epochs = 25

classes = 3
co_ile_walidacja = 3

import torch
print(torch.__version__)  # Sprawdź wersję PyTorch


print(torch.cuda.is_available())  # Powinno zwrócić: True
print(torch.cuda.device_count())  # Powinno zwrócić: liczba GPU
print(torch.cuda.get_device_name(0))  # Powinno zwrócić: nazwę twojej karty graficznej


# git
class classification(nn.Module):
    def __init__(self):
        super(classification, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1),
            nn.SiLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, classes)
        )

    def forward(self, x):
        x = self.features(x)
        predicitons = self.classifier(x)

        return predicitons

def test():
    model = classification()
    x, y = target_size
    input_tensor = torch.randn(1, 3, x, y)
    # Próbne wejście
    output = model(input_tensor)
    print(output.shape)  # Powinno dać (1, num_classes)

# test()

def pil_loader(path):
    # Ładowanie obrazu z Pillow
    img = Image.open(path)

    # Jeśli obraz jest w trybie P (paleta) i ma przezroczystość
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")

    return img

def dataloading():
    # Ścieżki do datasetu
    train_dir = "C:/Users/mateu/Desktop/pojazdy_dataset/train"
    train_dir = "C:/Users/mateu/Desktop/datasety/Klasyfikacja_pojazdów/train"
    valid_dir = "C:/Users/mateu/Desktop/datasety/Klasyfikacja_pojazdów/valid"
    test_dir = "C:/Users/mateu/Desktop/datasety/Klasyfikacja_pojazdów/test"

    # Transformacje dla obrazów
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(80),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ]),
        'valid': transforms.Compose([
            transforms.Resize(90),
            transforms.CenterCrop(80),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Resize(90),
            transforms.CenterCrop(80),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ]),
    }

    # Wczytanie datasetów
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
    }

    def balance_dataset(dataset):
        # Policz próbki w każdej klasie
        labels = [label for _, label in dataset]
        class_counts = Counter(labels)
        min_class_size = min(class_counts.values())

        # Zbierz indeksy próbek do wyrównania
        balanced_indices = []
        for cls in class_counts.keys():
            class_indices = [i for i, (_, label) in enumerate(dataset) if label == cls]
            balanced_indices.extend(random.sample(class_indices, min_class_size))

        # Stwórz zrównoważony subset
        return Subset(dataset, balanced_indices)

    balanced_train_dataset = balance_dataset(image_datasets['train'])
    balanced_valid_dataset = balance_dataset(image_datasets['valid'])

    # Stworzenie DataLoaderów
    batch_size = 32
    dataloaders = {
        'train': DataLoader(balanced_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'valid': DataLoader(balanced_valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=0),
    }

    # Sprawdzenie klasy
    class_names = image_datasets['train'].classes
    print(f"Klasy: {class_names}")
    print(f"Liczba próbek po balansowaniu: train={len(balanced_train_dataset)}, valid={len(balanced_valid_dataset)}")

    return dataloaders, class_names

criterion = CrossEntropyLoss()

def training_loop(dataloaders):
    # model = torch.load("klasyfikacja_model2.pth")
    model = classification()
    model.to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    strata_trening = []
    strata_walidacja = []


    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloaders['train']:
            images, labels = batch
            images, labels = images.to("cuda"), labels.to("cuda")

            predictions = model(images)

            loss = criterion(predictions, labels.long())


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        strata_trening.append(total_loss / len(dataloaders['train']))
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloaders['train'])}")
        print()

        if epoch % co_ile_walidacja == 0:
            total_loss2 = 0

            w_skut_0 = 0
            w_skut_1 = 0
            w_skut_2 = 0
            # w_skut_3 = 0
            # w_skut_4 = 0
            # w_skut_5 = 0

            il_0 = 0
            il_1 = 0
            il_2 = 0
            # il_3 = 0
            # il_4 = 0
            # il_5 = 0


            y_true = []
            y_pred = []
            for batch in dataloaders['valid']:
                images, labels = batch
                images, labels = images.to("cuda"), labels.to("cuda")

                predictions = model(images)
                loss = criterion(predictions, labels.long())
                total_loss2 += loss.item()

                predictions = predictions.argmax(1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(predictions.cpu().tolist())

                for i, (real, pred) in enumerate(zip(labels.tolist(), predictions.tolist())):

                    if real == 0:
                        il_0 += 1
                        if real == pred:
                            w_skut_0 += 1

                    if real == 1:
                        il_1 += 1
                        if real == pred:
                            w_skut_1 += 1

                    if real == 2:
                        il_2 += 1
                        if real == pred:
                            w_skut_2 += 1


                    # if real == 4:
                    #     il_4 += 1
                    #     if real == pred:
                    #         w_skut_4 += 1
                    #
                    # if real == 5:
                    #     il_5 += 1
                    #     if real == pred:
                    #         w_skut_5 += 1

            skut = (w_skut_0 + w_skut_1 + w_skut_2 ) / (il_0 + il_1 + il_2 )
            skut_0 = w_skut_0 / il_0
            skut_1 = w_skut_1 / il_1
            skut_2 = w_skut_2 / il_2

            # skut_4 = w_skut_4 / il_4
            # skut_5 = w_skut_5 / il_5

            print(f"Skut: {skut}")
            print(f"Skit 0: {skut_0}")
            print(f"Skit 1: {skut_1}")
            print(f"Skit 2: {skut_2}")

            # print(f"Skit 4: {skut_4}")
            # print(f"Skit 5: {skut_5}")
            print()
            disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred))
            disp.plot()
            plt.show()


            strata_walidacja.append(total_loss2 / len(dataloaders['valid']))


    torch.save(model.state_dict(), "lolol_maly_model.pth")
    torch.save(model, "lolol_maly_model.pth")

    return strata_trening, strata_walidacja

def conf_matrix(dataloaders):
    model = torch.load("klasyfikacja_model2 — kopia.pth")
    model.eval()
    model.to("cuda")
    total_loss2 = 0
    w_skut_0 = 0
    w_skut_1 = 0
    w_skut_2 = 0
    il_0 = 0
    il_1 = 0
    il_2 = 0

    y_true = []
    y_pred = []
    for batch in dataloaders['valid']:
        images, labels = batch
        images, labels = images.to("cuda"), labels.to("cuda")

        predictions = model(images)
        loss = criterion(predictions, labels.long())
        total_loss2 += loss.item()

        predictions = predictions.argmax(1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

        for i, (real, pred) in enumerate(zip(labels.tolist(), predictions.tolist())):
            if real == 0:
                il_0 += 1
                if real == pred:
                    w_skut_0 += 1
            if real == 1:
                il_1 += 1
                if real == pred:
                    w_skut_1 += 1
            if real == 2:
                il_2 += 1
                if real == pred:
                    w_skut_2 += 1

    skut = (w_skut_0 + w_skut_1 + w_skut_2 ) / (il_0 + il_1 + il_2 )
    skut_0 = w_skut_0 / il_0
    skut_1 = w_skut_1 / il_1
    skut_2 = w_skut_2 / il_2

    print(f"Skut: {skut}")
    print(f"Skit 0: {skut_0}")
    print(f"Skit 1: {skut_1}")
    print(f"Skit 2: {skut_2}")

    print(total_loss2 / len(dataloaders['valid']))

    print()
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred))
    disp.plot()
    plt.show()

def wizualizacja_starty(z1, z2):
    l1 = len(z1)
    X = np.arange(l1)
    X2 = np.arange(l1, step=co_ile_walidacja)

    try:
        plt.plot(X, z1)
        plt.plot(X2, z2)
        plt.ylabel("Strata modelu")
        plt.xlabel("Ilość epok")
        plt.legend(["Trening", "Walidacja"])
        plt.show()
    except ValueError:
        plt.plot(X, z1)
        plt.ylabel("Strata modelu")
        plt.xlabel("Ilość epok")
        plt.title("Strata dla danych treningowych")
        plt.show()
        plt.clf()

        X2 = len(z2)
        plt.plot(X2, z2)
        plt.ylabel("Strata modelu")
        plt.title("Strata dla danych walidacyjnych")
        plt.show()

def random_images(test_dir, num_samples):

    # Zbierz wszystkie ścieżki do plików obrazów w folderze test (w tym podfoldery)
    all_images = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                all_images.append(os.path.join(root, file))

    # Upewnij się, że mamy wystarczającą liczbę obrazów
    if len(all_images) < num_samples:
        raise ValueError(f"Za mało obrazów w folderze test. Znaleziono {len(all_images)} obrazów.")

    # Losowe wybieranie `num_samples` obrazów
    random_images = random.sample(all_images, num_samples)
    return random_images

def preprocess_images(image_paths, transform):

    images = []
    for path in image_paths:
        image = Image.open(path).convert('RGB')
        tensor = transform(image)
        images.append(tensor)

    batch = torch.stack(images)
    return batch

def wyswietlanie_zdjec(image_paths, preds):

    num_images = len(image_paths)
    cols = 3
    rows = 3

    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axs.flatten()

    preds2 = preds
    preds = []
    for i in preds2:
        if i == 0:
            preds.append("Bike")
        elif i == 1:
            preds.append("Car")
        elif i == 2:
            preds.append("Truck")


    for i, (image, pred) in enumerate(zip(image_paths, preds)):
        image = Image.open(image).convert('RGB')
        axes[i].imshow(image)
        axes[i].set_title(pred)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def walidacja():

    model = classification()
    model.load_state_dict(torch.load("klasyfikacja_model2 — kopia.pth", weights_only=True))
    model.eval()

    num_samples = 3
    test_dir_bikes = "C:/Users/mateu/Desktop/datasety/Klasyfikacja_pojazdów/test/Bike"
    test_dir_cars = "C:/Users/mateu/Desktop/datasety/Klasyfikacja_pojazdów/test/Car"
    test_dir_trucks = "C:/Users/mateu/Desktop/datasety/Klasyfikacja_pojazdów/test/Truck"


    l = random_images(test_dir_cars, num_samples)
    l2 = random_images(test_dir_bikes, num_samples)
    l3 = random_images(test_dir_trucks, num_samples)

    dirs = l + l2 + l3

    data_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])

    batch = preprocess_images(dirs, data_transform)
    preds = model(batch)
    preds = preds.argmax(1)

    wyswietlanie_zdjec(dirs, preds)

def trenowanie_yolo():

    model2 = YOLO("yolo11n.pt")

    model2.train(
        data="C:/Users/mateu/Desktop/datasety/Highway CCTV Images for Vehicle Detection Dataset.v6i.yolov11/data.yaml",
        epochs=10,
        imgsz=640,
        batch=16,
        device=0,
        project="C:/Users/mateu/PycharmProjects/widzeni_komputerowe/modele_niby",
        workers=4
    )

    model2.save('yolov11_coco8.pt')

def odtwarzanie_yolo():
    model_path = 'yolov11_coco8.pt'
    model = YOLO(model_path)
    # model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()


    video_path = "C:/Users/mateu/Downloads/Film bez tytułu ‐ Wykonano za pomocą Clipchamp.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Błąd wczytywania filmu!")
        exit()

    # pobieramy informacje o filmie
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Utwórz obiekt do zapisu przetworzonego filmu
    output_video_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Kodowanie do mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Przetwarzaj każdą klatkę wideo
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Jeśli nie ma więcej klatek, zakończ

        # Użyj modelu YOLO do detekcji obiektów w klatce
        results = model.predict(frame)  # Użyj predict zamiast bezpośredniego wywołania modelu

        # Przetwórz wyniki i rysuj bounding boxy
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Współrzędne bbox
            confidences = result.boxes.conf.cpu().numpy()  # Pewności
            classes = result.boxes.cls.cpu().numpy()  # Klasy obiektów

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                conf = confidences[i]
                cls = int(classes[i])

                if conf > 0.5:  # Jeśli pewność detekcji > 0.5
                    label = f"{model.names[cls]}: {conf:.2f}"  # Etykieta z nazwą klasy
                    # Rysowanie prostokąta (bounding box) na klatce
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Zapisz przetworzoną klatkę do pliku wyjściowego
        out.write(frame)

        # Wyświetl klatkę (opcjonalnie, żeby widzieć w czasie rzeczywistym)
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Zatrzymanie na 'q'
            break

    # Zakończ przetwarzanie
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Przetworzony film zapisano jako {output_video_path}")

def classify_crop(crop, class_model, class_names):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    # dodajemy wymiary batcha
    crop_tensor = transform(crop).unsqueeze(0)
    crop_tensor = crop_tensor.to("cuda")

    with torch.no_grad():
        output = class_model(crop_tensor)
        pred = output.argmax(1).item()
        pred_class = class_names[pred]

    return pred_class

def draw_boxes_with_labels(frame, results, classification_model, class_names):
    detections = []

    class_colrs ={
        "Bike": (0, 0, 255),
        "Car": (0, 255, 0),
        "Truck": (255, 0, 0)
    }

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            szer = (x2 - x1)/2
            wyso = (y2 - y1)/2
            x2 = int(x2+0.3*szer)
            x1 = int(x1-0.3*szer)
            y2 = int(y2+0.3*wyso)
            y1 = int(y1-0.3*wyso)

            centroid = [(x1 + x2)/2, (y1 + y2)/2]


            conf = confidences[i].item()
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue


            crop_pil = Image.fromarray(crop)
            class_label = classify_crop(crop_pil, classification_model, class_names)

            # narysuj bounding box z danymi
            if conf > 0.6:

                detections.append(Detection(points=np.array(centroid), scores=np.array([conf])))

                label = f"{class_label}:"
                color = class_colrs.get(class_label, (255, 255, 255))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, detections

def polaczenie_modeli():
    model1 = torch.load("lolol_maly_model.pth")
    model_path = 'yolov11_coco8.pt'
    model2 = YOLO(model_path)
    class_names = ["Bike", "Car", "Truck"]

    model1.eval()
    model2.eval()

    video_path = "C:/Users/mateu/Downloads/Film bez tytułu ‐ Wykonano za pomocą Clipchamp.mp4"
    # video_path = "C:/Users/mateu/Downloads/5.4 4K Camera Road in Thailand.mp4"
    # video_path = "C:/Users/mateu/Downloads/5.4 4K Camera Road in Thailand (online-video-cutter.com).mp4"

    output_path = "Video_przetworzone_tajlandia.mp4"

    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    tracker = Tracker(distance_function="euclidean", distance_threshold=30)

    vehicle_count = 0

    tracked_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model2.predict(frame)

        frame_with_changed_labels, detections = draw_boxes_with_labels(frame, results, model1, class_names)
        tracked_objects = tracker.update(detections)

        # draw_tracked_boxes(frame_with_changed_labels, tracked_objects)

        for obj in tracked_objects:
            if obj.id not in tracked_ids:
                tracked_ids.add(obj.id)
                vehicle_count += 1

        textt = f"Licznik pojazdow:  {vehicle_count}"
        cv2.putText(frame_with_changed_labels, textt, org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
        cv2.imshow("Detekcja i klasyfikacja na moim modelu", frame_with_changed_labels)
        out.write(frame_with_changed_labels)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def menu():

    while True:
        print()
        print("Co chcesz zrobić")
        print("1. Wytrenować model")
        print("2. Testowac wymiary")
        print("3. zaldować model i go ewaluaować")
        print("4. Wyjść")
        wybor = int(input("Wpisz odpowiedni numer:\t"))


        if wybor == 1:
            zmienna = input("Jesteś pewny? Jeśli tak wpisz Y:\t")
            if zmienna == 'Y':
                dataloaders, class_names = dataloading()
                z1, z2 = training_loop(dataloaders, class_names)
                wizualizacja_starty(z1, z2)
        elif wybor == 2:
            test()
        elif wybor == 3:
            walidacja()
        elif wybor == 4:
            break
        else:
            print("Wpisałeś zły numer")

# menu()

# if __name__ == "__main__":
#     dataloaders, class_names = dataloading()
#     z1, z2 = training_loop(dataloaders)
#     wizualizacja_starty(z1, z2)

# if __name__ == '__main__':
#     trenowanie_yolo()

# if __name__ == '__main__':
#     odtwarzanie_yolo()

if __name__ == '__main__':
    polaczenie_modeli()

# if __name__ == '__main__':
    # dataloaders, class_namqqes = dataloading()
    # conf_matrix(dataloaders)