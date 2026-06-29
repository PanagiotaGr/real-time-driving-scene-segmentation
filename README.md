# Real-Time Driving Scene Semantic Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

---

## Ελληνικός Deep-Dive Οδηγός

Το παρόν repository είναι ένα αναλυτικό framework για **semantic segmentation σε οδικές σκηνές**. Ο στόχος είναι να εκπαιδεύσουμε νευρωνικά δίκτυα που παίρνουν ως είσοδο RGB εικόνες δρόμου και επιστρέφουν μια μάσκα όπου κάθε pixel έχει αντιστοιχιστεί σε μία κατηγορία, όπως δρόμος, πεζοδρόμιο, αυτοκίνητο, πεζός, κτίριο, βλάστηση ή ουρανός.

Το project έχει σχεδιαστεί με ερευνητική λογική. Δεν περιλαμβάνει μόνο ένα μοντέλο, αλλά οργανώνει ολόκληρη τη διαδικασία: dataset loading, preprocessing, data augmentation, training, validation, metrics, benchmarking και αποθήκευση αποτελεσμάτων.

---

## Περιεχόμενα

1. [Τι είναι το project](#1-τι-είναι-το-project)
2. [Τι είναι semantic segmentation](#2-τι-είναι-semantic-segmentation)
3. [Γιατί χρησιμοποιείται σε driving scenes](#3-γιατί-χρησιμοποιείται-σε-driving-scenes)
4. [Πλήρης ροή συστήματος](#4-πλήρης-ροή-συστήματος)
5. [Dataset CamVid](#5-dataset-camvid)
6. [Εγκατάσταση](#6-εγκατάσταση)
7. [Δομή repository](#7-δομή-repository)
8. [Αναλυτική εξήγηση αρχείων](#8-αναλυτική-εξήγηση-αρχείων)
9. [Μοντέλα](#9-μοντέλα)
10. [Training pipeline](#10-training-pipeline)
11. [Loss functions](#11-loss-functions)
12. [Metrics και benchmarking](#12-metrics-και-benchmarking)
13. [Evaluation και αποτελέσματα](#13-evaluation-και-αποτελέσματα)
14. [Πώς παρουσιάζεται το project](#14-πώς-παρουσιάζεται-το-project)
15. [Μελλοντικές επεκτάσεις](#15-μελλοντικές-επεκτάσεις)

---

# 1. Τι είναι το project

Το **Real-Time Driving Scene Semantic Segmentation** είναι ένα project Υπολογιστικής Όρασης που εστιάζει στην κατανόηση οδικών σκηνών σε επίπεδο pixel. Αντί το μοντέλο να απαντά απλώς ότι μια εικόνα περιέχει δρόμο ή αυτοκίνητα, προσπαθεί να ταξινομήσει κάθε σημείο της εικόνας σε συγκεκριμένη σημασιολογική κατηγορία.

Με απλά λόγια:

```text
RGB εικόνα δρόμου
        ↓
Νευρωνικό δίκτυο semantic segmentation
        ↓
Μάσκα κατηγοριών ανά pixel
```

Αυτό σημαίνει ότι η έξοδος του συστήματος έχει την ίδια χωρική δομή με την είσοδο. Για κάθε pixel της αρχικής εικόνας, το μοντέλο προβλέπει αν ανήκει στον δρόμο, στο πεζοδρόμιο, σε αυτοκίνητο, σε πεζό, στον ουρανό ή σε άλλη κατηγορία.

Το repository είναι κατάλληλο για πανεπιστημιακή εργασία γιατί παρουσιάζει ολόκληρη τη λογική ενός deep learning segmentation pipeline και όχι μόνο ένα μεμονωμένο script.

---

# 2. Τι είναι semantic segmentation

Η **semantic segmentation** είναι εργασία της Υπολογιστικής Όρασης όπου κάθε pixel μιας εικόνας παίρνει μία ετικέτα κλάσης.

| Τεχνική | Τι προβλέπει | Παράδειγμα |
|---|---|---|
| Image classification | Μία ετικέτα για όλη την εικόνα | «δρόμος» |
| Object detection | Κουτιά γύρω από αντικείμενα | κουτί γύρω από αυτοκίνητο |
| Semantic segmentation | Κατηγορία για κάθε pixel | κάθε pixel ταξινομείται ως road, car, sky κ.λπ. |

Μαθηματικά, αν η εικόνα εισόδου είναι:

```text
X ∈ R^(H × W × 3)
```

τότε το μοντέλο παράγει logits:

```text
Y ∈ R^(N × C × H × W)
```

όπου `N` είναι το batch size, `C` ο αριθμός των κλάσεων, και `H × W` η ανάλυση της εικόνας. Η τελική μάσκα προκύπτει με:

```text
prediction = argmax(Y, dim=class)
```

---

# 3. Γιατί χρησιμοποιείται σε driving scenes

Σε οδικές σκηνές, η semantic segmentation είναι σημαντική γιατί επιτρέπει στο σύστημα να κατανοήσει τη γεωμετρία και τη σημασιολογία του περιβάλλοντος. Δεν αρκεί να ξέρουμε ότι υπάρχει ένα αντικείμενο· πρέπει να γνωρίζουμε πού ακριβώς βρίσκεται μέσα στην εικόνα και ποια περιοχή καταλαμβάνει.

Παραδείγματα ερωτημάτων που απαντά η segmentation:

- Πού βρίσκεται ο δρόμος;
- Πού είναι το πεζοδρόμιο;
- Ποια pixels ανήκουν σε οχήματα;
- Υπάρχουν πεζοί;
- Ποια σημεία της εικόνας είναι ουρανός ή βλάστηση;

Η έξοδος μπορεί να χρησιμοποιηθεί ως είσοδος σε επόμενα συστήματα, όπως path planning, obstacle reasoning ή scene understanding.

---

# 4. Πλήρης ροή συστήματος

Η συνολική ροή του project είναι:

```text
CamVid RGB Image
        ↓
CamVidDataset
        ↓
Albumentations preprocessing
        ↓
PyTorch DataLoader
        ↓
Segmentation Model
        ↓
Logits (N, C, H, W)
        ↓
Loss Function
        ↓
Backpropagation
        ↓
Optimizer Step
        ↓
Validation
        ↓
Metrics: Pixel Accuracy, mIoU, Class IoU
        ↓
Checkpoint + Results
```

Η ροή αυτή δείχνει ότι το project δεν είναι μόνο μοντέλο. Είναι πλήρης μηχανισμός εκπαίδευσης, αξιολόγησης και πειραματικής καταγραφής.

---

# 5. Dataset CamVid

Το project χρησιμοποιεί το **CamVid dataset**. Κάθε δείγμα αποτελείται από μια RGB εικόνα και την αντίστοιχη ground-truth mask.

Η αναμενόμενη δομή είναι:

```text
data/camvid/
├── train/
│   ├── leftImg8bit/
│   └── gtSeg8bit/
└── val/
    ├── leftImg8bit/
    └── gtSeg8bit/
```

Ο φάκελος `leftImg8bit` περιέχει τις αρχικές εικόνες. Ο φάκελος `gtSeg8bit` περιέχει τις μάσκες. Το αρχείο `src/datasets/camvid.py` ελέγχει ότι υπάρχει σωστή αντιστοίχιση εικόνων και masks. Αν ο αριθμός τους δεν είναι ίδιος, η εκτέλεση σταματά, γιατί η εκπαίδευση θα ήταν λανθασμένη.

---

# 6. Εγκατάσταση

```bash
git clone https://github.com/PanagiotaGr/real-time-driving-scene-segmentation
cd real-time-driving-scene-segmentation
pip install -r requirements.txt
```

Βασικές βιβλιοθήκες:

- `torch`, `torchvision`: βαθιά μάθηση,
- `opencv-python`: ανάγνωση και επεξεργασία εικόνων,
- `albumentations`: data augmentation,
- `numpy`: αριθμητικοί υπολογισμοί,
- `matplotlib`: οπτικοποίηση,
- `pyyaml`: configuration,
- `tensorboard`: logging.

Εκπαίδευση:

```bash
python src/training/train.py
```

Αξιολόγηση:

```bash
python src/training/evaluate.py semantic-segmentation-driving/results/checkpoints/best.pth
```

Reproducible experiment:

```bash
python src/training/run_experiment.py --checkpoint semantic-segmentation-driving/results/checkpoints/best.pth
```

---

# 7. Δομή repository

```text
.
├── README.md
├── RESEARCH_PLAN.md
├── requirements.txt
├── experiments/
│   ├── BENCHMARK_TABLE.md
│   └── RESULTS_TEMPLATE.md
└── src/
    ├── config.py
    ├── datasets/
    │   ├── __init__.py
    │   └── camvid.py
    ├── models/
    │   ├── unet.py
    │   ├── enet.py
    │   └── modules/
    │       └── edge_refinement.py
    ├── training/
    │   ├── train.py
    │   ├── trainer.py
    │   ├── evaluate.py
    │   ├── run_experiment.py
    │   └── benchmark_runner.py
    └── utils/
        ├── losses.py
        ├── metrics.py
        ├── advanced_metrics.py
        ├── benchmark.py
        ├── model_profiler.py
        └── losses/
            ├── boundary_loss.py
            └── hybrid_loss.py
```

---

# 8. Αναλυτική εξήγηση αρχείων

## `src/config.py`

Το `config.py` είναι το κεντρικό σημείο ρυθμίσεων. Περιέχει dataclasses για dataset, model, training και experiment configuration.

Παραδείγματα παραμέτρων:

- `batch_size`: πόσες εικόνες περνούν μαζί στο μοντέλο,
- `lr`: learning rate,
- `epochs`: αριθμός εποχών,
- `model_name`: επιλογή μοντέλου,
- `num_classes`: αριθμός κατηγοριών segmentation.

Έτσι οι ρυθμίσεις είναι συγκεντρωμένες και ο training code παραμένει καθαρός.

## `src/datasets/camvid.py`

Υλοποιεί την κλάση `CamVidDataset`. Η κλάση:

1. βρίσκει εικόνες και masks,
2. ελέγχει ότι υπάρχουν δεδομένα,
3. διαβάζει εικόνες με OpenCV,
4. μετατρέπει BGR σε RGB,
5. διαβάζει τις masks ως grayscale,
6. εφαρμόζει transforms,
7. επιστρέφει tensors.

Η συνάρτηση `get_camvid_transforms` δημιουργεί διαφορετικά transforms για training και validation. Στο training εφαρμόζονται augmentations όπως horizontal flip, brightness/contrast και rotation. Στο validation χρησιμοποιούνται μόνο resize, normalization και μετατροπή σε tensor.

## `src/datasets/__init__.py`

Περιέχει τη συνάρτηση `get_camvid_dataloaders`, η οποία δημιουργεί `train_loader` και `val_loader`. Ο train loader κάνει shuffle, ενώ ο validation loader όχι, ώστε η αξιολόγηση να είναι σταθερή.

## `src/models/unet.py`

Υλοποιεί την U-Net. Η U-Net έχει encoder, bottleneck και decoder με skip connections. Τα skip connections επιτρέπουν στο decoder να ανακτήσει λεπτομέρειες που χάνονται στα pooling layers.

Ροή:

```text
Input → Encoder → Bottleneck → Decoder + Skip Connections → 1x1 Conv → Logits
```

## `src/models/enet.py`

Υλοποιεί μια απλοποιημένη ENet-like αρχιτεκτονική. Η λογική της είναι να λειτουργεί ως ελαφρύτερο μοντέλο για real-time segmentation. Περιλαμβάνει encoder blocks, bottleneck, decoder με transposed convolutions και τελικό classifier.

## `src/models/modules/edge_refinement.py`

Περιέχει το `EdgeRefinementModule`, ένα ελαφρύ block που μπορεί να χρησιμοποιηθεί για βελτίωση ορίων. Χρησιμοποιεί residual σύνδεση και depthwise convolution ώστε να βελτιώνει λεπτομέρειες χωρίς μεγάλο κόστος.

## `src/training/train.py`

Είναι το βασικό script εκπαίδευσης. Η ροή του είναι:

```text
get_config()
  ↓
set seed
  ↓
load dataloaders
  ↓
initialize model
  ↓
initialize loss
  ↓
create optimizer
  ↓
create Trainer
  ↓
fit()
  ↓
save history
```

## `src/training/trainer.py`

Περιέχει την κλάση `Trainer`, η οποία αναλαμβάνει:

- training epoch,
- validation epoch,
- αποθήκευση checkpoints,
- early stopping,
- ιστορικό loss και metrics.

Σε κάθε batch κάνει forward pass, υπολογισμό loss, backpropagation και optimizer step.

## `src/training/evaluate.py`

Φορτώνει checkpoint, τρέχει validation και υπολογίζει Mean IoU, Pixel Accuracy, inference time και αριθμό παραμέτρων. Δημιουργεί επίσης visualization με είσοδο, ground truth και prediction.

## `src/training/run_experiment.py`

Εκτελεί πείραμα με checkpoint και αποθηκεύει αποτελέσματα σε JSON και CSV. Είναι χρήσιμο για σύγκριση διαφορετικών μοντέλων ή διαφορετικών loss functions.

## `src/utils/losses.py`

Περιέχει `DiceLoss` και `CombinedLoss`. Η Dice Loss βοηθά σε class imbalance, ενώ η Combined Loss συνδυάζει Cross Entropy και Dice.

## `src/utils/losses/boundary_loss.py`

Υπολογίζει boundary-aware loss με Sobel filters. Στόχος είναι να μειωθούν τα λάθη στα όρια αντικειμένων.

## `src/utils/losses/hybrid_loss.py`

Υλοποιεί:

```text
Hybrid Loss = α CrossEntropy + β Dice + γ Boundary
```

Αυτή η loss είναι κατάλληλη για ablation experiments.

## `src/utils/metrics.py`

Υπολογίζει Pixel Accuracy, IoU ανά κλάση και Mean IoU.

## `src/utils/advanced_metrics.py`

Υπολογίζει confusion matrix, precision, recall, F1, IoU και support ανά κλάση.

## `src/utils/benchmark.py` και `src/utils/model_profiler.py`

Μετρούν latency, FPS, αριθμό παραμέτρων και χρήση μνήμης.

---

# 9. Μοντέλα

## U-Net

Η U-Net είναι το βασικό baseline. Είναι καλή για segmentation επειδή συνδυάζει βαθιά σημασιολογικά features με χωρικές λεπτομέρειες μέσω skip connections.

Πλεονεκτήματα:

- καλή ακρίβεια,
- καθαρή αρχιτεκτονική,
- ισχυρό baseline.

Μειονεκτήματα:

- μεγαλύτερο υπολογιστικό κόστος,
- πιθανώς πιο αργή από real-time μοντέλα.

## ENet-like model

Το ENet-like μοντέλο είναι πιο ελαφρύ και δίνει έμφαση στην ταχύτητα. Χρησιμοποιείται για να μελετηθεί το trade-off ανάμεσα σε accuracy και inference speed.

## BiSeNet support hook

Τα training scripts έχουν επιλογή `bisenet`, που δείχνει ότι το framework έχει σχεδιαστεί ώστε να επεκταθεί σε BiSeNet-style αρχιτεκτονική. Αν δεν υπάρχει το `src/models/bisenet.py` στο clone, η επιλογή αυτή χρειάζεται πρώτα την αντίστοιχη υλοποίηση.

---

# 10. Training pipeline

Η εκπαίδευση γίνεται σε epochs. Σε κάθε epoch το μοντέλο βλέπει όλα τα training batches.

```text
Batch images + masks
        ↓
Forward pass
        ↓
Loss calculation
        ↓
Backward pass
        ↓
Optimizer step
        ↓
Validation
        ↓
Checkpoint if improved
```

Το validation γίνεται χωρίς gradients, ώστε να είναι πιο γρήγορο και να μην αλλάζει τα βάρη του μοντέλου.

---

# 11. Loss functions

## Cross Entropy

Βασική loss για ταξινόμηση κάθε pixel.

## Dice Loss

Μετρά επικάλυψη πρόβλεψης και ground truth. Είναι χρήσιμη όταν κάποιες κλάσεις είναι μικρές.

## Combined Loss

Συνδυάζει Cross Entropy και Dice.

## Hybrid Loss

Συνδυάζει Cross Entropy, Dice και Boundary Loss για να βελτιώνει ταξινόμηση, επικάλυψη και όρια.

---

# 12. Metrics και benchmarking

Βασικές μετρικές:

- **Pixel Accuracy**: ποσοστό σωστών pixels,
- **Class-wise IoU**: IoU για κάθε κλάση,
- **Mean IoU**: μέσο IoU όλων των κλάσεων,
- **Latency**: χρόνος inference,
- **FPS**: εικόνες ανά δευτερόλεπτο,
- **Parameters**: μέγεθος μοντέλου.

Για real-time segmentation, η αξιολόγηση πρέπει να συνδυάζει accuracy και speed. Ένα μοντέλο δεν αρκεί να έχει υψηλό mIoU· πρέπει να είναι και αρκετά γρήγορο.

---

# 13. Evaluation και αποτελέσματα

Η αξιολόγηση γίνεται με το `evaluate.py` ή το `run_experiment.py`. Το πρώτο δίνει άμεση αξιολόγηση και visualization. Το δεύτερο αποθηκεύει πιο οργανωμένα αποτελέσματα για πειράματα.

Τα αρχεία `experiments/BENCHMARK_TABLE.md` και `experiments/RESULTS_TEMPLATE.md` λειτουργούν ως templates για καταγραφή αποτελεσμάτων σε μορφή κατάλληλη για αναφορά ή εργασία.

---

# 14. Πώς παρουσιάζεται το project

Μια σύντομη παρουσίαση:

> Το project υλοποιεί ένα σύστημα semantic segmentation για οδικές σκηνές. Χρησιμοποιεί το CamVid dataset, φορτώνει εικόνες και ground-truth masks, εκπαιδεύει νευρωνικά δίκτυα όπως U-Net και ENet, αξιολογεί την απόδοση με Pixel Accuracy και Mean IoU, και μετρά την ταχύτητα inference μέσω latency και FPS. Ο σκοπός είναι να μελετηθεί η ισορροπία ανάμεσα σε ακρίβεια και πραγματικό χρόνο.

Τεχνική ροή παρουσίασης:

```text
Problem → Dataset → Preprocessing → Model → Training → Loss → Metrics → Benchmark → Future Work
```

---

# 15. Μελλοντικές επεκτάσεις

Προτεινόμενες επεκτάσεις:

1. πλήρης υλοποίηση BiSeNet,
2. ενσωμάτωση edge refinement στο κύριο μοντέλο,
3. πειράματα σε Cityscapes ή BDD100K,
4. robustness testing σε βροχή, νύχτα, θόρυβο και blur,
5. ablation studies για διαφορετικές loss functions,
6. knowledge distillation από μεγαλύτερο μοντέλο σε μικρότερο,
7. αναλυτική σύγκριση FPS/mIoU σε κοινό hardware.

---

# Τελικό συμπέρασμα

Το repository αποτελεί μια ολοκληρωμένη βάση για μελέτη semantic segmentation σε driving scenes. Συνδυάζει καθαρή δομή κώδικα, PyTorch training pipeline, multiple loss functions, metrics, benchmarking και research-oriented templates. Με επιπλέον πειράματα και πλήρη καταγραφή αποτελεσμάτων, μπορεί να αποτελέσει ισχυρή βάση για πανεπιστημιακή εργασία ή ερευνητική μελέτη στην Υπολογιστική Όραση.

---

*Developed for research and educational purposes in autonomous vehicle perception.*
