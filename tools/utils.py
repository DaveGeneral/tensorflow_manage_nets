import skimage
import skimage.io
import skimage.transform
import numpy as np
from datetime import datetime
import sys, os, csv
import itertools
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, average_precision_score, roc_curve, roc_auc_score, hamming_loss
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score
from sklearn.metrics import recall_score, precision_score, fbeta_score
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path, scale=255, xrange=[0, 1], dim_image=224):
    # load image
    img = skimage.io.imread(path)
    img = img / scale
    # assert (xrange[0] <= img).all() and (img <= xrange[1]).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (dim_image, dim_image), mode='constant')
    return resized_img


def load_image_withoutscale(path, xrange=[0, 1], dim_image=224):
    # load image
    img = skimage.io.imread(path)
    # assert (xrange[0] <= img).all() and (img <= xrange[1]).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (dim_image, dim_image), mode='constant')
    return resized_img


def save_image(path_source, path_dest, name_image, transform=False, path_csv=None):

    name, ext = name_image.split('.')

    f = open(path_csv, "a+")

    if transform is False:

        img = load_image(path_source+name_image)
        skimage.io.imsave(path_dest + name + '.' + ext, img)
        f.write(",".join(map(str, [name, 0])) + "\n")

    else:
        img = load_image(path_source + name_image)
        img90 = skimage.transform.rotate(img, 90, resize=True)
        img180 = skimage.transform.rotate(img, 180, resize=True)
        img270 = skimage.transform.rotate(img, 270, resize=True)

        skimage.io.imsave(path_dest + name+'_0.'+ext, img)
        f.write(",".join(map(str, [name+'_0', 1])) + "\n")
        skimage.io.imsave(path_dest + name+'_90.'+ext, img90)
        f.write(",".join(map(str, [name+'_90', 1])) + "\n")
        skimage.io.imsave(path_dest + name+'_180.'+ext, img180)
        f.write(",".join(map(str, [name+'_180', 1])) + "\n")
        skimage.io.imsave(path_dest + name+'_270.'+ext, img270)
        f.write(",".join(map(str, [name+'_270', 1])) + "\n")

    f.close()
    print('Save image: ', name_image)


def save_image2(path_source, path_dest, name_image):
    name, ext = name_image.split('.')
    img = load_image(path_source + name_image)
    imgflip = np.fliplr(img)
    skimage.io.imsave(path_dest + name+'_.'+ext, imgflip)


# returns the top1 string
def print_prob(prob, file_path, top=5):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    # np.argsort -> ordena el array y almacena los INDICES de los numeros ordenados
    # x[::-1] -> invierte el orden de la lista 'x'
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    if top > 0:
        top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(top)]
        print(("Top"+str(top)+": ", top5))

    return top1


def print_prob_all(prob, file_path, top=5):
    synset = [l.strip() for l in open(file_path).readlines()]
    for i in range(len(prob)):
        _prob = prob[i]
        pred = np.argsort(_prob)[::-1]
        top1 = synset[pred[0]]
        print("    Top1: ", top1, _prob[pred[0]])

        if top > 0:
            topn = [(synset[pred[i]], _prob[pred[i]]) for i in range(top)]
            print("    Top" + str(top) + ": ", topn)


def print_accuracy(target, prob, matrix_confusion=None, predicted=[]):

    total = len(target)
    count = 0

    for i in range(total):
        true_result = np.argsort(prob[i])[::-1][0]
        if target[i] == true_result:
            count += 1

        predicted.append(true_result)
        matrix_confusion[target[i]][true_result] = matrix_confusion[target[i]][true_result] + 1

    accuracy = count / total
    print('    results[ Total:'+str(total)+' | True:'+str(count)+' | False:'+str(total-count)+' | Accuracy:'+str(accuracy)+' ]')
    return count, matrix_confusion, predicted


def process_prob(target, prob, predicted=[], plot_predicted=[]):

    total = len(target)
    count = 0
    num_class = len(prob[0])

    for i in range(total):
        true_result = np.argsort(prob[i])[::-1][0]
        if target[i] == true_result:
            count += 1

        predicted.append(true_result)
        plot_predicted.append(prob[i][num_class-1])

    accuracy = count / total
    print('    results[ Total:'+str(total)+' | True:'+str(count)+' | False:'+str(total-count)+' | Accuracy:'+str(accuracy)+' ]')
    return count, predicted, plot_predicted


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def time_epoch(millis):
    millis = int(millis)
    seconds = (millis/1000) % 60
    seconds = int(seconds)
    minutes = (millis/(1000*60)) % 60
    minutes = int(minutes)
    hours = (millis/(1000*60*60)) % 24

    return hours, minutes, seconds


def test():
    img = skimage.io.imread("./test_data/tiger.jpeg")[:, :, :3]
    ny = 300
    nx = int(img.shape[1] * ny / img.shape[0])
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/output.jpg", img)


def write_log(total_data, epoch, m_batch, l_rate, accuracy=0, file_npy='None', extra=''):
    now = datetime.now()
    id = int(now.timestamp()*1000000)
    date = now.strftime('%d-%m-%Y %H:%M:%S')
    file = sys.argv[0].split('/')[-1]

    f = open("log-server.txt", "a+")
    f.write('id:{}  date:{}  file:{}  input:{}  epoch:{}  m-batch:{}  l-rate:{}  accuracy:{:3.3f}  file_npy:{}  extra:{}\n'.format(id,date,file,total_data,epoch,m_batch,l_rate,accuracy,file_npy,extra))
    f.close()
    print('Create log in log-server.txt:', id)


def save_layer_output(out_layer, label, name="layer", dir='/'):

    shx = out_layer.shape

    if len(shx)==4:
        out_layer = np.reshape(out_layer, [-1, shx[1] * shx[2] * shx[3]])

    total = len(label)
    lab = np.reshape(label, (total, 1))
    res = np.concatenate((out_layer, lab), axis=1)

    f = open(dir+"output_"+name+".csv", "a+")
    for i in range(total):
        f.write(",".join(map(str, res[i])) + "\n")
    f.close()
    print("    Save feature extractor, "+name)


def save_layer_output_by_class(out_layer, label, name="layer", dir='/'):

    shx = out_layer.shape

    if len(shx) == 4:
        out_layer = np.reshape(out_layer, [-1, shx[1] * shx[2] * shx[3]])

    total = len(label)
    lab = np.reshape(label, (total, 1))
    res = np.concatenate((out_layer, lab), axis=1)

    for i in range(total):
        f = open(dir+"output_" + name + '_class' + str(int(label[i])) + ".csv", "a+")
        f.write(",".join(map(str, res[i])) + "\n")
        f.close()

    print("    Save feature extractor, "+name)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def metrics(y_true=[], y_pred=[], plot_pred=[], plot_graph=False):

    cm1 = confusion_matrix(y_true=y_true, y_pred=y_pred)
    total1 = sum(sum(cm1))

    print('Confusion Matrix : \n', cm1)
    # from confusion matrix calculate accuracy
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print('Total Correct : ', cm1[0, 0] + cm1[1, 1])
    print('Accuracy      : ', accuracy1)

    sensitivity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print('Sensitivity   : ', sensitivity1)

    specificity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print('specificity   : ', specificity1)

    average_precision = average_precision_score(y_true, y_pred)
    print('Average Precision : ', average_precision)

    area_under_roc = roc_auc_score(y_true, y_pred)
    print('Area Under RCO : ', area_under_roc)

    if plot_graph is True:
        plot_curve_roc([y_true], [plot_pred])

    return accuracy1


def metrics_multiclass(y_true=[], y_pred=[]):

    cm1 = confusion_matrix(y_true=y_true, y_pred=y_pred)
    total = sum(sum(cm1))

    accuracy = accuracy_score(y_true, y_pred)
    total_score = accuracy_score(y_true, y_pred, normalize=False)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    print('Confusion Matrix : \n', cm1)  
    print('Total Correct: ', total_score, '/', total)
    print('Accuracy     : ', accuracy)
    print('F1-macro     : ', f1_macro)
    print('F1-micro     : ', f1_micro)
    print('F1-weighted  : ', f1_weighted)
    return accuracy


def plot_curve_roc(y_true, y_pred, title=None):
    res = []
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')

    for i in range(len(y_true)):
        try:
            t = title[i]
        except:
            t = 'Graph_'+str(i)

        fpr, tpr, _ = roc_curve(y_true[i], y_pred[i])
        res.append([fpr.tolist(), tpr.tolist(), t])
        plt.plot(fpr.tolist(), tpr.tolist(), label=t)

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


def plot_curve_roc_multiclass(y_true, y_prob, num_class, title=None):
    # Convertimos [1-D] -> [1-D][num_class-D]
    y_true = label_binarize(y_true, classes=[i for i in range(num_class + 1)])[:, :num_class]
    y_prob = label_binarize(y_prob, classes=[i for i in range(num_class + 1)])[:, :num_class]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_class):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_class

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(num_class), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def precision_recall(y_true, y_prob, num_class):
    """
    :param y_true: [1-D] 
    :param y_prob: [1-D][num_class-D]
    :param num_class: numero de clases
    """
    # Convertimos [1-D] -> [1-D][num_class-D]
    y_true = label_binarize(y_true, classes=[i for i in range(num_class + 1)])[:, :num_class]
    #y_prob = label_binarize(y_prob, classes=[i for i in range(num_class + 1)])[:, :num_class]

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(num_class):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_prob[:, i])

    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(), y_prob.ravel())
    average_precision["micro"] = average_precision_score(y_true, y_prob, average="micro")

    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    lw = 1
    # Plot Precision-recall
    plt.figure(1)
    plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(average_precision["micro"]))

    for i, color in zip(range(num_class), colors):
        plt.plot(recall[i], precision[i], color=color, lw=lw,
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def hamming_score(y_true, y_pred):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''

    acc_list = []
    for i in range(np.shape(y_true)[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])

        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def metrics_multiLabel(y_true, y_pred):
    hs = hamming_score(y_true, y_pred)
    accs = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    hl = hamming_loss(y_true, y_pred)

    print('Hamming score: {0}'.format(hs))
    print('Subset accuracy: {0}'.format(accs))
    print('Hamming loss: {0}'.format(hl))
    return hs


def print_accuracy_multilabel(target, prob):

    total = len(target)
    y_pred = []

    for i in range(total):
        new_prob = np.round(prob[i]).astype(int)
        y_pred.append(new_prob)

    accuracy = hamming_score(target, y_pred)
    print('    results[ Total:'+str(total)+' | Accuracy:'+str(accuracy)+' ]')
    return y_pred


def generate_max_csvData(sources, path_save, has_label=True, no_NaN=True):
    max_i = []
    for file in sources:
        with open(file, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            max_i.append(np.amax(np.float_(data), axis=0))

    if has_label is True:
        maximo = np.amax(max_i, axis=0)[:-1]
    else:
        maximo = np.amax(max_i, axis=0)

    myMax = max(maximo)
    myMin = min(maximo)

    if myMin == 0.0 and no_NaN == True:
        maximo[maximo == 0] = 0.0001

    myMin = min(maximo)

    f = open(path_save, "w")
    f.write(",".join(map(str, maximo)) + "\n")
    f.close()
    print("Save max vector, Dim =", str(len(maximo)), ', min =', myMin, ', max =', myMax)


def generate_MinMax_csvData(sources, path_save, name_csv, has_label=True, no_NaN=True):
    max_i = []
    min_i = []
    for file in sources:
        with open(file, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            max_i.append(np.amax(np.float_(data), axis=0))
            min_i.append(np.amin(np.float_(data), axis=0))

    # Get Maximo
    # ----------
    if has_label is True:
        maximo = np.amax(max_i, axis=0)[:-1]
    else:
        maximo = np.amax(max_i, axis=0)

    myMax = max(maximo)
    myMin = min(maximo)

    if myMin == 0.0 and no_NaN == True:
        maximo[maximo == 0] = 0.0001

    myMin = min(maximo)

    f = open(path_save + 'maximo_' + name_csv + '.csv', "w")
    f.write(",".join(map(str, maximo)) + "\n")
    f.close()
    print("Save max vector, Dim =", str(len(maximo)), ', min =', myMin, ', max =', myMax)

    # Get Minimo
    # ----------
    if has_label is True:
        minimo = np.amin(min_i, axis=0)[:-1]
    else:
        minimo = np.amin(min_i, axis=0)

    myMax = max(minimo)
    myMin = min(minimo)

    f = open(path_save + 'minimo_' + name_csv + '.csv', "w")
    f.write(",".join(map(str, minimo)) + "\n")
    f.close()
    print("Save min vector, Dim =", str(len(minimo)), ', min =', myMin, ', max =', myMax)


def normalization_with_minMax(paths, path_save):
    csv_original = paths[0]
    csv_minimo = paths[1]
    csv_maximo = paths[2]

    minimo = load_max_csvData(csv_minimo)
    maximo = load_max_csvData(csv_maximo)

    diferencia = maximo - minimo

    #print(minimo)
    #print(maximo)

    with open(csv_original, 'r') as f:
        reader = csv.reader(f)
        dataset = list(reader)

        f = open(path_save, "w")
        for i in range(len(dataset)):
            data = np.float_(dataset[i])

            sample = data[:-1]
            label = data[-1]
            xi = (sample - minimo) / diferencia
            # print(",".join(map(str, np.concatenate((xi, [label]), axis=0))))
            f.write(",".join(map(str, np.concatenate((xi, [label]), axis=0))) + "\n")
        f.close()
        print('Created data normalized,', path_save)


def normalization_complete(sources, has_label=True, no_NaN=True):
    max_i = []
    min_i = []
    for file in sources:
        with open(file, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            max_i.append(np.amax(np.float_(data), axis=0))
            min_i.append(np.amin(np.float_(data), axis=0))

    # Get Maximo
    # ----------
    if has_label is True:
        maximo = np.amax(max_i, axis=0)[:-1]
    else:
        maximo = np.amax(max_i, axis=0)

    myMax = max(maximo)
    myMin = min(maximo)
    if myMin == 0.0 and no_NaN == True:
        maximo[maximo == 0] = 0.0001

    myMin = min(maximo)
    print("Save max vector, Dim =", str(len(maximo)), ', min =', myMin, ', max =', myMax)

    # Get Minimo
    # ----------
    if has_label is True:
        minimo = np.amin(min_i, axis=0)[:-1]
    else:
        minimo = np.amin(min_i, axis=0)

    myMax = max(minimo)
    myMin = min(minimo)
    print("Save min vector, Dim =", str(len(minimo)), ', min =', myMin, ', max =', myMax)

    diferencia = maximo - minimo

    for file in sources:
        with open(file, 'r') as f:
            reader = csv.reader(f)
            dataset = list(reader)

            path_save = file[:-4] + '-norm.csv'
            f = open(path_save, "w")
            for i in range(len(dataset)):
                data = np.float_(dataset[i])

                sample = data[:-1]
                label = data[-1]
                xi = (sample - minimo) / diferencia
                # print(",".join(map(str, np.concatenate((xi, [label]), axis=0))))
                f.write(",".join(map(str, np.concatenate((xi, [label]), axis=0))) + "\n")
            f.close()
            print('Created data normalized,', path_save)


def load_max_csvData(path_max):

    with open(path_max, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    print('Load file ', path_max, ' generate max array.')
    return np.float_(data)[0]


def directory_exist(pathname):
    directory = Path(pathname)
    if not directory.exists():
        os.makedirs(pathname)
        print('El directorio {0} ha sido creado.'.format(pathname))
    else:
        print('El directorio {0} existe.'.format(pathname))


def get_labels_and_predict(objData, y_prob):

    print(np.shape(y_prob))
    ax = np.reshape(objData.labels.values, [objData.total_inputs])
    ax = list(ax)
    f = open("rick.csv", "a+")
    f.write(",".join(map(str, ax)) + "\n")
    f.write(",".join(map(str, y_prob)) + "\n")
    f.close()


def precision_recall_score(y_true, y_pred):

    metrics_multiclass(y_true, y_pred)
    print('--------------------------')

    acc = accuracy_score(y_true, y_pred)
    xprecision = precision_score(y_true, y_pred, average='macro')
    xrecall = recall_score(y_true, y_pred, average='macro')
    fbeta = fbeta_score(y_true, y_pred, average='macro', beta=0.5)

    print('Accuracy:', acc)
    print('Precision:', xprecision)
    print('Recall:', xrecall)
    print('Fbeta:', fbeta)




if __name__ == "__main__":
    test()
