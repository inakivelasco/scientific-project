#!/usr/bin/env python
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt

def showPercentagePiePlot(pct, allvalues):
    #  Function to show percentage and absolute value in the pie plot
    absolute = int(pct / 100. * np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, absolute)

def getPiePlotClassDistributionAndSaveTrain(logDir, saveDir):
    names = []
    instances = []
    with open(os.path.join(logDir, 'log.txt'), 'r') as f:
        analyzing = False
        gettingValues = False
        for line in f.readlines():
            if analyzing:
                if gettingValues:
                    if 'total' in line:
                        break
                    line = line.replace('|', '')
                    line = line.replace(':', '')
                    line = line.replace('-', '')
                    line = line.replace('\n', '')
                    for elem in line.split(' '):
                        if len(elem) > 0:
                            if elem.isdigit():
                                instances.append(int(elem))
                            else:
                                names.append(elem)
                if 'category' in line and '#instances' in line:
                    gettingValues = True
            if f'Distribution of instances' in line:
                analyzing = True
    instances = np.array(instances)

    plt.clf()
    plt.figure(figsize=(15, 9))
    plt.pie(instances, labels=names, autopct=lambda pct: showPercentagePiePlot(pct, instances),
            textprops={'fontsize': 12})
    # plt.title('Class distribution for validation dataset')
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.savefig(os.path.join(saveDir, 'classDistributionPiePlot.png'), bbox_inches='tight')

    return names, instances

def getPiePlotClassDistributionAndSaveDataset(logDir, saveDir, datasetName):
    names = []
    instances = []
    with open(os.path.join(logDir, 'log.txt'), 'r') as f:
        analyzing = False
        gettingValues = False
        for line in f.readlines():
            if analyzing:
                if gettingValues:
                    if 'total' in line:
                        break
                    line = line.replace('|', '')
                    line = line.replace(':', '')
                    line = line.replace('-', '')
                    line = line.replace('\n', '')
                    for elem in line.split(' '):
                        if len(elem) > 0:
                            if elem.isdigit():
                                instances.append(int(elem))
                            else:
                                names.append(elem)
                if 'category' in line and '#instances' in line:
                    gettingValues = True
            if f'{datasetName}NewAnnotations' in line:
                analyzing = True
    instances = np.array(instances)

    plt.clf()
    plt.figure(figsize=(15, 9))
    plt.pie(instances, labels=names, autopct=lambda pct: showPercentagePiePlot(pct, instances),
            textprops={'fontsize': 12})
    # plt.title('Class distribution for validation dataset')
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.savefig(os.path.join(saveDir, 'classDistributionPiePlot.png'), bbox_inches='tight')

    return names, instances

def getBarPlotApPerCategoryAndSaveTrain(apMetrics, classNames, nClassInstances, saveDir):
    x = np.arange(len(classNames))  # label locations
    width = 0.35  # bars width

    plt.clf()
    plt.figure(figsize=(15, 9))
    fig, ax = plt.subplots(figsize=(15, 9))

    apPerClass = np.around(list(apMetrics.values())[1:1 + len(classNames)], 2)

    # Reorder apMetrics for this graph (necessary)
    apMetricsClassNames = [i.split('-')[1] for i in list(apMetrics.keys())[1:1 + len(classNames)]]
    apPerClassOrder = []
    for name in apMetricsClassNames:
        apPerClassOrder.append(classNames.index(name))
    apPerClass = np.array(apPerClass)[apPerClassOrder]

    idxOrder = np.argsort(apPerClass)

    rects1 = ax.bar(x - width / 2, apPerClass[idxOrder], width, label='AP', color='#FF7276', edgecolor='#540B0C')
    rects2 = ax.bar(x + width / 2, np.round(nClassInstances[idxOrder] / np.sum(nClassInstances) * 100, 2), width,
                    label='Percentage of total instances', color='#ADD8E6', edgecolor='b')
    ax.bar_label(rects1, padding=4, size=12)
    ax.bar_label(rects2, padding=4, size=12)

    # ax.set_title('Scores per class')
    ax.set_xticks(x, np.array(classNames)[idxOrder])
    ax.set_ylabel('')
    ax.legend(loc='upper left', prop={'size': 20})
    plt.xticks(size=20)
    plt.yticks(size=20)
    fig.savefig(os.path.join(saveDir, 'apAndDistributionPerClassBarPlot.png'), bbox_inches='tight')

def getBarPlotApPerCategoryAndSaveDataset(datasetEvaluationDir, datasetPlotDir):
    aps = []
    names = []
    with open(os.path.join(datasetEvaluationDir, 'log.txt'), 'r') as f:
        analyzing = False
        gettingValues = False
        for line in f.readlines():
            if analyzing:
                if gettingValues:
                    if 'Loading' in line:
                        break
                    line = line.replace('|', '')
                    line = line.replace(':', '')
                    line = line.replace('-', '')
                    line = line.replace('\n', '')
                    for elem in line.split(' '):
                        if len(elem) > 0:
                            if elem not in classNames:
                                aps.append(float(elem))
                            else:
                                names.append(elem)
                if 'category' in line and 'AP' in line:
                    gettingValues = True
            if 'Per-category bbox AP' in line:
                analyzing = True

    x = np.arange(len(names))  # label locations
    width = 0.35  # bars width

    plt.clf()
    plt.figure(figsize=(15, 9))
    fig, ax = plt.subplots(figsize=(15, 9))

    apPerClass = np.around(aps, 2)

    # Reorder apMetrics for this graph (necessary)
    apPerClassOrder = []
    for name in names:
        apPerClassOrder.append(names.index(name))
    apPerClass = np.array(apPerClass)[apPerClassOrder]

    idxOrder = np.argsort(apPerClass)

    rects1 = ax.bar(x - width / 2, apPerClass[idxOrder], width, label='AP', color='#FF7276', edgecolor='#540B0C')
    rects2 = ax.bar(x + width / 2, np.round(nClassInstances[idxOrder] / np.sum(nClassInstances) * 100, 2), width,
                    label='Percentage of total instances', color='#ADD8E6', edgecolor='b')
    ax.bar_label(rects1, padding=4, size=12)
    ax.bar_label(rects2, padding=4, size=12)

    # ax.set_title('Scores per class')
    ax.set_xticks(x, np.array(names)[idxOrder])
    ax.set_ylabel('')
    plt.xticks(size=20)
    plt.yticks(size=20)
    ax.legend(loc='upper left', prop={'size': 20})
    fig.savefig(os.path.join(datasetPlotDir, 'apAndDistributionPerClassBarPlot.png'), bbox_inches='tight')

def getHorizontalBarPlotDiffApMetricsTrain(apMetrics, classNames, saveDir):
    plt.clf()
    plt.figure(figsize=(15, 9))
    aps = []
    apsLabels = []
    for apLabel in apMetrics.keys():
        metric2save = True
        for className in classNames:
            if className in apLabel or apLabel in 'iteration':
                metric2save = False
        if metric2save:
            apsLabels.append(apLabel)
            aps.append(apMetrics[apLabel])
    aps = np.around(aps, 2)

    plt.barh(apsLabels, aps, color='#ADD8E6', edgecolor='b')
    # plt.title('AP general metrics')
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlim(0, 100)
    plt.xlabel('%', size=20)
    plt.savefig(os.path.join(saveDir, 'apsHrzBarPlot.png'), bbox_inches='tight')

def getHorizontalBarPlotDiffApMetricsDataset(datasetEvaluationDir, datasetPlotDir):
    aps = []
    names = []
    with open(os.path.join(datasetEvaluationDir, 'log.txt'), 'r') as f:
        gettingValues = False
        for line in f.readlines():
            if gettingValues:
                if 'evaluation' in line:
                    break
                line = line.replace('|', '')
                line = line.replace(':', '')
                line = line.replace('-', '')
                line = line.replace('\n', '')
                for elem in line.split(' '):
                    if len(elem) > 0:
                        if elem.replace('.', '').isdigit():
                            aps.append(float(elem))
                        else:
                            names.append(elem)
            if 'Evaluation results for bbox' in line:
                gettingValues = True

    aps = np.around(aps, 2)
    plt.clf()
    plt.barh(names, aps, color='#ADD8E6', edgecolor='b')
    # plt.title('AP general metrics')
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlim(0, 100)
    plt.xlabel('%', size=20)
    plt.savefig(os.path.join(datasetPlotDir, 'apsHrzBarPlot.png'), bbox_inches='tight')

def getInferenceSpeedPlot(datasetEvaluationDir, datasetPlotDir):
    fps = []
    with open(os.path.join(datasetEvaluationDir, 'log.txt'), 'r') as f:
        gettingValues = False
        for line in f.readlines():
            if gettingValues:
                if 'Total inference time:' in line:
                    break
                if len(line) > 0:
                    fps.append(np.round(2/float(line.split(' ')[11]), 2))
            if 'Start inference on' in line:
                gettingValues = True

    plt.clf()
    plt.figure(figsize=(15, 9))
    # plt.title('Inference speed')
    plt.plot(fps, label='CPU %')
    plt.xlabel('Inference step', size=20)
    plt.ylabel('FPS', size=20)
    plt.xlim(0, len(fps)-1)
    plt.ylim(np.min(fps)-4*np.std(fps), np.max(fps)+4*np.std(fps))
    plt.savefig(os.path.join(datasetPlotDir, 'inferenceSpeedPlot.png'), bbox_inches='tight')

def getConsumptionPlot(datasetPlotDir):
    with open(os.path.join(datasetEvaluationDir, 'consumptionLog.txt'), 'r') as f:
        consumptions = [triplet[:-1] for triplet in f.readlines()[1:]]

    cpuUsage = []
    ramUsage = []
    gpuUsage = []
    for triplet in consumptions:
        values = triplet.split(' ')
        cpuUsage.append(float(values[0]))
        ramUsage.append(float(values[1]))
        gpuUsage.append(float(values[2]))

    xValues = np.arange(0, len(cpuUsage)/2, 0.5)  # It depends on threading.Timer in evaluate_lazy_config.py

    plt.clf()
    plt.figure(figsize=(15, 9))
    plt.plot(xValues, cpuUsage, label='CPU %')
    plt.plot(xValues, ramUsage, label='RAM %')
    plt.plot(xValues, gpuUsage, label='GPU %')
    # plt.title('Consumption percentages')
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlim(0, len(cpuUsage)//2)
    plt.ylim(0, 100)
    plt.xlabel('Time (s)', size=20)
    plt.ylabel('Usage (%)', size=20)
    plt.legend(loc="lower center", prop={'size': 20})
    plt.savefig(os.path.join(datasetPlotDir, 'consumptionPlot.png'), bbox_inches='tight')

def getApAndArMetrics(datasetEvaluationDir):
    apAndArTxt = ''
    with open(os.path.join(datasetEvaluationDir, 'log.txt'), 'r') as f:
        gettingValues = False
        for line in f.readlines():
            if gettingValues:
                if 'd2.evaluation.testing' in line:
                    break
                if len(line) > 0:
                    apAndArTxt = f'{apAndArTxt}{line}'
            if 'index created!' in line:
                gettingValues = True

    with open(os.path.join(datasetEvaluationDir, 'apAndAr.txt'), 'w') as f:
        f.write(apAndArTxt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate report script')
    parser.add_argument("--model-name", help="Model name (type str)", type=str, required=True)
    parser.add_argument("--dataset-name", help="Dataset name (type str)", type=str, required=True)
    args = parser.parse_args()

    modelName = args.model_name
    datasetName = args.dataset_name

    # Prepare directories ----------------------------------------------------------------------------------------------
    modelDir = os.path.abspath(os.path.join(os.pardir, 'models', modelName))
    datasetEvaluationDir = os.path.abspath(os.path.join(os.pardir, 'models', modelName, 'evaluation', datasetName))

    trainPlotDir = os.path.join(modelDir, 'plots', 'train')
    datasetPlotDir = os.path.join(modelDir, 'plots', datasetName)
    os.makedirs(trainPlotDir, exist_ok=True)
    os.makedirs(datasetPlotDir, exist_ok=True)

    # Train graphs -----------------------------------------------------------------------------------------------------
    # Graphs pre-inference -----
    with open(os.path.join(modelDir, 'metrics.json'), 'r') as f:
        metrics = [json.loads(line) for line in f]
        iterations = [iterData['iteration'] for iterData in metrics[:-1]]
        apMetrics = metrics[-1]

        # Total loss / iter
        totalLoss = [iterData['total_loss'] for iterData in metrics[:-1]]
        plt.figure(figsize=(15, 9))
        plt.plot(iterations, totalLoss)
        # plt.title('Training total_loss')
        plt.xlabel('Iteration', size=20)
        plt.ylabel('Total loss', size=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.savefig(os.path.join(trainPlotDir, 'totalLossPerIter.png'), bbox_inches='tight')

        # Some graphs for the FCOS models
        loss_fcos_cls = [iterData['loss_fcos_cls'] for iterData in metrics[:-1]]
        plt.figure(figsize=(15, 9))
        plt.plot(iterations, loss_fcos_cls)
        # plt.title('Training loss_fcos_cls')
        plt.xlabel('Iteration', size=20)
        plt.ylabel('loss_fcos_cls', size=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.savefig(os.path.join(trainPlotDir, 'lossFcosClsPerIter.png'), bbox_inches='tight')

        loss_fcos_ctr = [iterData['loss_fcos_ctr'] for iterData in metrics[:-1]]
        plt.figure(figsize=(15, 9))
        plt.plot(iterations, loss_fcos_ctr)
        # plt.title('Training loss_fcos_ctr')
        plt.xlabel('Iteration', size=20)
        plt.ylabel('loss_fcos_ctr', size=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.savefig(os.path.join(trainPlotDir, 'lossFcosCtrPerIter.png'), bbox_inches='tight')

        loss_fcos_loc = [iterData['loss_fcos_loc'] for iterData in metrics[:-1]]
        plt.figure(figsize=(15, 9))
        plt.plot(iterations, loss_fcos_loc)
        # plt.title('Training loss_fcos_loc')
        plt.xlabel('Iteration', size=20)
        plt.ylabel('loss_fcos_loc', size=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.savefig(os.path.join(trainPlotDir, 'lossFcosLocPerIter.png'), bbox_inches='tight')

    # Pie plot for class distribution (the count for some classes is incorrect, just by 1)
    classNames, nClassInstances = getPiePlotClassDistributionAndSaveTrain(modelDir, trainPlotDir)

    # Bar plot for AP per category
    getBarPlotApPerCategoryAndSaveTrain(apMetrics, classNames, nClassInstances, trainPlotDir)

    # Horizontal bar plot for different AP metrics
    getHorizontalBarPlotDiffApMetricsTrain(apMetrics, classNames, trainPlotDir)

    # Dataset graphs ---------------------------------------------------------------------------------------------------
    # Pie plot for class distribution (the count for some classes is incorrect, just by 1)
    classNames, nClassInstances = getPiePlotClassDistributionAndSaveDataset(datasetEvaluationDir, datasetPlotDir,
                                                                            datasetName)

    # Bar plot for AP per category
    getBarPlotApPerCategoryAndSaveDataset(datasetEvaluationDir, datasetPlotDir)

    # Horizontal bar plot for different AP metrics
    getHorizontalBarPlotDiffApMetricsDataset(datasetEvaluationDir, datasetPlotDir)

    # CPU RAM GPU Consumption plots
    getConsumptionPlot(datasetPlotDir)

    # Inference speed plot
    getInferenceSpeedPlot(datasetEvaluationDir, datasetPlotDir)

    # AP and AR metrics ------------------------------------------------------------------------------------------------
    getApAndArMetrics(datasetEvaluationDir)

