#!/usr/bin/env python
import argparse, os, random, sys
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import ParagraphStyle


global lineHeight, figureIdx, pageNumber
lineHeight = 200 * mm
figureIdx = 1
pageNumber = 1

def changeLineHeight(timesLineSpace: int):
    global lineHeight, pageNumber
    lineHeight -= timesLineSpace * lineSpace
    if lineHeight < bottomMargin:
        nextPage()

def nextPage():
    global lineHeight, pageNumber
    report.showPage()
    report.drawRightString(report._pagesize[0] - 20*mm, bottomMargin, str(pageNumber))
    lineHeight = topMargin
    pageNumber += 1

def drawBoldString(width: int, string: str, newSize: int = 12, timesLineSpace: int = 0):
    global lineHeight, pageNumber
    changeLineHeight(timesLineSpace)
    report.setFont(f'{font}-Bold', newSize)
    report.drawString(width, lineHeight, string)
    report.setFont(font, fontSize)

def drawBoldCentredString(width: int, string: str, newSize: int = 12, timesLineSpace: int = 0):
    global lineHeight
    changeLineHeight(timesLineSpace)
    report.setFont(f'{font}-Bold', newSize)
    report.drawCentredString(width, lineHeight, string)
    report.setFont(font, fontSize)

def writeParagraph(text: str, newSize: int = 12):
    global lineHeight
    p = Paragraph(text, ParagraphStyle('intro', fontSize=newSize, alignment=TA_JUSTIFY))
    p.wrapOn(report, 500, lineSize)
    changeLineHeight(len(p.blPara.lines) + 1)
    p.drawOn(report, 20 * mm, lineHeight)

def checkImageFits(imageHeight: int):
    global lineHeight
    if lineHeight / mm + imageHeight / mm > topMargin / mm:
        lineHeight = topMargin - imageHeight

def drawFigureTitle(hrzFigCenter: int, height: int, string: str):
    global figureIdx
    report.setFont(font, 8)
    report.drawCentredString(hrzFigCenter, height, f'Figure {figureIdx}. {string}')
    figureIdx += 1
    report.setFont(font, fontSize)

def drawImage(imagePath : str, x: int, title: str, width: int = 72*mm, height: int = 72*mm):
    global lineHeight
    image = ImageReader(imagePath)
    report.drawImage(image, x, lineHeight, width=width, height=height, preserveAspectRatio=True)
    drawFigureTitle(x + width // 2, lineHeight, title)

def draw2PlotsPerLine(plotSpecificFolder: str, generalTitle: str, plotFilenames: list, figureTitles: list):
    global lineHeight, figureIdx
    drawBoldString(20 * mm, generalTitle, 16, 5)

    plotsDir = os.path.abspath(os.path.join(os.pardir, 'models', modelName, 'plots', plotSpecificFolder))

    for i, plotName in enumerate(plotFilenames):
        if i % 2 == 0:
            if i == 0:
                changeLineHeight(20)
            else:
                changeLineHeight(18)
            checkImageFits(72 * mm)
            drawImage(os.path.join(plotsDir, plotName), 20 * mm, figureTitles[i])
        else:
            drawImage(os.path.join(plotsDir, plotName), 116 * mm, figureTitles[i])

def writeModelArchitecture(modelArchPath: str):
    # This function is UNUSED in this script
    global lineHeight
    nextPage()
    drawBoldString(20 * mm, 'Model Architecture', 16)
    with open(modelArchPath, 'r') as f:
        changeLineHeight(2)
        for line in f.readlines():
            report.drawString(20 * mm, lineHeight, line[:-1])
            changeLineHeight(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate report script')
    parser.add_argument("--model-name", help="Model name (type str)", type=str, required=True)
    parser.add_argument("--dataset-dir", help="Full dataset directory (type str)", type=str, required=True)
    parser.add_argument("--extra-intro-dir", help="Optional extra introduction text directory (type str)", type=str,
                        required=False, default='This is extra optional introduction. '*10)
    args = parser.parse_args()

    modelName = args.model_name
    datasetDir = args.dataset_dir
    datasetName = datasetDir.replace('\\', '/').split('/')[-1]
    extraIntro = args.extra_intro_dir

    # PDF general settings ---------------------------------------------------------------------------------------------
    fileName = f'report_{modelName}_{datasetName}.pdf'
    documentTitle = 'Scientific Project report'

    font = 'Helvetica'
    fontSize = 12
    topMargin = 280 * mm
    bottomMargin = 20 * mm
    leftMargin = 20 * mm
    lineSize = 90
    lineSpace = 4 * mm

    # Generation -------------------------------------------------------------------------------------------------------
    reportPath = os.path.abspath(os.path.join(os.pardir, 'reports', fileName))
    report = canvas.Canvas(reportPath, pagesize=A4)
    report.setFont(font, fontSize)
    report.setTitle(documentTitle)

    drawBoldCentredString(100 * mm, f'Report', 20)
    drawBoldCentredString(100 * mm, f'Model: {modelName}', 16, 2)
    drawBoldCentredString(100 * mm, f'Dataset: {datasetName}', 16, 2)
    nextPage()

    # Introduction -----------------------------------------------------------------------------------------------------
    lineHeight = topMargin
    drawBoldString(20 * mm, 'Introduction', 16)

    drawBoldString(20 * mm, 'Dataset description', 14, 3)
    with open(os.path.join(datasetDir, 'description.txt'), 'r') as f:
        writeParagraph(f.read())

    drawBoldString(20 * mm, 'Extra information', 14, 3)
    if '--extra-intro-dir' in sys.argv:
        with open(args.extra_intro_dir, 'r') as f:
            extraIntro = f.read()
    writeParagraph(extraIntro)

    # Train Plots ------------------------------------------------------------------------------------------------------
    trainPlotFilenames = ['classDistributionPiePlot.png', 'apAndDistributionPerClassBarPlot.png', 'apsHrzBarPlot.png',
                          'totalLossPerIter.png']
    trainFigureTitles = ['Class distribution for validation dataset', 'AP and percentage distribution per class',
                         'General AP metrics', 'Training total loss per iteration']
    draw2PlotsPerLine('train', 'Train plots', trainPlotFilenames, trainFigureTitles)

    # Dataset Plots ----------------------------------------------------------------------------------------------------
    datasetPlotFilenames = ['classDistributionPiePlot.png', 'apAndDistributionPerClassBarPlot.png', 'apsHrzBarPlot.png']
    datasetFigureTitles = [f'Class distribution for {datasetName} dataset', 'AP and percentage distribution per class',
                           'General AP metrics']
    draw2PlotsPerLine(datasetName, f'Dataset plots [{datasetName}]', datasetPlotFilenames, datasetFigureTitles)

    # Classified images ------------------------------------------------------------------------------------------------
    predictedImagesDir = os.path.abspath(os.path.join(os.pardir, 'models', modelName, 'evaluation', datasetName,
                                                      'images'))

    mainImages = []
    for file in os.listdir(predictedImagesDir):
        if os.path.isfile(os.path.join(predictedImagesDir, file)):
            mainImages.append(os.path.join(predictedImagesDir, file))

    nextPage()
    drawBoldString(20 * mm, 'Images', 16)

    # Images with predictions of every class -----
    drawBoldString(20 * mm, 'Predictions for every class', 14, 3)
    for i, imagePath in enumerate(random.sample(mainImages, 4)):
        imageName = imagePath.replace('\\', '/').split('/')[-1]
        if i % 2 == 0:
            changeLineHeight(18)
            checkImageFits(72 * mm)
            drawImage(imagePath, 20 * mm, f'Predictions for every class [{imageName}]')
        else:
            drawImage(imagePath, 116 * mm, f'Predictions for every class [{imageName}]')

    # Images with predictions for each class separately -----
    classLabels = ['Bark', 'Maize', 'Weeds']  # HARDCODED
    for classLabel in classLabels:
        drawBoldString(20 * mm, classLabel, 14, 3)
        for i, imageName in enumerate(random.sample(os.listdir(os.path.join(predictedImagesDir, classLabel)), 4)):
            imagePath = os.path.join(predictedImagesDir, classLabel, imageName)
            if i % 2 == 0:
                changeLineHeight(18)
                checkImageFits(72 * mm)
                drawImage(imagePath, 20 * mm, f'Predictions for {classLabel} [{imageName}]')
            else:
                drawImage(imagePath, 116 * mm, f'Predictions for {classLabel} [{imageName}]')

    report.save()
    print(f'Report generated in {reportPath}')
