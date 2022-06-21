import argparse, os, random

# import reportlab
import sys

from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.utils import ImageReader


def changeLineHeight(lineHeight: int, timesLineSpace: int):
    lineHeight -= timesLineSpace * lineSpace
    if lineHeight < bottomMargin:
        report.showPage()
        lineHeight = topMargin
    return lineHeight

def nextPage():
    report.showPage()
    lineHeight = topMargin
    return lineHeight

def drawBoldString(width: int, lineHeight: int, string: str, newSize: int = 12, timesLineSpace: int = 0):
    lineHeight = changeLineHeight(lineHeight, timesLineSpace)
    report.setFont(f'{font}-Bold', newSize)
    report.drawString(width, lineHeight, string)
    report.setFont(font, fontSize)
    return lineHeight

def drawBoldCentredString(width: int, lineHeight: int, string: str, newSize: int = 12, timesLineSpace: int = 0):
    lineHeight = changeLineHeight(lineHeight, timesLineSpace)
    report.setFont(f'{font}-Bold', newSize)
    report.drawCentredString(width, lineHeight, string)
    report.setFont(font, fontSize)
    return lineHeight

def writeParagraph(text: str, lineHeight: int, newSize: int = 12):
    p = Paragraph(text, ParagraphStyle('intro', fontSize=newSize, alignment=TA_JUSTIFY))
    p.wrapOn(report, 500, lineSize)
    lineHeight = changeLineHeight(lineHeight, len(p.blPara.lines) + 1)
    p.drawOn(report, 20 * mm, lineHeight)
    return lineHeight

def checkImageFits(lineHeight: int, imageHeight: int):
    if lineHeight / mm + imageHeight / mm > topMargin / mm:
        lineHeight = topMargin - imageHeight
    return lineHeight

def drawFigureTitle(hrzFigCenter: int, height: int, string: str, figureIdx: int = None):
    report.setFont(font, 8)
    if figureIdx is not None:
        report.drawCentredString(hrzFigCenter, height, f'Figure {figureIdx}. {string}')
        figureIdx += 1
    else:
        report.drawCentredString(hrzFigCenter, height, string)
    report.setFont(font, fontSize)
    return figureIdx

def drawImage(imagePath : str, x: int, y: int, title: str, figureIdx: int, width: int = 72*mm, height: int = 72*mm):
    image = ImageReader(imagePath)
    report.drawImage(image, x, y, width=width, height=height,
                     preserveAspectRatio=True)
    figureIdx = drawFigureTitle(x + width / 2, lineHeight, title, figureIdx)
    return figureIdx

def writeModelArchitecture(modelArchPath: str):
    # This function is UNUSED in this script
    lineHeight = nextPage()
    drawBoldString(20 * mm, lineHeight, 'Model Architecture', 16)
    with open(modelArchPath, 'r') as f:
        lineHeight = changeLineHeight(lineHeight, 2)
        for line in f.readlines():
            report.drawString(20 * mm, lineHeight, line[:-1])
            lineHeight = changeLineHeight(lineHeight, 1)
    return lineHeight


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate report script')
    parser.add_argument("--dataset-dir", help="Full dataset directory (type str)", type=str, required=True)
    parser.add_argument("--model-name", help="Model name (type str)", type=str, required=True)
    parser.add_argument("--extra-intro-dir", help="Optional extra introduction text directory (type str)", type=str,
                        required=False, default='This is extra optional introduction. '*10)
    args = parser.parse_args()

    modelName = args.model_name
    datasetDir = args.dataset_dir
    datasetName = datasetDir.replace('\\', '/').split('/')[-1]
    extraIntro = args.extra_intro_dir

    fileName = f'report_{modelName}_{datasetName}.pdf'
    documentTitle = 'Scientific Project report'

    # PDF general settings
    font = 'Helvetica'
    fontSize = 12
    topMargin = 280 * mm
    bottomMargin = 20 * mm
    leftMargin = 20 * mm
    lineSize = 90
    lineSpace = 4 * mm
    lineHeight = 280 * mm
    figureIdx = 1

    # Generation -------------------------------------------------------------------------------------------------------
    report = canvas.Canvas(os.path.abspath(os.path.join(os.pardir, 'reports', fileName)), pagesize=A4)
    report.setFont(font, fontSize)
    report.setTitle(documentTitle)

    lineHeight = drawBoldCentredString(100 * mm, lineHeight, f'Report', 20)
    lineHeight = drawBoldCentredString(100 * mm, lineHeight, f'Model: {modelName}', 16, 2)
    lineHeight = drawBoldCentredString(100 * mm, lineHeight, f'Dataset: {datasetName}', 16, 2)

    # Introduction -----------------------------------------------------------------------------------------------------
    lineHeight = 252 * mm
    lineHeight = drawBoldString(20 * mm, lineHeight, 'Introduction', 16)

    lineHeight = drawBoldString(20 * mm, lineHeight, 'Dataset description', 14, 3)
    with open(os.path.join(datasetDir, 'description.txt'), 'r') as f:
        lineHeight = writeParagraph(f.read(), lineHeight)

    lineHeight = drawBoldString(20 * mm, lineHeight, 'Extra information', 14, 3)
    if '--extra-intro-dir' in sys.argv:
        with open(args.extra_intro_dir, 'r') as f:
            extraIntro = f.read()
    lineHeight = writeParagraph(extraIntro, lineHeight)

    # Train Plots ------------------------------------------------------------------------------------------------------
    lineHeight = drawBoldString(20 * mm, lineHeight, 'Train plots', 16, 5)

    trainPlotsDir = os.path.abspath(os.path.join(os.pardir, 'models', modelName, 'plots', 'train'))

    lineHeight = changeLineHeight(lineHeight, 20)
    lineHeight = checkImageFits(lineHeight, 72 * mm)
    figureIdx = drawImage(os.path.join(trainPlotsDir, 'classDistributionPiePlot.png'), 20 * mm, lineHeight,
                          'Class distribution for validation dataset', figureIdx)

    figureIdx = drawImage(os.path.join(trainPlotsDir, 'apAndDistributionPerClassBarPlot.png'), 116 * mm, lineHeight,
                          'AP and percentage distribution per class', figureIdx)

    lineHeight = changeLineHeight(lineHeight, 18)
    lineHeight = checkImageFits(lineHeight, 72 * mm)
    figureIdx = drawImage(os.path.join(trainPlotsDir, 'apsHrzBarPlot.png'), 20 * mm, lineHeight,
                          'General AP metrics', figureIdx)

    figureIdx = drawImage(os.path.join(trainPlotsDir, 'totalLossPerIter.png'), 116 * mm, lineHeight,
                          'Training total loss per iteration', figureIdx)

    # Dataset Plots ----------------------------------------------------------------------------------------------------
    lineHeight = drawBoldString(20 * mm, lineHeight, f'Dataset plots [{datasetName}]', 16, 5)

    datasetPlotsDir = os.path.abspath(os.path.join(os.pardir, 'models', modelName, 'plots', datasetName))

    lineHeight = changeLineHeight(lineHeight, 20)
    lineHeight = checkImageFits(lineHeight, 72 * mm)
    figureIdx = drawImage(os.path.join(datasetPlotsDir, 'classDistributionPiePlot.png'), 20 * mm, lineHeight,
                          f'Class distribution for {datasetName} dataset', figureIdx)

    figureIdx = drawImage(os.path.join(datasetPlotsDir, 'apAndDistributionPerClassBarPlot.png'), 116 * mm, lineHeight,
                          'AP and percentage distribution per class', figureIdx)

    lineHeight = changeLineHeight(lineHeight, 18)
    lineHeight = checkImageFits(lineHeight, 72 * mm)
    figureIdx = drawImage(os.path.join(datasetPlotsDir, 'apsHrzBarPlot.png'), 20 * mm, lineHeight,
                          'General AP metrics', figureIdx)

    # Classified images ------------------------------------------------------------------------------------------------
    predictedImagesDir = os.path.abspath(os.path.join(os.pardir, 'models', modelName, 'evaluation', datasetName,
                                                      'images'))

    mainImages = []
    for file in os.listdir(predictedImagesDir):
        if os.path.isfile(os.path.join(predictedImagesDir, file)):
            mainImages.append(os.path.join(predictedImagesDir, file))

    lineHeight = nextPage()
    lineHeight = drawBoldString(20 * mm, lineHeight, 'Images', 16)

    lineHeight = drawBoldString(20 * mm, lineHeight, 'Predictions for every class', 14, 3)
    for i, imagePath in enumerate(random.sample(mainImages, 4)):
        if i % 2 == 0:
            lineHeight = changeLineHeight(lineHeight, 18)
            lineHeight = checkImageFits(lineHeight, 72 * mm)
            figureIdx = drawImage(imagePath, 20 * mm, lineHeight, f'Predictions for every class', figureIdx)
        else:
            figureIdx = drawImage(imagePath, 116 * mm, lineHeight, f'Predictions for every class', figureIdx)

    classLabels = ['Bark', 'Maize', 'Weeds']
    for classLabel in classLabels:
        lineHeight = drawBoldString(20 * mm, lineHeight, classLabel, 14, 3)
        for i, imageName in enumerate(random.sample(os.listdir(os.path.join(predictedImagesDir, classLabel)), 4)):
            imagePath = os.path.join(predictedImagesDir, classLabel, imageName)
            if i % 2 == 0:
                lineHeight = changeLineHeight(lineHeight, 18)
                lineHeight = checkImageFits(lineHeight, 72 * mm)
                figureIdx = drawImage(imagePath, 20 * mm, lineHeight, f'Predictions for {classLabel}', figureIdx)
            else:
                figureIdx = drawImage(imagePath, 116 * mm, lineHeight, f'Predictions for {classLabel}', figureIdx)

    report.save()
