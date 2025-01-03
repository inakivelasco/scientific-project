#!/usr/bin/env python

"""
Script that will generate the final report.

Args:
    --model-name (str): name of the already trained and evaluated model
    --dataset-dir (str): directory of the dataset that will be used for the report
    --extra-intro-dir (str): optional extra information for the introduction. This parameter corresponds to the .txt
                             file path to be shown

Returns:
    The report will be saved in the reports folder as report_{--model-name}_{--dataset-dir}.pdf
"""

import argparse, json, os, random, sys
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
    """
        Changes line height by timeLineSpace times. It will check if it is inside the limits.

        Args:
            timesLineSpace: number of blank lines

        Returns:
            lineHeight global value will be changed accordingly
    """
    global lineHeight, pageNumber
    lineHeight -= timesLineSpace * lineSpace
    if lineHeight < bottomMargin:
        nextPage()

def nextPage():
    """
        Creates a new blank page.

        Returns:
            lineHeight and pageNumber global values will be changed accordingly
    """
    global lineHeight, pageNumber
    report.showPage()
    report.drawRightString(report._pagesize[0] - 20*mm, bottomMargin, str(pageNumber))
    lineHeight = topMargin
    pageNumber += 1

def drawBoldString(width: float, string: str, newSize: int = 12, timesLineSpace: int = 0):
    """
        Writes a bold string at the desired location.

        Args:
            width: horizontal location of the string
            string: text to be written
            newSize: text size (default 12)
            timesLineSpace: number of blank lines to be inserted before the string (default 0)

        Returns:
            Bold string written at the desired location
    """
    global lineHeight
    changeLineHeight(timesLineSpace)
    report.setFont(f'{font}-Bold', newSize)
    report.drawString(width, lineHeight, string)
    report.setFont(font, fontSize)

def drawBoldCentredString(width: float, string: str, newSize: int = 12, timesLineSpace: int = 0):
    """
        Same as drawBoldString() but centered horizontally.

        Args:
            width: horizontal location of the string
            string: text to be written
            newSize: text size (default 12)
            timesLineSpace: number of blank lines to be inserted before the string (default 0)

        Returns:
            Bold string written at the desired location
    """
    global lineHeight
    changeLineHeight(timesLineSpace)
    report.setFont(f'{font}-Bold', newSize)
    report.drawCentredString(width, lineHeight, string)
    report.setFont(font, fontSize)

def writeParagraph(text: str, newSize: int = 12):
    """
        Writes a text paragraph.

        Args:
            text: string to be written
            newSize: text size (default 12)

        Returns:
            Written paragraph
    """
    global lineHeight
    p = Paragraph(text, ParagraphStyle('intro', fontSize=newSize, alignment=TA_JUSTIFY))
    p.wrapOn(report, 500, lineSize)
    changeLineHeight(len(p.blPara.lines) + 1)
    p.drawOn(report, 20 * mm, lineHeight)

def checkImageFits(imageHeight: float):
    """
        Check if the image fits considering the image height and the page top margin.

        Args:
            imageHeight: image height

        Returns:
            lineHeight will be changed if the image doesn't fit and will remain the same if it fits
    """
    global lineHeight
    if lineHeight / mm + imageHeight / mm > topMargin / mm:
        lineHeight = topMargin - imageHeight

def drawFigureTitle(hrzFigCenter: float, string: str):
    """
        Writes the title for a figure.

        Args:
            hrzFigCenter: horizontal coordinate of the figure center
            string: figure title

        Returns:
            Title is written and global variable figureIdx is changed accordingly
    """
    global lineHeight, figureIdx
    report.setFont(font, 8)
    report.drawCentredString(hrzFigCenter, lineHeight, f'Figure {figureIdx}. {string}')
    figureIdx += 1
    report.setFont(font, fontSize)

def drawImage(imagePath: str, x: float, title: str, width: float = 72*mm, height: float = 72*mm):
    """
        Draws an image at the desired location with the desired size.

        Args:
            imagePath: path where image is located
            x: horizontal coordinate where the image will be drawn
            title: image title
            width: image width (default 72*mm) (it preserves its aspect ratio)
            height: image height (default 72*mm) (it preserves its aspect ratio)

        Returns:
            Image with its corresponding title is drawn
    """
    global lineHeight
    image = ImageReader(imagePath)
    report.drawImage(image, x, lineHeight, width=width, height=height, preserveAspectRatio=True)
    drawFigureTitle(x + width // 2, title)

def draw2PlotsPerLine(plotSpecificFolder: str, generalTitle: str, plotFilenames: list, figureTitles: list):
    """
        Draws 2 plots per line, one next to each other. It iterates over the list.

        Args:
            plotSpecificFolder: directory where plots are
            generalTitle: general title for this plot section
            plotFilenames: each of the plots filenames
            figureTitles: each of the plots titles

        Returns:
            Draws 2 plots per line with a main title and specific plot titles
    """
    drawBoldString(20 * mm, generalTitle, 16)

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
    """
        CURRENTLY UNUSED IN THIS SCRIPT
        Writes the whole model architecture read from the corresponding file

        Args:
            modelArchPath: file path where the model architecture is written

        Returns:
            Writes the whole model architecture
        """
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
    datasetEvaluationDir = os.path.abspath(os.path.join(os.pardir, 'models', modelName, 'evaluation', datasetName))
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

    # Introduction -----------------------------------------------------------------------------------------------------
    nextPage()
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
    nextPage()
    trainPlotFilenames = ['classDistributionPiePlot.png', 'apAndDistributionPerClassBarPlot.png', 'apsHrzBarPlot.png',
                          'totalLossPerIter.png']
    trainFigureTitles = ['Class distribution for validation dataset', 'AP and percentage distribution per class',
                         'General AP metrics', 'Training total loss per iteration']
    draw2PlotsPerLine('train', 'Train plots', trainPlotFilenames, trainFigureTitles)

    # Dataset Plots ----------------------------------------------------------------------------------------------------
    nextPage()
    datasetPlotFilenames = ['classDistributionPiePlot.png', 'apAndDistributionPerClassBarPlot.png', 'apsHrzBarPlot.png',
                            'consumptionPlot.png', 'inferenceSpeedPlot.png']
    datasetFigureTitles = [f'Class distribution for {datasetName} dataset', 'AP and percentage distribution per class',
                           'General AP metrics', 'Consumption percentages', 'Inference speed']
    draw2PlotsPerLine(datasetName, f'Dataset plots [{datasetName}]', datasetPlotFilenames, datasetFigureTitles)

    # More dataset metrics ---------------------------------------------------------------------------------------------
    nextPage()
    drawBoldString(20 * mm, f'More dataset [{datasetName}] metrics', 16)
    with open(os.path.join(datasetEvaluationDir, 'apAndAr.txt'), 'r') as f:
        for line in f.readlines():
            writeParagraph(line)

    # Evaluating harwdare specs ----------------------------------------------------------------------------------------
    nextPage()
    drawBoldString(20 * mm, 'Evaluating hardware specs', 16)
    with open(os.path.join(datasetEvaluationDir, 'hardwareSpecs.json'), 'r') as f:
        specs = json.load(f)
    for key in specs.keys():
        drawBoldString(20 * mm, key.upper(), 14, 3)
        for spec in specs[key].values():
            if isinstance(spec, list):
                writeParagraph('\n'.join(spec))
            else:
                writeParagraph(spec)

    # Classified images ------------------------------------------------------------------------------------------------
    nextPage()
    predictedImagesDir = os.path.join(datasetEvaluationDir, 'images')

    mainImages = []
    for file in os.listdir(predictedImagesDir):
        if os.path.isfile(os.path.join(predictedImagesDir, file)):
            mainImages.append(os.path.join(predictedImagesDir, file))

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
    print(f'\nReport generated in {reportPath}')
