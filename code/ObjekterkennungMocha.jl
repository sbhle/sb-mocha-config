classes = open("/<Pfad zu>/ilsvrc_words.txt")
classifier = ImageClassifier(netDef, :prob, channel_order=(3,2,1), classes=classes)
img = imread("/<Pfad zu>/alpaca.jpeg")
prob, class = classify(classifier, img)
