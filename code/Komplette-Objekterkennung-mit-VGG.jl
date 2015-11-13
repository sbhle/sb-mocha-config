########################################
# Konfiguration der Umgebung
########################################
ENV["MOCHA_USE_NATIVE_EXT"] = "true"

using Mocha

backend = CPUBackend()
init(backend)

img_width, img_height, img_channels = (256, 256, 3)
crop_size = (224, 224)
batch_size = 1

println("ACHTUNG: Pfade müssen angepasst werden!")

########################################
# Erstellen der Netzwerkarchitektur
########################################
layers = [
    MemoryDataLayer(name="data", tops=[:data], batch_size=batch_size,
                    transformers=[(:data, DataTransformers.Scale(scale=255)), (:data, DataTransformers.SubMean(mean_file="/home/sbhle/Mocha.jl/examples/ijulia/ilsvrc12/model/ilsvrc12_mean.hdf5"))],
                    data = Array[zeros(img_width, img_height, img_channels, batch_size)])
    CropLayer(name="crop", tops=[:cropped], bottoms=[:data], crop_size=crop_size)
    ConvolutionLayer(name="conv1_1", tops=[:conv1_1], bottoms=[:cropped], kernel=(3,3), pad=(1,1), n_filter=64, neuron=Neurons.ReLU())
    ConvolutionLayer(name="conv1_2", tops=[:conv1_2], bottoms=[:conv1_1], kernel=(3,3), pad=(1,1), n_filter=64, neuron=Neurons.ReLU())
    PoolingLayer(name="pool1", tops=[:pool1], bottoms=[:conv1_2], kernel=(2,2), stride=(2,2), pooling=Pooling.Max())
    ConvolutionLayer(name="conv2_1", tops=[:conv2_1], bottoms=[:pool1], kernel=(3,3), pad=(1,1), n_filter=128, neuron=Neurons.ReLU())
    ConvolutionLayer(name="conv2_2", tops=[:conv2_2], bottoms=[:conv2_1], kernel=(3,3), pad=(1,1), n_filter=128, neuron=Neurons.ReLU())
    PoolingLayer(name="pool2", tops=[:pool2], bottoms=[:conv2_2], kernel=(2,2), stride=(2,2), pooling=Pooling.Max())
    ConvolutionLayer(name="conv3_1", tops=[:conv3_1], bottoms=[:pool2], kernel=(3,3), pad=(1,1), n_filter=256, neuron=Neurons.ReLU())
    ConvolutionLayer(name="conv3_2", tops=[:conv3_2], bottoms=[:conv3_1], kernel=(3,3), pad=(1,1), n_filter=256, neuron=Neurons.ReLU())
    ConvolutionLayer(name="conv3_3", tops=[:conv3_3], bottoms=[:conv3_2], kernel=(3,3), pad=(1,1), n_filter=256, neuron=Neurons.ReLU())
    ConvolutionLayer(name="conv3_4", tops=[:conv3_4], bottoms=[:conv3_3], kernel=(3,3), pad=(1,1), n_filter=256, neuron=Neurons.ReLU())
    PoolingLayer(name="pool3", tops=[:pool3], bottoms=[:conv3_4], kernel=(2,2), stride=(2,2), pooling=Pooling.Max())
    ConvolutionLayer(name="conv4_1", tops=[:conv4_1], bottoms=[:pool3], kernel=(3,3), pad=(1,1), n_filter=512, neuron=Neurons.ReLU())
    ConvolutionLayer(name="conv4_2", tops=[:conv4_2], bottoms=[:conv4_1], kernel=(3,3), pad=(1,1), n_filter=512, neuron=Neurons.ReLU())
    ConvolutionLayer(name="conv4_3", tops=[:conv4_3], bottoms=[:conv4_2], kernel=(3,3), pad=(1,1), n_filter=512, neuron=Neurons.ReLU())
    ConvolutionLayer(name="conv4_4", tops=[:conv4_4], bottoms=[:conv4_3], kernel=(3,3), pad=(1,1), n_filter=512, neuron=Neurons.ReLU())
    PoolingLayer(name="pool4", tops=[:pool4], bottoms=[:conv4_4], kernel=(2,2), stride=(2,2), pooling=Pooling.Max())
    ConvolutionLayer(name="conv5_1", tops=[:conv5_1], bottoms=[:pool4], kernel=(3,3), pad=(1,1), n_filter=512, neuron=Neurons.ReLU())
    ConvolutionLayer(name="conv5_2", tops=[:conv5_2], bottoms=[:conv5_1], kernel=(3,3), pad=(1,1), n_filter=512, neuron=Neurons.ReLU())
    ConvolutionLayer(name="conv5_3", tops=[:conv5_3], bottoms=[:conv5_2], kernel=(3,3), pad=(1,1), n_filter=512, neuron=Neurons.ReLU())
    ConvolutionLayer(name="conv5_4", tops=[:conv5_4], bottoms=[:conv5_3], kernel=(3,3), pad=(1,1), n_filter=512, neuron=Neurons.ReLU())
    PoolingLayer(name="pool5", tops=[:pool5], bottoms=[:conv5_4], kernel=(2,2), stride=(2,2), pooling=Pooling.Max())
    InnerProductLayer(name="fc6", tops=[:fc6], bottoms=[:pool5], output_dim=4096, neuron=Neurons.ReLU())
    DropoutLayer(name="drop6", bottoms=[:fc6], ratio=0.5)
    InnerProductLayer(name="fc7", tops=[:fc7], bottoms=[:fc6], output_dim=4096, neuron=Neurons.ReLU())
    DropoutLayer(name="drop7", bottoms=[:fc7], ratio=0.5)
    InnerProductLayer(name="fc8", tops=[:fc8], bottoms=[:fc7], output_dim=1000)
    SoftmaxLayer(name="prob", tops=[:prob], bottoms=[:fc8])
]

net = Net("VGG", backend, layers)

########################################
# Drucken/Bilderzeugung der Netzwerkarchitektur
########################################
# println(net)

# open("net.dot", "w") do out net2dot(out, net) end
# run(pipeline(`dot -Tpng net.dot`, "net.png"))

# using Images
# imread("net.png")

########################################
# Importieren von Gewichtungen
########################################
using HDF5
h5open("/home/sbhle/caffe/test_convert/vgg_19layer.hdf5", "r") do h5
  load_network(h5, net)
end
init(net)

########################################
# Training mit neuen Daten fortsetzen
########################################

# solver
base_dir = "VGG"
method = SGD()
train_params  = make_solver_parameters(method, max_iter=div(60000,batch_size),
  regu_coef=0.0, mom_policy=MomPolicy.Fixed(0.0),
  lr_policy=LRPolicy.Fixed(0.001), load_from=base_dir)
solver = Solver(method, train_params)

# coffee break config
add_coffee_break(solver, TrainingSummary(), every_n_iter=1000)
add_coffee_break(solver, Snapshot(base_dir), every_n_iter=3000)
solve(solver, net)

########################################
# Klassifizierung
########################################
classes = open("/home/sbhle/Mocha.jl/examples/ijulia/ilsvrc12/synset_words.txt") do s map(x -> replace(strip(x), r"^n[0-9]+ ", ""), readlines(s)) end

require(joinpath(Pkg.dir("Mocha"), "tools/image-classifier.jl"))
classifier = ImageClassifier(net, :prob, channel_order=(3,2,1), classes=classes)
println("Classifier constructed")

# load image for prediction
img = imread("/home/sbhle/Mocha.jl/examples/ijulia/ilsvrc12/images/alpaca.jpeg")

prob, class = classify(classifier, img)
println(class)

# Plot des Ergebnisses
using Gadfly

n_plot = 5
n_best = sortperm(vec(prob), rev=true)[1:n_plot]
best_probs = prob[n_best]
best_labels = classes[n_best]

my_plot = plot(x=1:length(best_probs), y=best_probs, color=best_labels, Geom.bar, Guide.ylabel("probability"))
Pkg.add("Cairo")
using Cairo
draw(PNG("myplot.png", 12inch, 6inch), my_plot)
