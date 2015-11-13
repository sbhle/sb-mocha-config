# install OpenBLAS
sudo apt-get install libopenblas-dev git unzip

# install Anaconda
wget http://09c8d0b2229f813c1b93-c95ac804525aac4b6dba79b00b39d1d3.r79.cf1.rackcdn.com/Anaconda-2.1.0-Linux-x86_64.sh
bash Anaconda-2.1.0-Linux-x86.sh

# install Boost
sudo apt-get install libboost-all-dev

# install protobuf
pip install protobuf

# install caffe
git clone https://github.com/BVLC/caffe
cp code/ImportGewichtungen/Makefile.config caffe/Makefile.config
cd caffe

make all
make test
make runtest
