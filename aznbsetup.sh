export PATH=/home/nbuser/anaconda3_501/bin:$PATH

wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzvf ta-lib-0.4.0-src.tar.gz
cd ta-lib
sh ./configure --prefix=/home/nbuser/talib
make
make install
export TA_LIBRARY_PATH=/home/nbuser/talib/lib
export TA_INCLUDE_PATH=/home/nbuser/talib/include
pip install --user Ta-Lib

pip3 install --user selenium
pip3 install --user kiteconnect
pip3 install --user plotly
