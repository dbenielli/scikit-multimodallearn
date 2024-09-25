apt-get  update -y 
apt-get upgrade -y 
apt-get install -y --no-install-recommends 
apt-get install -y --no-install-recommends        python3 
apt-get install -y --no-install-recommends       python3-setuptools 
apt-get install -y --no-install-recommends        python3-sklearn
apt-get install -y --no-install-recommends        python3-pytest 
apt-get install -y --no-install-recommends        python3-pytest-cov 
apt-get install -y --no-install-recommends        python3-nose 
apt-get install -y --no-install-recommends        python3-sphinx 
apt-get install -y --no-install-recommends       python3-numpydoc 
apt-get install -y --no-install-recommends        python3-sphinx-gallery 
apt-get install -y --no-install-recommends        python3-matplotlib 
apt-get install -y --no-install-recommends        python3-pil 
apt-get install -y --no-install-recommends        python3-tk  
apt-get install -y --no-install-recommends        python3-pip 
apt-get clean
apt-get install -y --no-install-recommends locales 
apt-get clean    locale-gen en_US.UTF-8  update-locale en_US.UTF-8 
echo "export LC_ALL=$(locale -a | grep en_US)" >> /root/.bashrc 
echo "export LANG=$(locale -a | grep en_US)" >>  /root/.bashrc
pip3 install -r requirements.txt